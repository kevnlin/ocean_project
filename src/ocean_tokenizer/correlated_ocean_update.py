"""PSD-consistent, complete-profile numerical updates for a local ocean query.

This is an isolated experimental component, not a replacement of the frozen OI
anchor. ``forward(scene, h)`` returns a *candidate standardized anomaly*, a
conditional covariance diagnostic, and diagnostics. The caller decides how to
combine that candidate with its anchor. The variance is learned covariance
under this local Gaussian model; it is not a calibrated full-model uncertainty.

For every query, observation values are expressed in a common unit using
``(physical_anomaly[l] - training_mean[l]) / training_std[query_level]``.
Subtracting the observation-level mean preserves the zero standardized prior;
scaling by the query-level standard deviation permits cross-depth covariance.
No target values, first guesses, or frozen-anchor values enter this update.

The state covariance is B = F F^T, with an independent query-only nugget. Query
and observation factors use the same context-conditioned feature map. R has
positive independent noise, depth-common T/S biases per profile identity, and
a common-source component. Its inverse is applied analytically, followed by a
rank-by-rank Woodbury solve. No dense (K * L * 2)-square matrix is constructed.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CorrelatedOceanUpdate(nn.Module):
    """Condition a rank-limited local state on complete nearby profiles.

    Required scene keys are ``local_profile_values/valid [Q,K,L,2]``,
    ``profile_depths [L]``, ``profile_normalization_mean/std [L,2]``,
    ``query_level [Q]``, ``local_offsets [Q,K,4]`` (east/north km, days,
    standardized SLA difference), ``local_profile_ids [Q,K]`` and
    ``query_coord [Q,4]``. Profile IDs identify records, not merely a float;
    reusing one ID for distinct measurements incorrectly marks them redundant.
    Optional ``local_profile_noise_variance [Q,K,L,2]`` adds a supplied physical
    measurement variance to the learned independent noise. It must be finite
    and nonnegative at valid observations.

    Correlated mode normalizes the precision of repeated IDs at each depth and
    channel, making exact record duplication invariant. Diagonal mode treats
    each record as independent and disables both correlated R components. Both
    modes retain the same PSD state covariance and parameter count.
    """

    def __init__(self, width: int, mode: str = "correlated", rank: int = 8):
        super().__init__()
        if width <= 0 or rank <= 0:
            raise ValueError("width and rank must be positive")
        if mode not in {"diagonal", "correlated"}:
            raise ValueError("mode must be diagonal or correlated")
        self.width, self.mode, self.rank = width, mode, rank
        self.feature_projection = nn.Linear(10, rank)
        # rank gates, state amplitude, nugget(2), noise(2), profile bias(2),
        # and common-source variance. Bounded positive quantities avoid
        # singular covariances even before training.
        self.context = nn.Linear(width, rank + 8)
        self.log_scales = nn.Parameter(torch.tensor([500., 500., 30., 1., 500.]).log())
        with torch.no_grad():
            self.context.weight.zero_()
            self.context.bias.zero_()
            self.context.bias[rank:] = torch.tensor([
                self._logit(1., .05, 4.),
                self._logit(.05, 1e-4, 1.), self._logit(.05, 1e-4, 1.),
                self._logit(.16, 1e-4, 4.), self._logit(.16, 1e-4, 4.),
                self._logit(.08, 1e-5, 2.), self._logit(.08, 1e-5, 2.),
                self._logit(.02, 1e-5, 1.),
            ])

    @staticmethod
    def _logit(value: float, low: float, high: float) -> float:
        probability = (value - low) / (high - low)
        return math.log(probability / (1. - probability))

    @staticmethod
    def _bounded(value: Tensor, low: float, high: float) -> Tensor:
        return low + (high - low) * value.sigmoid()

    @staticmethod
    def _linear(layer: nn.Linear, value: Tensor) -> Tensor:
        # Explicit FP32/64 projection also supports a manually half-cast module.
        return F.linear(value, layer.weight.to(value.dtype), layer.bias.to(value.dtype))

    @staticmethod
    def _groups(ids: Tensor) -> Tensor:
        """Dense identity indices [Q,K], without an O(K^2) equality matrix."""
        q, k = ids.shape
        if not k:
            return ids.clone()
        order = ids.argsort(dim=1, stable=True)
        sorted_ids = ids.gather(1, order)
        first = torch.ones((q, 1), device=ids.device, dtype=torch.bool)
        changes = torch.cat((first, sorted_ids[:, 1:] != sorted_ids[:, :-1]), 1)
        sorted_group = changes.long().cumsum(1) - 1
        return torch.empty_like(ids).scatter(1, order, sorted_group)

    def _factors(self, offset: Tensor, depths: Tensor, query_depth: Tensor,
                 gates: Tensor, state_variance: Tensor) -> Tensor:
        """Shared factor map for both query and observation locations."""
        q, k, l, _ = offset.shape
        scales = self.log_scales.to(offset.dtype).clamp(math.log(.1), math.log(1e5)).exp()
        geo = offset / scales[:4]
        relative = (depths - query_depth[:, None, None]) / scales[4]
        absolute = depths / 2000.
        proximity = (-.5 * geo[..., :3].square().sum(-1)).exp()
        vertical = (-.5 * relative.square()).exp()
        base = torch.cat((geo, relative[..., None], absolute[..., None],
                          proximity[..., None], vertical[..., None]), -1)
        base = base[..., None, :].expand(q, k, l, 2, 8)
        variable = torch.eye(2, device=offset.device, dtype=offset.dtype)
        variable = variable[None, None, None].expand(q, k, l, 2, 2)
        features = torch.cat((base, variable), -1)
        factors = self._linear(self.feature_projection, features).tanh()
        amplitude = (state_variance / self.rank).sqrt()
        # The same envelope is evaluated at all state locations, preserving PSD.
        envelope = (proximity * vertical)[..., None, None]
        return factors * envelope * gates[:, None, None, None] * amplitude[:, None, None, None, None]

    @staticmethod
    def _inverse_error(rhs: Tensor, precision: Tensor, groups: Tensor,
                       profile_variance: Tensor, common_variance: Tensor) -> Tensor:
        """Apply R^-1 by profile/channel blocks and a common rank-one update.

        rhs [Q,K,L,2,M], precision [Q,K,L,2], groups [Q,K]. Padding has
        zero precision and therefore never contributes to a matrix solve.
        """
        q, k, _, _, m = rhs.shape
        weighted = rhs * precision[..., None]
        group_index = groups[..., None].expand(q, k, 2)
        summed_precision = precision.new_zeros((q, k, 2)).scatter_add(
            1, group_index, precision.sum(2))
        summed_rhs = rhs.new_zeros((q, k, 2, m)).scatter_add(
            1, group_index[..., None].expand(q, k, 2, m), weighted.sum(2))
        coefficient = profile_variance[:, None] / (
            1. + profile_variance[:, None] * summed_precision)
        grouped_correction = (coefficient[..., None] * summed_rhs).gather(
            1, group_index[..., None].expand(q, k, 2, m))
        inverse_rhs = weighted - precision[..., None] * grouped_correction[:, :, None]
        # R0^-1 1 is particularly simple inside each constant-bias block.
        denominator = (1. + profile_variance[:, None] * summed_precision).gather(
            1, group_index)
        inverse_one = precision / denominator[:, :, None]
        common_coefficient = common_variance / (
            1. + common_variance * inverse_one.sum((1, 2, 3)))
        return inverse_rhs - inverse_one[..., None] * (
            common_coefficient[:, None, None, None, None]
            * inverse_rhs.sum((1, 2, 3))[:, None, None, None])

    @staticmethod
    def _cholesky(system: Tensor) -> tuple[Tensor, Tensor]:
        """Normally exact; report numerical ridge for failed rounded systems.

        Mathematically the matrix is I + F^T R^-1 F. Very large evidence
        precision can round away that identity in FP32. Only failed queries
        receive a small scale-relative ridge; it is exposed in diagnostics.
        """
        factor, info = torch.linalg.cholesky_ex(system)
        jitter = system.new_zeros(system.shape[0])
        eye = torch.eye(system.shape[-1], device=system.device, dtype=system.dtype)[None]
        scale = system.diagonal(dim1=-2, dim2=-1).abs().amax(-1).detach().clamp_min(1.)
        base = scale * (8. * torch.finfo(system.dtype).eps)
        for retry in range(3):
            failed = info != 0
            if not failed.any():
                return factor, jitter
            jitter = torch.where(failed, base * (10. ** retry), jitter)
            factor, info = torch.linalg.cholesky_ex(system + jitter[:, None, None] * eye)
        if (info != 0).any():
            raise RuntimeError("local covariance Cholesky failed after numerical regularization")
        return factor, jitter

    def forward(self, scene: Mapping[str, Tensor], h: Tensor
                ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if h.ndim != 2 or h.shape[1] != self.width:
            raise ValueError("h must have shape [Q,width]")
        values = scene["local_profile_values"]
        if values.ndim != 4 or values.shape[0] != h.shape[0] or values.shape[-1] != 2:
            raise ValueError("local_profile_values must have shape [Q,K,L,2]")
        q, k, l, _ = values.shape
        required_shapes = {
            "local_profile_valid": (q, k, l, 2), "profile_depths": (l,),
            "profile_normalization_mean": (l, 2),
            "profile_normalization_std": (l, 2), "query_level": (q,),
            "local_offsets": (q, k, 4), "local_profile_ids": (q, k),
            "query_coord": (q, 4),
        }
        for key, shape in required_shapes.items():
            if tuple(scene[key].shape) != shape:
                raise ValueError(f"{key} must have shape {shape}")
        levels, ids = scene["query_level"], scene["local_profile_ids"]
        if levels.dtype not in (torch.int32, torch.int64) or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_level and local_profile_ids must be integer tensors")
        if q and ((levels < 0).any() or (levels >= l).any()):
            raise ValueError("query_level is outside profile_depths")
        normalization_mean, normalization_std = (scene["profile_normalization_mean"],
                                                   scene["profile_normalization_std"])
        if (not torch.isfinite(normalization_mean).all()
                or not torch.isfinite(normalization_std).all()
                or (normalization_std <= 0).any()):
            raise ValueError("normalization must be finite with strictly positive std")
        if not torch.isfinite(scene["query_coord"]).all() or not torch.isfinite(h).all():
            raise ValueError("query coordinates and context must be finite")
        if not torch.isfinite(scene["profile_depths"]).all():
            raise ValueError("profile depths must be finite")

        dtype = torch.float64 if h.dtype == torch.float64 or self.context.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=h.device.type, enabled=False):
            context = self._linear(self.context, h.to(dtype))
            gates = 2. * context[:, :self.rank].sigmoid()
            parameters = context[:, self.rank:]
            state_variance = self._bounded(parameters[:, 0], .05, 4.)
            nugget = self._bounded(parameters[:, 1:3], 1e-4, 1.)
            noise = self._bounded(parameters[:, 3:5], 1e-4, 4.)
            profile_variance = self._bounded(parameters[:, 5:7], 1e-5, 2.)
            common_variance = self._bounded(parameters[:, 7], 1e-5, 1.)
            if self.mode == "diagonal":
                profile_variance = torch.zeros_like(profile_variance)
                common_variance = torch.zeros_like(common_variance)

            query_depth = scene["query_coord"][:, 2].to(dtype)
            depths = scene["profile_depths"].to(dtype)[None, None].expand(q, k, l)
            offsets = scene["local_offsets"].to(dtype)
            valid = (scene["local_profile_valid"].bool() & torch.isfinite(values)
                     & torch.isfinite(offsets).all(-1)[:, :, None, None])
            clean_offset = torch.nan_to_num(offsets, nan=0., posinf=0., neginf=0.)
            obs_factor = self._factors(clean_offset[:, :, None].expand(q, k, l, 4),
                                       depths, query_depth, gates, state_variance)
            obs_factor = torch.where(valid[..., None], obs_factor, 0.)
            query_factor = self._factors(h.new_zeros((q, 1, 1, 4), dtype=dtype),
                                         query_depth[:, None, None], query_depth,
                                         gates, state_variance)[:, 0, 0]
            query_std = normalization_std.to(dtype)[levels.long()]
            # Depth-specific training means, shared query-level units.
            clean_value = torch.where(valid, values.to(dtype), normalization_mean.to(dtype)[None, None])
            evidence = (clean_value - normalization_mean.to(dtype)[None, None]) / query_std[:, None, None]
            evidence = torch.where(valid, evidence, 0.)
            variance = noise[:, None, None].expand(q, k, l, 2)
            if "local_profile_noise_variance" in scene:
                supplied_noise = scene["local_profile_noise_variance"]
                if supplied_noise.shape != values.shape:
                    raise ValueError("local_profile_noise_variance must match profile values")
                if (valid & (~torch.isfinite(supplied_noise) | (supplied_noise < 0))).any():
                    raise ValueError("supplied valid-observation noise variance must be finite and nonnegative")
                variance = variance + torch.where(valid, supplied_noise.to(dtype), 0.) / query_std[:, None, None].square()
            precision = valid.to(dtype) / variance
            groups = self._groups(ids.long())
            if self.mode == "correlated" and k:
                element_groups = groups[:, :, None, None].expand(q, k, l, 2)
                multiplicity = precision.new_zeros((q, k, l, 2)).scatter_add(
                    1, element_groups, valid.to(dtype)).gather(1, element_groups)
                precision = precision / multiplicity.clamp_min(1.)

            rhs = torch.cat((obs_factor, evidence[..., None]), -1)
            inverse_rhs = self._inverse_error(rhs, precision, groups,
                                              profile_variance, common_variance)
            factors = obs_factor.reshape(q, k * l * 2, self.rank)
            inverse_factors = inverse_rhs[..., :self.rank].reshape(q, k * l * 2, self.rank)
            inverse_evidence = inverse_rhs[..., self.rank].reshape(q, k * l * 2, 1)
            eye = torch.eye(self.rank, device=h.device, dtype=dtype)[None]
            system = eye + factors.transpose(1, 2) @ inverse_factors
            # Suppress asymmetric rounding; the analytic matrix is I + F^T R^-1 F.
            system = .5 * (system + system.transpose(1, 2))
            cholesky, solve_jitter = self._cholesky(system)
            latent_mean = torch.cholesky_solve(factors.transpose(1, 2) @ inverse_evidence, cholesky)
            candidate = (query_factor @ latent_mean).squeeze(-1)
            posterior_query = torch.cholesky_solve(query_factor.transpose(1, 2), cholesky)
            posterior_variance = (query_factor * posterior_query.transpose(1, 2)).sum(-1) + nugget
            prior_variance = query_factor.square().sum(-1) + nugget
            # Nonnegative by construction, with a final roundoff guard.
            posterior_variance = posterior_variance.clamp_min(1e-4)
            group_active = torch.zeros((q, k), device=h.device, dtype=torch.long).scatter_add(
                1, groups, valid.any((2, 3)).long())
            diagnostics = {
                "prior_variance": prior_variance,
                "variance_reduction": (prior_variance - posterior_variance).clamp_min(0.),
                "observation_count": valid.sum((1, 2, 3)),
                "unique_profile_count": (group_active > 0).sum(1),
                "independent_noise_variance": noise,
                "profile_bias_variance": profile_variance,
                "common_source_variance": common_variance,
                "query_nugget_variance": nugget,
                "solve_jitter": solve_jitter,
            }
        return candidate.to(h.dtype), posterior_variance.to(h.dtype), diagnostics
