"""Linear observation residual updates of a standardized T/S column.

This module implements a conditional Gaussian *mechanism*. Its uncertainty is
conditional on the supplied covariance and observation-error model; it is not a
claim that a profile/OI prior and surface products made from the same CESM2
truth constitute independent Bayesian evidence. In particular, TEOS-10 steric
height is nonlinear. An SSH row must be supplied as an explicitly linearized
Jacobian and affine offset by a caller who knows the physical reference state.
No approximate SLA formula is invented here.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ObservationOperatorUpdate(nn.Module):
    """Update a column with a small linear observation system.

    Columns have shape ``[Q,L,2]`` and flatten in depth-major T/S order.
    Observations have shape ``[Q,M]``, with ``M <= n_modalities``; the first two
    modalities are conventionally SST and SSS, and the third is an optional
    externally linearized SSH operator. ``operator`` and ``operator_offset``
    predict observations in their own standardized units, ``H @ mean + offset``.

    The covariance is ``diag(column_variance) + U @ U.T`` when a factor U with
    shape ``[Q,L*2,rank]`` is supplied. ``observation_noise`` is optional known
    *variance* in observation units, added to a positive context-dependent
    learned diagonal R. The same gain updates the mean and posterior marginal
    variance. Queries never interact. Matrix operations run in at least FP32,
    including under automatic mixed precision.
    """

    def __init__(self, width: int, n_levels: int = 20, n_modalities: int = 3,
                 *, initial_noise_variance: float = 0.25,
                 min_variance: float = 1e-6) -> None:
        super().__init__()
        if min(width, n_levels, n_modalities) <= 0:
            raise ValueError("width, n_levels and n_modalities must be positive")
        if not 0 < min_variance < initial_noise_variance:
            raise ValueError("require 0 < min_variance < initial_noise_variance")
        self.width, self.n_levels = width, n_levels
        self.n_modalities, self.min_variance = n_modalities, min_variance
        self.noise_head = nn.Linear(width, n_modalities)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.constant_(self.noise_head.bias,
                          math.log(math.expm1(initial_noise_variance - min_variance)))

    def forward(self, column_mean: Tensor, column_variance: Tensor,
                observation: Tensor, operator: Tensor, operator_offset: Tensor,
                valid: Tensor, h: Tensor, *, observation_noise: Tensor | None = None,
                column_covariance_factor: Tensor | None = None,
                ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        q, levels, variables = column_mean.shape
        if (levels, variables) != (self.n_levels, 2):
            raise ValueError("column_mean must have shape [Q,n_levels,2]")
        if column_variance.shape != column_mean.shape:
            raise ValueError("column_variance must match column_mean")
        if observation.ndim != 2 or observation.shape[0] != q:
            raise ValueError("observation must have shape [Q,M]")
        m, d = observation.shape[1], levels * 2
        if not 0 < m <= self.n_modalities:
            raise ValueError("require 0 < M <= n_modalities")
        if operator.shape != (q, m, d):
            raise ValueError("operator must have shape [Q,M,n_levels*2]")
        if operator_offset.shape != (q, m) or valid.shape != (q, m):
            raise ValueError("operator_offset and valid must have shape [Q,M]")
        if h.shape != (q, self.width):
            raise ValueError("h must have shape [Q,width]")
        if observation_noise is not None and observation_noise.shape != (q, m):
            raise ValueError("observation_noise must have shape [Q,M]")
        if column_covariance_factor is not None and (
                column_covariance_factor.ndim != 3
                or column_covariance_factor.shape[:2] != (q, d)):
            raise ValueError("column_covariance_factor must have shape [Q,L*2,rank]")
        if not torch.isfinite(column_mean).all() or not torch.isfinite(column_variance).all():
            raise ValueError("prior column mean and variance must be finite")
        if (column_variance < 0).any():
            raise ValueError("prior column variance must be nonnegative")
        if column_covariance_factor is not None and not torch.isfinite(column_covariance_factor).all():
            raise ValueError("column covariance factor must be finite")
        if not torch.isfinite(h).all():
            raise ValueError("query context must be finite")

        learned_noise = self.noise_head(h)[..., :m]
        compute_dtype = torch.float64 if column_mean.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=column_mean.device.type, enabled=False):
            mean = column_mean.flatten(1).to(compute_dtype)
            diagonal = column_variance.flatten(1).to(compute_dtype)
            covariance = torch.diag_embed(diagonal)
            if column_covariance_factor is not None:
                factor = column_covariance_factor.to(compute_dtype)
                covariance = covariance + factor @ factor.transpose(-1, -2)

            usable = valid.bool() & torch.isfinite(observation)
            usable = usable & torch.isfinite(operator_offset) & torch.isfinite(operator).all(-1)
            noise = F.softplus(learned_noise.to(compute_dtype)) + self.min_variance
            if observation_noise is not None:
                if (observation_noise[usable] < 0).any():
                    raise ValueError("provided observation variance must be nonnegative")
                usable = usable & torch.isfinite(observation_noise)
                supplied_noise = torch.where(usable, observation_noise, 0.).to(compute_dtype)
                noise = noise + supplied_noise

            # Missing modalities become independent identity rows with zero H
            # and zero innovation, so they cannot alter any available row.
            H = torch.where(usable[..., None], operator, 0.).to(compute_dtype)
            offset = torch.where(usable, operator_offset, 0.).to(compute_dtype)
            y = torch.where(usable, observation, 0.).to(compute_dtype)
            predicted = (H @ mean[..., None]).squeeze(-1) + offset
            innovation = torch.where(usable, y - predicted, 0.)
            cross = covariance @ H.transpose(-1, -2)
            r = torch.where(usable, noise, torch.ones_like(noise))
            system = H @ cross + torch.diag_embed(r)
            gain = torch.linalg.solve(system, cross.transpose(-1, -2)).transpose(-1, -2)
            correction = (gain @ innovation[..., None]).squeeze(-1)
            posterior_mean = mean + correction
            prior_marginal = covariance.diagonal(dim1=-2, dim2=-1)
            reduction = (gain * cross).sum(-1)
            posterior_variance = (prior_marginal - reduction).clamp_min(0.)

            # The explicit branch also preserves original low-precision values
            # bit-for-bit where the query received no available observations.
            any_valid = usable.any(-1)
            posterior_mean = torch.where(any_valid[:, None], posterior_mean, mean)
            posterior_variance = torch.where(any_valid[:, None], posterior_variance, prior_marginal)
            diagnostics = {
                "predicted_observation": predicted,
                "innovation": innovation,
                "observation_variance": torch.where(usable, noise, 0.),
                "valid": usable,
                "gain": gain,
                "mean_correction": correction.reshape(q, levels, 2),
                "variance_reduction": reduction.reshape(q, levels, 2),
                "prior_marginal_variance": prior_marginal.reshape(q, levels, 2),
            }
        # Preserve FP32 posterior variances under AMP: casting tiny variances
        # to FP16 could create zero variance in the likelihood calculation.
        return (posterior_mean.reshape(q, levels, 2).to(column_mean.dtype),
                posterior_variance.reshape(q, levels, 2), diagnostics)


def build_surface_observation_operator(
        profile_mean: Tensor, profile_scale: Tensor,
        surface_mean: Tensor, surface_scale: Tensor, *,
        n_levels: int | None = None, n_queries: int | None = None,
        surface_depth_weights: Tensor | None = None,
        profile_climatology: Tensor | None = None,
        surface_climatology: Tensor | None = None,
        ssh_jacobian: Tensor | None = None,
        ssh_offset: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
    """Construct exact affine SST/SSS operators in standardized units.

    ``profile_mean/scale`` are ``[L,2]`` or ``[Q,L,2]``. Surface statistics
    are ``[2 or 3]`` or ``[Q,2 or 3]``. Climatologies, when provided, have the
    corresponding shapes (``surface_climatology`` may contain only SST/SSS).
    The default operator reads the first native column level, matching the
    benchmark's CESM2 5 m SST/SSS; it does not extrapolate to 0 m. Optional
    ``surface_depth_weights [L] or [Q,L]`` define a linear depth measurement.

    For x standardized as ``(physical-profile_climatology-profile_mean)/scale``
    and y as ``(physical_surface-surface_climatology-surface_mean)/surface_scale``,
    this helper includes their potentially different climatologies in offset.
    Omitted climatologies mean *both measurements use the same zero anomaly
    basis*. Callers must verify that assumption; neither interpolation nor
    equality of two independently prepared climatologies is inferred here.

    SSH stays unavailable unless both ``ssh_jacobian [Q,L*2]`` and
    ``ssh_offset [Q]`` are supplied. Those are already in standardized units
    and describe a caller-specified local linearization, never exact TEOS-10.
    Returned validity describes operator availability; combine it with the
    actual observation availability before the update.
    """
    if profile_mean.ndim not in (2, 3) or profile_mean.shape[-1] != 2:
        raise ValueError("profile_mean must have shape [L,2] or [Q,L,2]")
    levels = profile_mean.shape[-2]
    if n_levels is not None and n_levels != levels:
        raise ValueError("n_levels differs from profile statistics")
    query_candidates = [x.shape[0] for x in (
        profile_mean, profile_scale, profile_climatology) if x is not None and x.ndim == 3]
    query_candidates += [x.shape[0] for x in (
        surface_mean, surface_scale, surface_climatology, surface_depth_weights,
        ssh_jacobian) if x is not None and x.ndim == 2]
    if ssh_offset is not None and ssh_offset.ndim == 1:
        query_candidates.append(ssh_offset.shape[0])
    q = n_queries if n_queries is not None else (query_candidates[0] if query_candidates else 1)
    if q < 0 or any(candidate != q for candidate in query_candidates):
        raise ValueError("inconsistent query dimension in operator statistics")
    device, dtype = profile_mean.device, profile_mean.dtype

    def expand_profile(x: Tensor, name: str) -> Tensor:
        if x.shape not in ((levels, 2), (q, levels, 2)):
            raise ValueError(f"{name} must have shape [L,2] or [Q,L,2]")
        return x.to(device=device, dtype=dtype).expand(q, levels, 2)

    def expand_surface(x: Tensor, name: str) -> Tensor:
        if x.ndim not in (1, 2) or x.shape[-1] not in (2, 3):
            raise ValueError(f"{name} must contain two or three surface statistics")
        if x.ndim == 2 and x.shape[0] != q:
            raise ValueError(f"{name} has inconsistent query dimension")
        return x[..., :2].to(device=device, dtype=dtype).expand(q, 2)

    mu = expand_profile(profile_mean, "profile_mean")
    scale = expand_profile(profile_scale, "profile_scale")
    smu, sscale = expand_surface(surface_mean, "surface_mean"), expand_surface(surface_scale, "surface_scale")
    if not torch.isfinite(scale).all() or not torch.isfinite(sscale).all() or (scale <= 0).any() or (sscale <= 0).any():
        raise ValueError("normalization scales must be finite and positive")
    if (profile_climatology is None) != (surface_climatology is None):
        raise ValueError("supply both profile and surface climatologies or neither")
    if profile_climatology is not None:
        mu = mu + expand_profile(profile_climatology, "profile_climatology")
        smu = smu + expand_surface(surface_climatology, "surface_climatology")
    if surface_depth_weights is None:
        weights = torch.zeros(q, levels, device=device, dtype=dtype)
        weights[:, 0] = 1.
    else:
        if surface_depth_weights.shape not in ((levels,), (q, levels)):
            raise ValueError("surface_depth_weights must have shape [L] or [Q,L]")
        weights = surface_depth_weights.to(device=device, dtype=dtype).expand(q, levels)
        if not torch.isfinite(weights).all():
            raise ValueError("surface depth weights must be finite")

    H = torch.zeros(q, 3, levels, 2, device=device, dtype=dtype)
    H[:, 0, :, 0] = weights * scale[:, :, 0] / sscale[:, 0, None]
    H[:, 1, :, 1] = weights * scale[:, :, 1] / sscale[:, 1, None]
    offset = torch.zeros(q, 3, device=device, dtype=dtype)
    # Unused deep levels may have missing climatology. Zero depth support
    # must remove them before multiplication so 0 * NaN cannot hide valid SST.
    supported_mu = torch.where(weights[..., None] != 0., mu, 0.)
    offset[:, :2] = ((weights[..., None] * supported_mu).sum(1) - smu) / sscale
    available = torch.zeros(q, 3, device=device, dtype=torch.bool)
    available[:, :2] = torch.isfinite(offset[:, :2])
    if (ssh_jacobian is None) != (ssh_offset is None):
        raise ValueError("SSH requires both a standardized Jacobian and offset")
    if ssh_jacobian is not None:
        if ssh_jacobian.shape != (q, levels * 2) or ssh_offset.shape != (q,):
            raise ValueError("SSH Jacobian/offset must have shape [Q,L*2] and [Q]")
        H[:, 2] = ssh_jacobian.to(device=device, dtype=dtype).reshape(q, levels, 2)
        offset[:, 2] = ssh_offset.to(device=device, dtype=dtype)
        available[:, 2] = torch.isfinite(ssh_jacobian).all(-1) & torch.isfinite(ssh_offset)
    return H.flatten(2), offset, available
