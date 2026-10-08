"""Query-independent alignment of complete, masked physical anomaly profiles.

This module reads only the *observed* local profiles and an already constructed
query representation.  It never reads held-out targets.  A shared displacement
uses precisely the same sampling depth for temperature and salinity, but their
observation masks remain separate.  Linear interpolation requires two valid
adjacent endpoints; an observed endpoint itself remains valid without its
neighbor.  There is no extrapolation or interpolation across missing levels.

The required values are physical **anomalies**.  Optional observed absolute
profiles and their source climatology enable full water-column alignment;
otherwise this implements anomaly-profile alignment.  A learned geometric
shift alone is not an identified isopycnal or thermocline displacement.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn


class FullProfileAlignment(nn.Module):
    """Encode full profiles and optionally sample them at bounded shifted depths.

    ``forward(scene, h)`` returns ``(updated_scene, diagnostics)``.  The returned
    scene includes ``profile_embedding [Q,K,width]`` for the caller's local
    context path, and replaces ``local_innovation`` and ``local_valid`` when
    alignment is enabled.  ``local_features`` remains [Q,K,66].  Mode ``none``
    is the full-profile-context control: its numerical innovations and masks
    are the exact original tensors.  All modes use the same profile encoder.

    Values [Q,K,L,2] must be unnormalized temperature/salinity anomalies.  Their
    sampled values are standardized with the **query level's** training mean
    and standard deviation, not the statistics of the shifted source level.
    The same-depth standardized first guess in ``local_features[...,56:58]``
    is subtracted to preserve the existing innovation convention.
    The final shift layer is initialized to zero.  At zero shift valid input
    innovations are preserved exactly, while interpolation retains a gradient
    to the shift wherever an adjacent valid segment exists.

    If both ``local_profile_absolute`` and ``local_profile_climatology`` are
    present [Q,K,L,2], absolute T/S values are encoded and interpolated.  The
    source profile's climatology at the *unshifted* destination depth is then
    subtracted.  ``aligned_profile_values`` / ``aligned_profile_valid`` expose
    the same displacement at every native level for a later covariance update.
    """

    def __init__(self, width: int, mode: str = "none", max_shift_m: float = 100.0) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if mode not in {"none", "shared", "independent"}:
            raise ValueError("mode must be none, shared or independent")
        if not math.isfinite(max_shift_m) or max_shift_m <= 0:
            raise ValueError("max_shift_m must be finite and positive")
        self.width = width
        self.mode = mode
        self.max_shift_m = float(max_shift_m)
        # Per depth: standardized values, channel masks, physical slopes scaled
        # by a common channel scale, slope masks, and two depth coordinates.
        self.level_encoder = nn.Sequential(
            nn.Linear(10, width), nn.SiLU(), nn.Linear(width, width),
        )
        self.profile_encoder = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.shift_head = None
        if mode != "none":
            self.shift_head = nn.Sequential(
                nn.Linear(2 * width + 66 + 5, width), nn.SiLU(),
                nn.Linear(width, 1 if mode == "shared" else 2),
            )
            nn.init.zeros_(self.shift_head[-1].weight)
            nn.init.zeros_(self.shift_head[-1].bias)

    def _inputs(self, scene: Mapping[str, Tensor], h: Tensor) -> tuple[Tensor, ...]:
        if h.ndim != 2 or h.shape[-1] != self.width:
            raise ValueError(f"h must have shape [Q,{self.width}]")
        values = scene["local_profile_values"]
        if values.ndim != 4 or values.shape[0] != h.shape[0] or values.shape[-1] != 2:
            raise ValueError("local_profile_values must have shape [Q,K,L,2]")
        q, k, levels, _ = values.shape
        if levels < 1:
            raise ValueError("profiles must contain at least one depth level")
        expected = {
            "local_profile_valid": (q, k, levels, 2),
            "profile_depths": (levels,),
            "profile_normalization_mean": (levels, 2),
            "profile_normalization_std": (levels, 2),
            "query_level": (q,),
            "query_coord": (q, 4),
            "local_features": (q, k, 66),
            "local_innovation": (q, k, 2),
            "local_valid": (q, k, 2),
        }
        for key, shape in expected.items():
            if scene[key].shape != shape:
                raise ValueError(f"{key} must have shape {list(shape)}")
        # AMP is appropriate for profile embeddings and shift prediction, but
        # not for physical interpolation or subtracting salinity near 35 PSU.
        # Keep the numeric path in FP32 (or preserve FP64 reference checks).
        numeric_dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        values = values.to(device=h.device, dtype=numeric_dtype)
        depths = scene["profile_depths"].to(device=h.device, dtype=numeric_dtype)
        mean = scene["profile_normalization_mean"].to(device=h.device, dtype=numeric_dtype)
        std = scene["profile_normalization_std"].to(device=h.device, dtype=numeric_dtype)
        if not torch.isfinite(depths).all() or (depths < 0).any():
            raise ValueError("profile_depths must be finite and nonnegative")
        if levels > 1 and not (depths[1:] > depths[:-1]).all():
            raise ValueError("profile_depths must be strictly increasing")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("profile normalization must be finite, with positive std")
        query_level = scene["query_level"].to(device=h.device)
        if query_level.dtype not in (torch.int32, torch.int64):
            raise ValueError("query_level must contain integer level indices")
        if ((query_level < 0) | (query_level >= levels)).any():
            raise ValueError("query_level outside profile depth range")
        valid = scene["local_profile_valid"].to(device=h.device).bool() & torch.isfinite(values)
        # Invalid values are removed before any arithmetic.  This also makes
        # poisoned padding unable to influence profile slopes or embeddings.
        values = torch.where(valid, values, 0.0)
        std = std.clamp_min(max(torch.finfo(numeric_dtype).eps, 1e-6))
        return values, valid, depths, mean, std, query_level.long()

    def _encode_profile(self, values: Tensor, valid: Tensor, depths: Tensor,
                        mean: Tensor, std: Tensor) -> Tensor:
        q, k, levels, _ = values.shape
        normalized = torch.where(valid, (values - mean) / std, 0.0)
        slopes = torch.zeros_like(values)
        slope_valid = torch.zeros_like(valid)
        if levels > 1:
            pairs = valid[:, :, 1:] & valid[:, :, :-1]
            # Scaling with one training-derived scale per variable preserves
            # physical vertical differences despite depth-dependent stds.
            # Expanded constant scales have zero strides.  Materialize before
            # median, whose CPU implementation can mishandle such views.
            scale = std.contiguous().median(dim=0).values.clamp_min(1e-6)
            differences = values[:, :, 1:] - values[:, :, :-1]
            gradient = differences / (depths[1:] - depths[:-1])[None, None, :, None]
            slopes[:, :, :-1] = torch.where(pairs, gradient * 100.0 / scale, 0.0)
            slope_valid[:, :, :-1] = pairs
        depth_features = torch.stack((depths / 2000.0,
                                      depths.log1p() / math.log(2001.0)), dim=-1)
        inputs = torch.cat((normalized, valid.to(values.dtype), slopes,
                            slope_valid.to(values.dtype),
                            depth_features[None, None].expand(q, k, -1, -1)), dim=-1)
        encoded = self.level_encoder(inputs)
        active = valid.any(dim=-1)
        pooled = (encoded * active[..., None]).sum(dim=-2)
        pooled = pooled / active.sum(dim=-1, keepdim=True).clamp_min(1)
        embedding = self.profile_encoder(pooled)
        return torch.where(active.any(dim=-1, keepdim=True), embedding, 0.0)

    @staticmethod
    def _interpolate(values: Tensor, valid: Tensor, depths: Tensor,
                     sample_depth: Tensor) -> tuple[Tensor, Tensor]:
        """Channel-specific adjacent interpolation, with exact endpoint support."""
        q, k, levels, _ = values.shape
        single_depth = sample_depth.ndim == 3
        if single_depth:
            sample_depth = sample_depth[:, :, None]
        finite_depth = torch.isfinite(sample_depth)
        safe_depth = torch.where(finite_depth, sample_depth, depths[0])
        if levels == 1:
            supported = valid[:, :, :1] & finite_depth & (safe_depth == depths[0])
            sampled = torch.where(supported, values[:, :, :1], 0.0)
            return (sampled.squeeze(2), supported.squeeze(2)) if single_depth else (sampled, supported)
        # searchsorted selects the right-hand segment at an interior endpoint.
        # Its integer index is piecewise constant; interpolation within that
        # observed segment remains differentiable in depth and profile values.
        hi = torch.searchsorted(depths, safe_depth.contiguous(), right=True).clamp(1, levels - 1)
        lo = hi - 1
        lower_depth, upper_depth = depths[lo], depths[hi]
        lower = torch.gather(values, 2, lo)
        upper = torch.gather(values, 2, hi)
        lower_valid = torch.gather(valid, 2, lo)
        upper_valid = torch.gather(valid, 2, hi)
        within = finite_depth & (safe_depth >= depths[0]) & (safe_depth <= depths[-1])
        pair = lower_valid & upper_valid & within
        at_lower = lower_valid & within & (safe_depth == lower_depth)
        at_upper = upper_valid & within & (safe_depth == upper_depth)
        fraction = (safe_depth - lower_depth) / (upper_depth - lower_depth)
        interpolated = lower + fraction * (upper - lower)
        # Isolated observed endpoints stay valid, but have no fabricated slope.
        endpoint = torch.where(at_lower, lower, upper)
        sampled = torch.where(pair, interpolated, endpoint)
        supported = pair | at_lower | at_upper
        sampled = torch.where(supported, sampled, 0.0)
        return (sampled.squeeze(2), supported.squeeze(2)) if single_depth else (sampled, supported)

    def forward(self, scene: Mapping[str, Tensor], h: Tensor
                ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        values, valid, depths, mean, std, query_level = self._inputs(scene, h)
        q, k, levels, _ = values.shape
        absolute_keys = ("local_profile_absolute", "local_profile_climatology")
        has_absolute = all(key in scene for key in absolute_keys)
        if any(key in scene for key in absolute_keys) and not has_absolute:
            raise ValueError("absolute alignment requires both absolute profiles and source climatology")
        source_climatology = None
        sampling_values, sampling_valid = values, valid
        encoder_mean, encoder_std = mean, std
        if has_absolute:
            for key in absolute_keys:
                if scene[key].shape != values.shape:
                    raise ValueError(f"{key} must have shape [Q,K,L,2]")
            absolute = scene["local_profile_absolute"].to(values)
            source_climatology = scene["local_profile_climatology"].to(values)
            sampling_valid = valid & torch.isfinite(absolute)
            sampling_values = torch.where(sampling_valid, absolute, 0.0)
            encoder_mean = values.new_tensor([0.0, 35.0]).expand(levels, -1)
            encoder_std = values.new_tensor([10.0, 2.0]).expand(levels, -1)
        embedding = self._encode_profile(sampling_values, sampling_valid, depths,
                                         encoder_mean, encoder_std)
        coord = scene["query_coord"].to(values)
        safe_coord = torch.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
        annual = safe_coord[:, 3] * (2.0 * math.pi / 365.25)
        coord_features = torch.stack((safe_coord[:, 0] / 90.0, safe_coord[:, 1] / 180.0,
                                      safe_coord[:, 2] / 2000.0,
                                      annual.sin(), annual.cos()), dim=-1)
        if self.shift_head is None:
            shift = values.new_zeros((q, k, 2))
        else:
            local = torch.nan_to_num(scene["local_features"].to(h),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            inputs = torch.cat((h[:, None].expand(-1, k, -1), embedding, local,
                                coord_features[:, None].expand(-1, k, -1)), dim=-1)
            predicted = self.max_shift_m * self.shift_head(inputs).to(values).tanh()
            shift = predicted.expand(-1, -1, 2) if self.mode == "shared" else predicted
            # Entirely unobserved neighbor slots cannot suggest a displacement.
            shift = torch.where(valid.any(dim=(-2, -1))[..., None], shift, 0.0)
        with torch.autocast(device_type=h.device.type, enabled=False):
            sample_depth = coord[:, None, 2:3] + shift
            sampled, aligned_valid = self._interpolate(sampling_values, sampling_valid, depths, sample_depth)
            profile_depth = depths[None, None, :, None] + shift[:, :, None]
            aligned_profiles, aligned_profile_valid = self._interpolate(
                sampling_values, sampling_valid, depths, profile_depth)
            if source_climatology is not None:
                climate_valid = torch.isfinite(source_climatology)
                safe_climate = torch.where(climate_valid, source_climatology, 0.0)
                query_climate, query_climate_valid = self._interpolate(
                    safe_climate, climate_valid, depths,
                    coord[:, None, 2:3].expand(-1, k, 2))
                sampled = sampled - query_climate
                aligned_valid = aligned_valid & query_climate_valid
                aligned_profiles = aligned_profiles - safe_climate
                aligned_profile_valid = aligned_profile_valid & climate_valid
            # This identity preserves exact stored anomalies at initialization
            # while retaining gradients from the observed absolute/anomaly curve.
            profile_identity = values + (aligned_profiles - aligned_profiles.detach())
            aligned_profiles = torch.where(shift[:, :, None] == 0, profile_identity, aligned_profiles)
            aligned_profile_valid = aligned_profile_valid & valid.logical_or(shift[:, :, None] != 0)
            aligned_profiles = torch.where(aligned_profile_valid, aligned_profiles, 0.0)
            normalized = (sampled - mean[query_level, None]) / std[query_level, None]
            local_first_guess = scene["local_features"][..., 56:58].to(values)
            first_guess_valid = torch.isfinite(local_first_guess)
            normalized = normalized - torch.where(first_guess_valid, local_first_guess, 0.0)
            aligned_valid = aligned_valid & first_guess_valid
            normalized = torch.where(aligned_valid, normalized, 0.0)
        updated = dict(scene)
        updated["profile_embedding"] = embedding
        updated["aligned_profile_values"] = aligned_profiles
        updated["aligned_profile_valid"] = aligned_profile_valid
        if self.mode != "none":
            original = scene["local_innovation"].to(values)
            original_valid = scene["local_valid"].to(device=h.device).bool() & torch.isfinite(original)
            at_zero = shift == 0
            # Preserve the original zero-shift arithmetic exactly.  The added
            # term is exactly zero in value and has the interpolation gradient.
            identity = original + (normalized - normalized.detach())
            aligned_valid = aligned_valid & (~at_zero | original_valid)
            innovation = torch.where(at_zero, identity, normalized)
            updated["local_innovation"] = torch.where(aligned_valid, innovation, 0.0)
            updated["local_valid"] = aligned_valid
        else:
            # The control never changes numerical evidence, including masks.
            aligned_valid = scene["local_valid"].to(device=h.device).bool()
        count = aligned_valid.sum().clamp_min(1)
        diagnostics = {
            "profile_shift_m": shift,
            "profile_sample_depth_m": sample_depth,
            "profile_aligned_valid": aligned_valid,
            "profile_absolute_used": values.new_tensor(float(has_absolute)),
            "profile_observed_channels": valid.any(dim=-2).detach(),
            "profile_shift_abs_mean_m": ((shift.abs() * aligned_valid).sum() / count).detach(),
            "profile_aligned_coverage": aligned_valid.to(values.dtype).mean().nan_to_num().detach(),
        }
        return updated, diagnostics
