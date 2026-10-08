"""Isolated research variants; the original matched campaign stays frozen.

All corrections replace/refine the same OI anchor using the same observations.
The OI and learned local state are not treated as independent posterior evidence.
"""
from dataclasses import dataclass
from typing import Mapping, Any

import torch
from torch import nn

from .latent_ocean import LatentOceanConfig, LatentOceanModel, OceanAttention
from .profile_alignment import FullProfileAlignment
from .correlated_ocean_update import CorrelatedOceanUpdate
from .ocean_observation_operator import ObservationOperatorUpdate


RECIPES = (
    "local_control", "local_latent_off", "local_off", "profile_none",
    "profile_shared", "profile_independent", "cov_diagonal", "cov_correlated",
    "aligned_correlated", "operator_direct", "operator_update", "all_modules",
)


@dataclass(frozen=True)
class InnovationOceanConfig(LatentOceanConfig):
    recipe: str = "local_control"
    max_shift_m: float = 100.0
    covariance_rank: int = 8
    n_profile_levels: int = 20

    def __post_init__(self):
        super().__post_init__()
        if self.recipe not in RECIPES:
            raise ValueError(f"Unknown innovation recipe: {self.recipe}")
        if self.n_profile_levels < 2 or self.covariance_rank < 1 or self.max_shift_m <= 0:
            raise ValueError("Invalid profile/covariance settings")
        if self.recipe == "local_latent_off":
            object.__setattr__(self, "use_latent", False)
        if self.recipe == "local_off":
            object.__setattr__(self, "use_local", False)


class InnovationOceanModel(LatentOceanModel):
    def __init__(self, config):
        super().__init__(config)
        r, w = config.recipe, config.width
        self.alignment = None
        if r.startswith("profile_") or r in ("aligned_correlated", "all_modules"):
            mode = r.removeprefix("profile_") if r.startswith("profile_") else "shared"
            self.alignment = FullProfileAlignment(w, mode=mode, max_shift_m=config.max_shift_m)
            self.profile_reader = OceanAttention(w, config.n_heads)
            self.profile_norm = nn.LayerNorm(w)
            self.profile_null = nn.Parameter(torch.zeros(1, w))
        self.covariance = None
        if r in ("cov_diagonal", "cov_correlated", "aligned_correlated", "all_modules"):
            self.covariance = CorrelatedOceanUpdate(w, mode="diagonal" if r == "cov_diagonal" else "correlated",
                                                   rank=config.covariance_rank)
        self.operator = None
        if r in ("operator_direct", "operator_update", "all_modules") and config.n_sat_features > 0:
            self.column_projection = nn.Sequential(nn.Linear(56, w), nn.SiLU(), nn.Linear(w, w))
            self.column_reader = OceanAttention(w, config.n_heads)
            self.column_norm = nn.LayerNorm(w)
            self.column_mean = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, config.n_profile_levels * 2))
            self.column_log_variance = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, config.n_profile_levels * 2))
            self.column_gate = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 2))
            self.operator = ObservationOperatorUpdate(w, n_levels=config.n_profile_levels)
            nn.init.zeros_(self.column_mean[-1].weight)
            nn.init.zeros_(self.column_mean[-1].bias)
            nn.init.zeros_(self.column_gate[-1].weight)
            nn.init.zeros_(self.column_gate[-1].bias)

    def _local_evidence(self, scene, h, baseline_correction):
        if not self.config.use_local:
            return super()._local_evidence(scene, h, baseline_correction)
        diagnostics = {}
        if self.alignment is not None:
            scene, diagnostic = self.alignment(scene, h)
            diagnostics.update({k: v.detach() for k, v in diagnostic.items() if torch.is_tensor(v)})
            embedding = scene["profile_embedding"]
            valid = scene["local_profile_valid"].bool().any((-1, -2))
            memory = torch.cat((embedding, self.profile_null[None].expand(len(h), -1, -1)), 1)
            mask = torch.cat((valid, valid.new_ones((len(h), 1))), 1)
            bias = h.new_zeros(mask.shape).masked_fill(~mask, float("-inf"))
            h = h + self.profile_reader(self.profile_norm(h)[:, None], self.profile_norm(memory), bias)[:, 0]
            if self.covariance is not None:
                scene = {**scene, "local_profile_values": scene["aligned_profile_values"],
                         "local_profile_valid": scene["aligned_profile_valid"]}
        h, correction, stats, local_diagnostic = super()._local_evidence(scene, h, baseline_correction)
        diagnostics.update(local_diagnostic)
        if self.covariance is not None:
            candidate, variance, cov_diagnostic = self.covariance(scene, h)
            coverage = scene["local_profile_valid"].bool().any((1, 2)).to(h.dtype)
            stats = torch.cat((coverage, variance.to(h.dtype)), -1)
            gate = self.local_gate(torch.cat((h, stats), -1)).tanh() * coverage
            correction = gate * (candidate.to(h.dtype) - scene["baseline"])
            diagnostics.update({k: v.detach() for k, v in cov_diagnostic.items() if torch.is_tensor(v)})
            diagnostics["covariance_gate"] = gate.detach()
        return h, correction, stats, diagnostics

    def decode(self, scene: Mapping[str, torch.Tensor], state: Mapping[str, Any]):
        result = super().decode(scene, state)
        if self.operator is None:
            return result
        q = len(scene["query_features"])
        coord = scene["query_coord"].clone()
        coord[:, 2] = 0
        h = self.column_projection(scene["query_features"][:, :56]) + self.coordinate_projection(self.coordinate_encoding(coord))
        if self.config.use_latent:
            h = h + self.column_reader(self.column_norm(h), state["latent"])
        physical = scene["local_profile_values"]
        valid = scene["local_profile_valid"].bool() & torch.isfinite(physical)
        distance = scene["local_offsets"][..., :2].square().sum(-1) / (500.0 ** 2)
        weights = torch.exp(-0.5 * distance)[..., None, None] * valid
        coverage = weights.sum(1)
        candidate = (weights * torch.nan_to_num(physical)).sum(1) / coverage.clamp_min(1e-6)
        means, scales = scene["profile_normalization_mean"], scene["profile_normalization_std"]
        prior = torch.where(coverage > 0, (candidate - means) / scales, 0)
        prior = prior + self.column_mean(h).reshape(q, self.config.n_profile_levels, 2)
        prior_variance = 0.01 + self.column_log_variance(h).reshape_as(prior).sigmoid()
        if self.config.recipe != "operator_direct":
            column, variance, diagnostic = self.operator(prior, prior_variance, scene["surface_observation"],
                scene["surface_operator"], scene["surface_operator_offset"], scene["surface_operator_valid"], h)
            result["diagnostics"].update({k: v.detach() for k, v in diagnostic.items() if torch.is_tensor(v)})
        else:
            column, variance = prior, prior_variance
        index = torch.arange(q, device=h.device)
        level = scene["query_level"]
        gate = self.column_gate(h).tanh()
        # Signed learned refinement, not a second independent assimilation of OI.
        original = result["mean"]
        result["mean"] = original + gate * (column[index, level] - original)
        # Use this module's mean/variance together through the same bounded gate.
        result["std"] = ((1 - gate.abs()) * result["std"].square()
                         + gate.abs() * variance[index, level]).clamp_min(self.config.std_min ** 2).sqrt()
        result["diagnostics"]["operator_gate"] = gate.detach()
        return result
