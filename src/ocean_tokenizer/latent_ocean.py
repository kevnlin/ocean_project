"""Ocean observation-to-latent models anchored to an existing reconstruction.

The public contract is an unbatched *scene*: encode its observation set once,
then independently decode as many query chunks as desired.  Argo records are
individual depth measurements, never depth-band averages.  A frozen or jointly
trained background/interpolation prediction is supplied by the caller, and an
explicit local path continues to read the original numerical innovations.

``dense`` uses an ordinary latent Transformer. ``soft_moe`` replaces its
feed-forward blocks with the dispatch/combine construction of Puigcerver et al.,
ICLR 2024 (https://arxiv.org/abs/2308.00951). ``local_transformer`` additionally
contextualizes each query's local observation set.  None of these variants
allows different target queries to attend to one another.

Coordinates are [latitude degrees, longitude degrees, depth metres, days].
Local offsets are [east km, north km, elapsed days, standardized SLA difference].
Features are supplied by the data pipeline, so this module does not create a
second, potentially inconsistent normalization or satellite lookup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LatentOceanConfig:
    n_obs_features: int
    n_query_features: int
    n_local_features: int
    n_sat_features: int = 56
    variant: str = "dense"
    width: int = 128
    n_latents: int = 64
    n_heads: int = 4
    n_blocks: int = 4
    n_query_blocks: int = 2
    n_experts: int = 4
    slots_per_expert: int = 2
    mlp_ratio: float = 3.0
    std_min: float = 0.03
    std_max: float = 3.0
    initial_std: float = 0.6
    use_latent: bool = True
    use_local: bool = True

    def __post_init__(self) -> None:
        if self.variant not in {"dense", "soft_moe", "local_transformer"}:
            raise ValueError(f"unknown ocean latent variant: {self.variant}")
        if min(self.n_obs_features, self.n_query_features, self.n_local_features,
               self.n_sat_features) < 0:
            raise ValueError("feature dimensions must be nonnegative")
        if self.width <= 0 or self.n_heads <= 0 or self.width % self.n_heads:
            raise ValueError("width must be positive and divisible by n_heads")
        if min(self.n_latents, self.n_blocks, self.n_query_blocks,
               self.n_experts, self.slots_per_expert) <= 0:
            raise ValueError("latent, block, expert and slot counts must be positive")
        if not 0 < self.std_min < self.initial_std < self.std_max:
            raise ValueError("require 0 < std_min < initial_std < std_max")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(x: Tensor) -> Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class OceanCoordinateEncoding(nn.Module):
    """Spherical geography, depth scales and cyclic absolute/relative dates."""

    out_features = 42

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(4), persistent=False)

    def forward(self, coord: Tensor) -> Tensor:
        if coord.ndim < 2 or coord.shape[-1] != 4:
            raise ValueError("coordinates must end in [lat, lon, depth, day]")
        c = _finite(coord)
        lat = c[..., 0] * (math.pi / 180.0)
        lon = c[..., 1] * (math.pi / 180.0)
        xyz = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(),
                           lat.sin()), dim=-1)
        geo_phase = xyz[..., :, None] * self.frequencies * math.pi
        geo = torch.cat((xyz, geo_phase.sin().flatten(-2),
                         geo_phase.cos().flatten(-2)), dim=-1)
        depth = c[..., 2].clamp(min=0.0, max=12000.0)
        depth_scale = torch.stack((depth / 2000.0,
                                   torch.log1p(depth) / math.log(2001.0)), dim=-1)
        depth_phase = depth[..., None] / 2000.0 * self.frequencies * math.pi
        dep = torch.cat((depth_scale, depth_phase.sin(), depth_phase.cos()), dim=-1)
        # No raw epoch-day feature: its magnitude depends on the dataset epoch.
        annual = c[..., 3] * (2.0 * math.pi / 365.25)
        monthly = c[..., 3] * (2.0 * math.pi / 30.4375)
        time = torch.stack((annual.sin(), annual.cos(), monthly.sin(),
                            monthly.cos(), torch.ones_like(annual)), dim=-1)
        return torch.cat((geo, dep, time), dim=-1)


class OceanAttention(nn.Module):
    """Cross-attention with optional additive key prior; supports batched locals."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = width // n_heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query: Tensor, memory: Tensor,
                key_bias: Tensor | None = None) -> Tensor:
        unbatched = query.ndim == 2
        if unbatched:
            query, memory = query[None], memory[None]
            if key_bias is not None:
                key_bias = key_bias[None]
        batch, nq, width = query.shape
        nk = memory.shape[1]
        if nq == 0:
            return query[0] if unbatched else query
        q = self.q(query).reshape(batch, nq, self.n_heads, self.head_width).transpose(1, 2)
        k = self.k(memory).reshape(batch, nk, self.n_heads, self.head_width).transpose(1, 2)
        v = self.v(memory).reshape(batch, nk, self.n_heads, self.head_width).transpose(1, 2)
        bias = None if key_bias is None else key_bias[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = self.output(out.transpose(1, 2).reshape(batch, nq, width))
        return out[0] if unbatched else out


def _ffn(width: int, ratio: float) -> nn.Sequential:
    hidden = int(width * ratio)
    return nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))


class SoftMoE(nn.Module):
    """Original Soft MoE: dispatch over tokens, combine over expert slots.

    Experts process pooled slots, not top-k individual tokens. All experts are
    evaluated; no token-dropping, sparse-compute, or evidence-invariance claim
    is implied. The layer is used only in the shared context representation.
    """

    def __init__(self, width: int, n_experts: int, slots_per_expert: int,
                 mlp_ratio: float) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.slots_per_expert = slots_per_expert
        self.router = nn.Parameter(torch.randn(width, n_experts * slots_per_expert) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.experts = nn.ModuleList(_ffn(width, mlp_ratio) for _ in range(n_experts))

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = (F.normalize(x, dim=-1) @ F.normalize(self.router, dim=0)
                  * self.log_scale.clamp(-4.0, 4.0).exp())
        dispatch = logits.softmax(dim=-2)   # each slot pools a distribution of tokens
        combine = logits.softmax(dim=-1)   # each token reads a distribution of slots
        slots = dispatch.transpose(-2, -1) @ x
        slots = slots.reshape(self.n_experts, self.slots_per_expert, x.shape[-1])
        processed = torch.stack([expert(slots[i])
                                 for i, expert in enumerate(self.experts)], dim=0)
        out = combine @ processed.flatten(0, 1)
        usage = combine.reshape(-1, self.n_experts, self.slots_per_expert).sum(-1).mean(0)
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        return out, {"expert_usage": usage.detach(), "expert_entropy": entropy.detach()}


class LatentBlock(nn.Module):
    def __init__(self, config: LatentOceanConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.width)
        self.attention = OceanAttention(config.width, config.n_heads)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = (SoftMoE(config.width, config.n_experts, config.slots_per_expert,
                            config.mlp_ratio) if config.variant == "soft_moe"
                    else _ffn(config.width, config.mlp_ratio))

    def forward(self, latent: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        h = self.attn_norm(latent)
        latent = latent + self.attention(h, h)
        if isinstance(self.ffn, SoftMoE):
            update, diagnostics = self.ffn(self.ffn_norm(latent))
        else:
            update, diagnostics = self.ffn(self.ffn_norm(latent)), {}
        return latent + update, diagnostics


class IndependentQueryBlock(nn.Module):
    def __init__(self, config: LatentOceanConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.width)
        self.memory_norm = nn.LayerNorm(config.width)
        self.attention = OceanAttention(config.width, config.n_heads)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = _ffn(config.width, config.mlp_ratio)

    def forward(self, query: Tensor, latent: Tensor) -> Tensor:
        query = query + self.attention(self.query_norm(query), self.memory_norm(latent))
        return query + self.ffn(self.ffn_norm(query))


class LocalSetTransformer(nn.Module):
    """Independent local neighborhoods; no attention across target queries."""

    def __init__(self, config: LatentOceanConfig) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(config.width)
        self.local_attention = OceanAttention(config.width, config.n_heads)
        self.ffn_norm = nn.LayerNorm(config.width)
        self.ffn = _ffn(config.width, config.mlp_ratio)
        self.query_norm = nn.LayerNorm(config.width)
        self.read_norm = nn.LayerNorm(config.width)
        self.read_attention = OceanAttention(config.width, config.n_heads)

    def forward(self, query: Tensor, local: Tensor,
                mask: Tensor) -> tuple[Tensor, Tensor]:
        bias = torch.zeros_like(mask, dtype=local.dtype).masked_fill(~mask, float("-inf"))
        h = self.local_norm(local)
        local = local + self.local_attention(h, h, bias)
        local = local + self.ffn(self.ffn_norm(local))
        update = self.read_attention(self.query_norm(query)[:, None],
                                     self.read_norm(local), bias)[:, 0]
        return query + update, local


class LatentOceanModel(nn.Module):
    """Depth-preserving multimodal latent with anchored mean and uncertainty.

    Required context keys: ``obs_features [N,Cobs]``, ``obs_coord [N,4]``,
    ``obs_innovation [N,2]``, ``obs_valid [N,2]``. Optional surface tokens use
    ``satellite_features [S,Csat]``, ``satellite_coord [S,4]`` and optional
    ``satellite_valid [S]``. Observation features must not contain unmasked
    measured values: the explicit innovation input is masked inside the model.

    Required query keys: ``query_features [Q,Cq]``, ``query_coord [Q,4]``,
    ``baseline [Q,2]`` and ``query_background [Q,2]``. Optional local evidence
    requires all four ``local_features [Q,K,Clocal]``, ``local_offsets [Q,K,4]``,
    ``local_innovation [Q,K,2]``, ``local_valid [Q,K,2]`` keys.

    The initial mean equals ``baseline`` exactly. An unconstrained learned
    global residual and a bounded learned local correction then refine it.
    Positive uncertainty is predicted in the same standardized anomaly units.
    Model configuration plus the ordinary state_dict fully specifies replay.
    """

    def __init__(self, config: LatentOceanConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = LatentOceanConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass a config or keyword configuration, not both")
        self.config = config
        w = config.width
        self.coordinate_encoding = OceanCoordinateEncoding()
        self.coordinate_projection = nn.Linear(OceanCoordinateEncoding.out_features, w)
        self.observation_projection = nn.Sequential(nn.Linear(config.n_obs_features, w),
                                                     nn.SiLU(), nn.Linear(w, w))
        self.innovation_encoders = nn.ModuleList(nn.Sequential(nn.Linear(2, w), nn.SiLU(),
                                                                nn.Linear(w, w)) for _ in range(2))
        self.satellite_projection = (nn.Sequential(nn.Linear(config.n_sat_features, w), nn.SiLU(),
                                                   nn.Linear(w, w)) if config.n_sat_features else None)
        self.modality_embedding = nn.Parameter(torch.randn(2, w) * 0.02)
        self.null_observation = nn.Parameter(torch.randn(1, w) * 0.02)
        self.memory_norm = nn.LayerNorm(w)
        self.latent = nn.Parameter(torch.randn(config.n_latents, w) * 0.02)
        self.register_buffer("latent_coord", self._latent_coordinates(config.n_latents))
        self.evidence_norms = nn.ModuleList(nn.LayerNorm(w) for _ in range(config.n_blocks))
        self.evidence_attention = nn.ModuleList(OceanAttention(w, config.n_heads)
                                                for _ in range(config.n_blocks))
        self.evidence_gates = nn.Parameter(torch.full((config.n_blocks,), -1.0))
        self.latent_blocks = nn.ModuleList(LatentBlock(config) for _ in range(config.n_blocks))
        self.query_projection = nn.Sequential(nn.Linear(config.n_query_features + 4, w),
                                               nn.SiLU(), nn.Linear(w, w))
        self.query_blocks = nn.ModuleList(IndependentQueryBlock(config)
                                          for _ in range(config.n_query_blocks))
        self.output_norm = nn.LayerNorm(w)
        # Separate T and S nonlinear heads after a shared ocean representation.
        self.mean_head = nn.ModuleList(nn.Sequential(nn.Linear(w, w), nn.SiLU(),
                                                      nn.Linear(w, 1)) for _ in range(2))
        self.local_projection = nn.Sequential(nn.Linear(config.n_local_features + 8, w),
                                               nn.SiLU(), nn.Linear(w, w))
        self.local_scorer = nn.Sequential(nn.Linear(w * 2, w), nn.SiLU(), nn.Linear(w, 2))
        self.local_null = nn.Parameter(torch.randn(1, w) * 0.02)
        self.local_null_score = nn.Linear(w, 2)
        self.log_local_scales = nn.Parameter(torch.tensor([500.0, 500.0, 30.0, 1.0]).log().repeat(2, 1))
        self.local_gate = nn.Sequential(nn.Linear(w + 4, w), nn.SiLU(), nn.Linear(w, 2))
        self.local_transformer = (LocalSetTransformer(config)
                                  if config.variant == "local_transformer" else None)
        self.std_head = nn.Sequential(nn.Linear(w + 4, w), nn.SiLU(), nn.Linear(w, 2))
        for head in self.mean_head:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)
        nn.init.zeros_(self.std_head[-1].weight)
        frac = (config.initial_std - config.std_min) / (config.std_max - config.std_min)
        nn.init.constant_(self.std_head[-1].bias, math.log(frac / (1.0 - frac)))

    @staticmethod
    def _latent_coordinates(n: int) -> Tensor:
        # Quasi-uniform spherical positions avoid a learned global latent having
        # to discover its geographic frame from scratch. Depths span the water column.
        idx = torch.arange(n, dtype=torch.float32)
        z = 1.0 - 2.0 * (idx + 0.5) / n
        lat = z.asin() * (180.0 / math.pi)
        lon = ((idx * (math.pi * (3.0 - math.sqrt(5.0))) * (180.0 / math.pi)
                + 180.0) % 360.0) - 180.0
        depth = torch.tensor([0.0, 50.0, 150.0, 300.0, 700.0, 1500.0, 2000.0])[idx.long() % 7]
        return torch.stack((lat, lon, depth, torch.zeros_like(lat)), dim=-1)

    @staticmethod
    def _check_matrix(name: str, x: Tensor, rows: int, columns: int) -> None:
        if x.ndim != 2 or x.shape != (rows, columns):
            raise ValueError(f"{name} must have shape [{rows},{columns}], got {tuple(x.shape)}")

    def encode(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        """Encode context only; target queries and their values are never read."""
        features = scene["obs_features"]
        if not self.config.use_latent:
            return {"latent": features.new_empty((0, self.config.width)),
                    "aux_loss": features.new_zeros(()),
                    "diagnostics": {"latent_disabled": features.new_ones(())}}
        n = features.shape[0]
        self._check_matrix("obs_features", features, n, self.config.n_obs_features)
        self._check_matrix("obs_coord", scene["obs_coord"], n, 4)
        self._check_matrix("obs_innovation", scene["obs_innovation"], n, 2)
        self._check_matrix("obs_valid", scene["obs_valid"], n, 2)
        valid = (scene["obs_valid"].bool() & torch.isfinite(scene["obs_innovation"])
                 & torch.isfinite(scene["obs_coord"]).all(-1, keepdim=True))
        innovation = torch.where(valid, _finite(scene["obs_innovation"]), 0.0)
        active = valid.any(-1)
        # Select before encoding to prevent padding values from influencing any state.
        obs = (self.observation_projection(_finite(features[active]))
               + self.coordinate_projection(self.coordinate_encoding(scene["obs_coord"][active]))
               + self.modality_embedding[0])
        for channel, encoder in enumerate(self.innovation_encoders):
            flags = valid[active, channel].to(features.dtype)
            value_flags = torch.stack((innovation[active, channel], flags), dim=-1)
            obs = obs + encoder(value_flags) * flags[:, None]
        groups = [obs]
        if "satellite_features" in scene:
            if self.satellite_projection is None:
                raise ValueError("satellite tokens require n_sat_features > 0")
            sf = scene["satellite_features"]
            ns = sf.shape[0]
            self._check_matrix("satellite_features", sf, ns, self.config.n_sat_features)
            self._check_matrix("satellite_coord", scene["satellite_coord"], ns, 4)
            smask = torch.isfinite(scene["satellite_coord"]).all(-1)
            if "satellite_valid" in scene:
                if scene["satellite_valid"].shape != (ns,):
                    raise ValueError("satellite_valid must have shape [S]")
                smask = smask & scene["satellite_valid"].bool()
            sat = (self.satellite_projection(_finite(sf[smask]))
                   + self.coordinate_projection(self.coordinate_encoding(scene["satellite_coord"][smask]))
                   + self.modality_embedding[1])
            groups.append(sat)
        else:
            groups.append(features.new_empty((0, self.config.width)))
        present = sum(group.shape[0] > 0 for group in groups)
        # Equal total prior per present modality. The fallback token keeps
        # attention finite, including genuinely empty observation contexts.
        prior = []
        for group in groups:
            if group.shape[0]:
                prior.append(group.new_full((group.shape[0],), math.log(0.95 / present / group.shape[0])))
        prior.append(features.new_tensor([math.log(0.05) if present else 0.0]))
        memory = self.memory_norm(torch.cat((*groups, self.null_observation), dim=0))
        key_bias = torch.cat(prior)
        latent = self.latent + self.coordinate_projection(self.coordinate_encoding(self.latent_coord))
        diagnostic_blocks = []
        for i, block in enumerate(self.latent_blocks):
            latent = latent + self.evidence_gates[i].sigmoid() * self.evidence_attention[i](
                self.evidence_norms[i](latent), memory, key_bias)
            latent, diagnostics = block(latent)
            if diagnostics:
                diagnostic_blocks.append(diagnostics)
        diagnostics = {"n_argo_tokens": active.sum().detach(),
                       "n_satellite_tokens": features.new_tensor(groups[1].shape[0])}
        if diagnostic_blocks:
            diagnostics["expert_usage"] = torch.stack([d["expert_usage"] for d in diagnostic_blocks]).mean(0)
            diagnostics["expert_entropy"] = torch.stack([d["expert_entropy"] for d in diagnostic_blocks]).mean()
        return {"latent": latent, "aux_loss": latent.new_zeros(()), "diagnostics": diagnostics}

    def _local_evidence(self, scene: Mapping[str, Tensor], h: Tensor,
                        baseline_correction: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        q = h.shape[0]
        keys = ("local_features", "local_offsets", "local_innovation", "local_valid")
        if not self.config.use_local or not any(key in scene for key in keys):
            zero = h.new_zeros((q, 2))
            return h, zero, h.new_zeros((q, 4)), {"local_gate": zero.detach(),
                                                    "local_coverage": zero.detach()}
        if not all(key in scene for key in keys):
            raise ValueError(f"local evidence requires all keys: {keys}")
        lf, offset, innovation, supplied_valid = [scene[key] for key in keys]
        if lf.ndim != 3 or lf.shape[0] != q or lf.shape[-1] != self.config.n_local_features:
            raise ValueError("local_features must have shape [Q,K,n_local_features]")
        k = lf.shape[1]
        if offset.shape != (q, k, 4) or innovation.shape != (q, k, 2) or supplied_valid.shape != (q, k, 2):
            raise ValueError("local offsets/innovations/validity have inconsistent shapes")
        valid = (supplied_valid.bool() & torch.isfinite(innovation)
                 & torch.isfinite(offset).all(-1, keepdim=True))
        innovation = torch.where(valid, _finite(innovation), 0.0)
        geometry = _finite(offset)
        normalized = geometry / geometry.new_tensor([500.0, 500.0, 30.0, 1.0])
        local_input = torch.cat((_finite(lf), normalized, innovation,
                                 valid.to(h.dtype)), dim=-1)
        local = self.local_projection(local_input)
        token_valid = valid.any(-1)
        local = torch.where(token_valid[..., None], local, 0.0)
        if self.local_transformer is not None and q:
            local_with_null = torch.cat((local, self.local_null[None].expand(q, -1, -1)), dim=1)
            mask_with_null = torch.cat((token_valid, token_valid.new_ones((q, 1))), dim=1)
            h, contextual = self.local_transformer(h, local_with_null, mask_with_null)
            local = contextual[:, :k]
        paired = torch.cat((h[:, None].expand(-1, k, -1), local), dim=-1)
        scores = self.local_scorer(paired)
        scales = self.log_local_scales.clamp(-5.0, 12.0).exp()
        distance = (geometry[:, :, None, :] / scales[None, None]).square().sum(-1)
        scores = (scores - 0.5 * distance).masked_fill(~valid, float("-inf"))
        scores = torch.cat((scores, self.local_null_score(h)[:, None]), dim=1)
        weights = scores.softmax(dim=1)[:, :k]
        local_candidate = (weights * innovation).sum(dim=1)
        coverage = weights.sum(dim=1)
        centered = innovation - local_candidate[:, None]
        variance = (weights * centered.square()).sum(dim=1)
        evidence_stats = torch.cat((coverage, variance), dim=-1)
        gate = self.local_gate(torch.cat((h, evidence_stats), dim=-1)).tanh()
        # No evidence in a variable means no local overwrite of its known baseline.
        gate = gate * valid.any(dim=1).to(gate.dtype)
        correction = gate * (local_candidate - baseline_correction)
        return h, correction, evidence_stats, {"local_gate": gate.detach(),
                                               "local_coverage": coverage.detach()}

    def decode(self, scene: Mapping[str, Tensor], state: Mapping[str, Any]) -> dict[str, Any]:
        """Read one query chunk from a previously encoded scene context."""
        features = scene["query_features"]
        q = features.shape[0]
        self._check_matrix("query_features", features, q, self.config.n_query_features)
        self._check_matrix("query_coord", scene["query_coord"], q, 4)
        self._check_matrix("baseline", scene["baseline"], q, 2)
        self._check_matrix("query_background", scene["query_background"], q, 2)
        if not torch.isfinite(scene["baseline"]).all() or not torch.isfinite(scene["query_background"]).all():
            raise ValueError("baseline and query_background must be finite")
        baseline = scene["baseline"]
        background = scene["query_background"]
        qinput = torch.cat((_finite(features), background, baseline - background), dim=-1)
        h = (self.query_projection(qinput)
             + self.coordinate_projection(self.coordinate_encoding(scene["query_coord"])))
        if self.config.use_latent:
            for block in self.query_blocks:
                h = block(h, state["latent"])
        h, local_correction, evidence_stats, local_diagnostics = self._local_evidence(
            scene, h, baseline - background)
        h = self.output_norm(h)
        global_residual = torch.cat([head(h) for head in self.mean_head], dim=-1)
        mean = baseline + global_residual + local_correction
        std_logits = self.std_head(torch.cat((h, evidence_stats), dim=-1))
        std = self.config.std_min + (self.config.std_max - self.config.std_min) * std_logits.sigmoid()
        diagnostics = {**state.get("diagnostics", {}), **local_diagnostics,
                       "global_residual_rms": global_residual.square().mean().sqrt().detach()
                       if q else h.new_zeros(())}
        return {"mean": mean, "std": std, "aux_loss": state["aux_loss"],
                "diagnostics": diagnostics}

    def forward(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        return self.decode(scene, self.encode(scene))


def gaussian_objective(prediction: Mapping[str, Tensor], target: Tensor,
                       valid: Tensor, *, mse_weight: float = 1.0,
                       nll_weight: float = 0.0) -> dict[str, Tensor]:
    """Masked MSE/Gaussian NLL with one explicit loss weight per variable.

    Individual T/S losses are averaged equally when present. This prevents a
    field with more valid measurements from silently setting the objective.
    The caller can train the mean first and add NLL or calibrate std separately.
    This helper does not enforce causal input selection; the data loader does.
    """
    mean, std = prediction["mean"], prediction["std"]
    if mean.shape != target.shape or valid.shape != target.shape or mean.shape[-1] != 2:
        raise ValueError("predictions, targets and validity must have matching [Q,2] shapes")
    mask = valid.bool() & torch.isfinite(target)
    safe_target = torch.where(mask, _finite(target), mean.detach())
    error = mean - safe_target
    maskf = mask.to(mean.dtype)
    counts = maskf.sum(0)
    active = counts > 0
    denom = counts.clamp_min(1.0)
    mse_fields = (error.square() * maskf).sum(0) / denom
    nll_fields = ((std.log() + 0.5 * (error / std).square()
                   + 0.5 * math.log(2.0 * math.pi)) * maskf).sum(0) / denom
    n_active = active.sum().clamp_min(1)
    mse = (mse_fields * active).sum() / n_active
    nll = (nll_fields * active).sum() / n_active
    loss = mse_weight * mse + nll_weight * nll + prediction.get("aux_loss", mean.new_zeros(()))
    return {"loss": loss, "mse": mse, "nll": nll, "mse_fields": mse_fields,
            "nll_fields": nll_fields, "valid_counts": counts}
