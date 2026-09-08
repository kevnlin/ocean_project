"""The GODAS registered-row model (mentor doc §2.2-§2.6 assembled).

    tokens (§2.1)
      -> observation mass          dfs | uniform | count      (§2.2)
      -> conservative transport into 32 slots                 (§2.2)
      -> + 8 availability-conditioned reference slots         (§2.3)
      -> 32 global latents, mass-biased cross-attention       (§2.2/§2.3)
      -> independent-query decoder, no query self-attention   (§2.3)
      -> query-local refiner + T/S channel experts            (§2.4)
      -> optional frozen-OI residual with 8 lead/channel gates (§2.5/§2.6)

Rows differ **only** in the mass mode and whether the OI residual wraps them,
which is what makes `uniform` a matched mechanism control rather than a
different model: it runs the identical transport and mass-biased blocks with
``omega_i = 1``.  ``build_row`` therefore constructs every row from one class,
and a test asserts `dfs` and `uniform` have byte-identical parameter sets.

``count`` is the conventional Perceiver-resampler control: a fixed-query
resampler with no mass correction.  Its reported unit ``omega`` is diagnostic
only and does not drive its transport (doc §2.2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batched_dfs import (RandomFourierBasis, integrate_support, dfs_omega,
                          ConservativeResampler, PerceiverResampler,
                          variable_group_coords, N_FEATURES, LENGTH_SCALES,
                          LENGTH_SCALES_KM, to_physical, BASIS_SEED,
                          extent_for, length_scales_km_for,
                          vertical_quadrature)
from .godas_obs import N_MODALITIES, N_CHANNELS, N_VARIABLE_GROUPS
from .objective_interpolation import ObjectiveInterpolation, OISettings
from .oi_residual import OIResidual
from .query_decoder import (IndependentQueryDecoder, ChannelExpertHead,
                            ReferenceSlots)

ROWS = (
    "objective_interpolation",
    "count_expertlocal_cbottle",
    "uniform_expertlocal_cbottle",
    "dfs_expertlocal_cbottle",
    # the work plan's control ladder: what operational centres actually do.
    # Count-independent by CONSTRUCTION -- no learned mass anywhere -- so if
    # these match dfs on the redundancy regimes, the claim becomes "the first
    # differentiable, in-operator version of preprocessing" rather than a win.
    "thin_expertlocal_cbottle",
    "superob_expertlocal_cbottle",
    "count_oi_expert_cbottle",
    "uniform_oi_expert_cbottle",
    "dfs_oi_expert_cbottle",
)
#: mass modes that carry unit mass and instead remove redundancy by
#: PREPROCESSING the token set before it ever reaches the model
PREPROCESS_MODES = {"thin": "thin", "superob": "superob"}

D_MODEL = 64
N_HEADS = 4
N_LATENT_BLOCKS = 2
N_DEC_BLOCKS = 2
N_SLOTS = 32
N_REF_SLOTS = 8
N_LATENTS = 32
_LOG_EPS = 1e-12


class _TokenEncoder(nn.Module):
    """(value, value_mask, coord, modality) -> token embedding.

    The mask is an *input channel*, not just a filter: a zero-filled missing
    value and a genuine zero measurement must not look alike to the encoder.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.value = nn.Linear(2 * N_CHANNELS, d_model)   # values + finite flags
        self.coord = nn.Linear(4, d_model)
        self.modality = nn.Embedding(N_MODALITIES, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, value, value_mask, coord, modality):
        x = torch.cat([value, value_mask.to(value.dtype)], dim=-1)
        h = self.value(x) + self.coord(coord.to(value.dtype)) \
            + self.modality(modality)
        return self.norm(h)


class _MassBiasedFusion(nn.Module):
    """Latents read the slots with an additive ``log(mass)`` attention prior."""

    def __init__(self, d_model: int, n_heads: int, n_latents: int, n_blocks: int):
        super().__init__()
        self.latent0 = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.h, self.dh = n_heads, d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            IndependentQueryDecoder(d_model, 1, n_heads) for _ in range(n_blocks))

    def forward(self, kv, mass, kv_mask):
        B = kv.shape[0]
        z = self.latent0[None].expand(B, -1, -1)
        q = self.wq(self.norm_q(z)).view(B, -1, self.h, self.dh).transpose(1, 2)
        kvn = self.norm_kv(kv)
        k = self.wk(kvn).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.wv(kvn).view(B, -1, self.h, self.dh).transpose(1, 2)
        bias = torch.log(mass.clamp(min=_LOG_EPS)).to(kv.dtype)
        bias = bias.masked_fill(~kv_mask, float("-inf"))[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        z = z + self.wo(a.transpose(1, 2).reshape(B, z.shape[1], -1))
        for blk in self.blocks:
            z = blk(z, z)
        return z


class GodasRowModel(nn.Module):
    """One registered row.  ``mass_mode`` and ``use_oi`` are the only knobs."""

    def __init__(self, mass_mode: str = "dfs", use_oi: bool = False,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_slots: int = N_SLOTS, n_latents: int = N_LATENTS,
                 n_ref_slots: int = N_REF_SLOTS,
                 n_latent_blocks: int = N_LATENT_BLOCKS,
                 n_dec_blocks: int = N_DEC_BLOCKS,
                 n_features: int = N_FEATURES, physical_units: bool = True,
                 noise_scale: float = 1.0, n_vertical_nodes: int = 3,
                 provenance_rho: float = 0.0, region: str | None = None):
        super().__init__()
        assert mass_mode in ("dfs", "uniform", "count", "thin", "superob")
        self.mass_mode = mass_mode
        # thin/superob are unit-mass rows whose redundancy handling happens in
        # a preprocessing step, exactly as an operational centre would do it
        self.preprocess = PREPROCESS_MODES.get(mass_mode)
        # ``noise_scale`` is the Phase-0 sweep knob: it multiplies the declared
        # observation-error variance densities, moving s = support/noise
        # without touching the geometry.  1.0 is the registered prior.
        self.physical_units = bool(physical_units)
        self.noise_scale = float(noise_scale)
        # Phase 1a: >1 integrates the basis over each layer (3 nodes converge);
        # 1 is the pre-Phase-1 midpoint rule.
        self.n_vertical_nodes = int(n_vertical_nodes)
        # Phase 1b: correlation between tokens sharing a provenance group.
        # 0.0 keeps the pre-Phase-1 independent-noise behaviour.
        self.provenance_rho = float(provenance_rho)
        self.last_omega = None
        self.encoder = _TokenEncoder(d_model)
        # the kernel spans (x, y, z, t) PLUS a variable-group axis, so tokens
        # carrying different physical quantities are not consolidated as
        # duplicates just because they share a location
        # ``region`` selects the box whose physical span sets the kernel's
        # length scales. None keeps the historical Gulf Stream extent, so every
        # run made before this existed is unchanged.
        self.region = region
        self.box_extent = extent_for(region)
        base_scales = (length_scales_km_for(region) if physical_units
                       else LENGTH_SCALES)
        self.basis = RandomFourierBasis(
            n_features,
            base_scales + variable_group_coords.scales(N_VARIABLE_GROUPS),
            BASIS_SEED)
        # `count` is a DIFFERENT transport, not the conservative one fed unit
        # masses — otherwise it would be a copy of `uniform` and could not
        # separate an OI-residual gain from a DFS gain (doc §2.4)
        self.resampler = (PerceiverResampler(d_model, n_slots)
                          if mass_mode == "count"
                          else ConservativeResampler(d_model, n_slots))
        self.ref_slots = ReferenceSlots(n_ref_slots, d_model, N_MODALITIES)
        self.fusion = _MassBiasedFusion(d_model, n_heads, n_latents,
                                        n_latent_blocks)
        self.q_proj = nn.Linear(4, d_model)
        self.decoder = IndependentQueryDecoder(d_model, n_dec_blocks, n_heads)
        self.experts = ChannelExpertHead(d_model, n_heads, geographic=False)
        self.oi = ObjectiveInterpolation(OISettings())
        self.oi_residual = OIResidual() if use_oi else None

    # ---- §2.2 observation mass -----------------------------------------
    def observation_mass(self, s: dict) -> torch.Tensor:
        """(N,) mass per token.  Masked tokens always carry exactly zero."""
        mask = s["mask"]
        dev = s["coord"].device
        if self.mass_mode == "dfs":
            grp = variable_group_coords(s["variable_group"], N_VARIABLE_GROUPS)
            # Phase 0: the kernel runs in km/m/months, and the quadrature
            # weight is the token's physical footprint in km² instead of a
            # dimensionless 1, so lambda = noise_area * support_area and the
            # per-token operating point s = support/noise is interpretable.
            coord = (to_physical(s["coord"], self.box_extent)
                     if self.physical_units
                     else s["coord"])
            if self.physical_units and "support_area" in s:
                area = s["support_area"].to(torch.float64)
                if self.n_vertical_nodes > 1 and "support_dz" in s:
                    # Phase 1a: integrate the basis over the layer, so a token
                    # is a VOLUME.  Splitting a layer in two halves each child's
                    # thickness, so a vertical resampling that conserves total
                    # thickness conserves total evidence.
                    zq, wq = vertical_quadrature(coord[:, 2],
                                                 s["support_dz"].to(torch.float64),
                                                 self.n_vertical_nodes)
                    Q = zq.shape[1]
                    nodes = coord[:, None, :].repeat(1, Q, 1).clone()
                    nodes[:, :, 2] = zq
                    weight = wq * area[:, None]
                else:
                    nodes = coord[:, None, :]
                    weight = area[:, None]
            else:
                nodes = coord[:, None, :]
                weight = torch.ones(s["coord"].shape[0], 1,
                                    dtype=torch.float64, device=dev)
            g = grp.to(coord.device)[:, None, :].expand(-1, nodes.shape[1], -1)
            full = torch.cat([nodes, g], dim=-1)
            psi, lam = integrate_support(
                self.basis, full, weight,
                s["noise_density"] * self.noise_scale)
            # Phase 1b: provenance drives the NOISE correlation, so tokens from
            # one platform / product stream collapse along n_eff rather than
            # voting independently.  rho = 0 reproduces diagonal whitening.
            w = dfs_omega(psi, lam, mask & s["support_mask"],
                          s.get("provenance"), self.provenance_rho)
            self.last_omega = w.detach()
        else:
            # `uniform`, `count`, `thin` and `superob` all report unit mass;
            # for `count` it is diagnostic only and does not drive the
            # transport (doc §2.2), and for thin/superob the redundancy has
            # already been removed upstream by `prep`
            w = torch.ones(mask.shape[0], dtype=torch.float64, device=dev)
        return w * mask.to(w.dtype)

    # ---- §2.5 frozen background ----------------------------------------
    def oi_background(self, s: dict) -> torch.Tensor:
        live = s["mask"] & s["value_mask"].all(dim=-1)
        return self.oi(s["query"], torch.zeros(s["query"].shape[0],
                                               dtype=s["query"].dtype,
                                               device=s["query"].device),
                       s["coord"][live], s["value"][live].to(s["query"].dtype),
                       s["noise_density"][live]).to(torch.float32)

    def prep(self, s: dict) -> dict:
        """Apply the row's observation preprocessing, if it has any.

        Called at the top of ``forward``.  ``observation_mass`` deliberately
        does NOT call it, so a caller measuring mass directly gets the raw
        token set and cannot double-apply the merge.
        """
        if self.preprocess is None:
            return s
        from .godas_obs import superob_tokens
        return superob_tokens(s, self.preprocess)

    def forward(self, s: dict) -> torch.Tensor:
        s = self.prep(s)
        emb = self.encoder(s["value"], s["value_mask"], s["coord"],
                           s["modality"])[None]                  # (1,N,d)
        mask = s["mask"][None]
        mass = self.observation_mass(s)[None]

        slots, slot_mask, slot_mass = self.resampler(emb, mass, mask)

        dev = emb.device
        ref = self.ref_slots(s["modality_available"].to(emb.dtype)[None])
        kv = torch.cat([slots, ref], dim=1)
        kv_mask = torch.cat(
            [slot_mask, torch.ones(1, ref.shape[1], dtype=torch.bool,
                                   device=dev)], dim=1)
        # reference slots sit at unit mass: always present, never shouting over
        # real evidence when any exists
        kv_mass = torch.cat(
            [slot_mass, torch.ones(1, ref.shape[1], dtype=slot_mass.dtype,
                                   device=dev)], dim=1).to(kv.dtype)
        latent = self.fusion(kv, kv_mass, kv_mask)

        q = s["query"].to(emb.dtype)[None]
        lead = torch.full(q.shape[:2], int(s.get("lead", 0)), dtype=torch.long,
                          device=dev)
        h = self.decoder(self.q_proj(q), latent)
        # evidence and coordinates are carried in float64 for the DFS solve;
        # cast at the boundary into the float32 network rather than letting
        # torch promote halfway through and mismatch a Linear
        neural = self.experts(h, q, lead, emb=emb,
                              coord=s["coord"][None].to(emb.dtype),
                              tau=mass.to(emb.dtype),
                              time_offset=s["coord"][None, :, 3].to(emb.dtype),
                              mask=mask)[0]
        if self.oi_residual is None:
            return neural
        return self.oi_residual(self.oi_background(s), neural, lead[0])


def build_row(row: str, **kw) -> GodasRowModel:
    """Construct a registered row by name (doc §6.1)."""
    if row not in ROWS:
        raise ValueError(f"unknown row {row!r}; registered rows are {ROWS}")
    if row == "objective_interpolation":
        raise ValueError(
            "objective_interpolation is deterministic and has no model; call "
            "ObjectiveInterpolation directly")
    return GodasRowModel(mass_mode=row.split("_")[0],
                         use_oi="oi_expert" in row, **kw)
