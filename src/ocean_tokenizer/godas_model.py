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
                          ConservativeResampler, N_FEATURES, LENGTH_SCALES,
                          BASIS_SEED)
from .godas_obs import N_MODALITIES, N_CHANNELS
from .objective_interpolation import ObjectiveInterpolation, OISettings
from .oi_residual import OIResidual
from .query_decoder import (IndependentQueryDecoder, ChannelExpertHead,
                            ReferenceSlots)

ROWS = (
    "objective_interpolation",
    "count_expertlocal_cbottle",
    "uniform_expertlocal_cbottle",
    "dfs_expertlocal_cbottle",
    "count_oi_expert_cbottle",
    "uniform_oi_expert_cbottle",
    "dfs_oi_expert_cbottle",
)

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
                 n_dec_blocks: int = N_DEC_BLOCKS):
        super().__init__()
        assert mass_mode in ("dfs", "uniform", "count")
        self.mass_mode = mass_mode
        self.encoder = _TokenEncoder(d_model)
        self.basis = RandomFourierBasis(N_FEATURES, LENGTH_SCALES, BASIS_SEED)
        self.resampler = ConservativeResampler(d_model, n_slots)
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
        if self.mass_mode == "dfs":
            psi, lam = integrate_support(
                self.basis, s["coord"][:, None, :],
                torch.ones(s["coord"].shape[0], 1, dtype=torch.float64),
                s["noise_density"])
            w = dfs_omega(psi, lam, mask & s["support_mask"])
        else:
            # `uniform` and `count` both report unit mass; for `count` it is
            # diagnostic only and does not drive the transport (doc §2.2)
            w = torch.ones(mask.shape[0], dtype=torch.float64)
        return w * mask.to(w.dtype)

    # ---- §2.5 frozen background ----------------------------------------
    def oi_background(self, s: dict) -> torch.Tensor:
        live = s["mask"] & s["value_mask"].all(dim=-1)
        return self.oi(s["query"], torch.zeros(s["query"].shape[0],
                                               dtype=s["query"].dtype),
                       s["coord"][live], s["value"][live].to(s["query"].dtype),
                       s["noise_density"][live]).to(torch.float32)

    def forward(self, s: dict) -> torch.Tensor:
        emb = self.encoder(s["value"], s["value_mask"], s["coord"],
                           s["modality"])[None]                  # (1,N,d)
        mask = s["mask"][None]
        mass = self.observation_mass(s)[None]

        if self.mass_mode == "count":
            # conventional fixed-query resampler: unit masses, no correction
            slots, slot_mask, slot_mass = self.resampler(
                emb, torch.ones_like(mass), mask)
        else:
            slots, slot_mask, slot_mass = self.resampler(emb, mass, mask)

        ref = self.ref_slots(s["modality_available"].to(emb.dtype)[None])
        kv = torch.cat([slots, ref], dim=1)
        kv_mask = torch.cat(
            [slot_mask, torch.ones(1, ref.shape[1], dtype=torch.bool)], dim=1)
        # reference slots sit at unit mass: always present, never shouting over
        # real evidence when any exists
        kv_mass = torch.cat(
            [slot_mass, torch.ones(1, ref.shape[1], dtype=slot_mass.dtype)],
            dim=1).to(kv.dtype)
        latent = self.fusion(kv, kv_mass, kv_mask)

        q = s["query"].to(emb.dtype)[None]
        lead = torch.zeros(q.shape[:2], dtype=torch.long)
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
