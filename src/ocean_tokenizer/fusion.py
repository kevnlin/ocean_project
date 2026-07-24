"""Task 3 — comparable fusion variants for the shared-latent model.

Three variants, differing ONLY in the token->latent fusion rule (identical
modality encoders, model dim, latent token count, latent self-attention
trunk, coordinate query decoder, optimizer, data):

* ``StandardPerceiver``     (Variant A) — ordinary masked cross-attention
                            softmax(QK^T/sqrt d)V.  The most important
                            architecture baseline.
* ``FixedBudgetResampler``  (Variant B) — each modality is first compressed to
                            a fixed number of tokens by learned queries (no
                            observation-mass correction), then fused as in A.
                            Tests whether a fixed token budget alone fixes
                            token-multiplicity bias.
* ``MBCA``                  (Variant C) — Measure-Balanced Cross-Attention:
                            every token carries a nonnegative support mass
                            mu_mi; masses are normalised within each modality
                            (mu_bar = mu/sum mu), scaled by an equal modality
                            prior pi_m over *present* modalities
                            (renormalised when a modality is missing), and
                            enter attention as an additive prior:

                                MBCA(Q,K,V; w) =
                                    softmax(QK^T/sqrt d + log(w + eps)) V.

                            Exactly invariant to ideal token partitioning:
                            splitting a token (k, v, w) into n children
                            (k, v, w/n) leaves every attention output
                            unchanged (n * exp(s + log(w/n)) = exp(s + log w)).

First-implementation choices (kept deliberately simple / interpretable):
equal pi_m across available modalities; no learned quality gate; no
uncertainty prediction.  Tokens without a support mass (encoders predating
Task 4 metadata) default to uniform mass within their modality, which makes
MBCA differ from A only through the modality-prior rebalancing.

A learned "null" token is appended to every key set so attention is always
well-defined (zero-profile / empty-observation batches never NaN); it has
negligible MBCA mass (eps) and does not break partition invariance.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_api import (TokenBatch, SharedLatentModel, MODALITIES,
                        coord_features, N_COORD_FEATS)

_LOG_EPS = 1e-12          # inside log(w + eps): keeps masked/zero-mass finite
_NULL_BIAS = math.log(_LOG_EPS)


# --------------------------------------------------------------------------
# Attention primitives
# --------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional additive per-key logit bias.

    ``key_bias`` (B, N) is added to every query's attention logits for that
    key (the MBCA log-mass prior); ``key_mask`` (B, N) True = attend.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, kv, key_bias=None, key_mask=None):
        B, Lq, _ = q.shape
        N = kv.shape[1]
        qh = self.wq(q).view(B, Lq, self.h, self.dh).transpose(1, 2)
        kh = self.wk(kv).view(B, N, self.h, self.dh).transpose(1, 2)
        vh = self.wv(kv).view(B, N, self.h, self.dh).transpose(1, 2)
        bias = torch.zeros(B, 1, 1, N, device=q.device, dtype=q.dtype)
        if key_bias is not None:
            bias = bias + key_bias[:, None, None, :]
        if key_mask is not None:
            bias = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        return self.wo(out.transpose(1, 2).reshape(B, Lq, -1))


class SelfBlock(nn.Module):
    """Pre-LN transformer block over the latent array."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CrossAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d_model))

    def forward(self, z):
        h = self.ln1(z)
        z = z + self.attn(h, h)
        return z + self.mlp(self.ln2(z))


# --------------------------------------------------------------------------
# MBCA observation-measure weights
# --------------------------------------------------------------------------
def mbca_weights(mass, modality, mask, eps: float = 1e-8) -> torch.Tensor:
    """Token support masses -> attention prior weights w (B, N).

    Within each modality m: mu_bar_mi = mu_mi / sum_j mu_mj; the modality
    prior pi_m is equal across modalities *present* in the batch item
    (renormalised when a modality is missing); w_mi = pi_m * mu_bar_mi.
    Valid tokens therefore always sum to 1 per batch item.  ``mass`` None
    (encoder without Task-4 metadata) -> uniform mass per valid token.
    """
    maskf = mask.to(torch.float32)
    if mass is None:
        mass = maskf
    mass = mass.to(torch.float32).clamp(min=0.0) * maskf
    n_mod = len(MODALITIES)
    onehot = F.one_hot(modality.clamp(min=0), n_mod).to(torch.float32)
    onehot = onehot * maskf[..., None]                      # (B,N,M)
    tot = torch.einsum("bnm,bn->bm", onehot, mass)          # (B,M)
    present = tot > eps
    pi = present.float() / present.float().sum(-1, keepdim=True).clamp(min=1.0)
    denom = torch.einsum("bnm,bm->bn", onehot, tot.clamp(min=eps))
    mu_bar = mass / denom.clamp(min=eps)
    pi_tok = torch.einsum("bnm,bm->bn", onehot, pi)
    return pi_tok * mu_bar                                  # (B,N)


# --------------------------------------------------------------------------
# Shared trunk: latent array + fusion cross-attn + self blocks + query decoder
# --------------------------------------------------------------------------
class AttnFusionModel(SharedLatentModel):
    """encode -> (variant fusion) -> latent self-attention -> query decode.

    ``anchor_grid=(nlat_a, nlon_a)`` switches the latent array to
    *geographically anchored* tokens: one latent per coarse map cell, whose
    initial state is a learned free vector plus the shared coordinate
    featurisation of the cell centre (month/depth features held constant).
    Fusion cross-attention then has an immediate geographic matching signal
    between latent queries and observation-token keys — the same signal the
    query decoder uses on the way out — instead of having to discover
    geography from scratch (Week-4 finding: an unstructured global latent
    reaches only ~6 % skill before month-memorisation overtakes learning,
    failing hardest in the SST/SSS-rich top 100 m).  The latent remains a
    shared, modality-agnostic state array; only its parameterisation gains a
    spatial prior (GraphDOP-style mesh latent).
    """

    def __init__(self, encoders, d_model: int = 128, n_latent: int = 128,
                 n_heads: int = 4, n_self_blocks: int = 4, c_out: int = 2,
                 mlp_ratio: float = 2.0, anchor_grid: tuple[int, int] | None = None):
        super().__init__(encoders, d_model)
        if anchor_grid is not None:
            na, no = anchor_grid
            n_latent = na * no
            lat_c = torch.linspace(-90 + 90.0 / na, 90 - 90.0 / na, na)
            lon_c = torch.linspace(180.0 / no, 360 - 180.0 / no, no)
            coords = torch.stack([
                lat_c[:, None].expand(na, no).reshape(-1),
                lon_c[None, :].expand(na, no).reshape(-1),
                torch.zeros(na * no), torch.zeros(na * no)], dim=-1)
            self.register_buffer("anchor_coord", coords)      # (n_latent, 4)
            self.anchor_proj = nn.Linear(N_COORD_FEATS, d_model)
        else:
            self.anchor_coord = None
            self.anchor_proj = None
        self.n_latent = n_latent
        self.latent0 = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fuse_ln_q = nn.LayerNorm(d_model)
        self.fuse_ln_kv = nn.LayerNorm(d_model)
        self.fuse_attn = CrossAttention(d_model, n_heads)
        self.blocks = nn.ModuleList(
            SelfBlock(d_model, n_heads, mlp_ratio) for _ in range(n_self_blocks))
        self.q_proj = nn.Linear(N_COORD_FEATS, d_model)
        self.dec_ln_q = nn.LayerNorm(d_model)
        self.dec_ln_kv = nn.LayerNorm(d_model)
        self.dec_attn = CrossAttention(d_model, n_heads)
        hidden = 2 * d_model
        self.head = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, c_out))

    # ---- variant hooks -------------------------------------------------
    def _prepare_tokens(self, tokens: TokenBatch):
        """-> (kv (B,N,d), mask (B,N), key_bias (B,N) or None)."""
        return tokens.emb, tokens.mask, None

    # ---- shared fuse / decode ------------------------------------------
    def fuse(self, tokens: TokenBatch) -> torch.Tensor:
        kv, mask, bias = self._prepare_tokens(tokens)
        B = kv.shape[0]
        kv = torch.cat([kv, self.null_token.expand(B, 1, -1)], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool,
                                           device=mask.device)], dim=1)
        if bias is not None:
            bias = torch.cat([bias, torch.full((B, 1), _NULL_BIAS,
                                               device=bias.device,
                                               dtype=bias.dtype)], dim=1)
        z = self.latent0
        if self.anchor_proj is not None:
            z = z + self.anchor_proj(coord_features(self.anchor_coord))
        z = z[None].expand(B, -1, -1)
        z = z + self.fuse_attn(self.fuse_ln_q(z), self.fuse_ln_kv(kv),
                               key_bias=bias, key_mask=mask)
        for blk in self.blocks:
            z = blk(z)
        return z

    def decode(self, latent, query_coord):
        q = self.q_proj(coord_features(query_coord))
        a = self.dec_attn(self.dec_ln_q(q), self.dec_ln_kv(latent))
        return self.head(torch.cat([a, q], dim=-1))


# --------------------------------------------------------------------------
# Variant A — standard Perceiver cross-attention
# --------------------------------------------------------------------------
class StandardPerceiver(AttnFusionModel):
    """Ordinary masked cross-attention: softmax(QK^T/sqrt d)V."""
    pass


# --------------------------------------------------------------------------
# Variant B — fixed-budget modality resampler (no mass correction)
# --------------------------------------------------------------------------
class FixedBudgetResampler(AttnFusionModel):
    """Each modality -> ``k_per_modality`` tokens via learned queries, then
    standard fusion.  No observation-mass correction anywhere."""

    def __init__(self, encoders, d_model: int = 128, n_latent: int = 128,
                 n_heads: int = 4, n_self_blocks: int = 4, c_out: int = 2,
                 mlp_ratio: float = 2.0, k_per_modality: int = 32,
                 anchor_grid: tuple[int, int] | None = None):
        super().__init__(encoders, d_model, n_latent, n_heads, n_self_blocks,
                         c_out, mlp_ratio, anchor_grid=anchor_grid)
        self.k = k_per_modality
        self.res_query = nn.Parameter(
            torch.randn(len(MODALITIES), k_per_modality, d_model) * 0.02)
        self.res_ln_kv = nn.LayerNorm(d_model)
        self.res_attn = CrossAttention(d_model, n_heads)

    def _prepare_tokens(self, tokens: TokenBatch):
        B = tokens.emb.shape[0]
        kv_in = torch.cat([tokens.emb, self.null_token.expand(B, 1, -1)], dim=1)
        kv_in = self.res_ln_kv(kv_in)
        one = torch.ones(B, 1, dtype=torch.bool, device=tokens.mask.device)
        outs, masks = [], []
        for m_id in MODALITIES.values():
            sel = (tokens.modality == m_id) & tokens.mask          # (B,N)
            present = sel.any(dim=1)                               # (B,)
            q = self.res_query[m_id][None].expand(B, -1, -1)
            out = self.res_attn(q, kv_in,
                                key_mask=torch.cat([sel, one], dim=1))
            outs.append(out * present[:, None, None])
            masks.append(present[:, None].expand(B, self.k))
        return (torch.cat(outs, dim=1), torch.cat(masks, dim=1), None)


# --------------------------------------------------------------------------
# Variant C — Measure-Balanced Cross-Attention
# --------------------------------------------------------------------------
class MBCA(AttnFusionModel):
    """Fusion cross-attention with the log observation-measure prior."""

    def _prepare_tokens(self, tokens: TokenBatch):
        w = mbca_weights(tokens.support_mass, tokens.modality, tokens.mask)
        return tokens.emb, tokens.mask, torch.log(w + _LOG_EPS)


# --------------------------------------------------------------------------
# Builder — identical encoders/trunk for every variant
# --------------------------------------------------------------------------
VARIANTS = {"perceiver": StandardPerceiver,
            "resampler": FixedBudgetResampler,
            "mbca": MBCA}


def build_fusion_model(variant: str, grid, d_model: int = 128,
                       n_latent: int = 128, n_heads: int = 4,
                       n_self_blocks: int = 4, patch=(10, 12),
                       seed: int | None = None,
                       anchor_grid: tuple[int, int] | None = None, **kw):
    """Wire the project's three modality encoders into a fusion variant.

    With the same ``seed``, every variant starts from identical encoder,
    trunk, and decoder weights (variant-specific extras excepted), so
    comparisons isolate the fusion rule.
    """
    from .token_api import ProfileEncoder, GridPatchEncoder
    if seed is not None:
        torch.manual_seed(seed)
    encoders = {
        "profiles": ProfileEncoder(grid.depth, c_vars=2, d_model=d_model),
        "surf": GridPatchEncoder(2, d_model=d_model, patch=patch,
                                 modality="surf_grid"),
        "woa": GridPatchEncoder(2, d_model=d_model, patch=patch,
                                modality="woa_grid"),
    }
    cls = VARIANTS[variant]
    return cls(encoders, d_model=d_model, n_latent=n_latent, n_heads=n_heads,
               n_self_blocks=n_self_blocks, anchor_grid=anchor_grid, **kw)
