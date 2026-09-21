"""Task 3 — comparable fusion variants for the shared-latent model.

Four variants, differing ONLY in the token->latent fusion rule (identical
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

* ``DFSAttention``          (Variant D) — the current method.  Token weights
                            are no longer hand-designed: each observation's
                            *degrees of freedom for signal* are estimated from
                            a three-dimensional, stratification- and
                            target-resolution-aware support kernel
                            (``dfs.dfs_scores``), transported through a
                            fixed-budget resampler without loss
                            (``dfs.EvidenceResampler``), and fused against a
                            climatological background.  See
                            ``docs/dfs_attention.md``; MBCA remains the
                            hand-weighted baseline it is compared to.

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

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dfs
from .token_api import (TokenBatch, SharedLatentModel, MODALITIES,
                        coord_features, N_COORD_FEATS)
from .query_decoder import (D4RTQueryDecoder, ReferenceSlots, check_causal,
                            DEFAULT_CHUNK)

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

    def _latent0(self, B: int) -> torch.Tensor:
        z = self.latent0
        if self.anchor_proj is not None:
            z = z + self.anchor_proj(coord_features(self.anchor_coord))
        return z[None].expand(B, -1, -1)

    # ---- shared fuse / decode ------------------------------------------
    def fuse(self, tokens: TokenBatch, target=None) -> torch.Tensor:
        kv, mask, bias = self._prepare_tokens(tokens)
        B = kv.shape[0]
        kv = torch.cat([kv, self.null_token.expand(B, 1, -1)], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool,
                                           device=mask.device)], dim=1)
        if bias is not None:
            bias = torch.cat([bias, torch.full((B, 1), _NULL_BIAS,
                                               device=bias.device,
                                               dtype=bias.dtype)], dim=1)
        z = self._latent0(B)
        z = z + self.fuse_attn(self.fuse_ln_q(z), self.fuse_ln_kv(kv),
                               key_bias=bias, key_mask=mask)
        for blk in self.blocks:
            z = blk(z)
        return z

    def decode(self, latent, query_coord, query_scale=None):
        """``query_scale`` is accepted (and ignored) by every variant except
        DFS-Attention, whose decoder is scale-conditioned; keeping it in the
        base signature lets one evaluation loop drive all four variants."""
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
# Variant D — DFS-Attention
# --------------------------------------------------------------------------
class DFSAttention(AttnFusionModel):
    """Scale-aware evidence fusion (docs/dfs_attention.md).

    Three stages replace MBCA's single log-mass prior, addressing the two
    limitations that retired it (hand-designed weights; evidence discarded at
    the resampler):

    1. **Set-level effective-evidence estimation.**  ``dfs.dfs_scores`` scores
       every observation token by its ridge leverage under a three-dimensional,
       stratification- and target-resolution-aware support kernel, so tau_i is
       the *marginal* information token i adds once physical overlap, vertical
       complementarity, timing, uncertainty and provenance are accounted for.
       The climatological background is not part of the evidence set — it is
       the prior the evidence is measured against.

    2. **Conservative evidence transport.**  ``dfs.EvidenceResampler``
       compresses the observation tokens to a fixed budget while moving their
       evidence with them: the outgoing masses sum to the incoming DFS exactly,
       so the total physical evidence survives the resampler instead of being
       flattened to equal weights.

    3. **Background-referenced latent fusion.**  The latent array first reads
       the climatological background, then attends to the transported evidence
       with an additive ``log nu_s`` prior against a background key held at
       ``log lambda_bg``.  The comparison is therefore between an absolute
       number of observed degrees of freedom and the background's effective
       weight: where evidence is thin the fusion falls back to climatology, and
       where it is rich the observations win.  lambda_bg is the ridge parameter
       of stage 1 wearing its other hat — the background precision.

    The decoder is additionally conditioned on the target resolution
    (``query_scale``), which is the same scale that set the kernel lengths, so
    "what is being asked for" is consistent between evidence and decode.
    """

    BACKGROUND = ("woa_grid",)

    def __init__(self, encoders, d_model: int = 128, n_latent: int = 128,
                 n_heads: int = 4, n_self_blocks: int = 4, c_out: int = 2,
                 mlp_ratio: float = 2.0, anchor_grid=None, k_slots: int = 32,
                 k_neighbors: int = 32, learn_scales: bool = False,
                 target_scale=None, s_cross: float = dfs.S_CROSS,
                 detach_evidence: bool = True, mass_mode: str = "dfs"):
        super().__init__(encoders, d_model, n_latent, n_heads, n_self_blocks,
                         c_out, mlp_ratio, anchor_grid=anchor_grid)
        self.scales = dfs.SupportScales(learn_residual=learn_scales)
        self.mass_mode = mass_mode
        self.resampler = dfs.EvidenceResampler(
            d_model, k_slots, n_heads,
            mode=("count" if mass_mode == "count" else "conservative"))
        self.k_neighbors = int(k_neighbors)
        self.s_cross = float(s_cross)
        self.target_scale = target_scale or dfs.PROTOCOL_SCALE
        # evidence is a property of the observing geometry, not a learned
        # quantity, so it is estimated without gradient unless the length-scale
        # residual is being learned
        self.detach_evidence = bool(detach_evidence) and not learn_scales
        self.bg_ln_kv = nn.LayerNorm(d_model)
        self.bg_attn = CrossAttention(d_model, n_heads)
        # the background is the REFERENCE, not an optional extra: the gate
        # starts at 1 so the latent is climatology-anchored from step 0 (a
        # zero-initialised gate would leave the whole background branch, WOA
        # encoder included, without gradient)
        self.bg_gate = nn.Parameter(torch.ones(1))
        # background precision: how many degrees of freedom of evidence it
        # takes to outweigh the climatology (log-space, learnable)
        self.log_lambda_bg = nn.Parameter(torch.zeros(1))
        self.scale_proj = nn.Linear(2, d_model)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        self._bg_ids = torch.tensor([MODALITIES[m] for m in self.BACKGROUND])
        self.last_evidence = None          # diagnostics of the last fuse()

    # ---- helpers -------------------------------------------------------
    def _extra_kv(self, tokens: TokenBatch):
        """Optional extra fusion keys, appended after the background key.

        ``None`` for plain DFS-Attention; :class:`D4RTFusion` returns its
        availability-conditioned reference slots here so the hook keeps the
        fuse body in one place.
        """
        return None

    def _split(self, tokens: TokenBatch):
        bg_ids = self._bg_ids.to(tokens.modality.device)
        is_bg = (tokens.modality[..., None] == bg_ids).any(-1)
        return tokens.mask & ~is_bg, tokens.mask & is_bg

    def evidence(self, tokens: TokenBatch, target=None) -> dfs.DFSResult:
        """Per-token degrees of freedom for signal at the target scale.

        ``mass_mode="uniform"`` is the MATCHED CONTROL: every observation token
        carries tau = 1 instead of its measured evidence, and nothing else
        changes -- same conservative transport, same background-referenced
        fusion, same parameter count, same seed. It isolates the evidence
        estimate as the single manipulated variable, which is what makes a
        DFS-minus-Uniform difference attributable to the mass rule rather than
        to capacity or optimisation.

        It is NOT the same as the `perceiver` variant. That one replaces the
        transport as well, so multiplicity feeds through the softmax -- which is
        the `count` control, a different claim.
        """
        obs_mask, _ = self._split(tokens)
        ctx = torch.no_grad() if self.detach_evidence else _nullctx()
        with ctx:
            res = dfs.dfs_scores(tokens, target or self.target_scale,
                                 self.scales, k_neighbors=self.k_neighbors,
                                 s_cross=self.s_cross, evidence_mask=obs_mask)
        if getattr(self, "mass_mode", "dfs") == "uniform":
            unit = obs_mask.to(res.tau.dtype)
            res = dataclasses.replace(
                res, tau=unit, total=unit.sum(dim=-1),
                by_modality={k: torch.ones_like(v)
                             for k, v in res.by_modality.items()})
        return res

    # ---- fuse ----------------------------------------------------------
    def fuse(self, tokens: TokenBatch, target=None) -> torch.Tensor:
        target = target or self.target_scale
        obs_mask, bg_mask = self._split(tokens)
        B = tokens.emb.shape[0]
        res = self.evidence(tokens, target)

        # (2) conservative transport: fixed budget, evidence carried along
        kv, kv_mask, nu = self.resampler(tokens.emb, obs_mask, res.tau,
                                         tokens.modality)
        self.last_evidence = dict(tau=res.tau, nu=nu, total=res.total,
                                  by_modality=res.by_modality)

        # (3a) background-referenced latent: read the climatology first
        z = self._latent0(B)
        if bool(bg_mask.any()):
            z = z + self.bg_gate * self.bg_attn(
                self.fuse_ln_q(z), self.bg_ln_kv(tokens.emb),
                key_mask=bg_mask)
        z_bg = z

        # (3b) observation increment against the background key
        kv = torch.cat([kv, self.null_token.expand(B, 1, -1)], dim=1)
        one = torch.ones(B, 1, dtype=torch.bool, device=kv.device)
        kv_mask = torch.cat([kv_mask, one], dim=1)
        bias = torch.cat([torch.log(nu + _LOG_EPS),
                          self.log_lambda_bg.expand(B, 1)], dim=1)
        extra = self._extra_kv(tokens)
        if extra is not None:
            e_kv, e_mask, e_bias = extra
            kv = torch.cat([kv, e_kv], dim=1)
            kv_mask = torch.cat([kv_mask, e_mask], dim=1)
            bias = torch.cat([bias, e_bias], dim=1)
        z = z_bg + self.fuse_attn(self.fuse_ln_q(z), self.fuse_ln_kv(kv),
                                  key_bias=bias, key_mask=kv_mask)
        for blk in self.blocks:
            z = blk(z)
        return z

    # ---- scale-conditioned decode --------------------------------------
    def decode(self, latent, query_coord, query_scale=None):
        q = self.q_proj(coord_features(query_coord))
        if query_scale is None:
            ts = self.target_scale
            query_scale = torch.tensor([ts.dx_km, ts.dz_m],
                                       device=query_coord.device,
                                       dtype=query_coord.dtype)
            query_scale = query_scale.expand(*query_coord.shape[:-1], 2)
        sf = torch.stack([torch.log(query_scale[..., 0].clamp(min=1e-3)
                                    / dfs.PROTOCOL_SCALE.dx_km),
                          torch.log(query_scale[..., 1].clamp(min=1e-3)
                                    / dfs.PROTOCOL_SCALE.dz_m)], dim=-1)
        q = q + self.scale_proj(sf)
        a = self.dec_attn(self.dec_ln_q(q), self.dec_ln_kv(latent))
        return self.head(torch.cat([a, q], dim=-1))

    def forward(self, obs: dict, query_coord: torch.Tensor, target=None,
                query_scale=None) -> torch.Tensor:
        tokens = self.encode(obs, batch=query_coord.shape[0],
                             device=query_coord.device)
        return self.decode(self.fuse(tokens, target), query_coord, query_scale)


# --------------------------------------------------------------------------
# Variant E — DFS-Attention backbone + D4RT causal space-time query decoder
# --------------------------------------------------------------------------
class D4RTFusion(DFSAttention):
    """DFS-Attention fusion with the mentor §2.3/§2.4 query path.

    The fusion stage is unchanged except for the availability-conditioned
    reference slots (§2.3).  What changes is the way out: instead of one
    cross-attention layer into the latent, a query carries a target time and
    passes through an independent-query decoder, a query-local refiner over
    the observation tokens, and temperature/salinity experts.

    ``fuse`` caches the encoded tokens because the local refiner needs them at
    decode time; the existing "fuse once per month, decode in chunks" loop in
    ``fullrun.eval_packs`` therefore keeps working unchanged.  This mirrors
    the ``last_evidence`` caching DFS-Attention already does.
    """

    def __init__(self, encoders, d_model: int = 64, n_latent: int = 32,
                 n_heads: int = 4, n_self_blocks: int = 2, c_out: int = 2,
                 mlp_ratio: float = 2.0, anchor_grid=None,
                 n_ref_slots: int = 8, n_dec_blocks: int = 2,
                 max_lead: int = 3, query_chunk: int = DEFAULT_CHUNK,
                 causal_check: bool = True, **kw):
        super().__init__(encoders, d_model, n_latent, n_heads, n_self_blocks,
                         c_out, mlp_ratio, anchor_grid=anchor_grid, **kw)
        self.ref_slots = ReferenceSlots(n_ref_slots, d_model, len(MODALITIES))
        self.qdec = D4RTQueryDecoder(d_model, n_dec_blocks, n_heads, max_lead)
        self.query_chunk = int(query_chunk)
        self.causal_check = bool(causal_check)
        self._tokens = None

    # ---- fusion: reference slots + token cache -------------------------
    def _extra_kv(self, tokens: TokenBatch):
        B = tokens.emb.shape[0]
        avail = self._availability(tokens)
        kv = self.ref_slots(avail)
        n = kv.shape[1]
        mask = torch.ones(B, n, dtype=torch.bool, device=kv.device)
        # neutral prior: the reference is what the latent falls back on, so it
        # must not out-shout real evidence when any exists
        bias = self.log_lambda_bg.expand(B, n)
        return kv, mask, bias

    def _availability(self, tokens: TokenBatch) -> torch.Tensor:
        """(B, M) 1.0 where that modality has at least one live token."""
        B = tokens.emb.shape[0]
        M = len(MODALITIES)
        oh = F.one_hot(tokens.modality.clamp(min=0), M).to(tokens.emb.dtype)
        oh = oh * tokens.mask[..., None].to(tokens.emb.dtype)
        return (oh.sum(1) > 0).to(tokens.emb.dtype).view(B, M)

    def fuse(self, tokens: TokenBatch, target=None) -> torch.Tensor:
        if self.causal_check and tokens.time_offset is not None:
            check_causal(tokens.time_offset, tokens.support_t, tokens.mask)
        self._tokens = tokens
        return super().fuse(tokens, target)

    # ---- D4RT decode ----------------------------------------------------
    def decode(self, latent, query_coord, query_scale=None, lead=None,
               chunk: int | None = None):
        tokens = self._tokens
        if tokens is None:
            raise RuntimeError(
                "decode() needs the encoded tokens for the query-local "
                "refiner; call fuse() first (the eval loop already does).")
        if lead is None:
            lead = torch.zeros(query_coord.shape[:2], dtype=torch.long,
                               device=query_coord.device)
        obs_mask, _ = self._split(tokens)
        tau = self.last_evidence["tau"]
        t_off = tokens.time_offset
        if t_off is None:
            t_off = torch.zeros_like(tau)
        return self.qdec(latent, query_coord, lead, emb=tokens.emb,
                         coord=tokens.coord, tau=tau, time_offset=t_off,
                         mask=obs_mask,
                         chunk=self.query_chunk if chunk is None else chunk)

    def forward(self, obs: dict, query_coord: torch.Tensor, target=None,
                query_scale=None, lead=None, chunk=None) -> torch.Tensor:
        tokens = self.encode(obs, batch=query_coord.shape[0],
                             device=query_coord.device)
        latent = self.fuse(tokens, target)
        return self.decode(latent, query_coord, query_scale, lead, chunk)


# --------------------------------------------------------------------------
# Variant F — DFS–GAOT-Argo: spatially addressed latent + multiscale neighbourhoods
# --------------------------------------------------------------------------
_KM_PER_DEG = 111.195


class GAOTFusion(D4RTFusion):
    """D4RT fusion with a GAOT-style geometry-aware encoder and processor.

    Replaces the two stages of :class:`D4RTFusion` that have no notion of
    place -- the 32 unaddressed resampler slots and the 32 free latent vectors
    -- with **anchors that have spatial addresses**, following the
    "DFS with a modern ocean reconstruction backbone" first-run proposal
    (GAOT, NeurIPS 2025). Everything downstream is kept: the reference slots,
    the evidence-vs-background competition at ``log lambda_bg``, the
    independent-query D4RT decoder with its local refiner, the loss and the
    data contract. Improvements are therefore attributable to the spatial
    representation alone.

    1. **Anchors.** ``nx * ny`` horizontal anchors on a fixed grid over the
       region box, times ``len(anchor_depths)`` depth anchors taken from the
       cohort level grid. Placement depends on the region box and level grid
       only, never on targets. Each anchor's initial state is the shared
       ``coord_features`` of its address, so anchors and queries speak the same
       coordinate language.

    2. **Multiscale neighbourhood aggregation.** For each scale ``r`` the logit
       from token ``i`` to anchor ``j`` is a content dot product plus a
       geometric bias ``-0.5 * ((d_h / l_h)^2 + (d_z / l_z)^2 + (d_t / l_t)^2)``
       built from horizontal km, depth metres and days scaled SEPARATELY (never
       a raw distance mixing degrees and metres), plus a learned relative-
       offset term. Pairs beyond ``cutoff`` e-folds are masked.

    3. **The arm's mass rule, at the same stage as before.** The arms still
       differ exactly where :class:`dfs.EvidenceResampler` makes them differ:

       * ``dfs`` / ``uniform`` (conservative): each token distributes its whole
         evidence ``e_i`` (DFS tau, or unit mass) over anchors AND scales -- a
         softmax over ``(r, j)`` -- so ``sum_{r,j} m_rj = sum_i e_i`` exactly.
         An anchor's content is the evidence-weighted mean of what it received.
       * ``count``: the softmax runs over TOKENS inside each anchor's
         neighbourhood, so multiplicity feeds straight through, and the mass is
         the neighbourhood token count.

       Evidence metadata travels as its own channel (``m_j``) and is never
       folded into the normalised features.

    4. **Evidence against the reference, per anchor.** Each anchor attends to
       its aggregated observation token at bias ``log m_j`` against the null key
       and the availability-conditioned reference slots at ``log lambda_bg`` --
       the same competition :class:`DFSAttention` runs globally. An anchor with
       no evidence in reach has its observation key masked and falls back to
       the reference: nothing is fabricated at an unobserved anchor.

    5. **Processor.** The inherited global self-attention blocks run over the
       anchor tokens, which keep their coordinate embeddings.

    The local refiner is unchanged and reads ``tau`` exactly as
    :class:`D4RTFusion` does, so the arms' semantics there are unchanged too.
    """

    def __init__(self, encoders, d_model: int = 64, n_latent: int = 32,
                 n_heads: int = 4, n_self_blocks: int = 2, c_out: int = 2,
                 mlp_ratio: float = 2.0, anchor_grid=None,
                 anchor_box: dict | None = None,
                 anchor_hw: tuple[int, int] = (8, 4),
                 anchor_depths=(5.0, 35.0, 105.0, 186.3, 326.9, 527.7,
                                984.7, 1400.5),
                 scales_km: tuple[float, ...] = (300.0, 900.0),
                 scales_m: tuple[float, ...] = (100.0, 400.0),
                 scale_t_d: float = 30.0, cutoff: float = 3.0, **kw):
        super().__init__(encoders, d_model, n_latent, n_heads, n_self_blocks,
                         c_out, mlp_ratio, anchor_grid=None, **kw)
        if anchor_box is None:
            raise ValueError("GAOTFusion needs anchor_box=dict(lat=(lo, hi), "
                             "lon=(lo, hi)) -- anchors are placed on the region")
        mode = "count" if self.mass_mode == "count" else "conservative"
        # the Perceiver-specific stages this backbone replaces: no unaddressed
        # resampler slots, no free latent vectors
        del self.resampler
        del self.latent0
        self.agg_mode = mode
        nx, ny = anchor_hw
        (la0, la1), (lo0, lo1) = anchor_box["lat"], anchor_box["lon"]
        lat_c = la0 + (torch.arange(ny) + 0.5) * (la1 - la0) / ny
        lon_c = lo0 + (torch.arange(nx) + 0.5) * (lo1 - lo0) / nx
        dep = torch.as_tensor(anchor_depths, dtype=torch.float32)
        g = torch.stack(torch.meshgrid(dep, lat_c, lon_c, indexing="ij"), -1)
        coords = torch.cat([g[..., 1:2], g[..., 2:3], g[..., 0:1],
                            torch.zeros_like(g[..., :1])], -1).reshape(-1, 4)
        self.register_buffer("gaot_anchor", coords.float())    # (J, 4)
        self.n_anchor = coords.shape[0]
        self.register_buffer("gaot_l_km", torch.tensor(scales_km).float())
        self.register_buffer("gaot_l_m", torch.tensor(scales_m).float())
        self.gaot_l_t = float(scale_t_d)
        self.gaot_cutoff = float(cutoff)
        R = len(scales_km)
        assert len(scales_m) == R
        self.gaot_anchor_proj = nn.Linear(N_COORD_FEATS, d_model)
        self.gaot_ln_kv = nn.LayerNorm(d_model)
        self.gaot_ln_q = nn.LayerNorm(d_model)
        # q/k shared across scales (scale identity enters through the geometric
        # bias and a per-scale query offset); v/o per scale so each scale can
        # carry its own summary
        self.gaot_wq = nn.Linear(d_model, d_model)
        self.gaot_wk = nn.Linear(d_model, d_model)
        self.gaot_scale_q = nn.Parameter(torch.zeros(R, d_model))
        self.gaot_wv = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(R))
        self.gaot_rel = nn.Linear(3, n_heads)
        nn.init.zeros_(self.gaot_rel.weight)
        nn.init.zeros_(self.gaot_rel.bias)
        self.gaot_mix = nn.Linear(R * d_model, d_model)

    # ---- geometry --------------------------------------------------------
    def _offsets(self, tcoord):
        """-> (dh_km, dz_m) each (B, N, J) from tokens to anchors."""
        a = self.gaot_anchor
        dlat = tcoord[..., 0, None] - a[:, 0]
        dlon = (tcoord[..., 1, None] - a[:, 1] + 180.0) % 360.0 - 180.0
        mlat = torch.deg2rad(0.5 * (tcoord[..., 0, None] + a[:, 0]))
        dy = dlat * _KM_PER_DEG
        dx = dlon * _KM_PER_DEG * torch.cos(mlat)
        dz = tcoord[..., 2, None] - a[:, 2]
        return dx, dy, dz

    def _anchor_tokens(self, B):
        return self.gaot_anchor_proj(coord_features(self.gaot_anchor))[None].expand(B, -1, -1)

    def aggregate(self, tokens: TokenBatch, obs_mask, e):
        """Observation tokens -> per-anchor content (B,J,d) and mass (B,J)."""
        B, N, _ = tokens.emb.shape
        J, R, h = self.n_anchor, self.gaot_l_km.numel(), self.fuse_attn.h
        dh = tokens.emb.shape[-1] // h
        coord = torch.nan_to_num(tokens.coord)
        dx, dy, dz = self._offsets(coord)                           # (B,N,J)
        t_off = (torch.zeros_like(e) if tokens.time_offset is None
                 else torch.nan_to_num(tokens.time_offset))
        dt2 = (t_off / self.gaot_l_t)[..., None] ** 2               # (B,N,1)
        kv = self.gaot_ln_kv(torch.nan_to_num(tokens.emb))
        kh = self.gaot_wk(kv).view(B, N, h, dh)
        anc = self.gaot_ln_q(self._anchor_tokens(B))                # (B,J,d)
        rel = self.gaot_rel(torch.stack([dx / 1000.0, dy / 1000.0,
                                         dz / 1000.0], -1))         # (B,N,J,h)
        logits, inside = [], []
        for r in range(R):
            lh, lz = self.gaot_l_km[r], self.gaot_l_m[r]
            d2 = (dx / lh) ** 2 + (dy / lh) ** 2 + (dz / lz) ** 2 + dt2
            qh = self.gaot_wq(anc + self.gaot_scale_q[r]).view(B, J, h, dh)
            s = torch.einsum("bnhd,bjhd->bhnj", kh, qh) / math.sqrt(dh)
            lg = s - 0.5 * d2[:, None] + rel.permute(0, 3, 1, 2)
            ok = (d2 <= self.gaot_cutoff ** 2) & obs_mask[..., None]  # (B,N,J)
            logits.append(lg.masked_fill(~ok[:, None], -float("inf")))
            inside.append(ok)
        lg = torch.stack(logits, 2)                                  # (B,h,R,N,J)
        ok = torch.stack(inside, 1)                                  # (B,R,N,J)
        if self.agg_mode == "count":
            # normalise over TOKENS in each anchor's neighbourhood: multiplicity
            # feeds straight through, mass is the neighbourhood token count
            A = torch.nan_to_num(torch.softmax(lg, dim=3), nan=0.0)
            w = A
            mass = ok.sum(dim=2).to(e.dtype)                         # (B,R,J)
        else:
            # conservative: every token spreads its evidence over (scale, anchor)
            flat = lg.permute(0, 1, 3, 2, 4).reshape(B, h, N, R * J)
            A = torch.nan_to_num(torch.softmax(flat, dim=-1), nan=0.0)
            A = A.reshape(B, h, N, R, J).permute(0, 1, 3, 2, 4)      # (B,h,R,N,J)
            w = A * e[:, None, None, :, None]
            mass = w.sum(dim=3).mean(dim=1)                          # (B,R,J)
        outs = []
        for r in range(R):
            vh = self.gaot_wv[r](kv).view(B, N, h, dh)
            num = torch.einsum("bhnj,bnhd->bjhd", w[:, :, r], vh)
            den = w[:, :, r].sum(dim=2).clamp(min=_LOG_EPS).transpose(1, 2)[..., None]
            outs.append((num / den).reshape(B, J, -1))
        content = self.gaot_mix(torch.cat(outs, -1))
        m = mass.sum(dim=1)                                          # (B,J)
        return content * (m > 0)[..., None].to(content.dtype), m

    # ---- fuse -------------------------------------------------------------
    def fuse(self, tokens: TokenBatch, target=None) -> torch.Tensor:
        if self.causal_check and tokens.time_offset is not None:
            check_causal(tokens.time_offset, tokens.support_t, tokens.mask)
        self._tokens = tokens
        target = target or self.target_scale
        obs_mask, bg_mask = self._split(tokens)
        B = tokens.emb.shape[0]
        res = self.evidence(tokens, target)
        e = torch.nan_to_num(res.tau, nan=0.0).clamp(min=0.0) * obs_mask.to(res.tau.dtype)
        content, m = self.aggregate(tokens, obs_mask, e)
        self.last_evidence = dict(tau=res.tau, nu=m, total=res.total,
                                  by_modality=res.by_modality)

        z = self._anchor_tokens(B)
        if bool(bg_mask.any()):
            z = z + self.bg_gate * self.bg_attn(
                self.fuse_ln_q(z), self.bg_ln_kv(tokens.emb), key_mask=bg_mask)
        J, d = z.shape[1], z.shape[2]

        # per-anchor competition: [own observation summary @ log m_j,
        #                           null @ log lambda_bg, reference slots]
        own = content.reshape(B * J, 1, d)
        own_mask = (m > 0).reshape(B * J, 1)
        null = self.null_token.expand(B * J, 1, -1)
        kv = torch.cat([own, null], dim=1)
        kv_mask = torch.cat([own_mask, torch.ones_like(own_mask)], dim=1)
        bias = torch.cat([torch.log(m.reshape(B * J, 1) + _LOG_EPS),
                          self.log_lambda_bg.expand(B * J, 1)], dim=1)
        extra = self._extra_kv(tokens)
        if extra is not None:
            e_kv, e_mask, e_bias = extra
            n = e_kv.shape[1]
            kv = torch.cat([kv, e_kv[:, None].expand(B, J, n, d).reshape(B * J, n, d)], 1)
            kv_mask = torch.cat([kv_mask, e_mask[:, None].expand(B, J, n).reshape(B * J, n)], 1)
            bias = torch.cat([bias, e_bias[:, None].expand(B, J, n).reshape(B * J, n)], 1)
        zq = z.reshape(B * J, 1, d)
        z = z + self.fuse_attn(self.fuse_ln_q(zq), self.fuse_ln_kv(kv),
                               key_bias=bias, key_mask=kv_mask).reshape(B, J, d)
        for blk in self.blocks:
            z = blk(z)
        return z



# --------------------------------------------------------------------------
# Variant G — DFS–LNO: Latent Neural Operator physics-cross-attention fuse
# --------------------------------------------------------------------------
class LNOFusion(D4RTFusion):
    """D4RT fusion whose fuse stage is Latent-Neural-Operator Physics-Cross-Attention.

    Replaces the Perceiver-IO trunk's unaddressed resampler slots and free latent
    array (Wang et al., NeurIPS 2024, "Latent Neural Operator"; the slot framing
    follows Set Transformer / Slot Attention):

    1. **Position-only attention.** Each observation token's weights over the
       ``n_slots`` latent slots come from its coordinates alone, through a small
       MLP on the shared ``coord_features``; the token's content enters only as
       the value. Observation positions and query coordinates are therefore
       decoupled — nothing about where a query sits is baked into the encoder.
    2. **Softmax over the SLOTS** (per token, per head): every token distributes
       itself across the slots, so slots compete for tokens. This is the Slot
       Attention / Transolver normalisation, deliberately NOT the LNO paper's
       own PhCA encoder, which normalises over the input points (its Eq. 4):
       normalising over slots is what keeps DFS evidence conserved. Hence a
       "PhCA-style" backbone. The same competition drives the mass rule:

         * ``dfs`` / ``uniform`` (conservative): token i spreads its evidence
           ``e_i`` (DFS tau, or 1) as ``w_ij = e_i a_ij``; the slot mass
           ``m_j = sum_i w_ij`` sums to ``sum_i e_i`` exactly.
         * ``count``: the softmax runs over TOKENS inside each slot, so
           multiplicity feeds straight through, and the mass is the slot's
           token count.

       A slot's content is the mass-weighted mean of what it received.
    3. **Evidence against the reference, per slot**, exactly as
       :class:`GAOTFusion` does for its anchors: each slot attends to its own
       content at ``log m_j`` against the null key and the availability-
       conditioned reference slots at ``log lambda_bg``.
    4. The inherited latent self-attention blocks run over the slots; the D4RT
       query decoder, local refiner, loss and data contract are unchanged, so a
       difference against the Perceiver trunk is attributable to the fuse stage.
    """

    def __init__(self, encoders, d_model: int = 64, n_latent: int = 32,
                 n_heads: int = 4, n_self_blocks: int = 2, c_out: int = 2,
                 mlp_ratio: float = 2.0, anchor_grid=None, n_slots: int = 32,
                 proj_hidden: int = 96, **kw):
        super().__init__(encoders, d_model, n_latent, n_heads, n_self_blocks,
                         c_out, mlp_ratio, anchor_grid=None, **kw)
        # the Perceiver-specific stages this backbone replaces
        del self.resampler
        del self.latent0
        self.agg_mode = "count" if self.mass_mode == "count" else "conservative"
        self.n_slots = int(n_slots)
        h = self.fuse_attn.h
        self.phca_proj = nn.Sequential(nn.Linear(N_COORD_FEATS, proj_hidden),
                                       nn.GELU(),
                                       nn.Linear(proj_hidden, h * self.n_slots))
        self.phca_ln = nn.LayerNorm(d_model)
        self.phca_v = nn.Linear(d_model, d_model)
        self.phca_out = nn.Linear(d_model, d_model)
        self.slot_emb = nn.Parameter(torch.randn(self.n_slots, d_model) * 0.02)

    def slot_weights(self, coord: torch.Tensor) -> torch.Tensor:
        """(B,N,4) physical coords -> (B,N,h,M) position-only logits."""
        B, N, _ = coord.shape
        h = self.fuse_attn.h
        return self.phca_proj(coord_features(torch.nan_to_num(coord))).view(
            B, N, h, self.n_slots)

    def aggregate(self, tokens: TokenBatch, obs_mask, e):
        """Observation tokens -> per-slot content (B,M,d) and mass (B,M)."""
        B, N, d = tokens.emb.shape
        h, M = self.fuse_attn.h, self.n_slots
        dh = d // h
        lg = self.slot_weights(tokens.coord)                     # (B,N,h,M)
        valid = obs_mask[..., None, None].to(lg.dtype)
        a_slot = torch.softmax(lg, dim=-1) * valid               # over slots
        if self.agg_mode == "count":
            # softmax over TOKENS within each slot: multiplicity feeds through
            w = torch.softmax(lg.masked_fill(~obs_mask[..., None, None],
                                             -float("inf")), dim=1)
            w = torch.nan_to_num(w, nan=0.0)
            mass = a_slot.sum(dim=1).mean(dim=1)                 # token count
        else:
            w = a_slot * e[:, :, None, None]
            mass = w.sum(dim=1).mean(dim=1)                      # sum = sum_i e_i
        v = self.phca_v(self.phca_ln(torch.nan_to_num(tokens.emb))).view(B, N, h, dh)
        num = torch.einsum("bnhm,bnhd->bmhd", w, v)
        den = w.sum(dim=1).transpose(1, 2)[..., None]            # (B,M,h,1)
        content = self.phca_out((num / den.clamp(min=_LOG_EPS)).reshape(B, M, d))
        return content * (mass > 0)[..., None].to(content.dtype), mass

    def fuse(self, tokens: TokenBatch, target=None) -> torch.Tensor:
        if self.causal_check and tokens.time_offset is not None:
            check_causal(tokens.time_offset, tokens.support_t, tokens.mask)
        self._tokens = tokens
        target = target or self.target_scale
        obs_mask, bg_mask = self._split(tokens)
        B = tokens.emb.shape[0]
        res = self.evidence(tokens, target)
        e = torch.nan_to_num(res.tau, nan=0.0).clamp(min=0.0) * obs_mask.to(res.tau.dtype)
        content, m = self.aggregate(tokens, obs_mask, e)
        self.last_evidence = dict(tau=res.tau, nu=m, total=res.total,
                                  by_modality=res.by_modality)

        z = self.slot_emb[None].expand(B, -1, -1)
        if bool(bg_mask.any()):
            z = z + self.bg_gate * self.bg_attn(
                self.fuse_ln_q(z), self.bg_ln_kv(tokens.emb), key_mask=bg_mask)
        M, d = z.shape[1], z.shape[2]
        own = content.reshape(B * M, 1, d)
        own_mask = (m > 0).reshape(B * M, 1)
        null = self.null_token.expand(B * M, 1, -1)
        kv = torch.cat([own, null], dim=1)
        kv_mask = torch.cat([own_mask, torch.ones_like(own_mask)], dim=1)
        bias = torch.cat([torch.log(m.reshape(B * M, 1) + _LOG_EPS),
                          self.log_lambda_bg.expand(B * M, 1)], dim=1)
        extra = self._extra_kv(tokens)
        if extra is not None:
            e_kv, e_mask, e_bias = extra
            n = e_kv.shape[1]
            kv = torch.cat([kv, e_kv[:, None].expand(B, M, n, d).reshape(B * M, n, d)], 1)
            kv_mask = torch.cat([kv_mask, e_mask[:, None].expand(B, M, n).reshape(B * M, n)], 1)
            bias = torch.cat([bias, e_bias[:, None].expand(B, M, n).reshape(B * M, n)], 1)
        zq = z.reshape(B * M, 1, d)
        z = z + self.fuse_attn(self.fuse_ln_q(zq), self.fuse_ln_kv(kv),
                               key_bias=bias, key_mask=kv_mask).reshape(B, M, d)
        for blk in self.blocks:
            z = blk(z)
        return z


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------
# Builder — identical encoders/trunk for every variant
# --------------------------------------------------------------------------
#: Any variant takes a `_uniform` or `_count` suffix, and both are MATCHED
#: CONTROLS: identical architecture, identical parameter count, identical
#: weights at a fixed seed.  Only the observation-mass rule moves.
#:
#:   `_uniform`  the evidence estimate is replaced by unit mass; transport is
#:               still conservative, so duplicates compete for a fixed budget.
#:   `_count`    the resampler normalises over TOKENS rather than slots and
#:               ignores tau entirely -- token multiplicity feeds straight
#:               through.  This is the Perceiver rule, applied inside the same
#:               parameters.
#:
#: `_count` is NOT `_uniform` with a different name: handing unit tau to the
#: conservative transport would collapse the two.  With an evidence estimate
#: that halves per copy, conservative total mass holds at 1.00x through k=8
#: while count grows 1.18x.
#:
#: The standalone `perceiver` variant is a different NETWORK (no resampler at
#: all, ~261k params), so it is a family comparison rather than a control; use
#: `<variant>_count` when the claim is about the mass rule alone.
VARIANTS = {"perceiver": StandardPerceiver,
            "resampler": FixedBudgetResampler,
            "mbca": MBCA,
            "dfs": DFSAttention,
            "d4rt": D4RTFusion,
            "gaot": GAOTFusion,
            "lno": LNOFusion}


def build_fusion_model(variant: str, grid, d_model: int = 128,
                       n_latent: int = 128, n_heads: int = 4,
                       n_self_blocks: int = 4, patch=(10, 12),
                       seed: int | None = None,
                       anchor_grid: tuple[int, int] | None = None,
                       with_ssh: bool = False, patch_surf=None, patch_woa=None,
                       **kw):
    """Wire the project's modality encoders into a fusion variant.

    With the same ``seed``, every variant starts from identical encoder,
    trunk, and decoder weights (variant-specific extras excepted), so
    comparisons isolate the fusion rule.

    ``with_ssh`` adds the pseudo-SSH / steric-height encoder (one channel, its
    own ``ssh_grid`` modality).  It is opt-in and appended LAST so that, at a
    fixed seed, every other encoder still draws the same initial weights as a
    run without it — an SSH ablation stays a controlled comparison.

    ``patch_surf`` / ``patch_woa`` override the patch size per stream.  A
    gridded token currently averages ``patch`` cells before the model sees it
    (10x12 = 120 cells at 1 deg), and the modality-dropout audit found that
    dropping BOTH gridded streams costs ~0.2 % — i.e. the dense fields are
    effectively unused.  Refining only the stream under test keeps the token
    count, and therefore the O(Q*N) query refiner, affordable.
    """
    from .token_api import ProfileEncoder, GridPatchEncoder
    if seed is not None:
        torch.manual_seed(seed)
    encoders = {
        "profiles": ProfileEncoder(grid.depth, c_vars=2, d_model=d_model),
        "surf": GridPatchEncoder(2, d_model=d_model,
                                 patch=(patch_surf or patch),
                                 modality="surf_grid"),
        "woa": GridPatchEncoder(2, d_model=d_model,
                                patch=(patch_woa or patch),
                                modality="woa_grid"),
    }
    if with_ssh:
        encoders["ssh"] = GridPatchEncoder(1, d_model=d_model,
                                           patch=(patch_surf or patch),
                                           modality="ssh_grid")
    mass_mode = kw.pop("mass_mode", None)
    for sfx in ("_uniform", "_count"):
        if variant.endswith(sfx):
            variant, mass_mode = variant[: -len(sfx)], sfx[1:]
            break
    cls = VARIANTS[variant]
    if mass_mode is not None:
        kw["mass_mode"] = mass_mode
    model = cls(encoders, d_model=d_model, n_latent=n_latent, n_heads=n_heads,
                n_self_blocks=n_self_blocks, anchor_grid=anchor_grid, **kw)
    if mass_mode is not None and not isinstance(model, DFSAttention):
        raise ValueError(f"mass_mode is only meaningful for the DFS family; "
                         f"{variant!r} has no evidence estimate")
    return model
