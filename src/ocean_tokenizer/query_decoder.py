"""D4RT-style causal space-time query decoder.

Spec: docs/superpowers/specs/2026-08-16-d4rt-query-decoder-design.md
(mentor document `dfs_d4rt_intern_plan.md` sections 2.3 and 2.4, mapped onto
the protocol_v1 / CESM2-LE line).

The query is ``(x, y, z, t_src, t_tgt)``.  Spatially it reuses the existing
68-dim ``coord_features``; the target time enters as an additive lead
embedding, so ``coord_features`` — and every ``Linear(68, .)`` trained against
it — is left untouched.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .token_api import coord_features, N_COORD_FEATS


class LeadEmbedding(nn.Module):
    """Forecast lead ``t_tgt - t_src`` (whole months) -> additive query bias.

    ``padding_idx=0`` pins row 0 at exactly zero *and* excludes it from
    gradients, so lead 0 contributes no lead signal for the entire life of
    training rather than merely at initialisation.  Reconstruction is
    therefore provably unaffected by the forecasting machinery.
    """

    def __init__(self, max_lead: int, d_model: int):
        super().__init__()
        self.max_lead = int(max_lead)
        self.emb = nn.Embedding(self.max_lead + 1, d_model, padding_idx=0)
        nn.init.zeros_(self.emb.weight)

    def forward(self, lead: torch.Tensor) -> torch.Tensor:
        self._check(lead)
        return self.emb(lead)

    def _check(self, lead: torch.Tensor) -> None:
        """Reject anything the embedding table would silently misread.

        A float lead would be rounded and a negative one would be a valid
        Python index off the end of the table, so both are refused loudly
        instead of quietly answering the wrong question.
        """
        if torch.is_floating_point(lead) or torch.is_complex(lead):
            raise TypeError(f"lead must be an integer tensor, got {lead.dtype}")
        if lead.numel() == 0:
            return
        lo, hi = int(lead.min()), int(lead.max())
        if lo < 0:
            raise ValueError(
                f"lead must be >= 0 (t_tgt >= t_src); got minimum {lo}")
        if hi > self.max_lead:
            raise ValueError(
                f"lead must be <= max_lead={self.max_lead}; got maximum {hi}")


class QueryCrossBlock(nn.Module):
    """Pre-norm cross-attention (queries -> latent) + feed-forward.

    Deliberately has NO self-attention over the query axis.  That absence is
    the load-bearing property of the D4RT decoder: every query attends only to
    the shared latent, so adding, deleting or reordering queries cannot change
    any other query's prediction.  It is also what lets the caller chunk a
    large query set purely for memory.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(),
                                nn.Linear(hidden, d_model))

    def forward(self, q: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        B, Q, _ = q.shape
        L = latent.shape[1]
        qn, kvn = self.norm_q(q), self.norm_kv(latent)
        qh = self.wq(qn).view(B, Q, self.h, self.dh).transpose(1, 2)
        kh = self.wk(kvn).view(B, L, self.h, self.dh).transpose(1, 2)
        vh = self.wv(kvn).view(B, L, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(qh, kh, vh)
        q = q + self.wo(a.transpose(1, 2).reshape(B, Q, -1))
        return q + self.ff(self.norm_ff(q))


class IndependentQueryDecoder(nn.Module):
    """Stack of ``n_blocks`` query-independent cross-attention blocks."""

    def __init__(self, d_model: int, n_blocks: int, n_heads: int,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            QueryCrossBlock(d_model, n_heads, mlp_ratio)
            for _ in range(n_blocks))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q = blk(q, latent)
        return self.out_norm(q)


_LOG_EPS = 1e-12
_NULL_BIAS = math.log(_LOG_EPS)
_DAYS_PER_MONTH = 30.436875

# spec §3.6 / mentor §2.4 initial local length scales, (dx, dy, dz, dt)
_L_INIT = (0.35, 0.35, 0.30, 3.0)
_GATE_INIT = 0.05
_MASS_EXPONENT = 1.0


class QueryLocalRefiner(nn.Module):
    """Direct query -> observation-token attention (mentor §2.4).

    The fixed global slot bottleneck can blur a nearby profile or patch, so
    each decoded query also attends straight to the encoded observation
    tokens.  The attention score is

        content dot product
        + negative Gaussian distance in (dx, dy, dz, t_obs - t_tgt)
        + beta_head * log(evidence_mass ** p)
        + learned relative features (dx, dy, dz, t_obs-t_src, t_obs-t_tgt)

    Coordinate units: ``dy`` is a latitude difference over 90; ``dx`` is the
    wrapped longitude difference over 180, scaled by cos(mean latitude) so a
    degree of longitude shrinks toward the poles; ``dz`` is a depth difference
    over 1000 m; ``dt`` is in months.  The four length scales are learned and
    initialised to :data:`_L_INIT`.

    An always-valid null key is appended so the softmax denominator is never
    empty; it carries :data:`_NULL_BIAS` so real evidence dominates whenever
    any exists.  This is the same device ``fusion.py`` uses for the fusion
    cross-attention.

    Deviation from the mentor text, recorded deliberately: §2.4 calls the
    relative terms "relative-value features".  Injecting them on the value
    side costs O(B*h*Q*N*dh) memory, ~213 MB per 128-query chunk at 6.5k
    tokens, and rules out ``scaled_dot_product_attention``.  They are
    therefore added to the score, which is O(B*h*Q*N) (~13 MB) and keeps the
    fused kernel.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))
        # learned, positive via softplus-free direct storage of log-scale
        self.log_scale = nn.Parameter(torch.log(torch.tensor(_L_INIT)))
        self.beta_head = nn.Parameter(torch.full((n_heads,), _MASS_EXPONENT))
        self.rel_proj = nn.Linear(5, n_heads)
        # per-feature residual gate; the refiner starts as a small correction
        self.gate = nn.Parameter(torch.full((d_model,), _GATE_INIT))

    def _offsets(self, qcoord, lead, tcoord, time_offset):
        """-> (dx, dy, dz, dt_src, dt_tgt), each (B, Q, N)."""
        qlat, qlon, qdep = qcoord[..., 0], qcoord[..., 1], qcoord[..., 2]
        tlat, tlon, tdep = tcoord[..., 0], tcoord[..., 1], tcoord[..., 2]
        dy = (qlat[:, :, None] - tlat[:, None, :]) / 90.0
        dlon = (qlon[:, :, None] - tlon[:, None, :] + 180.0) % 360.0 - 180.0
        mean_lat = 0.5 * (qlat[:, :, None] + tlat[:, None, :])
        dx = dlon / 180.0 * torch.cos(torch.deg2rad(mean_lat))
        dz = (qdep[:, :, None] - tdep[:, None, :]) / 1000.0
        t_obs = (time_offset / _DAYS_PER_MONTH)[:, None, :]      # (B,1,N)
        dt_src = t_obs.expand_as(dy)
        dt_tgt = t_obs - lead.to(t_obs.dtype)[:, :, None]
        return dx, dy, dz, dt_src, dt_tgt.expand_as(dy)

    def forward(self, q, query_coord, lead, emb, coord, tau, time_offset,
                mask):
        B, Q, _ = q.shape
        N = emb.shape[1]
        # A masked slot must be inert.  Its score is driven to -inf below, but
        # attention still forms 0 * value, and 0 * NaN is NaN — so the content
        # of masked slots is neutralised here, before it can reach the values.
        keep = mask[..., None].to(emb.dtype)
        emb = torch.nan_to_num(emb) * keep
        tau = torch.nan_to_num(tau, nan=_LOG_EPS)
        coord = torch.nan_to_num(coord)
        time_offset = torch.nan_to_num(time_offset)
        dx, dy, dz, dt_src, dt_tgt = self._offsets(query_coord, lead, coord,
                                                   time_offset)
        ell = torch.exp(self.log_scale).clamp(min=1e-4)
        gauss = -0.5 * ((dx / ell[0]) ** 2 + (dy / ell[1]) ** 2
                        + (dz / ell[2]) ** 2 + (dt_tgt / ell[3]) ** 2)

        rel = torch.stack([dx, dy, dz, dt_src, dt_tgt], dim=-1)   # (B,Q,N,5)
        rel_bias = self.rel_proj(rel).permute(0, 3, 1, 2)          # (B,h,Q,N)

        # evidence mass: clamp before log so a masked or zero-mass token can
        # never put -inf/NaN into the score
        logm = _MASS_EXPONENT * torch.log(tau.clamp(min=_LOG_EPS))
        mass_bias = self.beta_head[None, :, None, None] * logm[:, None, None, :]

        bias = gauss[:, None] + rel_bias + mass_bias               # (B,h,Q,N)
        bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))

        # always-valid null key so the softmax is never over an empty set
        null = self.null_kv.expand(B, 1, -1)
        kv = torch.cat([self.norm_kv(emb), self.norm_kv(null)], dim=1)
        bias = torch.cat([bias, torch.full((B, self.h, Q, 1), _NULL_BIAS,
                                           device=bias.device,
                                           dtype=bias.dtype)], dim=-1)

        qn = self.norm_q(q)
        qh = self.wq(qn).view(B, Q, self.h, self.dh).transpose(1, 2)
        kh = self.wk(kv).view(B, N + 1, self.h, self.dh).transpose(1, 2)
        vh = self.wv(kv).view(B, N + 1, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        a = self.wo(a.transpose(1, 2).reshape(B, Q, -1))
        return q + self.gate * a


class ChannelExpertHead(nn.Module):
    """Temperature/salinity output experts (mentor §2.4).

    Temperature is read from the shared physical-state query after the shared
    local refiner, via the temperature component of the shared head.  Salinity
    gets a learned variable embedding, its **own** local refiner and a scalar
    head, so the salinity path can specialise without dragging the shared
    physical state with it.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.shared_refiner = QueryLocalRefiner(d_model, n_heads)
        self.salt_refiner = QueryLocalRefiner(d_model, n_heads)
        self.salt_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        hidden = 2 * d_model
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))
        self.salt_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, q, query_coord, lead, emb, coord, tau, time_offset,
                mask):
        kw = dict(emb=emb, coord=coord, tau=tau, time_offset=time_offset,
                  mask=mask)
        h = self.shared_refiner(q, query_coord, lead, **kw)
        temp = self.shared_head(h)[..., :1]
        s = self.salt_refiner(h + self.salt_embed, query_coord, lead, **kw)
        salt = self.salt_head(s)
        return torch.cat([temp, salt], dim=-1)


def check_causal(time_offset: torch.Tensor, support_t: torch.Tensor | None,
                 mask: torch.Tensor, cutoff_days: float = _DAYS_PER_MONTH / 2,
                 tol_days: float = 1e-3) -> None:
    """Raise if any active token carries information from after ``t_src``.

    ``time_offset`` is the signed offset in days from the analysis time, which
    is the *centre* of the source month.  The causal cutoff is therefore the
    **end** of that month, ``+DAYS_PER_MONTH/2``, not zero: this is a monthly
    product, and an observation representing month ``t_src`` legitimately has
    a ~30-day support straddling the month centre.  Setting the cutoff at zero
    would flag every ordinary monthly token as a leak.  What must never happen
    is support reaching into ``t_src + 1``.

    Two failures are possible and are reported separately, because a check
    that only looked at centres would pass a token whose *centre* is inside
    the source month but whose support still reaches into the next one — the
    leak hidden in the support geometry that mentor §3.3 warns about.  Masked
    slots are exempt: padding carries junk coordinates by construction.
    """
    if mask is None or not mask.any():
        return
    m = mask
    lim = cutoff_days + tol_days
    centre = time_offset[m]
    if centre.numel() and float(centre.max()) > lim:
        raise ValueError(
            f"non-causal token centre: max time_offset "
            f"{float(centre.max()):+.3f} d is past the end of the source "
            f"month ({cutoff_days:+.3f} d). An observation from after t_src "
            f"cannot be an input.")
    if support_t is None:
        return
    reach = (time_offset + support_t.abs() / 2.0)[m]
    if reach.numel() and float(reach.max()) > lim:
        raise ValueError(
            f"non-causal token support: centre is legal but support reaches "
            f"{float(reach.max()):+.3f} d, past the end of the source month "
            f"({cutoff_days:+.3f} d). The leak is in the support geometry, "
            f"not the centre.")


DEFAULT_CHUNK = 128           # mentor §2.4: memory only, never couples queries


class D4RTQueryDecoder(nn.Module):
    """The full §2.3 + §2.4 query path.

        coord_features(query) + lead_embed(t_tgt - t_src)
          -> independent-query cross-attention into the shared latent
          -> query-local refiner over the observation tokens
          -> temperature / salinity channel experts

    ``coord_features`` is used unmodified at 68 dims, so the target time never
    widens the coordinate contract that the encoders share.
    """

    def __init__(self, d_model: int = 64, n_blocks: int = 2, n_heads: int = 4,
                 max_lead: int = 3, mlp_ratio: float = 2.0):
        super().__init__()
        self.q_proj = nn.Linear(N_COORD_FEATS, d_model)
        self.lead_embed = LeadEmbedding(max_lead, d_model)
        self.decoder = IndependentQueryDecoder(d_model, n_blocks, n_heads,
                                               mlp_ratio)
        self.experts = ChannelExpertHead(d_model, n_heads)

    def forward(self, latent, query_coord, lead, emb, coord, tau,
                time_offset, mask, chunk: int | None = None):
        if chunk is None:
            return self._decode(latent, query_coord, lead, emb, coord, tau,
                                time_offset, mask)
        outs = [self._decode(latent, query_coord[:, i:i + chunk],
                             lead[:, i:i + chunk], emb, coord, tau,
                             time_offset, mask)
                for i in range(0, query_coord.shape[1], chunk)]
        return torch.cat(outs, dim=1)

    def _decode(self, latent, query_coord, lead, emb, coord, tau,
                time_offset, mask):
        q = self.q_proj(coord_features(query_coord)) + self.lead_embed(lead)
        h = self.decoder(q, latent)
        return self.experts(h, query_coord, lead, emb=emb, coord=coord,
                            tau=tau, time_offset=time_offset, mask=mask)


class ReferenceSlots(nn.Module):
    """Learned, availability-conditioned key/value slots (mentor §2.3).

    Eight slots accompany the transported observation slots into fusion, so
    the latent always has well-defined key mass to read even when a modality
    is dropped.  Conditioning on *which* modalities are present lets the
    fallback differ between "no profiles this month" and "no SSH this month"
    instead of collapsing to one generic null, which is the single-token
    behaviour this generalises.
    """

    def __init__(self, n_slots: int, d_model: int, n_modalities: int):
        super().__init__()
        self.base = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.avail_proj = nn.Linear(n_modalities, d_model)

    def forward(self, availability: torch.Tensor) -> torch.Tensor:
        """``availability`` (B, M) in {0,1} -> (B, n_slots, d_model)."""
        cond = self.avail_proj(availability.to(self.base.dtype))
        return self.base[None] + cond[:, None, :]
