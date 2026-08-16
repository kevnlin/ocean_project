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
