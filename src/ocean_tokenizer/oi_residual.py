"""Learned per-lead/per-channel neural residual over a frozen OI (mentor §2.6).

    prediction(q,c) = OI(q,c) + sigmoid(gate_logit[h,c]) * [neural(q,c) - OI(q,c)]

Eight trainable logits — four leads x two variables — and nothing else.  The
objective-interpolation background is **frozen**: detached, no autograd graph,
no parameters.  That is enforced here by detaching the OI tensor on entry, and
tested as an absence of gradient rather than assumed from the call site.

Gate initialisation follows §2.6: temperature 0.8, salinity 0.2.  Those are
*gate values*, so the stored logits are their inverse-sigmoids.  The asymmetry
encodes the prior the doc reports from its own runs — the neural branch earns
most of the temperature signal while OI remains the dominant salinity
component.

Why a residual rather than a plain blend: the gate multiplies
``neural - OI``, so at gate 0 the row degrades exactly to standalone OI.  A
row that cannot beat OI therefore cannot be *worse* than OI either, which is
what makes the comparison in §8 meaningful.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

MAX_LEAD = 3
N_CHANNELS = 2                      # TEMP, SALT
TEMP_GATE_INIT = 0.8
SALT_GATE_INIT = 0.2


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class OIResidual(nn.Module):
    """Blend a frozen OI background with a neural prediction, per lead/channel."""

    def __init__(self, max_lead: int = MAX_LEAD, n_channels: int = N_CHANNELS,
                 temp_init: float = TEMP_GATE_INIT,
                 salt_init: float = SALT_GATE_INIT):
        super().__init__()
        self.max_lead = int(max_lead)
        init = torch.empty(self.max_lead + 1, n_channels)
        init[:, 0] = _logit(temp_init)
        if n_channels > 1:
            init[:, 1:] = _logit(salt_init)
        self.gate_logit = nn.Parameter(init)

    def _check_lead(self, lead: torch.Tensor) -> None:
        if torch.is_floating_point(lead) or torch.is_complex(lead):
            raise TypeError(
                f"lead must be an integer tensor for gate indexing, got "
                f"{lead.dtype}; rounding would silently pick a different gate")
        if lead.numel() == 0:
            return
        lo, hi = int(lead.min()), int(lead.max())
        if lo < 0:
            raise ValueError(
                f"lead must be >= 0; got {lo}. A negative lead would index the "
                f"gate table from the end.")
        if hi > self.max_lead:
            raise ValueError(
                f"lead must be <= max_lead={self.max_lead}; got {hi}")

    def gates(self, lead: torch.Tensor) -> torch.Tensor:
        """(Q,) integer leads -> (Q, C) gate values in (0, 1)."""
        self._check_lead(lead)
        return torch.sigmoid(self.gate_logit)[lead]

    def forward(self, oi: torch.Tensor, neural: torch.Tensor,
                lead: torch.Tensor) -> torch.Tensor:
        """``oi``/``neural`` (Q, C); ``lead`` (Q,) integer in 0..max_lead."""
        g = self.gates(lead)
        # detach here, not at the call site: the frozen background must not
        # receive gradient no matter who assembled it
        return oi.detach() + g * (neural - oi.detach())
