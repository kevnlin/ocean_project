"""Masked training losses (mentor doc §2.6).

``CBottleMaskedLoss`` is the faithful cBottle-style **per-sample
valid-fraction** normalisation:

    valid_b = target_mask_b AND finite(target_b)
    L_b     = sum(valid_b * squared_error_b) / max(sum(valid_b), 1)
    L       = mean(L_b over samples with at least one valid target)

Per-sample normalisation is the whole point.  Pooling the numerator and
denominator across the batch lets a densely-supervised sample dominate a
sparsely-supervised one; normalising within each sample first gives every
sample the same weight regardless of how many of its targets survived masking.

The doc is explicit that the group-balanced variant is a *separate* thing and
"must not be attributed to cBottle".  It is therefore not implemented here at
all — adding it to this class, however convenient, would make the attribution
wrong the moment someone reads the class name.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CBottleMaskedLoss(nn.Module):
    """Per-sample valid-fraction normalised masked MSE."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                target_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``pred``/``target`` (B, N, C); ``target_mask`` (B, N, C) bool."""
        valid = torch.isfinite(target)
        if target_mask is not None:
            valid = valid & target_mask.to(torch.bool)
        vf = valid.to(pred.dtype)

        # Replace non-finite targets BEFORE the subtraction.  Masking after the
        # fact is not enough: 0 * NaN is NaN, so a single NaN target would
        # propagate through both the value and the backward pass.
        safe = torch.where(valid, target, torch.zeros_like(target))
        se = (pred - safe) ** 2 * vf

        dims = tuple(range(1, pred.ndim))
        num = se.sum(dim=dims)
        den = vf.sum(dim=dims)
        per_sample = num / den.clamp(min=1.0)

        contributing = den > 0
        if not bool(contributing.any()):
            return pred.sum() * 0.0            # finite, and keeps the graph
        return per_sample[contributing].mean()
