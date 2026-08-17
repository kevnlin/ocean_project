"""cBottle-style masked loss (mentor doc §2.6).

    valid_b = target_mask_b AND finite(target_b)
    L_b     = sum(valid_b * squared_error_b) / max(sum(valid_b), 1)
    L       = mean(L_b over samples with at least one valid target)

The doc is emphatic that this is *per-sample valid-fraction* normalisation and
that the group-balanced variant must not be attributed to cBottle, so the two
are tested as distinct things.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.losses import CBottleMaskedLoss


def test_all_valid_reduces_to_plain_mse():
    loss = CBottleMaskedLoss()
    p = torch.randn(3, 10, 2)
    y = torch.randn(3, 10, 2)
    m = torch.ones(3, 10, 2, dtype=torch.bool)
    assert torch.allclose(loss(p, y, m), (p - y).pow(2).mean(), atol=1e-6)


def test_masked_entries_do_not_contribute():
    loss = CBottleMaskedLoss()
    p = torch.zeros(1, 4, 1)
    y = torch.zeros(1, 4, 1)
    y[0, 2, 0] = 100.0                       # huge error, but masked out
    m = torch.ones(1, 4, 1, dtype=torch.bool)
    m[0, 2, 0] = False
    assert torch.allclose(loss(p, y, m), torch.zeros(()), atol=1e-6)


def test_nan_targets_are_dropped_even_when_the_mask_says_valid():
    """valid = mask AND finite — a NaN target must not poison the loss."""
    loss = CBottleMaskedLoss()
    p = torch.zeros(1, 4, 1)
    y = torch.zeros(1, 4, 1)
    y[0, 1, 0] = float("nan")
    m = torch.ones(1, 4, 1, dtype=torch.bool)
    out = loss(p, y, m)
    assert torch.isfinite(out) and torch.allclose(out, torch.zeros(()), atol=1e-6)


def test_normalisation_is_per_sample_not_pooled():
    """A sample with few valid targets weighs the same as a dense one.

    Pooling would let the dense sample dominate; that is exactly the
    difference the cBottle scheme exists to remove.
    """
    loss = CBottleMaskedLoss()
    p = torch.zeros(2, 100, 1)
    y = torch.zeros(2, 100, 1)
    m = torch.zeros(2, 100, 1, dtype=torch.bool)
    m[0, :99] = True                          # dense sample, error 0
    m[1, :1] = True                           # sparse sample, error 2
    y[1, 0, 0] = 2.0
    # per-sample: (0 + 4) / 2 = 2.0 ; pooled would be 4/100 = 0.04
    assert torch.allclose(loss(p, y, m), torch.tensor(2.0), atol=1e-6)


def test_samples_with_no_valid_target_are_excluded_from_the_mean():
    loss = CBottleMaskedLoss()
    p = torch.zeros(3, 4, 1)
    y = torch.zeros(3, 4, 1)
    y[1] = 2.0
    m = torch.zeros(3, 4, 1, dtype=torch.bool)
    m[1] = True                               # only sample 1 has any target
    # mean over the ONE contributing sample = 4.0, not 4/3
    assert torch.allclose(loss(p, y, m), torch.tensor(4.0), atol=1e-6)


def test_no_valid_targets_anywhere_gives_a_finite_zero():
    loss = CBottleMaskedLoss()
    p = torch.randn(2, 5, 1)
    y = torch.randn(2, 5, 1)
    m = torch.zeros(2, 5, 1, dtype=torch.bool)
    out = loss(p, y, m)
    assert torch.isfinite(out) and float(out) == 0.0


def test_loss_is_differentiable_through_the_prediction():
    loss = CBottleMaskedLoss()
    p = torch.randn(2, 6, 2, requires_grad=True)
    y = torch.randn(2, 6, 2)
    m = torch.ones(2, 6, 2, dtype=torch.bool)
    loss(p, y, m).backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_a_nan_target_does_not_produce_a_nan_gradient():
    """0 * NaN = NaN would poison the backward pass, not just the value."""
    loss = CBottleMaskedLoss()
    p = torch.randn(1, 4, 1, requires_grad=True)
    y = torch.zeros(1, 4, 1)
    y[0, 2, 0] = float("nan")
    m = torch.ones(1, 4, 1, dtype=torch.bool)
    loss(p, y, m).backward()
    assert torch.isfinite(p.grad).all()
