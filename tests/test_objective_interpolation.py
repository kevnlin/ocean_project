"""Causal local objective interpolation (mentor doc §2.5).

The row this backs is deterministic and has zero trainable parameters, so
every property below is a hard invariant rather than something training can
paper over.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.objective_interpolation import (
    OISettings, ObjectiveInterpolation, superobservations)


S = OISettings()          # the doc's frozen settings


def _obs(n=8, seed=0, t=0.0):
    """(n,4) coords in (x, y, z, t) and (n,2) T/S values."""
    g = torch.Generator().manual_seed(seed)
    c = torch.stack([torch.rand(n, generator=g),          # x in [0,1]
                     torch.rand(n, generator=g),          # y in [0,1]
                     torch.rand(n, generator=g),          # z in [0,1]
                     torch.full((n,), t)], dim=-1)
    v = torch.randn(n, 2, generator=g)
    return c, v


# --------------------------------------------------------------------------
# Zero trainable parameters
# --------------------------------------------------------------------------
def test_has_no_trainable_parameters():
    oi = ObjectiveInterpolation(S)
    assert sum(p.numel() for p in oi.parameters()) == 0


# --------------------------------------------------------------------------
# Background behaviour
# --------------------------------------------------------------------------
def test_no_observations_returns_the_zero_anomaly_background():
    oi = ObjectiveInterpolation(S)
    q = torch.rand(5, 4)
    out = oi(q, torch.zeros(5), torch.zeros(0, 4), torch.zeros(0, 2),
             torch.zeros(0))
    assert torch.equal(out, torch.zeros(5, 2))


def test_far_beyond_the_cutoff_returns_the_background():
    """Past the scaled-distance cutoff the kernel is dropped entirely."""
    oi = ObjectiveInterpolation(S)
    c = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[5.0, 5.0]])
    q = torch.tensor([[0.9, 0.9, 0.9, 0.0]])         # many length scales away
    out = oi(q, torch.zeros(1), c, v, torch.full((1,), 0.08))
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-6)


# --------------------------------------------------------------------------
# The doc's formula at an observation location
# --------------------------------------------------------------------------
def test_value_at_an_observation_follows_the_documented_formula():
    """OI = (rho0*0 + r*k*v) / (rho0 + r*k) with k=1 at zero distance.

    Note this is NOT the observation value: the doc says "exact causal point
    observations are reproduced exactly", but with rho0 = 0.30 the background
    always retains weight rho0/(rho0 + r). The formula is unambiguous, so the
    formula is what is implemented; the shrinkage is asserted here so the
    discrepancy with that sentence is visible rather than silent.
    """
    oi = ObjectiveInterpolation(S)
    c = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    v = torch.tensor([[2.0, -1.0]])
    sig = torch.full((1,), 0.08)
    out = oi(c, torch.zeros(1), c, v, sig)
    r = sig.item() ** (-S.noise_exponent)
    expect = v * r / (S.rho0 + r)
    assert torch.allclose(out, expect, atol=1e-6)
    assert not torch.allclose(out, v)        # shrinkage is real


def test_shrinkage_vanishes_as_the_background_precision_goes_to_zero():
    s0 = OISettings(rho0=0.0)
    oi = ObjectiveInterpolation(s0)
    c = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    v = torch.tensor([[2.0, -1.0]])
    out = oi(c, torch.zeros(1), c, v, torch.full((1,), 0.08))
    assert torch.allclose(out, v, atol=1e-6)


# --------------------------------------------------------------------------
# Causality (doc §3.4)
# --------------------------------------------------------------------------
def test_an_observation_later_than_the_query_cutoff_is_ignored():
    oi = ObjectiveInterpolation(S)
    c = torch.tensor([[0.5, 0.5, 0.5, 3.0]])        # 3 months in the future
    v = torch.tensor([[9.0, 9.0]])
    q = torch.tensor([[0.5, 0.5, 0.5, 3.0]])
    out = oi(q, torch.zeros(1), c, v, torch.full((1,), 0.08))
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-6), \
        "a future observation must not reach a causal query"


def test_a_past_observation_is_used():
    oi = ObjectiveInterpolation(S)
    c = torch.tensor([[0.5, 0.5, 0.5, -1.0]])
    v = torch.tensor([[9.0, 9.0]])
    q = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    out = oi(q, torch.zeros(1), c, v, torch.full((1,), 0.08))
    assert out.abs().sum() > 0.1


# --------------------------------------------------------------------------
# Superobservations — exact duplicates merge, they do not accumulate
# --------------------------------------------------------------------------
def test_exact_duplicates_merge_into_one_superobservation():
    c = torch.tensor([[0.5, 0.5, 0.5, 0.0]]).repeat(4, 1)
    v = torch.tensor([[2.0, 1.0]]).repeat(4, 1)
    sig = torch.full((4,), 0.08)
    cc, vv, ss = superobservations(c, v, sig)
    assert cc.shape[0] == 1
    assert torch.allclose(vv, torch.tensor([[2.0, 1.0]]))


def test_superobservation_value_is_inverse_noise_weighted():
    c = torch.tensor([[0.5, 0.5, 0.5, 0.0]]).repeat(2, 1)
    v = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    sig = torch.tensor([1.0, 0.5])           # second is 4x more reliable (1/s^2)
    _, vv, _ = superobservations(c, v, sig)
    w = torch.tensor([1.0, 2.0])             # inverse-noise weights, exponent 1
    expect = (w[:, None] * v).sum(0) / w.sum()
    assert torch.allclose(vv[0], expect, atol=1e-6)


def test_distinct_observations_are_not_merged():
    c, v = _obs(6, seed=3)
    cc, _, _ = superobservations(c, v, torch.full((6,), 0.08))
    assert cc.shape[0] == 6


def test_duplicates_do_not_double_the_pull_on_the_analysis():
    """Four copies of one observation must not out-vote the background 4x."""
    oi = ObjectiveInterpolation(S)
    q = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    c1 = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    v1 = torch.tensor([[2.0, 2.0]])
    s1 = torch.full((1,), 0.08)
    one = oi(q, torch.zeros(1), c1, v1, s1)
    four = oi(q, torch.zeros(1), c1.repeat(4, 1), v1.repeat(4, 1), s1.repeat(4))
    assert torch.allclose(one, four, atol=1e-6)
