"""Frozen-OI residual with learned lead/channel gates (mentor doc §2.6).

    prediction(q,c) = OI(q,c) + sigmoid(gate_logit[h,c]) * [neural(q,c) - OI(q,c)]

Eight logits: four leads x two variables.  Temperature gates initialise at
0.8, salinity at 0.2.  OI is frozen, detached, and run without an autograd
graph — so "frozen" is tested as an absence of gradient, not assumed.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.oi_residual import OIResidual, MAX_LEAD, N_CHANNELS


def _pieces(Q=7):
    oi = torch.randn(Q, N_CHANNELS)
    neural = torch.randn(Q, N_CHANNELS)
    lead = torch.randint(0, MAX_LEAD + 1, (Q,))
    return oi, neural, lead


# --------------------------------------------------------------------------
# Gate shape and initialisation
# --------------------------------------------------------------------------
def test_there_are_exactly_eight_gate_logits():
    r = OIResidual()
    assert r.gate_logit.shape == (MAX_LEAD + 1, N_CHANNELS) == (4, 2)
    assert r.gate_logit.numel() == 8


def test_temperature_gates_start_at_0_8_and_salinity_at_0_2():
    r = OIResidual()
    g = torch.sigmoid(r.gate_logit)
    assert torch.allclose(g[:, 0], torch.full((4,), 0.8), atol=1e-5)
    assert torch.allclose(g[:, 1], torch.full((4,), 0.2), atol=1e-5)


def test_gates_are_trainable():
    r = OIResidual()
    assert r.gate_logit.requires_grad
    assert sum(p.numel() for p in r.parameters()) == 8


# --------------------------------------------------------------------------
# The blend itself
# --------------------------------------------------------------------------
def test_a_closed_gate_returns_oi_exactly():
    r = OIResidual()
    with torch.no_grad():
        r.gate_logit.fill_(-30.0)              # sigmoid ~ 0
    oi, neural, lead = _pieces()
    assert torch.allclose(r(oi, neural, lead), oi, atol=1e-6)


def test_an_open_gate_returns_the_neural_prediction_exactly():
    r = OIResidual()
    with torch.no_grad():
        r.gate_logit.fill_(30.0)               # sigmoid ~ 1
    oi, neural, lead = _pieces()
    assert torch.allclose(r(oi, neural, lead), neural, atol=1e-6)


def test_the_blend_matches_the_documented_formula():
    r = OIResidual()
    oi, neural, lead = _pieces()
    g = torch.sigmoid(r.gate_logit)[lead]      # (Q, C)
    assert torch.allclose(r(oi, neural, lead), oi + g * (neural - oi), atol=1e-6)


def test_each_lead_uses_its_own_gate():
    """Changing the lead-2 gate must not move a lead-1 prediction."""
    r = OIResidual()
    oi = torch.randn(1, N_CHANNELS)
    neural = torch.randn(1, N_CHANNELS)
    lead1 = torch.tensor([1])
    before = r(oi, neural, lead1)
    with torch.no_grad():
        r.gate_logit[2] += 5.0
    assert torch.allclose(r(oi, neural, lead1), before, atol=1e-6)


def test_each_channel_uses_its_own_gate():
    r = OIResidual()
    oi = torch.randn(1, N_CHANNELS)
    neural = torch.randn(1, N_CHANNELS)
    lead = torch.tensor([0])
    before = r(oi, neural, lead)
    with torch.no_grad():
        r.gate_logit[0, 1] += 5.0              # salinity only
    after = r(oi, neural, lead)
    assert torch.allclose(after[:, 0], before[:, 0], atol=1e-6), "TEMP moved"
    assert not torch.allclose(after[:, 1], before[:, 1]), "SALT did not move"


# --------------------------------------------------------------------------
# OI is frozen — tested as an absence of gradient
# --------------------------------------------------------------------------
def test_no_gradient_flows_into_the_oi_branch():
    r = OIResidual()
    oi = torch.randn(4, N_CHANNELS, requires_grad=True)
    neural = torch.randn(4, N_CHANNELS, requires_grad=True)
    lead = torch.zeros(4, dtype=torch.long)
    r(oi, neural, lead).sum().backward()
    assert oi.grad is None or torch.allclose(oi.grad, torch.zeros_like(oi)), \
        "the frozen OI background must receive no gradient"
    assert neural.grad is not None and neural.grad.abs().sum() > 0


def test_gradient_reaches_the_gates():
    r = OIResidual()
    oi, neural, lead = _pieces()
    r(oi, neural.requires_grad_(True), lead).sum().backward()
    assert r.gate_logit.grad is not None
    assert torch.isfinite(r.gate_logit.grad).all()


# --------------------------------------------------------------------------
# Integer lead indexing (doc §3: "integer residual-gate indexing")
# --------------------------------------------------------------------------
def test_a_float_lead_raises_rather_than_rounding():
    r = OIResidual()
    oi, neural, _ = _pieces()
    with pytest.raises(TypeError, match="lead"):
        r(oi, neural, torch.full((7,), 1.5))


def test_an_out_of_range_lead_raises():
    r = OIResidual()
    oi, neural, _ = _pieces()
    with pytest.raises(ValueError, match="lead"):
        r(oi, neural, torch.full((7,), MAX_LEAD + 1, dtype=torch.long))


def test_a_negative_lead_raises_rather_than_indexing_from_the_end():
    r = OIResidual()
    oi, neural, _ = _pieces()
    with pytest.raises(ValueError, match="lead"):
        r(oi, neural, torch.full((7,), -1, dtype=torch.long))
