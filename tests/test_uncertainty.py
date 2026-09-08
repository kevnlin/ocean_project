"""P6's three required guarantees, plus the calibration diagnostics.

The plan lists exactly three tests for the uncertainty module. They are here as
tests rather than as a claim in a report, because "the mean did not change" is
the sort of thing that is true right up until a refactor makes it false.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from ocean_tokenizer.uncertainty import (CalibratedModel, UncertaintyHead,
                                         state_dict_hash, gaussian_nll,
                                         crps_gaussian, spread_skill,
                                         interval_coverage, reliability)


class FakeBase(nn.Module):
    """Stands in for a registered row: has dropout, and exposes last_omega."""
    def __init__(self, n_out=2):
        super().__init__()
        self.lin = nn.Linear(4, n_out)
        self.drop = nn.Dropout(0.5)
        self.last_omega = torch.rand(17)

    def forward(self, s):
        return self.drop(self.lin(s["query"].float()))


def sample(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"query": torch.rand(n, 4, generator=g, dtype=torch.float64),
            "target": torch.randn(n, 2, generator=g),
            "target_mask": torch.ones(n, 2, dtype=torch.bool)}


# --- the plan's three required tests -------------------------------------
def test_base_checkpoint_hash_unchanged_after_training_the_head():
    base = FakeBase()
    before = state_dict_hash(base.state_dict())
    m = CalibratedModel(base, UncertaintyHead(7))
    opt = torch.optim.Adam(m.head.parameters(), lr=1e-2)
    for i in range(20):
        s = sample(seed=i)
        mean, logvar = m(s)
        loss = gaussian_nll(mean, logvar, s["target"], s["target_mask"])
        opt.zero_grad(); loss.backward(); opt.step()
    assert state_dict_hash(m.base.state_dict()) == before
    assert m.base_hash == before


def test_zero_gradient_reaches_the_base_model():
    base = FakeBase()
    m = CalibratedModel(base, UncertaintyHead(7))
    s = sample()
    mean, logvar = m(s)
    loss = gaussian_nll(mean, logvar, s["target"], s["target_mask"])
    loss.backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in m.base.parameters())
    assert all(not p.requires_grad for p in m.base.parameters())


def test_mean_prediction_is_bit_identical_before_and_after_attaching_the_head():
    base = FakeBase()
    base.eval()
    s = sample()
    with torch.no_grad():
        before = base(s).clone()
    m = CalibratedModel(base, UncertaintyHead(7))
    after, _ = m(s)
    assert torch.equal(before, after)


def test_base_stays_in_eval_even_when_the_wrapper_is_put_in_train_mode():
    """Module.train() recurses; without the override, dropout comes back on."""
    base = FakeBase()
    m = CalibratedModel(base, UncertaintyHead(7))
    m.train()
    assert not m.base.training
    s = sample()
    a, _ = m(s)
    b, _ = m(s)
    assert torch.equal(a, b)          # deterministic => dropout is off


# --- scoring rule ---------------------------------------------------------
def test_nll_is_minimised_at_the_true_variance():
    """A proper score cannot be improved by misreporting the spread."""
    torch.manual_seed(0)
    y = torch.randn(20000, 1) * 2.0
    mean = torch.zeros_like(y)
    mask = torch.ones_like(y, dtype=torch.bool)
    true_lv = torch.full_like(y, float(np.log(4.0)))
    best = gaussian_nll(mean, true_lv, y, mask)
    for wrong in (np.log(1.0), np.log(16.0)):
        assert gaussian_nll(mean, torch.full_like(y, wrong), y, mask) > best


def test_nll_ignores_masked_entries():
    mean = torch.zeros(4, 2); lv = torch.zeros(4, 2)
    y = torch.tensor([[0., 0.], [0., 0.], [99., 99.], [99., 99.]])
    mask = torch.tensor([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=torch.bool)
    assert float(gaussian_nll(mean, lv, y, mask)) == pytest.approx(0.0, abs=1e-6)


# --- diagnostics ----------------------------------------------------------
def test_crps_is_smaller_for_a_sharper_correct_forecast():
    y = np.zeros(1000)
    sharp = crps_gaussian(np.zeros(1000), np.full(1000, 0.5), y).mean()
    broad = crps_gaussian(np.zeros(1000), np.full(1000, 5.0), y).mean()
    assert sharp < broad


def test_crps_penalises_a_confident_wrong_forecast():
    y = np.full(1000, 5.0)
    confident_wrong = crps_gaussian(np.zeros(1000), np.full(1000, 0.1), y).mean()
    honest = crps_gaussian(np.zeros(1000), np.full(1000, 5.0), y).mean()
    assert confident_wrong > honest


def test_spread_skill_slope_is_about_one_when_calibrated():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.2, 3.0, 40000)
    err = rng.normal(0, sigma)
    out = spread_skill(sigma, err)
    assert 0.85 < out["slope"] < 1.15


def test_spread_skill_detects_overconfidence():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.2, 3.0, 40000)
    err = rng.normal(0, 3 * sigma)          # errors 3x the claimed spread
    assert spread_skill(sigma, err)["slope"] > 2.0


def test_interval_coverage_matches_nominal_when_calibrated():
    rng = np.random.default_rng(2)
    n = 200000
    sigma = np.full(n, 1.5)
    y = rng.normal(0, 1.5, n)
    cov = interval_coverage(np.zeros(n), sigma, y)
    assert abs(cov["0.90"] - 0.90) < 0.01
    assert abs(cov["0.50"] - 0.50) < 0.01


def test_reliability_is_uniform_when_calibrated_and_not_otherwise():
    rng = np.random.default_rng(3)
    n = 100000
    y = rng.normal(0, 1.0, n)
    good = reliability(np.zeros(n), np.ones(n), y)
    bad = reliability(np.zeros(n), np.full(n, 0.3), y)   # over-confident
    assert good["tv_from_uniform"] < 0.02
    assert bad["tv_from_uniform"] > good["tv_from_uniform"]


def test_state_dict_hash_is_order_independent_and_change_sensitive():
    a = {"w": torch.ones(3), "b": torch.zeros(2)}
    b = {"b": torch.zeros(2), "w": torch.ones(3)}
    assert state_dict_hash(a) == state_dict_hash(b)
    c = {"w": torch.ones(3), "b": torch.zeros(2) + 1e-7}
    assert state_dict_hash(a) != state_dict_hash(c)
