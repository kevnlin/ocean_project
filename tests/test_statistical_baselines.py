"""121: damped persistence, one least-squares factor per lead, variable and depth band."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "stat121", Path(__file__).resolve().parents[1] / "experiments/synthetic/121_statistical_baselines.py")
stat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stat)

LEVELS_M = np.array([5., 150., 800.])


def test_the_factor_is_the_least_squares_damping_of_each_group():
    rng = np.random.default_rng(0)
    n = 30000
    lead, level = np.repeat([0, 6], n // 2), np.tile(np.arange(3), n // 3)
    base = rng.normal(size=(n, 2))
    damping = np.where(lead == 0, 1.0, 0.4)[:, None] * np.array([1., .8])
    target = damping * base + rng.normal(scale=.3, size=(n, 2))
    target[::7, 1] = np.nan
    alphas = stat.fit_alphas(base, target, lead, level, LEVELS_M)
    assert alphas["6"]["TEMP"]["0-100m"] == pytest.approx(.4, abs=.02)
    assert alphas["6"]["SALT"]["700-1400m"] == pytest.approx(.32, abs=.02)
    assert alphas["0"]["TEMP"]["100-300m"] == pytest.approx(1., abs=.02)


def test_applying_scales_each_cell_by_its_group_factor():
    alphas = {"6": {"TEMP": {"0-100m": .5, "700-1400m": .25}, "SALT": {"0-100m": .2, "700-1400m": .1}}}
    out = stat.damped(np.ones((2, 2)), np.array([6, 6]), np.array([0, 2]), LEVELS_M, alphas)
    assert out.tolist() == [[.5, .2], [.25, .1]]
