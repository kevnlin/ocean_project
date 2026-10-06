"""Closed-form tests for the fixed baselines at scattered query points.

Like tests/test_oi.py these pin the arithmetic, not a regression snapshot, and
need no data store.
"""
import numpy as np
import pytest

from ocean_tokenizer.oi import oi_level
from ocean_tokenizer.point_baselines import (Scores, band_of_levels, nearest_profile,
                                             oi_points, point_sweep)


def _obs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-10, 10, n), rng.uniform(0, 20, n), rng.normal(size=n)


# --------------------------------------------------------------------------
# nearest_profile
# --------------------------------------------------------------------------
def test_nearest_profile_takes_the_closest_value():
    lat = np.array([0.0, 0.0, 5.0]); lon = np.array([0.0, 10.0, 5.0])
    out = nearest_profile(lat, lon, np.array([1.0, 2.0, 3.0]),
                          np.array([0.1, 0.0, 4.9]), np.array([0.2, 9.5, 5.1]))
    assert np.array_equal(out, [1.0, 2.0, 3.0])


def test_nearest_profile_skips_non_finite_levels():
    """The closest profile has no value at this level: the next one is used."""
    out = nearest_profile(np.array([0.0, 0.0]), np.array([0.0, 3.0]),
                          np.array([np.nan, 7.0]), np.array([0.0]), np.array([0.1]))
    assert np.array_equal(out, [7.0])


def test_nearest_profile_crosses_the_date_line():
    """359.9 E is 0.2 deg from 0.1 E, not 359.8 deg."""
    out = nearest_profile(np.array([0.0, 0.0]), np.array([359.9, 180.0]),
                          np.array([4.0, 9.0]), np.array([0.0]), np.array([0.1]))
    assert np.array_equal(out, [4.0])


def test_nearest_profile_without_observations_is_background():
    out = nearest_profile(np.array([0.0]), np.array([0.0]), np.array([np.nan]),
                          np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.array_equal(out, [0.0, 0.0])


# --------------------------------------------------------------------------
# optimal interpolation at scattered queries
# --------------------------------------------------------------------------
def test_oi_points_equals_oi_level_on_the_same_points():
    """Scattered queries are the gridded solver's points, in any layout."""
    lat, lon, val = _obs()
    lat2d, lon2d = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(2, 18, 7),
                               indexing="ij")
    grid = oi_level(lat, lon, val, lat2d, lon2d, np.ones_like(lat2d, dtype=bool),
                    L_km=400.0, gamma=0.1, k=12)
    pts = oi_points(lat, lon, val, lat2d.ravel(), lon2d.ravel(),
                    L_km=400.0, gamma=0.1, k=12)
    assert pts.shape == (lat2d.size,)
    assert np.allclose(pts, grid.ravel(), rtol=0, atol=1e-10)


def test_oi_points_returns_the_observation_as_gamma_vanishes():
    """A query on top of an observation gets that value back when gamma -> 0."""
    lat, lon, val = _obs(n=25, seed=1)
    out = oi_points(lat, lon, val, lat[:5], lon[:5], L_km=300.0, gamma=1e-9, k=10)
    assert np.allclose(out, val[:5], atol=1e-5)


def test_oi_points_ignores_non_finite_observations():
    lat, lon, val = _obs(n=40, seed=2)
    holed = val.copy(); holed[::4] = np.nan
    keep = np.isfinite(holed)
    q_lat, q_lon = np.array([0.0, 3.0]), np.array([10.0, 12.0])
    a = oi_points(lat, lon, holed, q_lat, q_lon, L_km=500.0, gamma=0.1, k=8)
    b = oi_points(lat[keep], lon[keep], val[keep], q_lat, q_lon,
                  L_km=500.0, gamma=0.1, k=8)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_oi_points_without_observations_is_background():
    out = oi_points(np.array([0.0]), np.array([0.0]), np.array([np.nan]),
                    np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                    L_km=500.0, gamma=0.1, k=8)
    assert np.array_equal(out, [0.0, 0.0])


def test_point_sweep_sub_k_matches_a_fresh_solve():
    """One geometry at the largest k serves every smaller k exactly."""
    lat, lon, val = _obs(n=50, seed=3)
    q_lat, q_lon = np.array([-2.0, 4.0, 7.0]), np.array([5.0, 9.0, 15.0])
    sweep, v = point_sweep(lat, lon, val, q_lat, q_lon, k=20)
    assert np.allclose(sweep.sub_k(6).analyse(v, 350.0, 0.03),
                       oi_points(lat, lon, val, q_lat, q_lon, 350.0, 0.03, 6),
                       rtol=0, atol=1e-10)


# --------------------------------------------------------------------------
# pooled scoring (the arithmetic of score() in 62_sanity_train.py)
# --------------------------------------------------------------------------
LEVELS = np.array([5.0, 50.0, 200.0, 500.0, 985.0])
STD = {"TEMP": np.array([2.0, 1.0, 0.5, 0.25, 0.1]), "SALT": np.ones(5)}


def test_band_of_levels_follows_the_training_script():
    assert list(band_of_levels(LEVELS)) == ["0-100m", "0-100m", "100-300m",
                                            "300-700m", "700-1400m"]


def test_scores_hand_computed_case():
    s = Scores(LEVELS, STD)
    # two cells in 0-100 m (levels 0 and 1), one in 300-700 m (level 3)
    s.add("TEMP", [1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0, 1, 3])
    r = s.result()["TEMP"]
    # squared errors 1, 4, 1; squared targets 0, 4, 1; std^2 4, 1, 0.0625
    assert r["n"] == 3 and r["unit"] == "degC"
    assert r["rmse_z"] == pytest.approx(np.sqrt(6 / 3))
    assert r["climatology_z"] == pytest.approx(np.sqrt(5 / 3))
    assert r["J"] == pytest.approx(np.sqrt(6 / 5))
    assert r["rmse_physical"] == pytest.approx(np.sqrt((4 + 4 + 0.0625) / 3))
    assert r["by_band_z"]["0-100m"] == pytest.approx(np.sqrt(5 / 2))
    assert r["by_band_z"]["300-700m"] == pytest.approx(1.0)
    assert np.isnan(r["by_band_z"]["100-300m"])
    assert r["by_band_physical"]["0-100m"] == pytest.approx(np.sqrt(8 / 2))
    assert r["by_band_J"]["0-100m"] == pytest.approx(np.sqrt(8 / 4))
    assert r["by_band_J"]["300-700m"] == pytest.approx(1.0)


def test_scores_skip_non_finite_targets_and_pool_across_calls():
    s = Scores(LEVELS, STD)
    s.add("SALT", [1.0, 5.0], [0.0, np.nan], [0, 1])
    s.add("SALT", [3.0], [0.0], [2])
    r = s.result()
    assert r["SALT"]["n"] == 2
    assert r["SALT"]["rmse_z"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert "TEMP" not in r
    assert r["macro_z"] == pytest.approx(r["SALT"]["rmse_z"])


def test_zero_prediction_scores_j_one():
    """Climatology (zero anomaly) is the J = 1 floor by construction."""
    rng = np.random.default_rng(0)
    t, li = rng.normal(size=200), rng.integers(0, 5, 200)
    s = Scores(LEVELS, STD)
    s.add("TEMP", np.zeros(200), t, li)
    r = s.result()["TEMP"]
    assert r["J"] == pytest.approx(1.0)
    assert r["rmse_z"] == pytest.approx(r["climatology_z"])
