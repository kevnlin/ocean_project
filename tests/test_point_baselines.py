"""Closed-form tests for the fixed baselines at scattered query points.

Like tests/test_oi.py these pin the arithmetic, not a regression snapshot, and
need no data store.
"""
import numpy as np
import pytest

from ocean_tokenizer.oi import oi_level
from ocean_tokenizer.point_baselines import nearest_profile, oi_points, point_sweep


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
