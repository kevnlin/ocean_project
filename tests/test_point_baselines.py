"""Closed-form tests for the fixed baselines at scattered query points.

Like tests/test_oi.py these pin the arithmetic, not a regression snapshot, and
need no data store.
"""
import numpy as np
import pytest

from ocean_tokenizer.point_baselines import nearest_profile


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
