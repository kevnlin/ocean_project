"""Analytic unit tests for the optimal-interpolation baseline.

Every expectation here is a closed-form consequence of the OI weight equation

    zhat_g = C_go (C_oo + gamma I)^{-1} d,   C(r) = exp(-r^2 / (2 L^2))

so these tests pin the *math*, not a regression snapshot.  They need no data
store and run in well under a second.
"""
import numpy as np
import pytest

from ocean_tokenizer.oi import (LevelSweep, _chord_to_arc_km, _lonlat_to_xyz,
                                great_circle_km, oi_level)


def _tiny_grid():
    lat2d, lon2d = np.meshgrid(np.linspace(-10, 10, 21),
                               np.linspace(0, 20, 21), indexing="ij")
    return lat2d, lon2d, np.ones_like(lat2d, dtype=bool)


# --------------------------------------------------------------------------
# The five tests the plan specifies
# --------------------------------------------------------------------------
def test_zero_obs_returns_background():
    """No observations -> the analysis is exactly the background (zero anomaly)."""
    lat2d, lon2d, mask = _tiny_grid()
    out = oi_level(np.array([]), np.array([]), np.array([]),
                   lat2d, lon2d, mask)
    assert np.allclose(out[mask], 0.0)


def test_single_obs_shrinkage():
    """At the obs location with one obs: zhat = d / (1 + gamma).

    OI never interpolates exactly -- gamma shrinks the analysis toward the
    background.  Getting `d` back here would mean gamma was dropped.
    """
    lat2d, lon2d, mask = _tiny_grid()
    out = oi_level(np.array([0.0]), np.array([10.0]), np.array([2.0]),
                   lat2d, lon2d, mask, L_km=500.0, gamma=0.25, k=5)
    i, j = 10, 10   # grid cell exactly at (0, 10)
    assert abs(lat2d[i, j]) < 1e-9 and abs(lon2d[i, j] - 10.0) < 1e-9
    assert np.isclose(out[i, j], 2.0 / 1.25, atol=1e-6)


def test_duplicate_obs_saturate():
    """Two identical obs at one point: zhat = 2d/(2+gamma), NOT 2x the single-obs
    answer.  This is the redundancy handling that a naive weighted average lacks."""
    lat2d, lon2d, mask = _tiny_grid()
    g = 0.25
    out = oi_level(np.array([0.0, 0.0]), np.array([10.0, 10.0]),
                   np.array([2.0, 2.0]), lat2d, lon2d, mask,
                   L_km=500.0, gamma=g, k=5)
    assert np.isclose(out[10, 10], 2 * 2.0 / (2 + g), atol=1e-6)


def test_far_from_obs_returns_background():
    """~1500 km from the only obs with L=50 km -> correlation underflows, so the
    analysis must relax to the background rather than extrapolate."""
    lat2d, lon2d, mask = _tiny_grid()
    out = oi_level(np.array([0.0]), np.array([10.0]), np.array([2.0]),
                   lat2d, lon2d, mask, L_km=50.0, gamma=0.1, k=5)
    assert abs(out[0, 0]) < 1e-4


def test_localized_matches_full_solve():
    """With k = n the k-NN localisation is not an approximation: it must
    reproduce the dense global solve to machine precision."""
    rng = np.random.default_rng(0)
    lat2d, lon2d, mask = _tiny_grid()
    n, L, g = 12, 300.0, 0.1
    olat = rng.uniform(-8, 8, n); olon = rng.uniform(2, 18, n)
    d = rng.normal(size=n)
    out = oi_level(olat, olon, d, lat2d, lon2d, mask, L_km=L, gamma=g, k=n)
    q = _lonlat_to_xyz(np.array([lat2d[5, 5]]), np.array([lon2d[5, 5]]))
    o = _lonlat_to_xyz(olat, olon)
    r_go = _chord_to_arc_km(np.linalg.norm(q[:, None] - o[None], axis=-1))[0]
    r_oo = _chord_to_arc_km(np.linalg.norm(o[:, None] - o[None], axis=-1))
    w = np.linalg.solve(np.exp(-0.5 * (r_oo / L) ** 2) + g * np.eye(n),
                        np.exp(-0.5 * (r_go / L) ** 2))
    assert np.isclose(out[5, 5], w @ d, atol=1e-8)


# --------------------------------------------------------------------------
# Extra guards on the failure modes the plan's pitfall table calls out
# --------------------------------------------------------------------------
def test_distance_is_great_circle_km_not_chord_or_degrees():
    """The pitfall table's #1 OI bug: chord/degree/km mix-ups.

    Quarter of the equator along the equator must be a quarter of the
    circumference, and a 1 deg meridional step must be ~111.19 km.
    """
    quarter = great_circle_km(0.0, 0.0, 0.0, 90.0)
    assert np.isclose(quarter, 2 * np.pi * 6371.0 / 4, rtol=1e-9)
    assert np.isclose(great_circle_km(0.0, 0.0, 1.0, 0.0), 111.195, rtol=1e-4)
    # antipodal points must not wrap or NaN out
    assert np.isclose(great_circle_km(0.0, 0.0, 0.0, 180.0),
                      np.pi * 6371.0, rtol=1e-9)


def test_longitude_periodicity():
    """lon 359.5 and lon 0.5 are neighbours, not 359 deg apart.  A flat
    (lat, lon) metric would silently break every basin at the date line."""
    assert np.isclose(great_circle_km(0.0, 359.5, 0.0, 0.5),
                      great_circle_km(0.0, 0.5, 0.0, 1.5), rtol=1e-9)


def test_land_cells_stay_nan():
    """Cells outside the ocean mask must never receive an analysis value."""
    lat2d, lon2d, mask = _tiny_grid()
    mask = mask.copy()
    mask[0, :] = False
    out = oi_level(np.array([0.0]), np.array([10.0]), np.array([2.0]),
                   lat2d, lon2d, mask, L_km=500.0, gamma=0.1, k=5)
    assert np.all(np.isnan(out[0, :]))
    assert np.all(np.isfinite(out[1:, :]))


def test_nonfinite_obs_are_dropped():
    """NaN observations (below-seafloor levels of a profile) must be filtered,
    not propagated into the solve."""
    lat2d, lon2d, mask = _tiny_grid()
    clean = oi_level(np.array([0.0]), np.array([10.0]), np.array([2.0]),
                     lat2d, lon2d, mask, L_km=500.0, gamma=0.25, k=5)
    dirty = oi_level(np.array([0.0, 1.0, 2.0]), np.array([10.0, 11.0, 12.0]),
                     np.array([2.0, np.nan, np.nan]),
                     lat2d, lon2d, mask, L_km=500.0, gamma=0.25, k=5)
    assert np.allclose(clean[mask], dirty[mask], atol=1e-12)


def test_all_nan_obs_returns_background():
    """A depth level where every profile is below the seafloor: background."""
    lat2d, lon2d, mask = _tiny_grid()
    out = oi_level(np.array([0.0, 1.0]), np.array([10.0, 11.0]),
                   np.array([np.nan, np.nan]), lat2d, lon2d, mask)
    assert np.allclose(out[mask], 0.0)


@pytest.mark.parametrize("block", [None, 7, 64])
def test_blocking_does_not_change_the_answer(block):
    """Query-cell blocking is a memory optimisation only: identical numbers."""
    rng = np.random.default_rng(3)
    lat2d, lon2d, mask = _tiny_grid()
    n = 30
    olat = rng.uniform(-9, 9, n); olon = rng.uniform(1, 19, n)
    d = rng.normal(size=n)
    ref = oi_level(olat, olon, d, lat2d, lon2d, mask, L_km=400.0,
                   gamma=0.1, k=12, block=None)
    got = oi_level(olat, olon, d, lat2d, lon2d, mask, L_km=400.0,
                   gamma=0.1, k=12, block=block)
    assert np.allclose(ref[mask], got[mask], atol=1e-12)


def test_k_larger_than_n_is_clipped():
    """Asking for more neighbours than observations must not crash."""
    lat2d, lon2d, mask = _tiny_grid()
    out = oi_level(np.array([0.0, 1.0]), np.array([10.0, 11.0]),
                   np.array([1.0, -1.0]), lat2d, lon2d, mask, k=50)
    assert np.all(np.isfinite(out[mask]))


@pytest.mark.parametrize("L,g,k", [(300.0, 0.1, 5), (800.0, 0.03, 12),
                                   (500.0, 0.3, 20)])
def test_levelsweep_matches_oi_level(L, g, k):
    """The sweep fast path must be numerically identical to the reference
    implementation -- it is an optimisation, never a different method."""
    rng = np.random.default_rng(11)
    lat2d, lon2d, mask = _tiny_grid()
    n = 25
    olat = rng.uniform(-10, 10, n); olon = rng.uniform(0, 20, n)
    d = rng.normal(size=n)
    ref = oi_level(olat, olon, d, lat2d, lon2d, mask, L_km=L, gamma=g, k=k)
    got = LevelSweep(olat, olon, lat2d, lon2d, mask, k=k).analyse(d, L, g)
    assert np.allclose(ref[mask], got[mask], atol=1e-12)


@pytest.mark.parametrize("k", [3, 8, 15])
def test_levelsweep_sub_k_matches_a_freshly_built_geometry(k):
    """Slicing the k nearest columns out of a larger geometry must equal
    building that geometry from scratch -- this is what makes the k-sweep cheap,
    and it only holds because kd-tree neighbours come back distance-sorted."""
    rng = np.random.default_rng(17)
    lat2d, lon2d, mask = _tiny_grid()
    n = 30
    olat = rng.uniform(-10, 10, n); olon = rng.uniform(0, 20, n)
    d = rng.normal(size=n)
    big = LevelSweep(olat, olon, lat2d, lon2d, mask, k=20)
    sliced = big.sub_k(k).analyse(d, 600.0, 0.1)
    fresh = LevelSweep(olat, olon, lat2d, lon2d, mask, k=k).analyse(d, 600.0, 0.1)
    assert np.allclose(sliced[mask], fresh[mask], atol=1e-12)


def test_levelsweep_zero_obs_returns_background():
    lat2d, lon2d, mask = _tiny_grid()
    sw = LevelSweep(np.array([]), np.array([]), lat2d, lon2d, mask, k=5)
    out = sw.analyse(np.array([]), 500.0, 0.1)
    assert np.allclose(out[mask], 0.0)


def test_analysis_is_bounded_by_obs_magnitude():
    """With a positive-definite covariance and gamma>0 the analysis cannot
    exceed the largest observed anomaly -- a cheap guard against a sign or
    solve error blowing the field up."""
    rng = np.random.default_rng(5)
    lat2d, lon2d, mask = _tiny_grid()
    n = 40
    olat = rng.uniform(-10, 10, n); olon = rng.uniform(0, 20, n)
    d = rng.normal(size=n)
    out = oi_level(olat, olon, d, lat2d, lon2d, mask, L_km=600.0,
                   gamma=0.1, k=20)
    assert np.nanmax(np.abs(out)) <= np.abs(d).max() + 1e-9
