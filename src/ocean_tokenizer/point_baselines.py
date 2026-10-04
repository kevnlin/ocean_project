"""Fixed (no trainable parameters) baselines at scattered query points.

The synthetic Argo audit scores a model at held-out profiles' own positions
rather than on a grid. These helpers give the non-learned references that
interface -- observations at (lat, lon) with one value per level, queries at
(lat, lon), one analysis per query -- and reproduce the scoring arithmetic of
``experiments/real_data/62_sanity_train.py`` on numpy arrays, so a baseline and
a trained model are pooled by the same formulas.

Everything is in z-scored anomaly space: the background is zero, and a
non-finite observation (a level below the sea floor) is simply absent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .oi import LevelSweep, _lonlat_to_xyz


def _finite(obs_lat, obs_lon, obs_val):
    """Coordinates and values of the observations that are finite at this level."""
    v = np.asarray(obs_val, dtype=np.float64).ravel()
    ok = np.isfinite(v)
    return (np.asarray(obs_lat, dtype=np.float64).ravel()[ok],
            np.asarray(obs_lon, dtype=np.float64).ravel()[ok], v[ok])


def nearest_profile(obs_lat, obs_lon, obs_val, q_lat, q_lon):
    """(Q,) value of the nearest finite observation; 0 where there is none."""
    lat, lon, v = _finite(obs_lat, obs_lon, obs_val)
    q_lat = np.asarray(q_lat, dtype=np.float64).ravel()
    q_lon = np.asarray(q_lon, dtype=np.float64).ravel()
    if v.size == 0:
        return np.zeros(q_lat.size)
    # chord length on the unit sphere is monotone in great-circle distance
    _, idx = cKDTree(_lonlat_to_xyz(lat, lon)).query(_lonlat_to_xyz(q_lat, q_lon), k=1)
    return v[np.asarray(idx).ravel()]


def point_sweep(obs_lat, obs_lon, obs_val, q_lat, q_lon, k):
    """k-NN geometry over one level's finite observations, at scattered queries.

    Returns ``(sweep, values)``: an :class:`oi.LevelSweep` whose analysis has
    one entry per query, and the finite observation values it was built from
    (pass them to ``sweep.analyse``). ``sweep.sub_k`` gives any smaller k from
    the same geometry.
    """
    lat, lon, v = _finite(obs_lat, obs_lon, obs_val)
    q_lat = np.asarray(q_lat, dtype=np.float64).ravel()
    q_lon = np.asarray(q_lon, dtype=np.float64).ravel()
    sweep = LevelSweep(lat, lon, q_lat, q_lon, np.ones(q_lat.size, dtype=bool), k=k)
    return sweep, v


def oi_points(obs_lat, obs_lon, obs_val, q_lat, q_lon, L_km, gamma, k):
    """(Q,) optimal-interpolation analysis at scattered queries (see :mod:`oi`)."""
    sweep, v = point_sweep(obs_lat, obs_lon, obs_val, q_lat, q_lon, k)
    return sweep.analyse(v, L_km, gamma)
