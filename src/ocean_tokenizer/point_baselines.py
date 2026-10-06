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

CH = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
#: the depth bands of 62_sanity_train.py: same names, same edges
BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))


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


def band_of_levels(levels, bands=BANDS):
    """Band name of every level, by the rule of 62_sanity_train.py."""
    levels = np.asarray(levels, dtype=float)
    out = []
    for d in levels:
        for name, lo, hi in bands:
            if (lo < d <= hi) or (d <= levels.min() and lo <= 0):
                out.append(name)
                break
        else:
            out.append(bands[-1][0])
    return np.array(out)


class Scores:
    """Pooled errors over cells, accumulated as ``score()`` in 62 does.

    ``add`` takes one channel's cells (prediction and target in z units, and
    each cell's level index); cells with a non-finite target are skipped.
    ``result`` returns the ``scores`` block 62 writes to its summaries:
    ``rmse_z``, ``rmse_physical``, ``unit``, ``J`` (RMSE / RMS of the target),
    ``climatology_z`` (RMS of the target), ``n``, the three ``by_band_*``
    dictionaries and ``macro_z`` (mean ``rmse_z`` over channels).
    """

    def __init__(self, levels, std, bands=BANDS):
        self.bands = bands
        self.band = band_of_levels(levels, bands)
        self.std = {ch: np.asarray(std[ch], dtype=np.float64) for ch in CH}
        # per channel: sum e^2 (z), cells, sum e^2 (physical), sum target^2 (z)
        self.tot = {ch: [0.0, 0, 0.0, 0.0] for ch in CH}
        # per band: sum e^2 (z), cells, sum e^2 (physical), sum target^2 (physical)
        self.by = {ch: {b: [0.0, 0, 0.0, 0.0] for b, _, _ in bands} for ch in CH}

    def add(self, ch, pred_z, target_z, level_index):
        pred = np.asarray(pred_z, dtype=np.float64).ravel()
        tgt = np.asarray(target_z, dtype=np.float64).ravel()
        li = np.asarray(level_index, dtype=int).ravel()
        m = np.isfinite(tgt)
        if not m.any():
            return
        e = (pred[m] - tgt[m]) ** 2
        sd2 = self.std[ch][li[m]] ** 2
        t2 = tgt[m] ** 2
        t = self.tot[ch]
        t[0] += float(e.sum()); t[1] += int(m.sum())
        t[2] += float((e * sd2).sum()); t[3] += float(t2.sum())
        bl = self.band[li[m]]
        for b, _, _ in self.bands:
            k = bl == b
            if k.any():
                v = self.by[ch][b]
                v[0] += float(e[k].sum()); v[1] += int(k.sum())
                v[2] += float((e[k] * sd2[k]).sum())
                v[3] += float((t2[k] * sd2[k]).sum())

    def result(self):
        nan = float("nan")
        out = {}
        for ch in CH:
            se, n, sep, se0 = self.tot[ch]
            if n == 0:
                continue
            rz, r0 = float(np.sqrt(se / n)), float(np.sqrt(se0 / n))
            by = self.by[ch]
            out[ch] = {
                "rmse_z": rz, "rmse_physical": float(np.sqrt(sep / n)),
                "unit": UNITS[ch], "J": rz / max(r0, 1e-9),
                "climatology_z": r0, "n": n,
                "by_band_z": {b: (float(np.sqrt(v[0] / v[1])) if v[1] else nan)
                              for b, v in by.items()},
                "by_band_physical": {b: (float(np.sqrt(v[2] / v[1])) if v[1] else nan)
                                     for b, v in by.items()},
                "by_band_J": {b: (float(np.sqrt(v[2] / v[3])) if v[3] else nan)
                              for b, v in by.items()}}
        out["macro_z"] = float(np.mean([out[ch]["rmse_z"] for ch in CH if ch in out]))
        return out


#: latitude scale of baselines._point_features: the std of the 1-degree grid's
#: cell-centre latitudes (whose mean is 0)
MLP_LAT_STD = float(np.std(np.arange(180) - 89.5))
MLP_FEATURES = ("lat", "sin_lon", "cos_lon", "depth", "sin_month", "cos_month",
                "nearest_TEMP", "nearest_SALT", "nearest_distance")


def mlp_point_features(obs_lat, obs_lon, obs_z, q_lat, q_lon, levels, month):
    """Inputs of the gridded line's pointwise MLP, profiles only, at scattered queries.

    The features of ``baselines._point_features`` with the ``profiles`` input
    alone: position and calendar month, the horizontally nearest input
    profile's z-anomaly at the cell's level for each channel (0 where that
    profile has none) and the chord distance to it on the unit sphere. The
    neighbour is chosen by position, the same one for every level.

    ``obs_z`` is ``{channel: (n, L)}``. Returns ``(Q * L, 9)`` float32 in the
    order of ``MLP_FEATURES``, profile-major: row ``q * L + l`` is query ``q``
    at level ``l``.
    """
    levels = np.asarray(levels, dtype=np.float64)
    q_lat = np.asarray(q_lat, dtype=np.float64).ravel()
    q_lon = np.asarray(q_lon, dtype=np.float64).ravel()
    Q, L = q_lat.size, levels.size
    dist, idx = cKDTree(_lonlat_to_xyz(obs_lat, obs_lon)).query(
        _lonlat_to_xyz(q_lat, q_lon), k=1)
    lon = np.deg2rad(np.repeat(q_lon, L))
    cols = [np.repeat(q_lat, L) / MLP_LAT_STD, np.sin(lon), np.cos(lon),
            np.tile((levels - levels.mean()) / (levels.std() + 1e-6), Q),
            np.full(Q * L, np.sin(2 * np.pi * month / 12)),
            np.full(Q * L, np.cos(2 * np.pi * month / 12))]
    cols += [np.nan_to_num(np.asarray(obs_z[ch], dtype=np.float64)[idx], nan=0.0).ravel()
             for ch in CH]
    cols.append(np.repeat(dist, L))
    return np.stack(cols, axis=1).astype("float32")
