"""Optimal interpolation (objective analysis) baseline.

Why this exists
---------------
OI is the operational standard for gridded T/S reconstruction from profiles:
EN4, the Roemmich-Gilson Argo climatology and ISAS are all OI-family products.
Until a learned method beats a *properly tuned* OI it has not demonstrated
anything an operational centre would care about.  ``predict_nearest`` in
:mod:`baselines` is a distance-gated nearest fill, not OI -- it has no
covariance model and therefore cannot handle observation redundancy.

Method (Bretherton, Davis & Fandry 1976)
----------------------------------------
Level-by-level 2-D OI **in z-scored anomaly space** -- the same space every
learned method in this repo trains in.  In that space the background is the
train-only climatology, i.e. identically zero, and the background variance is
one, so the analysis at grid point ``g`` given the ``k`` nearest observed
anomalies ``d`` is

    zhat_g = C_go (C_oo + gamma I)^{-1} d,      C(r) = exp(-r^2 / (2 L^2))

with ``r`` the great-circle distance in km, ``L`` the background-error
correlation length scale and ``gamma`` the observation-error / background-error
variance ratio.  Sharing the anomaly target, the normalisation and the scoring
mask with the U-Net is what makes the comparison fair.

Two points worth understanding before tuning:

* OSSE profiles are noise-free, yet ``gamma`` must stay > 0.  It also absorbs
  *representativeness* error -- a point observation stands in for a 1 deg cell
  mean -- and it is what keeps ``C_oo`` invertible when two profiles land close
  together.  ``gamma = 0`` is both physically wrong and numerically fragile.
* The analysis relaxes to zero (= the climatology, = the reported RMSE floor)
  far from data, so OI can never score *worse* than the floor by much.  If a
  sweep returns an OI RMSE above the floor, suspect the distance units or the
  month used for the z-score before suspecting the method.

Localisation
------------
Using only the ``k`` nearest observations per grid point is an approximation of
the dense global solve (the standard operational practice -- the dense system
is ``n x n`` with ``n`` = all profiles).  With ``k = n`` it is exact, which is
what ``tests/test_oi.py::test_localized_matches_full_solve`` pins.  ``k`` is
swept in the tuning script so the bias it introduces is measured, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_EARTH_KM = 6371.0

#: default query-cell block size; bounds the (block, k, k, 3) temporary
DEFAULT_BLOCK = 8192


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def _lonlat_to_xyz(lat_deg, lon_deg):
    """(lat, lon) in degrees -> unit-sphere xyz.

    Working on the sphere rather than in (lat, lon) makes longitude periodicity
    and the pole convergence automatic; a flat metric would break every basin
    at the date line.
    """
    la, lo = np.radians(np.asarray(lat_deg, dtype=np.float64)), \
             np.radians(np.asarray(lon_deg, dtype=np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


def _chord_to_arc_km(chord):
    """Euclidean chord length on the unit sphere -> great-circle arc in km."""
    return 2.0 * R_EARTH_KM * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0.0, 1.0))


def great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    a = _lonlat_to_xyz(lat1, lon1)
    b = _lonlat_to_xyz(lat2, lon2)
    return _chord_to_arc_km(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1))


# --------------------------------------------------------------------------
# Core: one variable, one depth level, one month
# --------------------------------------------------------------------------
def oi_level(obs_lat, obs_lon, obs_val, lat2d, lon2d, ocean_mask,
             L_km=500.0, gamma=0.1, k=20, block=DEFAULT_BLOCK):
    """Optimal interpolation of one 2-D level.

    Parameters
    ----------
    obs_lat, obs_lon, obs_val : (n,)
        Observation coordinates (degrees) and **z-scored anomalies**.
        Non-finite ``obs_val`` entries are dropped (e.g. levels of a profile
        below the seafloor).
    lat2d, lon2d : (H, W)
        Analysis-grid coordinates, degrees.
    ocean_mask : (H, W) bool
        Cells to analyse.  Everything else stays NaN.
    L_km : float
        Background-error correlation length scale, km.
    gamma : float
        Observation-error to background-error variance ratio.
    k : int
        Number of nearest observations used per analysis point (localisation).
        Clipped to the number of available observations.
    block : int or None
        Analysis points processed per chunk.  Memory only -- the result is
        bit-identical for any blocking.  ``None`` processes everything at once.

    Returns
    -------
    (H, W) float64 analysis in z-anomaly space: 0 far from data (= background
    = climatology), NaN outside ``ocean_mask``.
    """
    ocean_mask = np.asarray(ocean_mask, dtype=bool)
    out = np.full(lat2d.shape, np.nan, dtype=np.float64)

    obs_val = np.asarray(obs_val, dtype=np.float64).ravel()
    finite = np.isfinite(obs_val)
    if finite.size == 0 or not finite.any():
        out[ocean_mask] = 0.0                       # no data -> background
        return out
    olat = np.asarray(obs_lat, dtype=np.float64).ravel()[finite]
    olon = np.asarray(obs_lon, dtype=np.float64).ravel()[finite]
    oval = obs_val[finite]

    k = int(min(max(int(k), 1), oval.size))
    oxyz_all = _lonlat_to_xyz(olat, olon)                       # (n,3)
    tree = cKDTree(oxyz_all)

    glat, glon = lat2d[ocean_mask], lon2d[ocean_mask]
    gxyz = _lonlat_to_xyz(glat, glon)                           # (G,3)
    G = gxyz.shape[0]
    res = np.empty(G, dtype=np.float64)

    step = G if block is None else int(block)
    eye = gamma * np.eye(k, dtype=np.float64)[None]             # (1,k,k)
    for s in range(0, G, step):
        e = min(s + step, G)
        chord, idx = tree.query(gxyz[s:e], k=k)
        chord = np.asarray(chord, dtype=np.float64).reshape(e - s, k)
        idx = np.asarray(idx).reshape(e - s, k)

        d_go = _chord_to_arc_km(chord)                          # (B,k)
        oxyz = oxyz_all[idx]                                    # (B,k,3)
        d_oo = _chord_to_arc_km(
            np.linalg.norm(oxyz[:, :, None, :] - oxyz[:, None, :, :], axis=-1))

        C_go = np.exp(-0.5 * (d_go / L_km) ** 2)                # (B,k)
        C_oo = np.exp(-0.5 * (d_oo / L_km) ** 2) + eye          # (B,k,k)
        w = np.linalg.solve(C_oo, C_go[..., None])[..., 0]      # (B,k)
        res[s:e] = (w * oval[idx]).sum(axis=1)

    out[ocean_mask] = res
    return out


# --------------------------------------------------------------------------
# Sweep helper: the geometry does not depend on (L, gamma)
# --------------------------------------------------------------------------
class LevelSweep:
    """Precomputed k-NN geometry for one level, reused across many (L, gamma).

    Profiling one global level (42k ocean cells, 1500 profiles, k=20): the
    kd-tree query and the two great-circle distance matrices cost ~1.4 s while
    building the covariances and solving costs ~0.44 s.  A sweep that rebuilds
    the geometry for every (L, gamma) therefore spends ~75 % of its time
    recomputing identical distances.  This class hoists that work out of the
    loop; ``tests/test_oi.py`` pins it against :func:`oi_level` so the two
    paths cannot silently diverge.

    Note the memory: ``d_oo`` is ``(G, k, k)`` float64 -- ~136 MB at k=20 and
    ~543 MB at k=40 for the global ocean.  Build one level at a time.
    """

    def __init__(self, obs_lat, obs_lon, lat2d, lon2d, ocean_mask, k=20):
        self.ocean_mask = np.asarray(ocean_mask, dtype=bool)
        self.shape = lat2d.shape
        olat = np.asarray(obs_lat, dtype=np.float64).ravel()
        olon = np.asarray(obs_lon, dtype=np.float64).ravel()
        self.n = olat.size
        if self.n == 0:
            self.k = 0
            return
        self.k = int(min(max(int(k), 1), self.n))
        oxyz_all = _lonlat_to_xyz(olat, olon)
        glat, glon = lat2d[self.ocean_mask], lon2d[self.ocean_mask]
        chord, idx = cKDTree(oxyz_all).query(_lonlat_to_xyz(glat, glon), k=self.k)
        G = glat.size
        self.idx = np.asarray(idx).reshape(G, self.k)
        self.d_go = _chord_to_arc_km(
            np.asarray(chord, dtype=np.float64).reshape(G, self.k))
        o = oxyz_all[self.idx]
        self.d_oo = _chord_to_arc_km(
            np.linalg.norm(o[:, :, None, :] - o[:, None, :, :], axis=-1))

    def sub_k(self, k):
        """A view of this geometry restricted to the ``k`` nearest neighbours.

        ``cKDTree.query`` returns neighbours sorted by ascending distance, so
        the first ``k`` columns *are* the k-NN geometry -- exactly, not
        approximately.  This lets a k-sweep build the largest geometry once
        and slice the smaller ones out of it for free.
        """
        k = int(min(max(int(k), 1), self.k)) if self.n else 0
        sub = LevelSweep.__new__(LevelSweep)
        sub.ocean_mask, sub.shape, sub.n, sub.k = self.ocean_mask, self.shape, self.n, k
        if self.n:
            sub.idx = self.idx[:, :k]
            sub.d_go = self.d_go[:, :k]
            sub.d_oo = self.d_oo[:, :k, :k]
        return sub

    def analyse(self, obs_val, L_km, gamma):
        """Solve this level for one (L_km, gamma).  ``obs_val`` must align with
        the ``obs_lat`` / ``obs_lon`` this geometry was built from."""
        out = np.full(self.shape, np.nan, dtype=np.float64)
        if self.n == 0:
            out[self.ocean_mask] = 0.0
            return out
        oval = np.asarray(obs_val, dtype=np.float64).ravel()
        C_go = np.exp(-0.5 * (self.d_go / L_km) ** 2)
        C_oo = (np.exp(-0.5 * (self.d_oo / L_km) ** 2)
                + gamma * np.eye(self.k, dtype=np.float64)[None])
        w = np.linalg.solve(C_oo, C_go[..., None])[..., 0]
        out[self.ocean_mask] = (w * oval[self.idx]).sum(axis=1)
        return out


# --------------------------------------------------------------------------
# Full-field prediction, mirroring the predict_* interface of baselines.py
# --------------------------------------------------------------------------
def _per_var(param, v):
    """Allow a scalar or a {var: value} dict for L_km / gamma / k."""
    if isinstance(param, dict):
        return param[v]
    return param


def predict_oi(sample, anorm, grid, L_km=500.0, gamma=0.1, k=20,
               block=DEFAULT_BLOCK):
    """Full-field OI prediction in **physical units**.

    Mirrors ``baselines.predict_nearest`` / ``predict_clim_floor``: takes the
    per-month ``sample`` dict from :func:`baselines.prepare_month`, returns
    ``{var: (D, H, W)}`` with land as NaN.

    ``L_km`` / ``gamma`` / ``k`` accept either a scalar (shared by TEMP and
    SALT) or a ``{"TEMP": ..., "SALT": ...}`` dict -- the tuning sweep finds
    different optima for the two variables.
    """
    from .baselines import VARS

    mo = sample["month"]
    lat2d, lon2d = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    out = {}
    for v in VARS:
        # z-score the observed columns with THIS month's climatology, exactly
        # as the learned methods do.  A month mismatch here is the classic bug.
        obs_z = anorm.z3d(v, sample["obs"][v], mo)              # (D,H,W)
        Lv, gv, kv = _per_var(L_km, v), _per_var(gamma, v), _per_var(k, v)
        zhat = np.zeros(obs_z.shape, dtype=np.float64)
        for d in range(grid.ndepth):
            m = np.isfinite(obs_z[d])
            zhat[d] = oi_level(lat2d[m], lon2d[m], obs_z[d][m],
                               lat2d, lon2d, grid.ocean,
                               L_km=Lv, gamma=gv, k=kv, block=block)
        pred = anorm.unz3d(v, zhat.astype("float32"), mo)
        out[v] = np.where(grid.ocean[None], pred, np.nan).astype("float32")
    return out
