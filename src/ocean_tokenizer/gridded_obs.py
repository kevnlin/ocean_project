"""Profiles onto the 1-degree grid, and a grid back onto points.

A gridded method (4DVarNet) needs scattered profiles as a field with gaps, and
its gridded answer has to be scored at the held-out profiles' own positions.
Both directions use the cohort's conventions: cell ``(grid_y, grid_x)`` =
``(floor(lat + 90), floor(lon mod 360))`` with centres at -89.5 ... 89.5 and
0.5 ... 359.5, and the bilinear stencil of
``experiments/synthetic/41_synth_argo_cohort.py``, periodic in longitude.
"""
from __future__ import annotations

import numpy as np


def bin_profiles(grid_y, grid_x, values, shape):
    """Mean of the profiles in each cell, level by level.

    ``values`` is ``(n, L)`` with NaN where a profile has no value. Returns
    ``(L, NY, NX)``, NaN in a cell that received no finite value at that level.
    """
    NY, NX = shape
    values = np.asarray(values, dtype=np.float64)
    flat = np.asarray(grid_y, dtype=np.int64) * NX + np.asarray(grid_x, dtype=np.int64)
    out = np.full((values.shape[1], NY * NX), np.nan)
    for l in range(values.shape[1]):
        ok = np.isfinite(values[:, l])
        n = np.bincount(flat[ok], minlength=NY * NX)
        s = np.bincount(flat[ok], weights=values[ok, l], minlength=NY * NX)
        m = n > 0
        out[l, m] = s[m] / n[m]
    return out.reshape(values.shape[1], NY, NX)


def sample_bilinear(field, lat, lon, lat0=-89.5, lon0=0.5):
    """``(C, NY, NX)`` on the 1-degree cell centres -> ``(P, C)`` at the points.

    Periodic in longitude; latitude is clamped to the first and last centre.
    """
    field = np.asarray(field)
    _, NY, NX = field.shape
    fy = np.clip(np.asarray(lat, dtype=np.float64) - lat0, 0.0, NY - 1.0)
    fx = (np.asarray(lon, dtype=np.float64) - lon0) % 360.0
    i0 = np.minimum(np.floor(fy).astype(int), NY - 1)
    i1 = np.minimum(i0 + 1, NY - 1)
    j0 = np.floor(fx).astype(int) % NX
    j1 = (j0 + 1) % NX
    wy, wx = fy - i0, fx - np.floor(fx)
    out = (field[:, i0, j0] * ((1 - wy) * (1 - wx)) + field[:, i0, j1] * ((1 - wy) * wx)
           + field[:, i1, j0] * (wy * (1 - wx)) + field[:, i1, j1] * (wy * wx))
    return out.T
