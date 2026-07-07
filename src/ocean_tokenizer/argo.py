"""Synthetic Argo-like profile generation.

Argo floats sample sparse vertical columns of the ocean.  We emulate this by
randomly drawing ocean (lat, lon) columns from the CESM2-LE 3D TEMP/SALT fields
of a given monthly snapshot and returning their full vertical profiles together
with location/time metadata.
"""
from __future__ import annotations
import numpy as np


def sample_profiles(fields: dict, t: int, grid, n_profiles: int, rng) -> dict:
    """Draw `n_profiles` random ocean columns from monthly field index `t`.

    Returns dict:
        ij      : (P,2) int grid indices (lat_i, lon_j)
        lat,lon : (P,)  coordinates
        month   : (P,)  calendar month
        TEMP,SALT : (P, D) vertical profiles
    """
    ocean = grid.ocean                       # (H,W) bool
    oi, oj = np.where(ocean)
    pick = rng.choice(oi.size, size=min(n_profiles, oi.size), replace=False)
    ii, jj = oi[pick], oj[pick]
    out = {
        "ij": np.stack([ii, jj], axis=1),
        "lat": grid.lat[ii],
        "lon": grid.lon[jj],
        "month": np.full(ii.size, fields["months"][t], dtype=int),
    }
    for v in ("TEMP", "SALT"):
        out[v] = fields[v][t][:, ii, jj].T   # (P, D)
    return out


def build_obs_grid(prof: dict, grid, var: str) -> np.ndarray:
    """Scatter sampled profiles back onto a (D,H,W) grid (NaN where unobserved).

    This is the gridded sparse-observation tensor consumed by the U-Net and the
    nearest/interpolation baseline.
    """
    D = grid.ndepth
    obs = np.full((D, grid.nlat, grid.nlon), np.nan, dtype="float32")
    ii, jj = prof["ij"][:, 0], prof["ij"][:, 1]
    obs[:, ii, jj] = prof[var].T             # (D,P) -> scatter
    return obs
