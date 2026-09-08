"""EN4 and ECCO as scoreable baselines at real float positions.

Both products were cited in this repo for a year as the operational ancestors of
our OI baseline and never actually scored.  The adapter makes them rows in the
table by asking the only question that puts them on equal footing with the
model: **what does this product say at the exact place and time a held-out float
measured, and how far is that from what the float measured?**

That is a fair comparison and a demanding one.  EN4 is an objective analysis
that *assimilated the very float we are scoring against*, so it should be very
hard to beat — it has seen the answer.  Reporting it honestly means saying so:
EN4 is an upper reference, not a peer.  ECCO V4r4 is a physics-constrained state
estimate that also ingests Argo, and it stops in 2017, so it cannot appear in
the development or holdout rows at all.

Interpolation
-------------
Trilinear in (lat, lon, depth) onto the float's own position, then the same
train-only z-scoring the model's targets use, so the RMSE is in the same units
as every other row.  Depth is handled separately from the horizontal because the
products' vertical grids differ from ours (EN4 has 26 levels below 1000 m where
we have 16) and a single 3-D interpolator would silently extrapolate past the
deepest common level.  Out-of-range points are NaN and drop out of the score
rather than being filled.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import xarray as xr

KELVIN_OFFSET = 273.15


class GriddedReference:
    """A monthly gridded T/S product, queryable at arbitrary float positions."""

    def __init__(self, name: str, files: list[str], tvar: str, svar: str,
                 latname: str, lonname: str, depthname: str,
                 temp_units: str):
        if not files:
            raise FileNotFoundError(f"{name}: no files")
        self.name = name
        ds = xr.open_mfdataset(sorted(files), combine="by_coords")
        self.temp = ds[tvar]
        self.salt = ds[svar]
        self.lat = np.asarray(ds[latname].values, float)
        lon = np.asarray(ds[lonname].values, float) % 360.0
        self.lon = lon
        self.depth = np.abs(np.asarray(ds[depthname].values, float))
        self.time = np.asarray(ds["time"].values, "datetime64[M]")
        self.temp_units = temp_units
        self._ds = ds
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def en4(root: str, region: str) -> "GriddedReference":
        f = sorted(glob.glob(os.path.join(root, "en4", f"*.{region}.*.nc")))
        return GriddedReference("EN4", f, "temperature", "salinity",
                                "lat", "lon", "depth", "K")

    @staticmethod
    def ecco(root: str, region: str) -> "GriddedReference":
        f = sorted(glob.glob(os.path.join(root, "ecco", f"*.{region}.*.nc")))
        return GriddedReference("ECCO", f, "THETA", "SALT",
                                "latitude", "longitude", "Z", "degC")

    def _month_fields(self, month_index: int):
        """(T, S) volumes for one month, in degC / g-kg, cached."""
        if month_index in self._cache:
            return self._cache[month_index]
        want = np.datetime64("2000-01", "M") + month_index
        j = np.flatnonzero(self.time == want)
        if j.size == 0:
            self._cache[month_index] = (None, None)
            return None, None
        t = np.asarray(self.temp.isel(time=int(j[0])).values, float)
        s = np.asarray(self.salt.isel(time=int(j[0])).values, float)
        if self.temp_units == "K":
            t = t - KELVIN_OFFSET
        # drop degenerate leading axes some products carry (e.g. a size-1 tile)
        t, s = np.squeeze(t), np.squeeze(s)
        self._cache[month_index] = (t, s)
        return t, s

    def _interp_column(self, vol: np.ndarray, lat: float, lon: float,
                       levels: np.ndarray) -> np.ndarray:
        """Bilinear in lat/lon, then linear in depth, no extrapolation."""
        if vol is None:
            return np.full(levels.size, np.nan)
        iy = np.searchsorted(self.lat, lat) - 1
        ix = np.searchsorted(self.lon, lon) - 1
        if not (0 <= iy < self.lat.size - 1 and 0 <= ix < self.lon.size - 1):
            return np.full(levels.size, np.nan)
        wy = (lat - self.lat[iy]) / (self.lat[iy + 1] - self.lat[iy])
        wx = (lon - self.lon[ix]) / (self.lon[ix + 1] - self.lon[ix])
        col = ((1 - wy) * (1 - wx) * vol[:, iy, ix] +
               (1 - wy) * wx * vol[:, iy, ix + 1] +
               wy * (1 - wx) * vol[:, iy + 1, ix] +
               wy * wx * vol[:, iy + 1, ix + 1])
        ok = np.isfinite(col)
        if ok.sum() < 2:
            return np.full(levels.size, np.nan)
        d, v = self.depth[ok], col[ok]
        order = np.argsort(d)
        d, v = d[order], v[order]
        out = np.interp(levels, d, v, left=np.nan, right=np.nan)
        out[(levels < d[0]) | (levels > d[-1])] = np.nan     # never extrapolate
        return out

    def predict(self, month_index: int, lat: np.ndarray, lon: np.ndarray,
                levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(R, L) temperature and salinity at R float positions."""
        t, s = self._month_fields(month_index)
        T = np.empty((lat.size, levels.size))
        S = np.empty((lat.size, levels.size))
        for i, (la, lo) in enumerate(zip(lat, np.asarray(lon) % 360.0)):
            T[i] = self._interp_column(t, la, lo, levels)
            S[i] = self._interp_column(s, la, lo, levels)
        return T, S

    def covers(self, month_index: int) -> bool:
        want = np.datetime64("2000-01", "M") + month_index
        return bool((self.time == want).any())
