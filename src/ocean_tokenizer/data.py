"""Unified data access on a common analysis grid.

Responsibilities
----------------
* open the three standardized zarr stores
* roll WOA23 longitudes onto the CESM2-LE (0-360) convention
* expose a single ocean land/sea mask and target depth grid
* extract dense monthly fields (TEMP/SALT/SST/SSS) as numpy tensors on the
  common grid, and interpolate the WOA23 monthly climatology onto it
* provide a small data-card summariser
"""
from __future__ import annotations
import numpy as np
import xarray as xr

from . import config as C


# --------------------------------------------------------------------------
# Dataset opening
# --------------------------------------------------------------------------
def open_raw(name: str) -> xr.Dataset:
    return xr.open_zarr(C.ZARR[name])


def _phys_clip(arr, var):
    """Set values outside the physically plausible range to NaN."""
    lo, hi = C.PHYS_RANGE[var]
    return np.where((arr >= lo) & (arr <= hi), arr, np.nan).astype(arr.dtype)


def _roll_to_0_360(ds: xr.Dataset) -> xr.Dataset:
    """Roll a (-180..180) longitude dataset onto the (0..360) convention."""
    lon = ds.lon.values
    if lon.min() < 0:
        new_lon = np.where(lon < 0, lon + 360.0, lon)
        ds = ds.assign_coords(lon=new_lon).sortby("lon")
    return ds


# --------------------------------------------------------------------------
# Common grid handles
# --------------------------------------------------------------------------
class CommonGrid:
    """Holds the analysis grid (lat, lon, target depths) and the ocean mask."""

    def __init__(self):
        gt = open_raw(C.GT_SOURCE)
        self.lat = gt.lat.values.astype("float64")
        self.lon = gt.lon.values.astype("float64")
        self.depth_idx = np.asarray(C.DEPTH_INDICES, dtype=int)
        self.depth = gt.depth.values[self.depth_idx].astype("float64")
        # ocean mask: MASK==1 is ocean
        if "MASK" in gt:
            self.ocean = (gt.MASK.values == 1)
        else:
            # derive from finite SST of first timestep
            self.ocean = np.isfinite(gt.SST.isel(time=0).values)
        self.nlat = self.lat.size
        self.nlon = self.lon.size
        self.ndepth = self.depth.size

    def __repr__(self):
        return (f"CommonGrid(lat={self.nlat}, lon={self.nlon}, "
                f"depth={self.ndepth} [{self.depth.min():.0f}-{self.depth.max():.0f} m], "
                f"ocean_frac={self.ocean.mean():.3f})")


# --------------------------------------------------------------------------
# Field extraction (ground truth = CESM2-LE)
# --------------------------------------------------------------------------
def select_month_indices(name: str, years: tuple[int, int]):
    ds = open_raw(name)
    yrs = np.array([t.year for t in ds.time.values])
    return np.where((yrs >= years[0]) & (yrs <= years[1]))[0]


def load_gt_fields(time_indices, grid: CommonGrid) -> dict:
    """Load CESM2-LE TEMP/SALT (depth-subset) and SST/SSS for given time indices.

    Returns dict with arrays:
        TEMP, SALT : (T, D, H, W)
        SST,  SSS  : (T, H, W)
        months     : (T,) calendar month (1-12)
    Land points are set to NaN using the ocean mask.
    """
    ds = open_raw(C.GT_SOURCE)
    sub = ds.isel(time=time_indices)
    out = {}
    T = len(time_indices)
    for v in C.VARS_3D:
        arr = sub[v].isel(depth=grid.depth_idx).values.astype("float32")  # (T,D,H,W)
        out[v] = _phys_clip(arr, v)
    for v in C.VARS_SURF:
        if v in sub:
            out[v] = _phys_clip(sub[v].values.astype("float32"), v)         # (T,H,W)
    # apply ocean mask
    ocean = grid.ocean[None]  # (1,H,W)
    for v in C.VARS_3D:
        out[v] = np.where(ocean[:, None], out[v], np.nan)
    for v in C.VARS_SURF:
        if v in out:
            out[v] = np.where(ocean, out[v], np.nan)
    out["months"] = np.array([t.month for t in sub.time.values], dtype=int)
    out["time"] = np.array([str(t)[:10] for t in sub.time.values])
    return out


# --------------------------------------------------------------------------
# WOA23 climatological prior on the common grid
# --------------------------------------------------------------------------
def woa_prior(grid: CommonGrid) -> dict:
    """Interpolate WOA23 monthly climatology onto the common (lon, depth) grid.

    Returns dict TEMP/SALT : (12, D, H, W) and SST/SSS : (12, H, W),
    indexed by calendar month-1.  Land -> NaN.
    """
    ds = _roll_to_0_360(open_raw("woa23"))
    # interpolate onto common lon (lat already identical), nearest for lon edge safety
    ds = ds.interp(lon=grid.lon, lat=grid.lat, method="linear",
                   kwargs={"fill_value": None})
    out = {}
    for v in C.VARS_3D:
        # interpolate onto target depths
        di = ds[v].interp(depth=grid.depth, method="linear",
                          kwargs={"fill_value": None}).values.astype("float32")  # (12,D,H,W)
        out[v] = _phys_clip(di, v)
    for v in C.VARS_SURF:
        if v in ds:
            out[v] = _phys_clip(ds[v].values.astype("float32"), v)  # (12,H,W)
    ocean = grid.ocean[None]
    for v in C.VARS_3D:
        out[v] = np.where(ocean[:, None], out[v], np.nan)
    for v in C.VARS_SURF:
        if v in out:
            out[v] = np.where(ocean, out[v], np.nan)
    return out


# --------------------------------------------------------------------------
# Data cards
# --------------------------------------------------------------------------
def data_card(name: str) -> str:
    ds = open_raw(name)
    lines = []
    a = ds.attrs
    lines.append(f"# Data Card — `{name}`  ({C.ZARR[name].split('/')[-1]})\n")
    if a:
        lines.append("## Provenance / attributes")
        for k, v in a.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    lines.append("## Dimensions")
    for d, n in ds.sizes.items():
        lines.append(f"- `{d}`: {n}")
    lines.append("")
    lines.append("## Coordinates")
    for cname in ds.coords:
        cv = ds[cname].values
        try:
            if cv.dtype.kind in "fi":
                lines.append(f"- `{cname}`: {cv.size} pts, range [{np.nanmin(cv):.3g}, {np.nanmax(cv):.3g}]")
            else:
                lines.append(f"- `{cname}`: {cv.size} pts, [{str(cv[0])[:19]} .. {str(cv[-1])[:19]}]")
        except Exception:
            lines.append(f"- `{cname}`: {cv.size} pts")
    lines.append("")
    lines.append("## Variables (units, sample statistics)")
    lines.append("")
    lines.append("| var | dims | units | min | mean | max | %finite |")
    lines.append("|-----|------|-------|-----|------|-----|---------|")
    for v in ds.data_vars:
        da = ds[v]
        units = da.attrs.get("units", a.get(f"{v.lower()}_units", ""))
        # sample one timestep to keep it cheap
        samp = da.isel(time=0) if "time" in da.dims else da
        vals = np.asarray(samp.values, dtype="float64")
        finite = np.isfinite(vals)
        if finite.any():
            mn, me, mx = np.nanmin(vals), np.nanmean(vals), np.nanmax(vals)
        else:
            mn = me = mx = np.nan
        dims = ",".join(da.dims)
        lines.append(f"| {v} | ({dims}) | {units} | {mn:.3g} | {me:.3g} | {mx:.3g} | {100*finite.mean():.1f}% |")
    lines.append("")
    return "\n".join(lines)
