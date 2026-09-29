"""A fixed synthetic Argo cohort from CESM2-LE, in the real cohort's layout.

The 2026-09-26 plan asks for an Argo-only interpolation task whose truth is
known, before any more work on real data. This script writes one: profiles at
**uniform random ocean positions** (continuous lat/lon, area-uniform on the
sphere), with CESM2 TEMP/SALT **bilinearly interpolated** to each position at
the real cohort's 20 levels (5-985 m, all CESM2 native levels, so nothing is
interpolated in depth).

  months   2000-01 .. 2005-12 (the 72 months in cesm2_le_full_standard.zarr)
  split    train 2000-2003 / validation 2004 / development (test) 2005
  density  7 600 profiles a month (the real global cohort's median), 80 % input
           ("cohort_float") and 20 % query ("heldout_float"), drawn once with a
           fixed seed, so every arm sees the same inputs and queries
  target   anomaly against CESM2's own TRAIN-month 1-deg monthly climatology
           (the synthetic stand-in for WOA23), stored two ways:
             CLIM_POS_*   climatology interpolated to the profile's position
             CLIM_CELL_*  climatology at the centre of the 1-deg cell

Each profile is its own "float" (unique id): with uniform positions there is no
float clustering to hold out.

The store has one data defect this script removes: 29 coastal columns whose
deepest wet level reads TEMP = 0, SALT = 0 in every month (a regridding fill).
Those grid values are set to NaN BEFORE interpolation, so a profile whose
bilinear stencil touches one loses that level instead of mixing a zero into it.
``TEMP_RAW`` / ``SALT_RAW`` keep what interpolation would have given with the
fill left in, for the salinity audit (42).

  .venv/bin/python experiments/synthetic/41_synth_argo_cohort.py
"""
from __future__ import annotations

import json, os, sys, time

import numpy as np
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "synthetic_argo", "cesm2_uniform.nc")
SEED = 20260926
N_PER_MONTH = 7600
QUERY_FRAC = 0.2
#: the real global cohort's levels (data/argo_cohort/global_global.nc)
LEVELS = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 85.0, 105.0, 125.0,
                   145.0, 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7,
                   707.6, 984.7])
SPLITS = {"train": (2000, 2003), "validation": (2004, 2004),
          "development": (2005, 2005)}

t0 = time.time()
ds = xr.open_zarr(os.path.join(ROOT, "data", "cesm2_le_full_standard.zarr"))
di = [int(np.argmin(np.abs(ds.depth.values - d))) for d in LEVELS]
assert np.allclose(ds.depth.values[di], LEVELS, atol=0.2), ds.depth.values[di]
lat_c = ds.lat.values.astype(float)          # -89.5 .. 89.5
lon_c = ds.lon.values.astype(float)          # 0.5 .. 359.5
NT, NL, NY, NX = ds.sizes["time"], len(di), lat_c.size, lon_c.size
raw = {v: ds[v].isel(depth=di).values.astype("float32") for v in ("TEMP", "SALT")}
fill = (raw["TEMP"] == 0) & (raw["SALT"] == 0)
clean = {v: np.where(fill, np.nan, a) for v, a in raw.items()}
print(f"loaded {NT} months x {NL} levels; fill values removed: {int(fill.sum())} "
      f"in {int(fill.any(axis=(0, 1)).sum())} columns  ({time.time()-t0:.0f}s)",
      flush=True)

month_index = np.arange(NT)                   # months since 2000-01
years = 2000 + month_index // 12
cal = month_index % 12                        # 0..11
train = (years >= SPLITS["train"][0]) & (years <= SPLITS["train"][1])
clim = {}
for v, a in clean.items():
    c = np.empty((12, NL, NY, NX), "float32")
    for m in range(12):
        c[m] = np.nanmean(a[train & (cal == m)], axis=0)
    clim[v] = c

wet0 = np.isfinite(clean["TEMP"][0, 0])       # static wet pattern at 5 m


def stencil(lat, lon):
    """Bilinear stencil on the cell-centre grid: indices and weights."""
    fy = lat - lat_c[0]
    fx = (lon - lon_c[0]) % 360.0
    i0 = np.floor(fy).astype(int)
    j0 = np.floor(fx).astype(int) % NX
    wy, wx = fy - i0, fx - np.floor(fx)
    i1, j1 = np.minimum(i0 + 1, NY - 1), (j0 + 1) % NX
    idx = [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]
    w = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]
    return idx, w


def interp(field, idx, w):
    """field (L, NY, NX) -> (P, L); NaN where any stencil corner is dry."""
    out = 0.0
    for (i, j), wk in zip(idx, w):
        out = out + field[:, i, j].T * wk[:, None]
    return out


rng = np.random.default_rng(SEED)
cols = {k: [] for k in ("lat", "lon", "month_index", "grid_y", "grid_x", "wmo",
                        "float_split")}
arrs = {k: [] for k in ("TEMP", "SALT", "TEMP_RAW", "SALT_RAW", "CLIM_POS_TEMP",
                        "CLIM_POS_SALT", "CLIM_CELL_TEMP", "CLIM_CELL_SALT")}
for t in range(NT):
    lat = np.empty(0); lon = np.empty(0)
    while lat.size < N_PER_MONTH:
        # area-uniform on the sphere, kept inside the outermost cell centres so
        # the stencil never leaves the grid
        la = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 4 * N_PER_MONTH)))
        lo = rng.uniform(0.0, 360.0, la.size)
        ok = np.abs(la) <= lat_c[-1]
        la, lo = la[ok], lo[ok]
        idx, _ = stencil(la, lo)
        # accept only positions whose whole stencil is wet at the surface
        wet = np.logical_and.reduce([wet0[i, j] for i, j in idx])
        lat = np.r_[lat, la[wet]]; lon = np.r_[lon, lo[wet]]
    lat, lon = lat[:N_PER_MONTH], lon[:N_PER_MONTH]
    idx, w = stencil(lat, lon)
    iy = np.floor(lat + 90.0).astype(int)
    ix = np.floor(lon).astype(int) % NX
    for v in ("TEMP", "SALT"):
        arrs[v].append(interp(clean[v][t], idx, w))
        arrs[f"{v}_RAW"].append(interp(raw[v][t], idx, w))
        arrs[f"CLIM_POS_{v}"].append(interp(clim[v][cal[t]], idx, w))
        cc = clim[v][cal[t]][:, iy, ix].T
        # the cell climatology is only defined where the interpolated value is
        # (the containing cell is one of the stencil's corners, so this never
        # masks a finite value; it keeps the NaN pattern identical)
        arrs[f"CLIM_CELL_{v}"].append(np.where(np.isfinite(arrs[v][-1]), cc, np.nan))
    q = np.zeros(N_PER_MONTH, bool)
    q[rng.choice(N_PER_MONTH, int(round(QUERY_FRAC * N_PER_MONTH)), replace=False)] = True
    cols["lat"].append(lat); cols["lon"].append(lon)
    cols["month_index"].append(np.full(N_PER_MONTH, t))
    cols["grid_y"].append(iy); cols["grid_x"].append(ix)
    cols["wmo"].append(np.array([f"S{t:02d}{i:04d}" for i in range(N_PER_MONTH)]))
    cols["float_split"].append(np.where(q, "heldout_float", "cohort_float"))

cols = {k: np.concatenate(v) for k, v in cols.items()}
arrs = {k: np.concatenate(v).astype("float32") for k, v in arrs.items()}
year = 2000 + cols["month_index"] // 12
ysplit = np.full(year.size, "unassigned", dtype=object)
for name, (lo, hi) in SPLITS.items():
    ysplit[(year >= lo) & (year <= hi)] = name

P = year.size
out = xr.Dataset(
    {**{k: (("profile", "level"), v) for k, v in arrs.items()},
     "TEMP_ERR": (("profile", "level"), np.full((P, NL), np.nan, "float32")),
     "SALT_ERR": (("profile", "level"), np.full((P, NL), np.nan, "float32")),
     "lat": ("profile", cols["lat"]), "lon": ("profile", cols["lon"]),
     "grid_y": ("profile", cols["grid_y"]), "grid_x": ("profile", cols["grid_x"]),
     "month_index": ("profile", cols["month_index"]), "year": ("profile", year),
     "wmo": ("profile", cols["wmo"].astype(str)),
     "year_split": ("profile", ysplit.astype(str)),
     "float_split": ("profile", cols["float_split"].astype(str))},
    coords={"level": LEVELS},
    attrs={"region": "synthetic", "grid": [NY, NX], "seed": SEED,
           "source": "cesm2_le_full_standard.zarr, bilinear at uniform random "
                     "ocean positions", "splits": json.dumps(SPLITS),
           "fill_values_removed": int(fill.sum()),
           "fill_columns": int(fill.any(axis=(0, 1)).sum())})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_netcdf(OUT)
nf = np.isfinite(arrs["SALT"]).mean(axis=0)
print(f"{P:,} profiles ({N_PER_MONTH}/month x {NT}), "
      f"{int((cols['float_split'] == 'heldout_float').sum()):,} queries; "
      f"finite fraction by level {nf.min():.3f}-{nf.max():.3f}; "
      f"values touched by the fill: {int((np.isfinite(arrs['SALT_RAW']) & ~np.isfinite(arrs['SALT'])).sum())}"
      f"\n-> {OUT}  ({time.time()-t0:.0f}s)")
