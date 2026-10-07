"""Surface fields for the synthetic Argo cohort: SST, SSS and a sea-level stand-in.

The satellite-type inputs of the synthetic multi-modal comparison, one 1-degree
field a month for 2000-01 ... 2005-12, written with the variable names of the
real satellite store so `62_sanity_train.py --region synthetic --surface` reads
them through its existing path.

  SST, SSS   CESM2's own surface fields (its 5 m level)
  SLA        steric height relative to 990 dbar, TEOS-10, from CESM2's TEMP and
             SALT on the cohort's 20 levels (`ocean_tokenizer.ssh`); NaN where
             the column does not reach the reference level

Each field is an anomaly against its own train-year (2000-2003) monthly
climatology, the way the cohort's T/S target is defined.

Two limitations, to be repeated wherever a result uses this store:

* SST and SSS are noise-free and ARE the 5 m level of the truth, so a method
  given them is handed the answer at the shallowest level.
* SLA is a vertical integral of the T and S being reconstructed: a derived,
  optimistic stand-in for altimetry.

  .venv/bin/python experiments/synthetic/46_synth_surface_fields.py     # ~2 min CPU
"""
from __future__ import annotations

import os, shutil, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORE = os.path.join(ROOT, "data", "cesm2_le_full_standard.zarr")
COHORT = os.path.join(ROOT, "data", "synthetic_argo", "cesm2_uniform.nc")
OUT = os.path.join(ROOT, "data", "synthetic_argo", "cesm2_surface_1deg.zarr")
TRAIN = (2000, 2003)
VARS = ("SST", "SLA", "SSS")
t0 = time.time()

LEVELS = np.asarray(xr.open_dataset(COHORT)["level"].values, dtype=float)
ds = xr.open_zarr(STORE)
di = [int(np.argmin(np.abs(ds.depth.values - d))) for d in LEVELS]
assert np.allclose(ds.depth.values[di], LEVELS, atol=0.2), ds.depth.values[di]
lat, lon = ds.lat.values.astype(float), ds.lon.values.astype(float)
NT = ds.sizes["time"]
assert NT == 72, NT
T = ds["TEMP"].isel(depth=di).values.astype("float32")        # (NT, L, NY, NX)
S = ds["SALT"].isel(depth=di).values.astype("float32")
fill = (T == 0) & (S == 0)                                    # the regridding fill 41 removes
T, S = np.where(fill, np.nan, T), np.where(fill, np.nan, S)
month_index = np.arange(NT)                                   # months since 2000-01
years, cal = 2000 + month_index // 12, month_index % 12
train = (years >= TRAIN[0]) & (years <= TRAIN[1])
print(f"loaded {NT} months x {LEVELS.size} levels ({time.time() - t0:.0f}s)", flush=True)

raw = {"SST": ds["SST"].values.astype("float32"), "SSS": ds["SSS"].values.astype("float32")}
lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
sla = np.full((NT, lat.size, lon.size), np.nan, dtype="float32")
for t in range(NT):
    # a column is used only where every level is finite, i.e. it reaches the reference
    ii, jj = np.where(np.isfinite(T[t]).all(axis=0) & np.isfinite(S[t]).all(axis=0))
    sla[t, ii, jj] = steric_height_columns(T[t][:, ii, jj], S[t][:, ii, jj], LEVELS,
                                           lat2d[ii, jj], lon2d[ii, jj])
raw["SLA"] = sla
print(f"steric height done ({time.time() - t0:.0f}s)", flush=True)


def anomaly(a):
    """a - its train-year monthly climatology; a is (NT, ...)."""
    clim = np.stack([np.nanmean(a[train & (cal == m)], axis=0) for m in range(12)])
    return (a - clim[cal]).astype("float32")


anom = {v: anomaly(raw[v]) for v in VARS}

# ---- diagnostics: what the fields contain
wet = np.isfinite(raw["SST"][0])
t_anom = anomaly(T)
band = (LEVELS > 100) & (LEVELS <= 300)
t_band = np.nanmean(t_anom[:, band], axis=1)                  # (NT, NY, NX)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


print(f"wet cells {int(wet.sum()):,}; SLA defined on "
      f"{100 * np.isfinite(sla[0]).sum() / wet.sum():.0f} % of them")
for v in VARS:
    print(f"  {v}: anomaly std over train months {np.nanstd(anom[v][train]):.4f}, "
          f"finite on {int(np.isfinite(anom[v][0]).sum()):,} cells")
print(f"  5 m check: max |SST - TEMP(5 m)| = {np.nanmax(np.abs(raw['SST'] - T[:, 0])):.2e}, "
      f"max |SSS - SALT(5 m)| = {np.nanmax(np.abs(raw['SSS'] - S[:, 0])):.2e}")
print(f"  corr(SLA anomaly, 100-300 m TEMP anomaly) = {corr(anom['SLA'], t_band):.3f}; "
      f"corr(SLA anomaly, SST anomaly) = {corr(anom['SLA'], anom['SST']):.3f}")

out = xr.Dataset(
    {v: (("time", "lat", "lon"), anom[v]) for v in VARS},
    coords={"time": np.arange("2000-01", "2006-01", dtype="datetime64[M]").astype("datetime64[ns]"),
            "lat": lat.astype("float32"), "lon": lon.astype("float32")},
    attrs={"source": "CESM2-LE 1 degree, data/cesm2_le_full_standard.zarr",
           "role": "satellite-type inputs of the synthetic Argo cohort",
           "anomaly": f"against each field's {TRAIN[0]}-{TRAIN[1]} monthly climatology",
           "SST_SSS": "the model's 5 m level, noise-free: the truth at the cohort's shallowest level",
           "SLA": f"steric height relative to {P_REF_DBAR:.0f} dbar from TEMP/SALT on the cohort's "
                  "20 levels (ocean_tokenizer.ssh); derived from the truth, not an independent "
                  "observation; NaN where the column does not reach the reference"})
if os.path.exists(OUT):
    shutil.rmtree(OUT)
out.to_zarr(OUT, mode="w")
print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
