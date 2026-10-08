"""Satellite values at every profile of the global Argo cohort.

The token line gives the model satellite fields as patch means on the 1-degree
monthly grid. The query-anchored model (``67_anchored_fusion.py``) reads them
at the profile itself instead. For each profile of
``data/argo_cohort/global_global.nc`` this writes, at the profile's own position:

  * the 1-degree monthly SST / SLA / SSS of ``data/real_obs_1deg.zarr`` (what the
    ``--surface`` arms of ``62_sanity_train.py`` are given), raw and as anomalies
    from a monthly climatology of the same store over ``--clim-years``;
  * the daily 0.25-degree sea-level anomaly on the profile's own DAY, at the
    profile and 0.5 degree to its east / west / north / south, from a directory
    of monthly files ``sla_daily_YYYY-MM.nc`` (variable ``sla``, dimensions
    time x latitude x longitude). Profiles outside those files get NaN.

Reads are bilinear and NaN-aware (a corner over land is left out; a read with
less than a quarter of its weight over water is NaN). Rows are in the order of
``ArgoCohort.load`` (stable sort by month), so the file lines up with the cohort.

  .venv/bin/python experiments/data/49_satellite_at_profiles.py --sla-dir <dir>
  -> data/argo_cohort/global_satellite.npz
"""
from __future__ import annotations

import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.audit_tools import cohort_path

ap = argparse.ArgumentParser()
ap.add_argument("--sla-dir", default=None,
                help="directory of sla_daily_YYYY-MM.nc files (daily 0.25-degree SLA); "
                     "omit to write the 1-degree monthly values only")
ap.add_argument("--clim-years", default="2016,2020",
                help="first,last year of the monthly climatology the anomalies are taken "
                     "from (the training years of the audit's split)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = args.out or os.path.join(ROOT, "data", "argo_cohort", "global_satellite.npz")
Y0, Y1 = (int(v) for v in args.clim_years.split(","))
t0 = time.time()

d = xr.open_dataset(cohort_path(ROOT, "global"))
mi = np.asarray(d["month_index"].values, int)
order = np.argsort(mi, kind="stable")            # the order ArgoCohort.load uses
mi = mi[order]
lat = np.asarray(d["lat"].values, float)[order]
lon = np.asarray(d["lon"].values, float)[order] % 360.0
day = np.asarray(d["time"].values)[order].astype("datetime64[D]")
d.close()
P = mi.size
print(f"{P:,} profiles ({time.time() - t0:.0f}s)", flush=True)


def bilinear(field, y, x, periodic=True):
    """NaN-aware bilinear read of ``field`` (ny, nx) at fractional indices."""
    ny, nx = field.shape
    y = np.clip(y, 0, ny - 1)
    y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, ny - 1)
    x0 = np.floor(x).astype(int)
    wy, wx = y - y0, x - x0
    if periodic:
        x0, x1 = x0 % nx, (x0 + 1) % nx
    else:
        x0 = np.clip(x0, 0, nx - 1); x1 = np.minimum(x0 + 1, nx - 1)
    num = np.zeros(y.shape); den = np.zeros(y.shape)
    for yy, xx, w in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x1, (1 - wy) * wx),
                      (y1, x0, wy * (1 - wx)), (y1, x1, wy * wx)):
        v = field[yy, xx]
        ok = np.isfinite(v)
        num += np.where(ok, v, 0.0) * w; den += ok * w
    return np.where(den > 0.25, num / np.maximum(den, 1e-12), np.nan)


# ------------------------------------------------------------------ 1-degree monthly
o = xr.open_zarr(os.path.join(ROOT, "data", "real_obs_1deg.zarr"))
smi = ((o.time.values.astype("datetime64[M]") - np.datetime64("2000-01", "M"))
       / np.timedelta64(1, "M")).astype(int)
cal = smi % 12
in_clim = (2000 + smi // 12 >= Y0) & (2000 + smi // 12 <= Y1)
out, clim1 = {}, {}
for v in ("SST", "SLA", "SSS"):
    a = np.asarray(o[v].values, float)                               # (T, 180, 360)
    clim1[v] = np.stack([np.nanmean(a[in_clim & (cal == m)], axis=0) for m in range(12)])
    val = np.full(P, np.nan); anom = np.full(P, np.nan)
    for k, m in enumerate(smi):
        sel = np.flatnonzero(mi == m)
        if sel.size:
            y, x = lat[sel] + 89.5, lon[sel] - 0.5
            val[sel] = bilinear(a[k], y, x)
            anom[sel] = bilinear(a[k] - clim1[v][cal[k]], y, x)
    out[f"{v}_1deg"] = val.astype("float32"); out[f"{v}_1deg_anom"] = anom.astype("float32")
    print(f"  {v}: anomaly sd {np.nanstd(anom):.4f} ({time.time() - t0:.0f}s)", flush=True)
sel = np.isin(mi, smi)
out["SLA_clim_1deg"] = np.full(P, np.nan, "float32")
for m in np.unique(mi[sel]):
    j = np.flatnonzero(mi == m)
    out["SLA_clim_1deg"][j] = bilinear(clim1["SLA"][m % 12], lat[j] + 89.5, lon[j] - 0.5)

# ------------------------------------------------------------------ daily 0.25-degree SLA
OFFS = {"c": (0.0, 0.0), "e": (0.0, 0.5), "w": (0.0, -0.5), "n": (0.5, 0.0), "s": (-0.5, 0.0)}
daily = {k: np.full(P, np.nan) for k in OFFS}
if args.sla_dir:
    for f in sorted(os.listdir(args.sla_dir)):
        if not (f.startswith("sla_daily_") and f.endswith(".nc")):
            continue
        ym = f[len("sla_daily_"):-3]
        m = (int(ym[:4]) - 2000) * 12 + int(ym[5:7]) - 1
        sel = np.flatnonzero(mi == m)
        if sel.size == 0:
            continue
        s = xr.open_dataset(os.path.join(args.sla_dir, f))
        days = s.time.values.astype("datetime64[D]")
        la, lo = s.latitude.values.astype(float), s.longitude.values.astype(float)
        arr = np.asarray(s["sla"].values, "float32")                 # (D, ny, nx)
        s.close()
        di = np.searchsorted(days, day[sel])
        ok = (di < days.size) & (days[np.minimum(di, days.size - 1)] == day[sel])
        for dd in np.unique(di[ok]):
            j = sel[ok & (di == dd)]
            for k, (dy, dx) in OFFS.items():
                y = (lat[j] + dy - la[0]) / (la[1] - la[0])
                x = (((lon[j] + dx + 180.0) % 360.0 - 180.0) - lo[0]) / (lo[1] - lo[0])
                daily[k][j] = bilinear(arr[dd], y, x)
    n = int(np.isfinite(daily["c"]).sum())
    both = np.isfinite(daily["c"]) & np.isfinite(out["SLA_1deg"])
    print(f"  daily SLA at {n:,} profiles; correlation with the 1-degree monthly mean "
          f"{np.corrcoef(daily['c'][both], out['SLA_1deg'][both])[0, 1]:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
for k in OFFS:
    out[f"SLA_day_{k}"] = daily[k].astype("float32")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, month_index=mi, **out)
print(f"wrote {OUT} ({time.time() - t0:.0f}s)")
