"""``cesm2_vlong_uniform``: 87's synthetic CESM2 Argo cohort from 1850-01, 1,872 months, 14.2 M profiles.

Two steps, each skipped when its output exists:

1. ``data/cesm2_long/cesm2_le_cohort_levels_1850_2005.zarr``: ``53_cesm2_long_levels.py`` run with
   ``--first-year 1850``, the raw member at the cohort's 20 levels for 1850-01 ... 2005-12. Months
   after 2005-12 are never read (2008 onwards is reserved for a sealed test). Check: every month of
   1964-2005 equals the existing training store value for value.
2. ``data/synthetic_argo/cesm2_vlong_uniform.nc``: every row of ``cesm2_long_uniform.nc`` unchanged
   and last (1964-2005, so selection on 2004-2005 and the OI anchor are unchanged), preceded by
   1850-01 ... 1963-12 drawn by 87's own ``draw_month`` (7,600 profiles a month, 20 % queries,
   generator ``[SEED, t + 10000]`` with t down to -1800, anomalies against the same 2000-2003
   climatology). Float ids of the added months start with "V"; their year split is "unassigned".
   Check: 87's climatology from the new store reproduces CLIM_POS of the existing cohort, and
   redrawing every 1964-1999 month from the new store reproduces the copied rows value for value,
   so the whole cohort is 87's recipe on one store.

Then the caches of ``150_wang_campaign.py``'s ``cohort_options`` (``cohort_caches.py``: 88/93/100's
recipes on the new store, copied rows checked against the existing caches; the 1964-01 ... 1964-06
rows gain lagged columns the old cache left at zero) and ``44_synth_argo_oi.py --cohort``, whose
selection must equal ``fixed_baselines_cesm2_long_uniform``'s.

  .venv/bin/python experiments/synthetic/153_cesm2_vlong_cohort.py --workers 8     # ~1 h, mostly step 1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import xarray as xr

import cohort_caches as H

NAME = "cesm2_vlong_uniform"
NEW = H.ROOT / f"data/synthetic_argo/{NAME}.nc"
STORE = H.ROOT / "data/cesm2_long/cesm2_le_cohort_levels_1850_2005.zarr"
OUTPUT = H.ROOT / f"outputs/cohort_{NAME}"
OI_SUMMARY = H.ROOT / f"outputs/audit/synthetic/fixed_baselines_{NAME}/summary.json"
FIRST_YEAR = 1850
FIRST_MONTH = (FIRST_YEAR - 2000) * 12       # -1800


def run(script, options, log_path):
    env = {**os.environ, "PYTHONPATH": str(H.ROOT / "src")}
    with log_path.open("a") as log:
        subprocess.run([sys.executable, "-u", str(H.ROOT / script), *options], cwd=H.ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def build_store(started):
    if not STORE.exists():
        run("experiments/data/53_cesm2_long_levels.py",
            ["--first-year", str(FIRST_YEAR), "--last-year", "2005", "--output", str(STORE)],
            OUTPUT / "levels_store.log")
        H.log(f"53 wrote {STORE}", started)
    new, old = xr.open_zarr(STORE, chunks=None), xr.open_zarr(H.OLD_STORE, chunks=None)
    months = np.asarray(new.month_index.values, dtype=int)
    if not np.array_equal(months, np.arange(FIRST_MONTH, 72)):
        raise SystemExit(f"{STORE.name} must hold {FIRST_YEAR}-01 ... 2005-12 exactly once each")
    old_months = np.asarray(old.month_index.values, dtype=int)
    for k in ("lat", "lon", "level"):
        if not np.array_equal(new[k].values, old[k].values):
            raise SystemExit(f"{k} differs between the stores")
    if not np.array_equal(new.MASK.values, old.MASK.values):
        raise SystemExit("MASK differs between the stores")
    at = np.searchsorted(months, old_months)
    for start in range(0, len(old_months), 12):
        rows = slice(start, start + 12)
        for v in ("TEMP", "SALT"):
            a = np.asarray(new[v][at[rows]].values)
            b = np.asarray(old[v][rows].values)
            if not np.array_equal(a, b, equal_nan=True):
                raise SystemExit(f"{v} of months {old_months[rows]} differ from {H.OLD_STORE.name}")
    new.close(), old.close()
    H.log(f"store: all {len(old_months)} months of 1964-2005 equal {H.OLD_STORE.name} value for value", started)


def build_cohort(started):
    store = H.Store(STORE)
    store.fit_climatology()
    old_ds = xr.open_dataset(H.OLD)
    if not np.allclose(store.levels, old_ds.level.values):
        raise SystemExit("store and cohort levels differ")
    store.check_climatology(old_ds)
    H.log(f"climatology from {STORE.name} reproduces CLIM_POS of {H.OLD.name}", started)
    slices = H.month_rows(old_ds)
    old = H.load_rows(old_ds)
    attrs = dict(old_ds.attrs)
    levels = old_ds.level.values
    old_ds.close()
    if [t for t in slices if t < 0] != list(range(H.OLD_FIRST_MONTH, 0)):
        raise SystemExit("the existing cohort does not hold exactly 1964-01 ... 1999-12 before 2000")

    # the copied 1964-1999 months are what 87's recipe gives on the new store
    for count, t in enumerate(range(H.OLD_FIRST_MONTH, 0), 1):
        col, arr = store.draw_87(t)
        if not H.rows_equal(col, arr, old, slices[t]):
            raise SystemExit(f"87's draw of month {t} from {STORE.name} does not reproduce the stored rows")
        if count % 144 == 0:
            H.log(f"redrawn {count}/432 copied months", started)
    H.log(f"87's recipe on {STORE.name} reproduces every 1964-1999 row of {H.OLD.name}", started)

    pieces = {k: [] for k in H.ALL_VARS}
    n_levels = levels.size
    for count, t in enumerate(range(FIRST_MONTH, H.OLD_FIRST_MONTH), 1):
        col, arr = store.draw_87(t)
        col["wmo"] = np.array([f"V{t - FIRST_MONTH:04d}{i:04d}" for i in range(H.N_PER_MONTH)])
        for k in H.COLUMNS:
            pieces[k].append(col[k] if k in ("wmo", "float_split") else col[k].astype(old[k].dtype))
        pieces["year"].append(np.full(H.N_PER_MONTH, 2000 + t // 12, old["year"].dtype))
        pieces["year_split"].append(np.full(H.N_PER_MONTH, "unassigned"))
        for k in H.ARRAYS:
            pieces[k].append(arr[k])
        for k in ("TEMP_ERR", "SALT_ERR"):
            pieces[k].append(np.full((H.N_PER_MONTH, n_levels), np.nan, "float32"))
        if count % 240 == 0:
            H.log(f"drew {count}/{H.OLD_FIRST_MONTH - FIRST_MONTH} months before 1964", started)
    store.close()
    for k in H.ALL_VARS:
        pieces[k].append(old[k])                                  # 1964-2005, unchanged and last
    data = {k: np.concatenate(v) for k, v in pieces.items()}
    del pieces
    if (np.diff(data["month_index"]) < 0).any():
        raise SystemExit("rows are not sorted by month")
    if np.unique(data["wmo"]).size != data["wmo"].size:
        raise SystemExit("float ids are not unique")
    n_new = data["lat"].size - old["lat"].size
    data_vars = {k: (("profile", "level") if data[k].ndim == 2 else "profile", data[k]) for k in H.ALL_VARS}
    attrs.update(source=f"{H.OLD.name} for 1964-2005 (every row, unchanged); {STORE.relative_to(H.ROOT)} "
                        "bilinear at uniform random ocean positions for 1850-1963",
                 extension=json.dumps({"first_month": f"{FIRST_YEAR}-01", "seed": H.SEED,
                                       "per_month_generator": "numpy default_rng([seed, month_index + 10000]) (87's draw_month)",
                                       "float_id_prefix_before_1964": "V",
                                       "rows_copied": int(old["lat"].size), "rows_added": int(n_new),
                                       "climatology": "2000-2003, identical to cesm2_long_uniform.nc",
                                       "store": str(STORE.relative_to(H.ROOT)),
                                       "script": "experiments/synthetic/153_cesm2_vlong_cohort.py"}))
    n = H.write_cohort(NEW, data_vars, levels, attrs)
    H.log(f"{n:,} profiles ({n_new:,} added over {H.OLD_FIRST_MONTH - FIRST_MONTH} months) -> {NEW}", started)


def copied_rows_positions():
    with xr.open_dataset(NEW) as ds:
        months = np.asarray(ds.month_index.values, int)
        wmo = np.asarray(ds.wmo.values).astype(str)
    wmo = wmo[np.argsort(months, kind="stable")]
    return np.flatnonzero(~np.char.startswith(wmo, "V"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_store(started)
    if NEW.exists():
        print(f"{NEW} exists: keeping it, building what is missing", flush=True)
    else:
        build_cohort(started)
    caches = [OUTPUT / f for f in ("satellites.npz", "lagged_surface.npz", "surface_grids.npz")]
    if all(f.exists() for f in caches):
        print("caches exist: keeping them", flush=True)
        checks = json.loads(str(np.load(caches[0], allow_pickle=False)["metadata"].item()))["checks"]
    else:
        checks = H.build_caches(NAME, STORE, copied_rows_positions(), OUTPUT, __file__, args.workers, started,
                                extra_metadata={"first_month": f"{FIRST_YEAR}-01"})
    if not OI_SUMMARY.exists():
        run("experiments/synthetic/44_synth_argo_oi.py", ["--cohort", NAME, "--workers", str(args.workers)],
            OUTPUT / "oi_baseline.log")
        H.log(f"OI baseline -> {OI_SUMMARY}", started)
    equal, detail = H.oi_selection_equal(OI_SUMMARY)
    checks["oi_selection_identical_to_cesm2_long_uniform"] = equal
    checks["oi_detail"] = detail
    (OUTPUT / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1), flush=True)
    if not equal:
        raise SystemExit("the OI selection differs from cesm2_long_uniform's")
    H.log("done", started)


if __name__ == "__main__":
    main()
