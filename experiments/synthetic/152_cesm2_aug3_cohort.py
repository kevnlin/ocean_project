"""``cesm2_long_aug3``: the 1964-2005 cohort with three draws of profiles in every month before 2000.

``data/synthetic_argo/cesm2_long_aug3.nc`` holds every row of ``cesm2_long_uniform.nc`` unchanged
(1964-1999 draw 0 and 2000-2005, so selection on 2004-2005 and the OI anchor are unchanged) plus,
for each month 1964-01 ... 1999-12, two more draws of 7,600 profiles made exactly as
``87_cesm2_long_cohort.py`` makes a month, from the generators

    numpy default_rng([SEED, t + 10000 + 100000 * d])      d = 1, 2

(draw 0 is 87's ``[SEED, t + 10000]``), against the same 2000-2003 climatology and the same level
store. Float ids of draw d start with "Dd" (87's start with "L", and "L1"/"L2" would be
ambiguous with 87's months 100-299); their year split is "unassigned". Rows stay sorted by
month, as in 87: within a month draw 0, then draw 1, then draw 2. Then the caches of
``150_wang_campaign.py``'s ``cohort_options`` are built (``cohort_caches.py``, 88/93/100's recipes)
and ``44_synth_argo_oi.py --cohort cesm2_long_aug3`` is run.

Checks: 87's climatology reproduces CLIM_POS of the existing cohort; redrawing draw 0 of every
1964-1999 month reproduces the stored rows value for value (so the draw code is 87's); the copied
rows' features, lagged columns and grids reproduce the existing caches; the OI selection equals
``fixed_baselines_cesm2_long_uniform``'s.

  .venv/bin/python experiments/synthetic/152_cesm2_aug3_cohort.py --workers 8
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

NAME = "cesm2_long_aug3"
NEW = H.ROOT / f"data/synthetic_argo/{NAME}.nc"
OUTPUT = H.ROOT / f"outputs/cohort_{NAME}"
OI_SUMMARY = H.ROOT / f"outputs/audit/synthetic/fixed_baselines_{NAME}/summary.json"
DRAWS = (1, 2)


def build_cohort(started):
    store = H.Store(H.OLD_STORE)
    store.fit_climatology()
    old_ds = xr.open_dataset(H.OLD)
    if not np.allclose(store.levels, old_ds.level.values):
        raise SystemExit("store and cohort levels differ")
    store.check_climatology(old_ds)
    H.log(f"climatology and stencil reproduce {H.OLD.name}", started)
    slices = H.month_rows(old_ds)
    old = H.load_rows(old_ds)
    attrs = dict(old_ds.attrs)
    levels = old_ds.level.values
    old_ds.close()
    early = [t for t in slices if t < 0]
    if early != list(range(H.OLD_FIRST_MONTH, 0)):
        raise SystemExit("the existing cohort does not hold exactly 1964-01 ... 1999-12 before 2000")

    pieces = {k: [] for k in H.ALL_VARS}
    n_levels = levels.size

    def append_old(rows):
        for k in H.ALL_VARS:
            pieces[k].append(old[k][rows])

    for count, t in enumerate(early, 1):
        month = store.month(t)
        col0, arr0 = store.draw(t, [H.SEED, t + 10_000], lambda i, t=t: f"L{t + 432:03d}{i:04d}", month)
        if not H.rows_equal(col0, arr0, old, slices[t]):
            raise SystemExit(f"draw 0 of month {t} does not reproduce the stored rows: the draw code is not 87's")
        append_old(slices[t])
        for d in DRAWS:
            col, arr = store.draw(t, [H.SEED, t + 10_000 + 100_000 * d], lambda i, t=t, d=d: f"D{d}{t + 432:03d}{i:04d}", month)
            for k in H.COLUMNS:
                pieces[k].append(col[k] if k == "wmo" or k == "float_split" else col[k].astype(old[k].dtype))
            pieces["year"].append(np.full(H.N_PER_MONTH, 2000 + t // 12, old["year"].dtype))
            pieces["year_split"].append(np.full(H.N_PER_MONTH, "unassigned"))
            for k in H.ARRAYS:
                pieces[k].append(arr[k])
            for k in ("TEMP_ERR", "SALT_ERR"):
                pieces[k].append(np.full((H.N_PER_MONTH, n_levels), np.nan, "float32"))
        if count % 48 == 0:
            H.log(f"month {count}/{len(early)}: draw 0 reproduced, draws {DRAWS} added", started)
    store.close()
    H.log(f"draw 0 of all {len(early)} months before 2000 reproduces the stored rows", started)
    append_old(slice(slices[0].start, old["lat"].size))          # 2000-2005, unchanged and last

    data = {k: np.concatenate(v) for k, v in pieces.items()}
    del pieces
    if (np.diff(data["month_index"]) < 0).any():
        raise SystemExit("rows are not sorted by month")
    if np.unique(data["wmo"]).size != data["wmo"].size:
        raise SystemExit("float ids are not unique")
    n_new = data["lat"].size - old["lat"].size
    data_vars = {k: (("profile", "level") if data[k].ndim == 2 else "profile", data[k]) for k in H.ALL_VARS}
    attrs.update(source=f"{H.OLD.name} (every row, unchanged); {H.OLD_STORE.relative_to(H.ROOT)} bilinear at "
                        "uniform random ocean positions for two more draws per month before 2000",
                 augmentation=json.dumps({"draws_before_2000": 3, "seed": H.SEED,
                                          "per_month_generator": "numpy default_rng([seed, month_index + 10000 + 100000 * draw])",
                                          "float_id_prefix": {d: f"D{d}" for d in DRAWS},
                                          "row_order": "by month; within a month draw 0, 1, 2",
                                          "rows_copied": int(old["lat"].size), "rows_added": int(n_new),
                                          "climatology": "2000-2003, identical to cesm2_long_uniform.nc",
                                          "script": "experiments/synthetic/152_cesm2_aug3_cohort.py"}))
    n = H.write_cohort(NEW, data_vars, levels, attrs)
    H.log(f"{n:,} profiles ({n_new:,} added) -> {NEW}", started)


def copied_rows_positions():
    """Positions of ``cesm2_long_uniform``'s rows (in its loaded order) within the new cohort's loaded order."""
    with xr.open_dataset(NEW) as ds:
        months = np.asarray(ds.month_index.values, int)
        wmo = np.asarray(ds.wmo.values).astype(str)
    order = np.argsort(months, kind="stable")
    wmo = wmo[order]
    return np.flatnonzero(~np.char.startswith(wmo, "D"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    started = time.time()
    if NEW.exists():
        print(f"{NEW} exists: keeping it, building what is missing", flush=True)
    else:
        build_cohort(started)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    caches = [OUTPUT / f for f in ("satellites.npz", "lagged_surface.npz", "surface_grids.npz")]
    if all(f.exists() for f in caches):
        print("caches exist: keeping them", flush=True)
        checks = json.loads(str(np.load(caches[0], allow_pickle=False)["metadata"].item()))["checks"]
    else:
        old_at = copied_rows_positions()
        checks = H.build_caches(NAME, H.OLD_STORE, old_at, OUTPUT, __file__, args.workers, started,
                                extra_metadata={"draws_before_2000": 3})
    if not OI_SUMMARY.exists():
        log = (OUTPUT / "oi_baseline.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(H.ROOT / "src")}
        subprocess.run([sys.executable, "-u", str(H.ROOT / "experiments/synthetic/44_synth_argo_oi.py"), "--cohort", NAME,
                        "--workers", str(args.workers)], cwd=H.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        log.close()
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
