"""The level store of the sealed CESM2 test v2: the raw member at the cohort's 20 levels, 2000-2010.

Spec ``docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md``. ``53_cesm2_long_levels.py`` run with
``--first-year 2000 --last-year 2010`` writes

  data/cesm2_long/cesm2_le_cohort_levels_2000_2010.zarr     132 months, 2000-01 ... 2010-12

reading the raw member's months 2000-01 ... 2010-12 and nothing after (53 keeps only the wanted
months; its own check: 2000-2005 equal ``cesm2_le_full_standard.zarr``). Then, before the store is
used by ``167``: its grid, levels and MASK equal the 1964-2007 store's; every month of 2000-01 ...
2007-12 equals ``cesm2_le_cohort_levels_1964_2007.zarr`` value for value; every month of 2000-2005
equals ``cesm2_le_cohort_levels.zarr`` and ``cesm2_le_cohort_levels_1850_2005.zarr``. The checks go
to ``outputs/cesm2_test2_20261022/store_checks.json``. Resumable: an existing store is checked, not
rebuilt.

  .venv/bin/python experiments/synthetic/166_cesm2_test2_store.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import xarray as xr

STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels_2000_2010.zarr"
REFERENCES = {"1964_2007": ROOT / "data/cesm2_long/cesm2_le_cohort_levels_1964_2007.zarr",
              "1964_2005": ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr",
              "1850_2005": ROOT / "data/cesm2_long/cesm2_le_cohort_levels_1850_2005.zarr"}
OUTPUT = ROOT / "outputs/cesm2_test2_20261022"
FIRST_YEAR, LAST_YEAR = 2000, 2010
FIRST_MONTH, LAST_MONTH = (FIRST_YEAR - 2000) * 12, (LAST_YEAR - 2000) * 12 + 11      # 0 ... 131


def log(message, started):
    print(f"{message} ({time.time() - started:.0f}s)", flush=True)


def build(started):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with (OUTPUT / "levels_store.log").open("a") as log_file:
        subprocess.run([sys.executable, "-u", str(ROOT / "experiments/data/53_cesm2_long_levels.py"),
                        "--first-year", str(FIRST_YEAR), "--last-year", str(LAST_YEAR), "--output", str(STORE)],
                       cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    log(f"53 wrote {STORE}", started)


def check(started):
    new = xr.open_zarr(STORE, chunks=None)
    months = np.asarray(new.month_index.values, dtype=int)
    if not np.array_equal(months, np.arange(FIRST_MONTH, LAST_MONTH + 1)):
        raise SystemExit(f"{STORE.name} must hold {FIRST_YEAR}-01 ... {LAST_YEAR}-12 exactly once each")
    checks = {"store": str(STORE.relative_to(ROOT)), "months": [int(months[0]), int(months[-1])],
              "n_months": int(len(months)), "references": {}}
    for name, path in REFERENCES.items():
        old = xr.open_zarr(path, chunks=None)
        for k in ("lat", "lon", "level"):
            if not np.array_equal(new[k].values, old[k].values):
                raise SystemExit(f"{k} differs between {STORE.name} and {path.name}")
        if not np.array_equal(new.MASK.values, old.MASK.values):
            raise SystemExit(f"MASK differs between {STORE.name} and {path.name}")
        old_months = np.asarray(old.month_index.values, dtype=int)
        shared = old_months[(old_months >= FIRST_MONTH) & (old_months <= LAST_MONTH)]
        at_new, at_old = np.searchsorted(months, shared), np.searchsorted(old_months, shared)
        compared = 0
        for start in range(0, len(shared), 12):
            rows = slice(start, start + 12)
            for v in ("TEMP", "SALT"):
                a = np.asarray(new[v][at_new[rows]].values)
                b = np.asarray(old[v][at_old[rows]].values)
                if not np.array_equal(a, b, equal_nan=True):
                    raise SystemExit(f"{v} of months {shared[rows]} differ between {STORE.name} and {path.name}")
                compared += a.size
        old.close()
        checks["references"][name] = {"path": str(path.relative_to(ROOT)), "months_equal": [int(shared[0]), int(shared[-1])],
                                      "n_months_equal": int(len(shared)), "values_compared": int(compared),
                                      "grid_levels_mask_equal": True}
        log(f"{len(shared)} shared months ({shared[0]} ... {shared[-1]}) equal {path.name} value for value", started)
    # the new months are finite where the mask is wet, at least at the surface
    wet = new.MASK.values > 0
    for i in np.flatnonzero(months > 95):
        t = np.asarray(new.TEMP[i, 0].values)
        if not np.isfinite(t[wet]).all() and np.isfinite(t[wet]).mean() < .99:
            raise SystemExit(f"month {months[i]}: 5 m temperature is not finite over the ocean")
    checks["new_months"] = [96, int(months[-1])]
    checks["new_months_surface_finite_over_ocean"] = True
    checks["store_attrs"] = {k: v for k, v in new.attrs.items() if k != "manifest"}
    new.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "store_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()
    started = time.time()
    if STORE.exists():
        print(f"{STORE} exists: checking it", flush=True)
    else:
        build(started)
    check(started)
    log("done", started)


if __name__ == "__main__":
    main()
