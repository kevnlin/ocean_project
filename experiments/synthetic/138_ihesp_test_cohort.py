"""The sealed iHESP test cohort: ``ihesp_hr_uniform`` plus the months 1997-07 ... 1999-12.

Spec ``docs/superpowers/specs/2026-10-21-ihesp-sealed-test.md``. The training cohort's rows are
copied unchanged; each new month is drawn with 85's recipe (7,600 area-weighted ocean cells,
20 % queries, a uniform point inside the cell) from its own generator
``default_rng([20261009, month_index + 10000])``, its values are the 0.1-degree model columns of
``data/ihesp/hr_test_levels`` (``51_ihesp_hr_subset.py --years 1997 1998 1999``), and its targets
are anomalies from the 2000-2003 monthly climatology of each drawn cell, computed as 85 does from
the training files. The new rows are outside every split ("unassigned").

Checks, none of which reads a statistic of a test value: the training rows are the cohort's own;
85's climatology recomputed at a sample of training rows equals their stored ``CLIM_POS``; every
new month has 7,600 profiles with 1,520 queries; no value is the 0/0 fill.

  .venv/bin/python experiments/synthetic/138_ihesp_test_cohort.py --workers 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

spec = importlib.util.spec_from_file_location("ihesp_cohort85", ROOT / "experiments/synthetic/85_ihesp_argo_cohort.py")
C85 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C85
spec.loader.exec_module(C85)
from ocean_tokenizer.ihesp_sources import FILL, HighResolution, year_month

TRAIN_COHORT = ROOT / "data/synthetic_argo/ihesp_hr_uniform.nc"
OUTPUT = ROOT / "data/synthetic_argo/ihesp_hr_test.nc"
TEST_FOLDER = ROOT / "data/ihesp/hr_test_levels"
TEST_MONTHS = tuple(range(-30, 0))           # 1997-07 ... 1999-12, counted from 2000-01
TRAIN_YEARS = (2000, 2003)


class TestYears(HighResolution):
    """The same reader on the test-year files."""

    def __init__(self, root):
        super().__init__(root)
        self.folder = Path(root) / "data/ihesp/hr_test_levels"


def draws(source, ocean):
    out = {}
    for t in TEST_MONTHS:
        rng = np.random.default_rng([C85.SEED, t + 10_000])
        cells = C85.draw_cells(source.area, ocean, C85.N_PER_MONTH, rng)
        held = C85.draw_queries(C85.N_PER_MONTH, C85.QUERY_FRAC, rng)
        offset = rng.uniform(-C85.IN_CELL, C85.IN_CELL, (C85.N_PER_MONTH, 2))
        out[t] = cells, held, offset
    return out


def calendar_month(job):
    """Test values and 2000-2003 climatology at the drawn cells of the test months of one calendar month."""
    root, month, cells = job
    train, test = HighResolution(root), TestYears(root)
    months = [t for t in TEST_MONTHS if t % 12 == month]
    union = np.concatenate([cells[t] for t in months])
    years = [12 * (y - 2000) + month for y in range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)]
    out = {}
    for variable in ("TEMP", "SALT"):
        clim = C85.train_mean(np.stack([C85.columns(train.field(variable, t), train.kmt, train.full_level_index, union)
                                        for t in years]), np.ones(len(years), bool))
        for i, t in enumerate(months):
            rows = slice(i * C85.N_PER_MONTH, (i + 1) * C85.N_PER_MONTH)
            value = C85.columns(test.field(variable, t), test.kmt, test.full_level_index, cells[t])
            out[(variable, t)] = value, clim[rows]
    return out


def check_training_climatology(raw, train, n=4000, seed=7):
    """85's climatology recomputed at ``n`` random training rows must equal their stored CLIM_POS."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(raw.sizes["profile"], n, replace=False))
    month = np.asarray(raw.month_index.values)[rows]
    cell = (np.asarray(raw.model_j.values)[rows] * train.kmt.shape[1] + np.asarray(raw.model_i.values)[rows])
    worst = 0.
    for m in np.unique(month % 12):
        pick = rows[month % 12 == m]
        cells = cell[month % 12 == m]
        years = [12 * (y - 2000) + int(m) for y in range(TRAIN_YEARS[0], TRAIN_YEARS[1] + 1)]
        for variable in ("TEMP", "SALT"):
            clim = C85.train_mean(np.stack([C85.columns(train.field(variable, t), train.kmt, train.full_level_index, cells)
                                            for t in years]), np.ones(len(years), bool))
            stored = np.asarray(raw[f"CLIM_POS_{variable}"].values)[pick]
            both = np.isfinite(stored)
            if not np.array_equal(np.isfinite(clim[both]), np.ones(both.sum(), bool)):
                raise SystemExit("the climatology recipe leaves values missing that the cohort has")
            worst = max(worst, float(np.abs(clim[both] - stored[both]).max()))
    if worst > 1e-6:
        raise SystemExit(f"85's climatology does not come back at the training rows ({worst:.3g})")
    return worst


def main():
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    started = time.time()
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists: the sealed cohort is built once")
    train, test = HighResolution(ROOT), TestYears(ROOT)
    for key in ("lat", "lon", "area", "kmt"):
        if not np.array_equal(getattr(train, key), getattr(test, key)):
            raise SystemExit(f"the test files' grid differs from the training files' ({key})")
    ocean = (train.kmt > 0) & (np.abs(train.lat) <= 90.0) & (train.area > 0) & (train.area < FILL)
    raw = xr.open_dataset(TRAIN_COHORT)
    worst = check_training_climatology(raw, train)
    print(f"85's climatology reproduced at 4,000 training rows (largest difference {worst:.2g}) "
          f"({time.time() - started:.0f}s)", flush=True)
    drawn = draws(train, ocean)
    cells = {t: drawn[t][0] for t in TEST_MONTHS}
    values = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(calendar_month, [(str(ROOT), m, cells) for m in range(12)])):
            values.update(result)
            print(f"calendar month {i + 1}/12 ({time.time() - started:.0f}s)", flush=True)

    flat = np.concatenate([cells[t] for t in TEST_MONTHS])
    lat, lon = C85.position_in_cell(train.lat.ravel()[flat], train.lon.ravel()[flat], train.area.ravel()[flat],
                                    np.concatenate([drawn[t][2] for t in TEST_MONTHS]))
    month_index = np.repeat(np.array(TEST_MONTHS), C85.N_PER_MONTH)
    iy, ix = C85.one_degree_cell(lat, lon)
    new = {}
    for variable in ("TEMP", "SALT"):
        value = np.concatenate([values[(variable, t)][0] for t in TEST_MONTHS])
        clim = np.concatenate([values[(variable, t)][1] for t in TEST_MONTHS])
        clim = np.where(np.isfinite(value), clim, np.nan).astype(np.float32)
        new.update({variable: value, f"{variable}_RAW": value, f"CLIM_POS_{variable}": clim,
                    f"CLIM_CELL_{variable}": clim})
    if int(((new["TEMP"] == 0) & (new["SALT"] == 0)).sum()):
        raise SystemExit("a test value reads TEMP = 0 and SALT = 0: a fill this script does not know")
    n_new, levels = new["TEMP"].shape
    new.update({"TEMP_ERR": np.full((n_new, levels), np.nan, "float32"), "SALT_ERR": np.full((n_new, levels), np.nan, "float32"),
                "lat": lat, "lon": lon, "grid_y": iy, "grid_x": ix,
                "model_j": flat // ocean.shape[1], "model_i": flat % ocean.shape[1], "month_index": month_index,
                "year": np.array([year_month(int(t))[0] for t in month_index]),
                "wmo": np.array([f"T{t + 100:03d}{i:04d}" for t in TEST_MONTHS for i in range(C85.N_PER_MONTH)]),
                "year_split": np.full(n_new, "unassigned"),
                "float_split": np.concatenate([np.where(drawn[t][1], "heldout_float", "cohort_float") for t in TEST_MONTHS])})
    data_vars = {}
    for name, var in raw.data_vars.items():
        old = np.asarray(var.values)
        extra = np.asarray(new[name])
        joined = (np.concatenate([old.astype(str), extra.astype(str)]) if old.dtype.kind == "U" or extra.dtype.kind == "U"
                  else np.concatenate([old, extra.astype(old.dtype)]))
        data_vars[name] = (var.dims, joined)
    missing = set(new) - set(raw.data_vars)
    if missing:
        raise SystemExit(f"new columns the training cohort does not have: {sorted(missing)}")
    counts = {int(t): int((month_index == t).sum()) for t in TEST_MONTHS}
    queries = {int(t): int(drawn[t][1].sum()) for t in TEST_MONTHS}
    if set(counts.values()) != {C85.N_PER_MONTH} or set(queries.values()) != {int(round(C85.QUERY_FRAC * C85.N_PER_MONTH))}:
        raise SystemExit("a test month has the wrong number of profiles or queries")
    attrs = dict(raw.attrs)
    attrs["test_extension"] = json.dumps({"months": "1997-07 to 1999-12", "seed": C85.SEED,
                                          "per_month_generator": "numpy default_rng([seed, month_index + 10000])",
                                          "source": str(TEST_FOLDER.relative_to(ROOT)), "rows_before": int(raw.sizes["profile"]),
                                          "climatology": "2000-2003 of each cell, as 85", "spec": "docs/superpowers/specs/2026-10-21-ihesp-sealed-test.md"})
    out = xr.Dataset(data_vars, coords={"level": raw.level.values}, attrs=attrs)
    raw.close()
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "rows": int(out.sizes["profile"]), "new_rows": n_new,
                      "new_queries": int(sum(queries.values())), "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
