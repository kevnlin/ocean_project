"""The sealed GLORYS12 test cohort: ``glorys12_uniform`` plus GLORYS 2019-01 .. 2020-12.

Spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``. The training cohort's rows are
copied unchanged; each test month (cohort 2006-2007, month index 72 .. 95) is drawn with 155's
recipe from its own generator ``default_rng([20261021, month_index + 10000])`` (the draws file
holds them, made before any training month was reduced), its values are the 1/12-degree
columns of the test months' union files (``experiments/data/155_glorys_months.py --test``), and
its targets are anomalies from the 1997-2016 monthly climatology of each drawn cell, from the
training months' union files as 155 computes it. The new rows are outside every split
("unassigned").

Checks, none of which reads a statistic of a test value: the climatology recomputed at 4,000
training rows equals their stored ``CLIM_POS``; every test month has 7,600 profiles with 1,520
queries; no test value is a fill.

  .venv/bin/python experiments/synthetic/157_glorys_test_cohort.py --workers 12
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

from ocean_tokenizer import glorys_sources as G

spec = importlib.util.spec_from_file_location("glorys_cohort155", ROOT / "experiments/synthetic/155_glorys_cohort.py")
C155 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C155
spec.loader.exec_module(C155)

TRAIN_COHORT = ROOT / "data/synthetic_argo/glorys12_uniform.nc"
OUTPUT = ROOT / "data/synthetic_argo/glorys12_test.nc"


def calendar_month(job):
    """Test values and 1997-2016 climatology at the drawn cells of the test months of one calendar month."""
    root, month, cells = job
    folders = G.Folders(root)
    months = [t for t in G.TEST_MONTHS if t % 12 == month]
    union = np.unique(np.concatenate([cells[t] for t in months]))
    clim_t, clim_s = G.climatology_at(folders, month, union)
    out = {}
    for t in months:
        where = np.searchsorted(union, cells[t])
        temp, salt = G.columns_at(folders, t, cells[t])
        out[t] = temp, salt, clim_t[where], clim_s[where]
    return out


def main():
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    started = time.time()
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists: the sealed cohort is built once")
    folders = G.Folders(ROOT)
    grid = G.Grid(folders)
    all_draws = G.draws(folders, grid)
    missing = [G.stamp(t) for t in G.TEST_MONTHS if not folders.columns_file(t).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test months have no union file yet (155_glorys_months.py --test), e.g. {missing[:3]}")
    raw = xr.open_dataset(TRAIN_COHORT)
    if int(raw.attrs["seed"]) != G.SEED or int(raw.attrs["calendar_offset_years"]) != G.OFFSET_YEARS:
        raise SystemExit("the training cohort was not built with this seed and calendar")
    train_cells = (np.asarray(raw.model_j.values) * grid.shape[1] + np.asarray(raw.model_i.values))
    worst = C155.check_climatology(folders, all_draws, train_cells, np.asarray(raw.month_index.values),
                                   {ch: np.asarray(raw[f"CLIM_POS_{ch}"].values) for ch in ("TEMP", "SALT")})
    print(f"155's climatology reproduced at 4,000 training rows (largest difference {worst:.2g}) "
          f"({time.time() - started:.0f}s)", flush=True)
    cells = {t: all_draws[t][0] for t in G.TEST_MONTHS}
    values = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(calendar_month, [(str(ROOT), m, cells) for m in range(12)])):
            values.update(result)
            print(f"calendar month {i + 1}/12 ({time.time() - started:.0f}s)", flush=True)

    months = list(G.TEST_MONTHS)
    flat = np.concatenate([cells[t] for t in months])
    lat, lon = G.position_in_cell(grid.lat2d.ravel()[flat], grid.lon2d.ravel()[flat],
                                  np.concatenate([all_draws[t][2] for t in months]))
    month_index = np.repeat(np.array(months), G.N_PER_MONTH)
    iy, ix = G.one_degree_cell(lat, lon)
    new = {}
    for k, variable in enumerate(("TEMP", "SALT")):
        value = np.concatenate([values[t][k] for t in months])
        clim = np.concatenate([values[t][2 + k] for t in months])
        clim = np.where(np.isfinite(value), clim, np.nan).astype(np.float32)
        new.update({variable: value, f"{variable}_RAW": value, f"CLIM_POS_{variable}": clim, f"CLIM_CELL_{variable}": clim})
    if int(((new["TEMP"] == 0) & (new["SALT"] == 0)).sum()):
        raise SystemExit("a test value reads TEMP = 0 and SALT = 0: a fill this script does not know")
    n_new, levels = new["TEMP"].shape
    glorys = np.array([G.glorys_year_month(int(t)) for t in month_index])
    new.update({"TEMP_ERR": np.full((n_new, levels), np.nan, "float32"), "SALT_ERR": np.full((n_new, levels), np.nan, "float32"),
                "lat": lat, "lon": lon, "grid_y": iy, "grid_x": ix,
                "model_j": flat // grid.shape[1], "model_i": flat % grid.shape[1], "month_index": month_index,
                "year": 2000 + month_index // 12, "glorys_year": glorys[:, 0], "glorys_month": glorys[:, 1],
                "wmo": np.array([f"T{t:03d}{i:04d}" for t in months for i in range(G.N_PER_MONTH)]),
                "year_split": np.full(n_new, "unassigned"),
                "float_split": np.concatenate([np.where(all_draws[t][1], "heldout_float", "cohort_float") for t in months])})
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
    counts = {int(t): int((month_index == t).sum()) for t in months}
    queries = {int(t): int(all_draws[t][1].sum()) for t in months}
    if set(counts.values()) != {G.N_PER_MONTH} or set(queries.values()) != {int(round(G.QUERY_FRAC * G.N_PER_MONTH))}:
        raise SystemExit("a test month has the wrong number of profiles or queries")
    attrs = dict(raw.attrs)
    attrs["test_extension"] = json.dumps({"months": "GLORYS 2019-01 to 2020-12 (cohort 2006-2007, month index 72-95)",
                                          "seed": G.SEED, "per_month_generator": "numpy default_rng([seed, month_index + 10000])",
                                          "source": "data/glorys/columns (155_glorys_months.py --test)",
                                          "rows_before": int(raw.sizes["profile"]),
                                          "climatology": "1997-2016 of each cell, as 155",
                                          "spec": "docs/superpowers/specs/2026-10-21-glorys-osse.md"})
    out = xr.Dataset(data_vars, coords={"level": raw.level.values}, attrs=attrs)
    raw.close()
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "rows": int(out.sizes["profile"]), "new_rows": n_new,
                      "new_queries": int(sum(queries.values())), "bytes": OUTPUT.stat().st_size,
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
