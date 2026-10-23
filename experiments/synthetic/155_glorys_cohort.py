"""The GLORYS12 training and selection cohort ``glorys12_uniform``, in the iHESP cohort's layout.

Spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``. 85's recipe on the 1/12-degree
reanalysis grid: 7,600 model columns a month at area-weighted random ocean cells, a uniform
point inside the cell, 80 % inputs and 20 % queries, for the 264 months GLORYS 1997-01 ..
2018-12, which the cohort calendar labels 1984-01 .. 2005-12 (month index -192 .. 71; train
1984-2003, validation 2004, development 2005). The target is the column's anomaly from the
1997-2016 monthly climatology of its cell. Values come from the per-month union files of
``experiments/data/155_glorys_months.py``; nothing is interpolated horizontally.

Checks before the file is written: every month holds 7,600 profiles and 1,520 queries; no
value is a fill; the climatology recomputed from the union files at 4,000 random rows equals
the stored one; the finite fraction by level is printed.

  .venv/bin/python experiments/synthetic/155_glorys_cohort.py --workers 12
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ocean_tokenizer import glorys_sources as G

OUTPUT = ROOT / "data/synthetic_argo/glorys12_uniform.nc"


def calendar_month(job):
    """Values and climatology at the drawn cells of every cohort month of one calendar month."""
    root, month, cells = job
    folders = G.Folders(root)
    months = [t for t in G.COHORT_MONTHS if t % 12 == month]
    union = np.unique(np.concatenate([cells[t] for t in months]))
    clim_t, clim_s = G.climatology_at(folders, month, union)
    out = {}
    for t in months:
        where = np.searchsorted(union, cells[t])
        temp, salt = G.columns_at(folders, t, cells[t])
        out[t] = temp, salt, clim_t[where], clim_s[where]
    return out


def check_climatology(folders, all_draws, cells_flat, month_index, clim, n=4000, seed=7):
    """The climatology recomputed from the union files at ``n`` random rows must equal the stored one."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(month_index), n, replace=False))
    worst = 0.
    for m in np.unique(month_index[rows] % 12):
        pick = rows[month_index[rows] % 12 == m]
        order = np.argsort(cells_flat[pick], kind="stable")
        pick = pick[order]
        again = G.climatology_at(folders, int(m), cells_flat[pick])
        for k, variable in enumerate(("TEMP", "SALT")):
            stored = clim[variable][pick]
            both = np.isfinite(stored)
            if not np.isfinite(again[k][both]).all():
                raise SystemExit("the climatology recipe leaves values missing that the cohort has")
            worst = max(worst, float(np.abs(again[k][both] - stored[both]).max()))
    if worst > 1e-6:
        raise SystemExit(f"the climatology does not come back at the sampled rows ({worst:.3g})")
    return worst


def main():
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    started = time.time()
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists")
    folders = G.Folders(ROOT)
    grid = G.Grid(folders)
    all_draws = G.draws(folders, grid)
    missing = [G.stamp(t) for t in G.COHORT_MONTHS if not folders.columns_file(t).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} months have no union file yet, e.g. {missing[:3]}")
    cells = {t: all_draws[t][0] for t in G.COHORT_MONTHS}
    values = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(calendar_month, [(str(ROOT), m, cells) for m in range(12)])):
            values.update(result)
            print(f"calendar month {i + 1}/12 ({time.time() - started:.0f}s)", flush=True)
    months = list(G.COHORT_MONTHS)
    flat = np.concatenate([cells[t] for t in months])
    lat, lon = G.position_in_cell(grid.lat2d.ravel()[flat], grid.lon2d.ravel()[flat],
                                  np.concatenate([all_draws[t][2] for t in months]))
    month_index = np.repeat(np.array(months), G.N_PER_MONTH)
    year = 2000 + month_index // 12
    year_split = np.full(year.size, "unassigned", dtype=object)
    for name, (lo, hi) in G.SPLITS.items():
        year_split[(year >= lo) & (year <= hi)] = name
    iy, ix = G.one_degree_cell(lat, lon)
    arrays = {}
    for k, variable in enumerate(("TEMP", "SALT")):
        value = np.concatenate([values[t][k] for t in months])
        clim = np.concatenate([values[t][2 + k] for t in months])
        clim = np.where(np.isfinite(value), clim, np.nan).astype(np.float32)
        arrays.update({variable: value, f"{variable}_RAW": value, f"CLIM_POS_{variable}": clim, f"CLIM_CELL_{variable}": clim})
    zero = int(((arrays["TEMP"] == 0) & (arrays["SALT"] == 0)).sum())
    if zero:
        raise SystemExit(f"{zero} values read TEMP = 0 and SALT = 0: a fill this script does not know")
    if (np.abs(arrays["TEMP"][np.isfinite(arrays["TEMP"])]) > 50).any() or (np.abs(arrays["SALT"][np.isfinite(arrays["SALT"])]) > 60).any():
        raise SystemExit("a value is outside the physical range: a fill this script does not know")
    held = np.concatenate([all_draws[t][1] for t in months])
    counts = {t: int((month_index == t).sum()) for t in months}
    queries = {t: int(all_draws[t][1].sum()) for t in months}
    if set(counts.values()) != {G.N_PER_MONTH} or set(queries.values()) != {int(round(G.QUERY_FRAC * G.N_PER_MONTH))}:
        raise SystemExit("a month has the wrong number of profiles or queries")
    worst = check_climatology(folders, all_draws, flat, month_index, {ch: arrays[f"CLIM_POS_{ch}"] for ch in ("TEMP", "SALT")})
    print(f"climatology reproduced at 4,000 rows (largest difference {worst:.2g}) ({time.time() - started:.0f}s)", flush=True)
    P, L = arrays["TEMP"].shape
    glorys = np.array([G.glorys_year_month(int(t)) for t in month_index])
    out = xr.Dataset(
        {**{k: (("profile", "level"), v) for k, v in arrays.items()},
         "TEMP_ERR": (("profile", "level"), np.full((P, L), np.nan, "float32")),
         "SALT_ERR": (("profile", "level"), np.full((P, L), np.nan, "float32")),
         "lat": ("profile", lat), "lon": ("profile", lon),
         "grid_y": ("profile", iy), "grid_x": ("profile", ix),
         "model_j": ("profile", flat // grid.shape[1]), "model_i": ("profile", flat % grid.shape[1]),
         "month_index": ("profile", month_index), "year": ("profile", year),
         "glorys_year": ("profile", glorys[:, 0]), "glorys_month": ("profile", glorys[:, 1]),
         "wmo": ("profile", np.array([f"S{t + 192:03d}{i:04d}" for t in months for i in range(G.N_PER_MONTH)]).astype(str)),
         "year_split": ("profile", year_split.astype(str)),
         "float_split": ("profile", np.where(held, "heldout_float", "cohort_float").astype(str))},
        coords={"level": G.LEVELS},
        attrs={"region": "synthetic", "grid": [180, 360], "seed": G.SEED,
               "source": f"{G.DESCRIBE}; model columns at area-weighted random ocean cells, levels interpolated "
                         "linearly in depth from the 36 GLORYS levels, no horizontal interpolation; position uniform "
                         "inside the cell", "model_grid": list(grid.shape),
               "calendar_offset_years": G.OFFSET_YEARS,
               "calendar": "cohort year = GLORYS year - 13 (month index 0 = GLORYS 2013-01); glorys_year/glorys_month hold the real dates",
               "climatology": "1997-2016 (cohort 1984-2003) monthly mean of each drawn cell",
               "splits": json.dumps(G.SPLITS), "spec": "docs/superpowers/specs/2026-10-21-glorys-osse.md"})
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, OUTPUT)
    finite = np.isfinite(arrays["SALT"]).mean(axis=0)
    anomaly = arrays["TEMP"] - arrays["CLIM_POS_TEMP"]
    train = year_split == "train"
    print(json.dumps({"output": str(OUTPUT), "profiles": P, "months": len(months), "queries": int(held.sum()),
                      "finite_fraction_by_level": [round(float(f), 3) for f in finite],
                      "temp_anomaly_std_5m": {"train": round(float(np.nanstd(anomaly[train, 0])), 3),
                                              "2004": round(float(np.nanstd(anomaly[year == 2004, 0])), 3)},
                      "bytes": OUTPUT.stat().st_size, "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
