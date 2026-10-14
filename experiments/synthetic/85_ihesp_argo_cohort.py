"""A synthetic Argo cohort from the iHESP CESM1.3 historical run, in the CESM2 cohort's layout.

The same task as ``41_synth_argo_cohort.py`` on another model: 7,600 profiles a
month at uniform random ocean positions for 2000-2005, 80 % inputs and 20 %
queries, train 2000-2003 / validation 2004 / development 2005, the same 20
depth levels (5-985 m, all native levels of both iHESP grids), target = anomaly
from the model's own training-year monthly climatology.

  --source hr   the 0.1-degree run, from data/ihesp/hr_cohort_levels
                (51_ihesp_hr_subset.py)            -> ihesp_hr_uniform.nc
  --source lr   the 1-degree run, from data/ihesp/lr -> ihesp_lr_uniform.nc

One difference from the CESM2 cohort, which interpolated a regular 1-degree
grid to continuous positions: here a profile IS a model column. Ocean cells are
drawn with probability proportional to their area (so positions are
area-uniform), without replacement inside a month; the profile carries the
cell's own values and sits at a uniform random point inside the cell. Nothing
is interpolated, so the 0.1-degree run keeps its eddies; ``CLIM_POS`` and
``CLIM_CELL`` are both the climatology of that cell.

The random point matters beyond realism: with every profile on a cell centre,
many pairs are at exactly the same distance, the k nearest profiles of a query
are then not unique, and two OI code paths that should agree pick different
ones (44_synth_argo_oi.py stops on that).

Land needs care in the 0.1-degree files: besides the fill value, a third of
the grid lies in land blocks the model eliminated, where every field is -1.
Cells are drawn only where ``KMT > 0``, and a level at or below the sea floor
(``level index >= KMT``) is missing.

  .venv/bin/python experiments/synthetic/85_ihesp_argo_cohort.py --source hr
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocean_tokenizer.ihesp_sources import (FILL, LEVELS, N_MONTHS, SOURCES, HighResolution,
                                           LowResolution, stored_values, year_month)
SEED = 20261009
N_PER_MONTH = 7600
QUERY_FRAC = 0.2
SPLITS = {"train": (2000, 2003), "validation": (2004, 2004), "development": (2005, 2005)}
R_EARTH_CM = 6.37122e8          # POP's sphere; cell areas are in cm^2
#: a profile lies within this fraction of the cell's width from its centre
IN_CELL = 0.45


def draw_cells(area, ocean, n, rng):
    """Flat indices of ``n`` distinct ocean cells, drawn with probability ~ area."""
    candidates = np.flatnonzero(np.asarray(ocean).ravel())
    if n > candidates.size:
        raise ValueError(f"{n} profiles asked of {candidates.size} ocean cells")
    weight = np.asarray(area, dtype=np.float64).ravel()[candidates]
    return np.sort(rng.choice(candidates, n, replace=False, p=weight / weight.sum()))


def draw_queries(n, fraction, rng):
    held = np.zeros(n, dtype=bool)
    held[rng.choice(n, int(round(fraction * n)), replace=False)] = True
    return held


def columns(field, kmt, full_level_index, cells):
    """``(len(cells), L)`` model columns; missing on land and at or below the sea floor.

    ``field`` is ``(L, ny, nx)`` at the wanted levels, ``full_level_index`` each
    level's index in the model's full depth axis, ``kmt`` the number of wet
    levels of each column.
    """
    levels = field.shape[0]
    out = field.reshape(levels, -1)[:, cells].T.astype(np.float32)
    floor = np.asarray(kmt).ravel()[cells][:, None]
    dry = (np.abs(out) > FILL) | (np.asarray(full_level_index)[None, :] >= floor)
    out[dry] = np.nan
    return out


def train_mean(values, train):
    """``(years, cells, L)`` -> ``(cells, L)`` mean over the training years, NaN where none."""
    picked = values[np.asarray(train, dtype=bool)]
    count = np.isfinite(picked).sum(0)
    total = np.where(np.isfinite(picked), picked, 0.).sum(0, dtype=np.float64)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def position_in_cell(lat, lon, area_cm2, offset):
    """A point inside each cell: the centre moved by ``offset`` cell widths north and east.

    ``offset`` is ``(n, 2)`` in units of the cell's width, taken as the square
    root of its area. Longitude comes back in 0-360.
    """
    width = np.rad2deg(np.sqrt(np.asarray(area_cm2, dtype=np.float64)) / R_EARTH_CM)
    lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
    east = offset[:, 1] * width / np.maximum(np.cos(np.deg2rad(lat)), 1e-6)
    return np.clip(lat + offset[:, 0] * width, -90.0, 90.0), (lon + east) % 360.0


def one_degree_cell(lat, lon):
    """Row and column of the 1-degree cell, the CESM2 cohort's ``grid_y`` / ``grid_x``."""
    iy = np.clip(np.floor(np.asarray(lat) + 90.0).astype(int), 0, 179)
    ix = np.floor(np.asarray(lon) % 360.0).astype(int) % 360
    return iy, ix


def calendar_month(job):
    """Values and climatology at the cells of the six months sharing one calendar month."""
    source_key, root, month, cells = job
    source = SOURCES[source_key](root)
    months = [month + 12 * y for y in range(N_MONTHS // 12)]
    union = np.concatenate([cells[t] for t in months])
    train = np.array([SPLITS["train"][0] <= year_month(t)[0] <= SPLITS["train"][1] for t in months])
    out = {}
    for variable in ("TEMP", "SALT"):
        values = np.stack([columns(source.field(variable, t), source.kmt, source.full_level_index, union)
                           for t in months])                       # (6 years, cells, L)
        clim = train_mean(values, train)
        for i, t in enumerate(months):
            rows = slice(i * N_PER_MONTH, (i + 1) * N_PER_MONTH)
            out[(variable, t)] = values[i, rows], clim[rows]
    return out


def main():
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES), required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    started = time.time()
    source = SOURCES[args.source](ROOT)
    ocean = (source.kmt > 0) & (np.abs(source.lat) <= 90.0) & (source.area > 0) & (source.area < FILL)
    print(f"{source.describe}: grid {ocean.shape}, ocean cells {int(ocean.sum()):,} "
          f"({100 * ocean.mean():.1f} %)", flush=True)
    rng = np.random.default_rng(SEED)
    cells, held, offset = {}, {}, {}
    for t in range(N_MONTHS):            # positions first, in month order, so they do not depend on the workers
        cells[t] = draw_cells(source.area, ocean, N_PER_MONTH, rng)
        held[t] = draw_queries(N_PER_MONTH, QUERY_FRAC, rng)
        offset[t] = rng.uniform(-IN_CELL, IN_CELL, (N_PER_MONTH, 2))
    values = {}
    jobs = [(args.source, str(ROOT), month, cells) for month in range(12)]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(calendar_month, jobs)):
            values.update(result)
            print(f"calendar month {i + 1}/12 ({time.time() - started:.0f}s)", flush=True)

    flat = np.concatenate([cells[t] for t in range(N_MONTHS)])
    lat, lon = position_in_cell(source.lat.ravel()[flat], source.lon.ravel()[flat], source.area.ravel()[flat],
                                np.concatenate([offset[t] for t in range(N_MONTHS)]))
    month_index = np.repeat(np.arange(N_MONTHS), N_PER_MONTH)
    year = 2000 + month_index // 12
    year_split = np.full(year.size, "unassigned", dtype=object)
    for name, (lo, hi) in SPLITS.items():
        year_split[(year >= lo) & (year <= hi)] = name
    iy, ix = one_degree_cell(lat, lon)
    arrays = {}
    for variable in ("TEMP", "SALT"):
        value = np.concatenate([values[(variable, t)][0] for t in range(N_MONTHS)])
        clim = np.concatenate([values[(variable, t)][1] for t in range(N_MONTHS)])
        clim = np.where(np.isfinite(value), clim, np.nan).astype(np.float32)
        arrays.update({variable: value, f"{variable}_RAW": value,
                       f"CLIM_POS_{variable}": clim, f"CLIM_CELL_{variable}": clim})
    zero = int(((arrays["TEMP"] == 0) & (arrays["SALT"] == 0)).sum())
    if zero:
        raise SystemExit(f"{zero} values read TEMP = 0 and SALT = 0: a fill this script does not know")
    P, L = arrays["TEMP"].shape
    out = xr.Dataset(
        {**{k: (("profile", "level"), v) for k, v in arrays.items()},
         "TEMP_ERR": (("profile", "level"), np.full((P, L), np.nan, "float32")),
         "SALT_ERR": (("profile", "level"), np.full((P, L), np.nan, "float32")),
         "lat": ("profile", lat), "lon": ("profile", lon),
         "grid_y": ("profile", iy), "grid_x": ("profile", ix),
         "model_j": ("profile", flat // ocean.shape[1]), "model_i": ("profile", flat % ocean.shape[1]),
         "month_index": ("profile", month_index), "year": ("profile", year),
         "wmo": ("profile", np.array([f"S{t:02d}{i:04d}" for t in range(N_MONTHS)
                                      for i in range(N_PER_MONTH)]).astype(str)),
         "year_split": ("profile", year_split.astype(str)),
         "float_split": ("profile", np.concatenate([np.where(held[t], "heldout_float", "cohort_float")
                                                    for t in range(N_MONTHS)]).astype(str))},
        coords={"level": LEVELS},
        attrs={"region": "synthetic", "grid": [180, 360], "seed": SEED,
               "source": f"{source.describe}; model columns at area-weighted random ocean cells, "
                         "no interpolation; position uniform inside the cell", "model_grid": list(ocean.shape),
               "splits": json.dumps(SPLITS)})
    path = ROOT / "data/synthetic_argo" / f"{source.name}.nc"
    temporary = path.with_name(path.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, path)
    finite = np.isfinite(arrays["SALT"]).mean(axis=0)
    anomaly = arrays["TEMP"] - arrays["CLIM_POS_TEMP"]
    train = year_split == "train"
    print(f"{P:,} profiles ({N_PER_MONTH}/month x {N_MONTHS}), "
          f"{int(sum(h.sum() for h in held.values())):,} queries; finite fraction by level "
          f"{finite.min():.3f}-{finite.max():.3f}; TEMP anomaly std at 5 m: train "
          f"{np.nanstd(anomaly[train, 0]):.3f}, 2004 {np.nanstd(anomaly[year == 2004, 0]):.3f} degC"
          f"\n-> {path}  ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
