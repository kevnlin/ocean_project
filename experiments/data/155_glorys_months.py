"""Download and reduce GLORYS12 months, one at a time, keeping only the cohort's products.

Spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``. For every requested cohort month:

  1. ``copernicusmarine subset`` of thetao, so and zos at 0-1100 m (36 levels), one global
     month (about 1.3 GB), unless the raw file is already there;
  2. the two 3-D fields interpolated to the cohort's 20 levels (``glorys_sources``), the wet
     mask checked against the grid file;
  3. ``data/glorys/boxes/boxes_YYYYMM.npz``: the 1-degree area-weighted box means of the
     20-level T and S, and the four surface box fields SST (5 m), SSS (5 m), TEOS-10 steric
     height of the box-mean column (where complete) and zos;
  4. ``data/glorys/columns/columns_YYYYMM.npz``: the 20-level T and S columns at the union of
     the cells drawn for this calendar month in every cohort and test year (the climatology of a
     drawn cell is the mean over the training years of these);
  5. the raw file is deleted.

``--init`` writes the grid file (lat, lon, area, the 20-level wet mask) from the raw 1997-01
file and the draws file; it must run first. ``--test`` selects the sealed test years 2019-2020,
which are downloaded only when the test cohort is built; without it only 1997-2018 are touched.

  .venv/bin/python experiments/data/155_glorys_months.py --init
  .venv/bin/python experiments/data/155_glorys_months.py --workers 3
  .venv/bin/python experiments/data/155_glorys_months.py --test --workers 3
"""
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ocean_tokenizer import glorys_sources as G

COPERNICUS = Path(sys.executable).with_name("copernicusmarine")
LOGS = ROOT / "outputs/glorys_20261021/logs"
SETUP = {}


def download(folders, t, attempts=4):
    """The raw month file; returns (seconds, bytes). Retries a failed or unreadable download."""
    path = folders.raw_file(t)
    year, month = G.glorys_year_month(t)
    last = calendar.monthrange(year, month)[1]
    command = [str(COPERNICUS), "subset", "-i", G.DATASET, *sum([["-v", v] for v in G.VARIABLES], []),
               "-z", "0", "-Z", "1100", "-t", f"{year}-{month:02d}-01", "-T", f"{year}-{month:02d}-{last}",
               "-o", str(path.parent), "-f", path.name, "--overwrite", "--disable-progress-bar"]
    for attempt in range(1, attempts + 1):
        started = time.time()
        if not path.exists():
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            if result.returncode:
                print(json.dumps({"month": G.stamp(t), "download_failed": attempt, "stderr": result.stderr[-500:]}), flush=True)
                path.unlink(missing_ok=True)
                time.sleep(30 * attempt)
                continue
        try:
            import netCDF4
            with netCDF4.Dataset(path) as nc:
                if nc["thetao"].shape[0] != 1 or nc["depth"].shape[0] != 36:
                    raise ValueError("wrong shape")
            return time.time() - started, path.stat().st_size
        except Exception as error:
            print(json.dumps({"month": G.stamp(t), "unreadable": attempt, "error": repr(error)}), flush=True)
            path.unlink(missing_ok=True)
            time.sleep(30 * attempt)
    raise RuntimeError(f"month {G.stamp(t)} could not be downloaded")


def reduce(folders, t, grid, union):
    """Boxes and union columns of one raw month; the raw file is deleted afterwards."""
    from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns
    temp, salt, zos, meta = G.read_month(folders.raw_file(t))
    if not (np.allclose(meta["lat"], grid.lat) and np.allclose(meta["lon"], grid.lon) and np.allclose(meta["depth"], grid.depth)):
        raise ValueError(f"{G.stamp(t)}: grid or depth axis differs from the grid file's")
    grid.check(temp, salt, G.stamp(t))
    wet = np.flatnonzero(grid.ocean.ravel())
    box = G.one_degree_box(grid.lat2d.ravel()[wet], grid.lon2d.ravel()[wet])
    weight = grid.area.ravel()[wet]
    temp_box = G.box_mean(temp.reshape(len(G.LEVELS), -1)[:, wet], weight, box)
    salt_box = G.box_mean(salt.reshape(len(G.LEVELS), -1)[:, wet], weight, box)
    zos_box = G.box_mean(zos.reshape(1, -1)[:, wet], weight, box)[0]
    lat = np.repeat(np.arange(180) - 89.5, 360)
    lon = np.tile(np.arange(360) + 0.5, 180)
    full = np.isfinite(temp_box).all(axis=0) & np.isfinite(salt_box).all(axis=0)
    steric = np.full(G.N_BOXES, np.nan)
    if full.any():
        steric[full] = steric_height_columns(temp_box[:, full], salt_box[:, full], G.LEVELS, lat[full], lon[full], P_REF_DBAR)
    surface = np.stack([temp_box[0], salt_box[0], steric, zos_box]).astype(np.float32)
    folders.boxes.mkdir(parents=True, exist_ok=True)
    folders.columns.mkdir(parents=True, exist_ok=True)
    temporary = folders.boxes_file(t).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, TEMP=temp_box.astype(np.float32), SALT=salt_box.astype(np.float32), surface=surface,
                        surface_names=np.array(["SST", "SSS", "steric_SSH", "zos"]), levels=G.LEVELS,
                        month_index=t, stamp=G.stamp(t), time_hours_since_1950=meta["time_hours_since_1950"])
    os.replace(temporary, folders.boxes_file(t))
    cells = union[t % 12]
    temporary = folders.columns_file(t).with_suffix(".tmp.npz")
    np.savez_compressed(temporary, cells=cells, TEMP=temp.reshape(len(G.LEVELS), -1)[:, cells].T.astype(np.float32),
                        SALT=salt.reshape(len(G.LEVELS), -1)[:, cells].T.astype(np.float32), levels=G.LEVELS,
                        month_index=t, stamp=G.stamp(t))
    os.replace(temporary, folders.columns_file(t))
    return {"boxes_bytes": folders.boxes_file(t).stat().st_size, "columns_bytes": folders.columns_file(t).stat().st_size,
            "finite_fraction_5m_boxes": float(np.isfinite(temp_box[0]).mean()),
            "finite_fraction_steric": float(np.isfinite(steric).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", action="store_true")
    p.add_argument("--test", action="store_true", help="the sealed test years 2019-2020")
    p.add_argument("--months", nargs="+", type=int, default=None, help="cohort month indices, else all of the set")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--keep-raw", action="store_true")
    args = p.parse_args()
    folders = G.Folders(ROOT)
    LOGS.mkdir(parents=True, exist_ok=True)
    if args.init:
        if folders.grid.exists():
            raise SystemExit(f"{folders.grid} exists")
        first = G.COHORT_MONTHS[0]
        if not folders.raw_file(first).exists():
            download(folders, first)
        temp, salt, zos, meta = G.read_month(folders.raw_file(first))
        G.write_grid(folders, temp, salt, meta)
        grid = G.Grid(folders)
        d = G.draws(folders, grid)
        print(json.dumps({"grid": str(folders.grid), "ocean_cells": int(grid.ocean.sum()),
                          "ocean_fraction": float(grid.ocean.mean()), "wet_fraction_by_level": np.round(grid.wet.mean(axis=(1, 2)), 3).tolist(),
                          "draws": str(folders.draws), "months_drawn": len(d),
                          "union_cells_per_calendar_month": [int(len(G.union_cells(d, m))) for m in range(12)]}), flush=True)
        return
    months = args.months or list(G.TEST_MONTHS if args.test else G.COHORT_MONTHS)
    if not args.test and any(t in G.TEST_MONTHS for t in months):
        raise SystemExit("test months need --test")
    started = time.time()
    log = {"months": [], "started_unix": started}
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(process_month, [(str(ROOT), int(t), args.keep_raw) for t in months]):
            log["months"].append(result)
            print(json.dumps(result), flush=True)
    done = [m for m in log["months"] if not m.get("skipped")]
    log.update(seconds=time.time() - started, downloaded_bytes=sum(m["raw_bytes"] for m in done),
               download_seconds=sum(m["download_seconds"] for m in done), processed=len(done),
               kept_bytes=sum(m["boxes_bytes"] + m["columns_bytes"] for m in done))
    name = "test" if args.test else "cohort"
    (LOGS / f"months_{name}_{int(started)}.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({k: v for k, v in log.items() if k != "months"}), flush=True)


def process_month(job):
    root, t, keep_raw = job
    folders = G.Folders(root)
    if folders.boxes_file(t).exists() and folders.columns_file(t).exists():
        return {"month": G.stamp(t), "month_index": t, "skipped": True}
    if "grid" not in SETUP:                       # once per worker process
        SETUP["grid"] = G.Grid(folders)
        SETUP["union"] = {m: G.union_cells(G.draws(folders), m) for m in range(12)}
    grid, union = SETUP["grid"], SETUP["union"]
    started = time.time()
    seconds, size = download(folders, t)
    reduced = reduce(folders, t, grid, union)
    if not keep_raw:
        folders.raw_file(t).unlink()
    return {"month": G.stamp(t), "month_index": t, "download_seconds": round(seconds, 1), "raw_bytes": size,
            "seconds": round(time.time() - started, 1), **reduced}


if __name__ == "__main__":
    main()
