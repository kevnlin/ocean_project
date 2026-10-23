"""Dense monthly surface grids for the 1964-2005 cohort: SST, SSS and steric SSH z-scores.

``88_cesm2_long_surface.py`` computes these fields on CESM2's 1-degree grid and keeps only their
values at the profiles. The gridded state model reads the whole field, so this script writes the
grids themselves: 88's ``month_fields`` for every cohort month, minus 88's 2000-2003 monthly
climatology, scaled by 88's mean and spread (all three read from 88's cache, not refitted). NaN
over land and wherever 88 has no steric height.

Check before writing: bilinear sampling of these grids at every profile of a few months gives
the SST/SSS/SSH columns of 88's cache.

  .venv/bin/python experiments/synthetic/100_surface_grids.py --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

_spec = importlib.util.spec_from_file_location("long_surface", ROOT / "experiments/synthetic/88_cesm2_long_surface.py")
_eighty_eight = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _eighty_eight  # the worker processes unpickle month_fields by module name
_spec.loader.exec_module(_eighty_eight)
month_fields, sample_grid, STORE = _eighty_eight.month_fields, _eighty_eight.sample_grid, _eighty_eight.STORE

SATELLITES = ROOT / "outputs/long_record_20261017/satellites.npz"
OUTPUT = ROOT / "outputs/grid_ka_20261019/surface_grids.npz"
CHECK_MONTHS = (-432, -1, 0, 47, 48, 71)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    started = time.time()
    import xarray as xr
    with np.load(SATELLITES, allow_pickle=False) as cache:
        climate, mean, std = cache["surface_climatology"], cache["surface_mean"], cache["surface_std"]
        wanted, grid_lat, grid_lon = cache["grid_month"], cache["grid_lat"], cache["grid_lon"]
        features, months, lat, lon = cache["features"], cache["month_index"], cache["lat"], cache["lon"]
        metadata = json.loads(str(cache["metadata"].item()))
    with xr.open_zarr(STORE, chunks=None) as ds:
        store_months = np.asarray(ds.month_index.values, dtype=int)
        levels = np.asarray(ds.level.values)
    position = {int(m): i for i, m in enumerate(store_months)}
    grids = np.full((len(wanted), 3, len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
    at = {position[int(m)]: i for i, m in enumerate(wanted)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, (where, out) in enumerate(pool.map(month_fields, ((position[int(m)], levels) for m in wanted)), 1):
            i = at[where]
            grids[i] = (out - climate[int(wanted[i]) % 12] - mean[:, None, None]) / std[:, None, None]
            if count % 48 == 0:
                print(f"surface grids {count}/{len(wanted)} months ({time.time() - started:.0f}s)", flush=True)
    largest = 0.
    for month in CHECK_MONTHS:
        rows = np.flatnonzero(months == month)
        sampled = sample_grid(grids[int(np.flatnonzero(wanted == month)[0])], lat[rows], lon[rows], grid_lat, grid_lon)
        for k in range(3):
            ok = features[rows, 45 + 2 * k] > 0
            if (np.isfinite(sampled[:, k]) != ok).any():
                raise SystemExit(f"month {month}: surface field {k} validity differs from 88's cache")
            largest = max(largest, float(np.abs(sampled[ok, k] - features[rows[ok], 44 + 2 * k]).max(initial=0.)))
    if largest > 1e-4:
        raise SystemExit(f"sampled grids differ from 88's cache by up to {largest:.3g}")
    print(f"sampled grids equal 88's cache at the profiles (largest difference {largest:.2g})", flush=True)
    record = {"cohort_fingerprint": metadata["cohort_fingerprint"], "surface_variables": ["SST", "SSS", "steric_SSH"],
              "units": "z-score of the anomaly against the 2000-2003 monthly climatology (88's constants)",
              "satellites_sha256": hashlib.sha256(SATELLITES.read_bytes()).hexdigest(),
              "largest_difference_at_profiles": largest, "check_months": list(CHECK_MONTHS),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez(temporary, grids=grids.astype(np.float16), grid_month=wanted, grid_lat=grid_lat, grid_lon=grid_lon,
             metadata=json.dumps(record))
    os.replace(temporary, args.output)
    print(json.dumps({"cache": str(args.output), "shape": list(grids.shape), "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
