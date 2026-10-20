"""Surface inputs of earlier months at every profile of the 1964-2005 cohort, for forecasts.

A forecast at lead L reads the observations of month t and is scored at month t + L. A scene's
query carries its own surface features (SST, SSS and steric SSH at the query, columns 44-50 of
the 56), which in a forecast scene would come from the target month: satellite data a forecaster
cannot have. This writes, for every profile p of month m and every lead L, the surface features of
month m - L sampled at p's position, from the fields and the 2000-2003 normalisation of
``88_cesm2_long_surface.py``, so ``forecast_scenes`` can put them in place of month m's:

    outputs/forecast_20261019/lagged_surface.npz
        features (P, n_leads, 7)  SST z, valid, SSS z, valid, SSH z, valid, (SST - 14) / 11
        leads, month_index, lat, lon, metadata

A profile whose month m - L is before 1964-01 gets zeros and validity 0. Before writing, the
same code at lead 0 must give the surface columns of ``outputs/long_record_20261017/satellites.npz``
exactly.

  .venv/bin/python experiments/synthetic/93_lagged_surface.py --workers 4
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
import xarray as xr

from ocean_tokenizer import synth_argo_eval as S
from ocean_tokenizer.latent_experiment import array_fingerprint

COHORT = "cesm2_long_uniform"
SURFACE = ROOT / "outputs/long_record_20261017/satellites.npz"
LEADS = (1, 3, 6)

_spec = importlib.util.spec_from_file_location("long_surface", ROOT / "experiments/synthetic/88_cesm2_long_surface.py")
_surface = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_surface)


def month_fields(task):
    """88's month of surface fields, from this module so worker processes can be handed it."""
    return _surface.month_fields(task)


def lagged_columns(c, z, fields, at, lead, grid):
    """(P, 7) surface columns of month m - lead at each profile of month m."""
    out = np.zeros((len(c.lat), 7), dtype=np.float32)
    for month in np.unique(c.month_index):
        source = at.get(int(month) - lead)
        if source is None:
            continue
        rows = c.month(int(month))
        sampled = _surface.sample_grid(z[source], c.lat[rows], c.lon[rows], *grid)
        absolute = _surface.sample_grid(fields[source][:1], c.lat[rows], c.lon[rows], *grid)
        for k in range(3):
            out[rows, 2 * k] = np.nan_to_num(sampled[:, k])
            out[rows, 2 * k + 1] = np.isfinite(sampled[:, k])
        out[rows, 6] = np.nan_to_num((absolute[:, 0] - 14.) / 11.)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/forecast_20261019/lagged_surface.npz")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    started = time.time()
    c, norm, obs = S.load(str(ROOT), cohort=COHORT)
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    with np.load(SURFACE, allow_pickle=False) as cache:
        meta = json.loads(str(cache["metadata"].item()))
        if meta["cohort_fingerprint"] != fingerprint:
            raise SystemExit(f"{SURFACE.name} was built on another cohort")
        climate, mean, std = cache["surface_climatology"], cache["surface_mean"], cache["surface_std"]
        grid_month = cache["grid_month"]
        grid = (cache["grid_lat"], cache["grid_lon"])
        reference = cache["features"][:, 44:51]
    store = xr.open_zarr(_surface.STORE, chunks=None)
    position = {int(m): i for i, m in enumerate(np.asarray(store.month_index.values, dtype=int))}
    store.close()
    fields = np.full((len(grid_month), 3) + climate.shape[-2:], np.nan, dtype=np.float32)
    at = {int(m): i for i, m in enumerate(grid_month)}
    index_of = {position[int(m)]: i for i, m in enumerate(grid_month)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = ((position[int(m)], np.asarray(c.levels)) for m in grid_month)
        for count, (where, out) in enumerate(pool.map(month_fields, tasks), 1):
            fields[index_of[where]] = out
            if count % 96 == 0:
                print(f"surface fields {count}/{len(grid_month)} months ({time.time() - started:.0f}s)", flush=True)
    z = (fields - climate[grid_month % 12] - mean[None, :, None, None]) / std[None, :, None, None]

    check = lagged_columns(c, z, fields, at, 0, grid)
    largest = float(np.abs(check - reference).max())
    if largest > 1e-6:
        raise SystemExit(f"lead 0 differs from {SURFACE.name} by up to {largest:.3g}")
    print(f"lead 0 reproduces {SURFACE.name} (largest difference {largest:.2g})", flush=True)
    features = np.stack([lagged_columns(c, z, fields, at, lead, grid) for lead in LEADS], axis=1)
    missing = {lead: int((features[:, i, 1] == 0).sum()) for i, lead in enumerate(LEADS)}
    metadata = {"cohort_fingerprint": fingerprint, "cohort": COHORT, "leads_months": list(LEADS),
                "columns": ["SST_z", "SST_valid", "SSS_z", "SSS_valid", "SSH_z", "SSH_valid", "SST_absolute_scaled"],
                "replaces_feature_columns": [44, 51], "normalization_years": [2000, 2003],
                "source_surface_cache": str(SURFACE.relative_to(ROOT)),
                "source_surface_sha256": hashlib.sha256(SURFACE.read_bytes()).hexdigest(),
                "lead0_largest_difference": largest, "profiles_without_earlier_month": missing,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=features, leads=np.array(LEADS), month_index=c.month_index,
                        lat=c.lat, lon=c.lon, metadata=json.dumps(metadata))
    os.replace(temporary, args.output)
    print(json.dumps({"cache": str(args.output), "shape": list(features.shape), "missing": missing,
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
