"""Surface inputs for the 1964-2005 synthetic cohort: SST, SSS and steric SSH.

``50_prepare_matched_satellites.py`` for ``cesm2_long_uniform`` (see
``87_cesm2_long_cohort.py``), from the 20-level store of
``53_cesm2_long_levels.py``:

  SST, SSS   CESM2's 5 m TEMP and SALT, noise free
  SSH        TEOS-10 steric height relative to 990 dbar from TEMP/SALT on the
             cohort's 20 levels (``ocean_tokenizer.ssh``); NaN where the column
             does not reach the reference level

Each field is an anomaly against its 2000-2003 monthly climatology and is
scaled by its 2000-2003 mean and spread, the same numbers 50 uses, so the
2000-2005 rows of the cache are the rows every earlier CESM2 surface result
was trained and scored with; the script checks this against
``outputs/radius_scale_20261008/satellites.npz`` before writing. The feature
layout is 50's: 44 position/season fields, then SST/SSS/SSH anomaly z-scores
with validity flags in columns 44-49 and absolute SST in column 50.

As in 50, SST/SSS are the truth's top level and SSH is a vertical integral of
the T/S being reconstructed: optimistic stand-ins for satellite data.

  .venv/bin/python experiments/synthetic/88_cesm2_long_surface.py --workers 4
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
from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns
from ocean_tokenizer.synthetic_latent_experiment import position_features

COHORT = "cesm2_long_uniform"
STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr"
PREVIOUS = ROOT / "outputs/radius_scale_20261008/satellites.npz"
TRAIN = (2000, 2003)

_spec = importlib.util.spec_from_file_location("prepare_satellite", ROOT / "experiments/synthetic/50_prepare_matched_satellites.py")
_fifty = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fifty)
sample_grid = _fifty.sample_grid


def month_fields(task):
    """(3, NY, NX) SST, SSS and steric height of one month, as 50 computes them."""
    position, levels = task
    with xr.open_zarr(STORE, chunks=None) as ds:
        temp = np.asarray(ds.TEMP[position].values, dtype=np.float32)
        salt = np.asarray(ds.SALT[position].values, dtype=np.float32)
        lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    fill = (temp == 0) & (salt == 0)
    temp[fill], salt[fill] = np.nan, np.nan
    out = np.full((3,) + temp.shape[1:], np.nan, dtype=np.float32)
    out[0], out[1] = temp[0], salt[0]
    wet = np.isfinite(temp).all(0) & np.isfinite(salt).all(0)
    iy, ix = np.where(wet)
    if len(iy):
        out[2, iy, ix] = steric_height_columns(temp[:, iy, ix], salt[:, iy, ix], levels,
                                               lat2d[iy, ix], lon2d[iy, ix], P_REF_DBAR)
    return position, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/long_record_20261017/satellites.npz")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    started = time.time()
    c, norm, obs = S.load(str(ROOT), cohort=COHORT)
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    cohort_fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    with xr.open_zarr(STORE, chunks=None) as ds:
        grid_lat, grid_lon = np.asarray(ds.lat.values), np.asarray(ds.lon.values)
        store_months = np.asarray(ds.month_index.values, dtype=int)
        if not np.allclose(ds.level.values, c.levels):
            raise SystemExit("store and cohort levels differ")
    wanted = np.unique(c.month_index)
    position = {int(m): i for i, m in enumerate(store_months)}
    if any(int(m) not in position for m in wanted):
        raise SystemExit("the store does not cover every cohort month")
    fields = np.full((len(wanted), 3, len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
    at = {position[int(m)]: i for i, m in enumerate(wanted)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = ((position[int(m)], np.asarray(c.levels)) for m in wanted)
        for count, (where, out) in enumerate(pool.map(month_fields, tasks), 1):
            fields[at[where]] = out
            if count % 48 == 0:
                print(f"surface fields {count}/{len(wanted)} months ({time.time() - started:.0f}s)", flush=True)
    year = 2000 + wanted // 12
    train = (year >= TRAIN[0]) & (year <= TRAIN[1])
    climate = np.stack([np.nanmean(fields[train & (wanted % 12 == m)], axis=0) for m in range(12)])
    anomalies = fields - climate[wanted % 12]
    mean = np.nanmean(anomalies[train], axis=(0, 2, 3))
    std = np.nanstd(anomalies[train], axis=(0, 2, 3))
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.)
    z = (anomalies - mean[None, :, None, None]) / std[None, :, None, None]
    features = position_features(c).astype(np.float32)
    sampled = np.empty((len(c.lat), 3), dtype=np.float32)
    absolute = np.empty_like(sampled)
    for i, month in enumerate(wanted):
        rows = c.month(int(month))
        sampled[rows] = sample_grid(z[i], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
        absolute[rows] = sample_grid(fields[i], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
    for k in range(3):
        features[:, 44 + 2 * k] = np.nan_to_num(sampled[:, k])
        features[:, 45 + 2 * k] = np.isfinite(sampled[:, k])
    features[:, 50] = np.nan_to_num((absolute[:, 0] - 14.) / 11.)

    # 50's guard: the 5 m samples are the cohort's own top level
    with xr.open_dataset(ROOT / f"data/synthetic_argo/{COHORT}.nc") as raw:
        order = np.argsort(np.asarray(raw.month_index.values), kind="stable")
        for k, ch in enumerate(("TEMP", "SALT")):
            expected = np.asarray(raw[ch].values)[order, 0]
            if not np.allclose(absolute[:, k], expected, atol=2e-6, rtol=2e-6, equal_nan=True):
                raise SystemExit(f"{ch} surface sampling differs from the cohort's 5 m values")
    # the 2000-2005 rows are the cache the earlier CESM2 surface runs used
    with np.load(PREVIOUS, allow_pickle=False) as previous:
        recent = c.month_index >= 0
        for name in ("month_index", "lat", "lon"):
            if not np.array_equal(previous[name], getattr(c, name)[recent]):
                raise SystemExit(f"2000-2005 {name} differs from {PREVIOUS.name}")
        largest = float(np.abs(previous["features"] - features[recent]).max())
        if largest > 1e-5:
            raise SystemExit(f"2000-2005 features differ from {PREVIOUS.name} by up to {largest:.3g}")
        for name, value in (("surface_mean", mean), ("surface_std", std)):
            if not np.allclose(previous[name], value, rtol=1e-6, atol=0):
                raise SystemExit(f"{name} differs from {PREVIOUS.name}")
    print(f"2000-2005 features equal {PREVIOUS.name} (largest difference {largest:.2g})", flush=True)

    metadata = {
        "cohort_fingerprint": cohort_fingerprint,
        "cohort": COHORT,
        "normalization_scope": "training years only",
        "normalization_years": list(TRAIN),
        "splits": {k: list(v) for k, v in S.SPLITS.items()},
        "recipe": "noise-free CESM2 TEMP/SALT at 5 m and TEOS-10 steric height; periodic bilinear sampling",
        "surface_variables": ["SST", "SSS", "steric_SSH"],
        "surface_feature_offsets": [44, 46, 48],
        "p_ref_dbar": P_REF_DBAR,
        "surface_is_independent_of_truth": False,
        "satellite_noise": "none",
        "source_store": str(STORE.relative_to(ROOT)),
        "matches_for_2000_2005": str(PREVIOUS.relative_to(ROOT)),
        "largest_difference_2000_2005": largest,
        "feature_fingerprint": array_fingerprint((features,)),
        "surface_fingerprint": array_fingerprint((wanted, z)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
                        metadata=json.dumps(metadata), surface_climatology=climate,
                        surface_mean=mean, surface_std=std, grid_month=wanted,
                        grid_lat=grid_lat, grid_lon=grid_lon,
                        normalization_mean=np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], -1),
                        normalization_std=np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], -1))
    os.replace(temporary, args.output)
    print(json.dumps({"cache": str(args.output), "profiles": len(c.lat), "months": len(wanted),
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
