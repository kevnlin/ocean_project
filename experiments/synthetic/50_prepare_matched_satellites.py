"""Prepare explicit CESM2 surface observations for the matched Argo benchmark.

SST/SSS are noise-free CESM2 5 m fields; sea level is TEOS-10 steric height
derived from the same T/S truth, not an independent altimeter observation.
This is a new, documented multimodal arm, not a replay of unlocated artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from ocean_tokenizer import synth_argo_eval as S
from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns
from ocean_tokenizer.synthetic_latent_experiment import position_features

ROOT = Path(__file__).resolve().parents[2]


def sample_grid(field, lat, lon, grid_lat, grid_lon):
    """Periodic-longitude bilinear sampling; any invalid corner remains NaN."""
    fy = (np.asarray(lat) - grid_lat[0]) / (grid_lat[1] - grid_lat[0])
    fx = ((np.asarray(lon) - grid_lon[0]) % 360) / (grid_lon[1] - grid_lon[0])
    i0 = np.floor(fy).astype(int)
    j0 = np.floor(fx).astype(int) % len(grid_lon)
    if ((i0 < 0) | (i0 >= len(grid_lat))).any():
        raise ValueError("profiles outside the surface latitude grid")
    i1, j1 = np.minimum(i0 + 1, len(grid_lat) - 1), (j0 + 1) % len(grid_lon)
    wy, wx = fy - i0, fx - np.floor(fx)
    return (field[..., i0, j0] * ((1-wy)*(1-wx))
            + field[..., i0, j1] * ((1-wy)*wx)
            + field[..., i1, j0] * (wy*(1-wx))
            + field[..., i1, j1] * (wy*wx)).T


def prepare(root: Path, output: Path):
    c, norm, obs = S.load(str(root))
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    cohort_fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo,
                                           c.levels, standardized))
    # The store chunks span all 72 months. Read each selected depth block once;
    # per-month Dask indexing otherwise repeatedly decompresses the same data.
    # Eager loading also avoids nested async/thread schedulers in restricted jobs.
    with xr.open_zarr(root / "data/cesm2_le_full_standard.zarr", chunks=None) as ds:
        grid_lat, grid_lon = np.asarray(ds.lat.values), np.asarray(ds.lon.values)
        months = ((ds.time.values.astype("datetime64[M]")
                   - np.datetime64("2000-01", "M")) / np.timedelta64(1, "M")).astype(int)
        wanted = np.unique(c.month_index)
        lookup = {int(m): i for i, m in enumerate(months)}
        if any(int(m) not in lookup for m in wanted):
            raise ValueError("CESM2 store does not cover the exact cohort months")
        depth_index = [int(np.abs(ds.depth.values-d).argmin()) for d in c.levels]
        if not np.allclose(ds.depth.values[depth_index], c.levels, atol=.2):
            raise ValueError("CESM2 and cohort depths differ")
        print("loading matched CESM2 depth blocks", flush=True)
        all_temp = np.asarray(ds.TEMP.isel(depth=depth_index).values, dtype=np.float32)
        all_salt = np.asarray(ds.SALT.isel(depth=depth_index).values, dtype=np.float32)
        print("CESM2 depth blocks loaded", flush=True)
        fields = np.full((len(wanted), 3, len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        lat2d, lon2d = np.meshgrid(grid_lat, grid_lon, indexing="ij")
        for i, month in enumerate(wanted):
            temp = all_temp[lookup[int(month)]]
            salt = all_salt[lookup[int(month)]]
            fill = (temp == 0) & (salt == 0)
            temp[fill], salt[fill] = np.nan, np.nan
            fields[i, 0], fields[i, 1] = temp[0], salt[0]
            wet = np.isfinite(temp).all(0) & np.isfinite(salt).all(0)
            iy, ix = np.where(wet)
            if len(iy):
                fields[i, 2, iy, ix] = steric_height_columns(
                    temp[:, iy, ix], salt[:, iy, ix], c.levels,
                    lat2d[iy, ix], lon2d[iy, ix], P_REF_DBAR)
            if (i+1) % 12 == 0:
                print(f"surface observations {i+1}/{len(wanted)} months", flush=True)
    train = (2000+wanted//12 >= 2000) & (2000+wanted//12 <= 2003)
    climate = np.stack([np.nanmean(fields[train & (wanted % 12 == m)], axis=0)
                        for m in range(12)])
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
    # Same 44 positional features and 12 reserved fields as the Argo-only arm.
    # Synthetic features use train-normalized surface anomalies plus validity.
    for k in range(3):
        features[:, 44+2*k] = np.nan_to_num(sampled[:, k])
        features[:, 45+2*k] = np.isfinite(sampled[:, k])
    features[:, 50] = np.nan_to_num((absolute[:, 0]-14.)/11.)
    with xr.open_dataset(root / "data/synthetic_argo/cesm2_uniform.nc") as raw:
        order = np.argsort(np.asarray(raw.month_index.values), kind="stable")
        for k, ch in enumerate(("TEMP", "SALT")):
            expected = np.asarray(raw[ch].values)[order, 0]
            if not np.allclose(absolute[:, k], expected, atol=2e-6, rtol=2e-6, equal_nan=True):
                raise ValueError(f"{ch} surface sampling differs from the old CESM2 cohort")
    metadata = {
        "cohort_fingerprint": cohort_fingerprint,
        "normalization_scope": "training years only",
        "splits": {k:list(v) for k,v in S.SPLITS.items()},
        "recipe": "noise-free CESM2 TEMP/SALT at 5 m and TEOS-10 steric height; periodic bilinear sampling",
        "surface_variables": ["SST", "SSS", "steric_SSH"],
        "surface_feature_offsets": [44, 46, 48],
        "p_ref_dbar": P_REF_DBAR,
        "surface_is_independent_of_truth": False,
        "satellite_noise": "none",
        "historical_pasted_multimodal_artifact_replayed": False,
        "source_store": "data/cesm2_le_full_standard.zarr",
        "feature_fingerprint": array_fingerprint((features,)),
        "surface_fingerprint": array_fingerprint((wanted, z)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=features, month_index=c.month_index,
        lat=c.lat, lon=c.lon, metadata=json.dumps(metadata),
        surface_z=z, surface_physical=fields, surface_climatology=climate,
        surface_mean=mean, surface_std=std, grid_month=wanted,
        grid_lat=grid_lat, grid_lon=grid_lon,
        normalization_mean=np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], -1),
        normalization_std=np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], -1))
    os.replace(temporary, output)
    print(json.dumps({"cache": str(output), "profiles": len(c.lat),
                      "months": len(wanted), "metadata": metadata}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/satellites.npz")
    args = parser.parse_args()
    prepare(args.root, args.output)


if __name__ == "__main__":
    main()
