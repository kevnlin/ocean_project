"""Inputs of the sealed 2024 real test, built after every model was frozen (2026-10-21).

The real cohort file already holds 2024 (117,111 profiles of 3,290 floats, splits stored per
float); the 1-degree satellite store ``data/real_obs_1deg.zarr`` ends in December 2023.
``41_download_real_obs.py --start 2023 --end 2024`` builds ``data/real_obs_1deg_2023_2024.zarr``
from the same three products (OISST v2.1, OISSS L4 v2, CMEMS DUACS SLA) with the same regrid. This
script then

  1. compares the new store's 2023 months with the old store's (same products, same recipe);
  2. stacks the old store's 2016-2023 months and the new store's 2024 months and builds the three
     caches of ``105_real_surface.py`` on the 2016-2024 cohort (``load_real(years=(2016, 2024))``)
     into ``outputs/real_2024_test``;
  3. requires every 2016-2023 row and month of them to equal the caches the 2022-2023 test
     registered (``outputs/real_global_20261019``), value for value;
  4. checks the satellite SST against the measured 5 m temperature of the 2024 INPUT floats only:
     no value of a held-out 2024 profile is read before the test is registered.

``--replicate`` builds the 2016-2023 caches from the old store alone and only runs step 3: the
recipe must give the registered caches back before any 2024 input is used.

  .venv/bin/python experiments/real_data/132_real_2024_inputs.py --replicate
  .venv/bin/python experiments/real_data/132_real_2024_inputs.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import xarray as xr

from ocean_tokenizer.audit_tools import cohort_path
from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.real_global import (REAL_SPLITS, cohort_fingerprint, load_real, profile_surface_columns,
                                         surface_z)
from ocean_tokenizer.synthetic_latent_experiment import position_features

OLD_STORE = ROOT / "data/real_obs_1deg.zarr"
NEW_STORE = ROOT / "data/real_obs_1deg_2023_2024.zarr"
REGISTERED = ROOT / "outputs/real_global_20261019"
OUTPUT = ROOT / "outputs/real_2024_test"
LEADS = (1, 3, 6)
VARIABLES = ("SST", "SSS", "SLA")


def save(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def read_store(path):
    with xr.open_zarr(path) as ds:
        t = ds.time.values.astype("datetime64[M]").astype(int)
        months = (t // 12 + 1970 - 2000) * 12 + t % 12
        fields = np.stack([ds[v].values for v in VARIABLES], 1).astype(np.float64)
        return months.astype(int), fields, ds.lat.values.astype(np.float64), ds.lon.values.astype(np.float64), \
            ds.attrs.get("provenance")


def compare_2023(old_months, old_fields, new_months, new_fields):
    """Per variable over the 2023 months: largest and mean absolute difference, and the field spread."""
    report = {}
    for k, name in enumerate(VARIABLES):
        a = old_fields[np.isin(old_months, new_months), k]
        b = new_fields[np.isin(new_months, old_months), k]
        both = np.isfinite(a) & np.isfinite(b)
        report[name] = {"max_abs": float(np.abs(a - b)[both].max()), "mean_abs": float(np.abs(a - b)[both].mean()),
                        "spread": float(np.nanstd(a)), "same_cells": bool(np.array_equal(np.isfinite(a), np.isfinite(b)))}
    return report


def build(c, months, fields, grid_lat, grid_lon):
    year = 2000 + months // 12
    train = (year >= REAL_SPLITS["train"][0]) & (year <= REAL_SPLITS["train"][1])
    z, climate, mean, spread = surface_z(fields, months, train)
    features = position_features(c).astype(np.float32)
    features[:, 44:51] = profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon,
                                                 c.month_index)
    lagged = np.stack([profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon,
                                               c.month_index, lag=lead) for lead in LEADS], 1).astype(np.float32)
    return z, climate, mean, spread, features, lagged


def same_as_registered(n_rows, months, z, features, lagged):
    """Every 2016-2023 row and month equals the registered caches; returns what was compared."""
    with np.load(REGISTERED / "surface_grids.npz") as g, np.load(REGISTERED / "satellites.npz") as s, \
            np.load(REGISTERED / "lagged_surface.npz") as l:
        old_months, old_grids = g["grid_month"], g["grids"]
        if not np.array_equal(months[:len(old_months)], old_months):
            raise SystemExit("the 2016-2023 months differ from the registered grids")
        checks = {"grids": np.array_equal(z[:len(old_months)].astype(np.float16), old_grids, equal_nan=True),
                  "features": np.array_equal(features[:n_rows], s["features"]),
                  "lagged": np.array_equal(lagged[:n_rows], l["features"]),
                  "rows": int(n_rows), "months": int(len(old_months))}
        if s["features"].shape[0] != n_rows:
            raise SystemExit("the registered caches hold another number of 2016-2023 rows")
    if not all(checks[k] for k in ("grids", "features", "lagged")):
        raise SystemExit(f"the recipe does not give the registered caches back: {checks}")
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replicate", action="store_true")
    args = p.parse_args()
    started = time.time()
    old_months, old_fields, grid_lat, grid_lon, old_provenance = read_store(OLD_STORE)
    if args.replicate:
        c, norm, obs = load_real(ROOT)
        z, climate, mean, spread, features, lagged = build(c, old_months, old_fields, grid_lat, grid_lon)
        print(json.dumps({"replicated": same_as_registered(len(c.lat), old_months, z, features, lagged),
                          "seconds": round(time.time() - started)}), flush=True)
        return
    if (OUTPUT / "satellites.npz").exists():
        raise SystemExit(f"{OUTPUT} holds caches: the 2024 inputs are built once")
    new_months, new_fields, new_lat, new_lon, new_provenance = read_store(NEW_STORE)
    if not (np.array_equal(new_lat, grid_lat) and np.array_equal(new_lon, grid_lon)):
        raise SystemExit("the two stores have different grids")
    wanted = np.arange(12 * 24, 12 * 25)
    if not np.isin(wanted, new_months).all() or not np.isin(np.arange(12 * 23, 12 * 24), new_months).all():
        raise SystemExit("the new store must hold every month of 2023 and 2024")
    overlap = compare_2023(old_months, old_fields, new_months, new_fields)
    print(json.dumps({"2023 overlap": overlap}), flush=True)
    for name, r in overlap.items():
        if not r["same_cells"] or r["mean_abs"] > .01 * r["spread"]:
            raise SystemExit(f"{name}: the new store's 2023 differs from the old store's; check the product")
    months = np.r_[old_months, wanted]
    fields = np.concatenate([old_fields, new_fields[np.isin(new_months, wanted)]], 0)

    c, norm, obs = load_real(ROOT, years=(2016, 2024))
    n_old = int(np.sum(c.year <= 2023))
    if not np.all(c.year[:n_old] <= 2023) or not np.all(c.year[n_old:] == 2024):
        raise SystemExit("the 2016-2023 rows must come first, as in the registered cohort")
    z, climate, mean, spread, features, lagged = build(c, months, fields, grid_lat, grid_lon)
    checks = same_as_registered(n_old, months, z, features, lagged)

    # the satellite SST at the 2024 input floats against their own measured 5 m temperature
    with xr.open_dataset(cohort_path(str(ROOT), "global")) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        raw_year = np.asarray(raw.year.values)[order]
        rows = order[(raw_year >= 2016) & (raw_year <= 2024)]
        if not np.array_equal(np.asarray(raw.lat.values)[rows], c.lat):
            raise SystemExit("the raw cohort rows differ from the loaded cohort")
        inputs_2024 = np.flatnonzero((c.year == 2024) & (c.float_split == "cohort_float"))
        measured = np.asarray(raw.TEMP.values[rows[inputs_2024], 0])
    sst = features[inputs_2024, 50] * 11. + 14.
    ok = (features[inputs_2024, 45] > 0) & np.isfinite(measured)
    corr = float(np.corrcoef(sst[ok], measured[ok])[0, 1])
    bias = float(np.mean(sst[ok] - measured[ok]))
    print(json.dumps({"sst_vs_argo_5m_2024_inputs": {"r": corr, "mean_difference_C": bias, "profiles": int(ok.sum())},
                      "registered_rows_equal": checks}), flush=True)
    if corr < .98 or abs(bias) > .5:
        raise SystemExit("the 2024 satellite SST does not match the measured 5 m temperature; check the grids")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    common = {"cohort_fingerprint": cohort_fingerprint(c, obs), "cohort": "real_global 2016-2024",
              "splits": {k: list(v) for k, v in REAL_SPLITS.items()}, "test_year": 2024,
              "normalization_scope": "training years only", "surface_variables": list(VARIABLES),
              "source": [str(OLD_STORE.relative_to(ROOT)) + " (2016-2023)", str(NEW_STORE.relative_to(ROOT)) + " (2024)"],
              "store_provenance": [old_provenance, new_provenance], "overlap_2023": overlap,
              "registered_rows_equal": checks,
              "sst_vs_argo_5m_2024_inputs": {"r": corr, "mean_difference_C": bias},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(OUTPUT / "surface_grids.npz", grids=z.astype(np.float16), grid_month=months, grid_lat=grid_lat,
         grid_lon=grid_lon, surface_climatology=climate.astype(np.float32), surface_mean=mean, surface_std=spread,
         metadata=json.dumps({**common, "units": "z-score of the anomaly against the 2016-2020 monthly climatology"}))
    save(OUTPUT / "satellites.npz", features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
         metadata=json.dumps({**common, "feature_fingerprint": array_fingerprint((features,))}))
    save(OUTPUT / "lagged_surface.npz", features=lagged, leads=np.array(LEADS), month_index=c.month_index,
         lat=c.lat, lon=c.lon, metadata=json.dumps(common))
    print(json.dumps({"output": str(OUTPUT), "profiles": int(len(c.lat)), "months": int(len(months)),
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
