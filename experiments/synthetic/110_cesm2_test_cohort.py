"""A sealed CESM2 test: the 1964-2005 cohort plus 2006-2007, never seen by any training or selection.

Every gridded and reference model so far was trained on ``cesm2_long_uniform`` (1964-2005) and its
checkpoint chosen on 2004-2005, so their scores on those years are validation scores. This builds
``cesm2_long_test``, the same cohort with 24 later months appended, and the caches a forecast reads:

  data/synthetic_argo/cesm2_long_test.nc
      the 3,830,400 profiles of ``cesm2_long_uniform.nc``, every variable unchanged and in order,
      then 2006-01 ... 2007-12 drawn exactly as ``87`` draws a month (7,600 area-uniform ocean
      positions, 20 % held-out floats, generator ``[SEED, month + 10000]``, CESM2 TEMP/SALT
      bilinear from the 1964-2007 level store of ``53``, anomalies against the same 2000-2003
      climatology). Their year split is "unassigned", like 1964-1999.
  outputs/cesm2_test_20261019/satellites.npz        88's 56 features (old rows copied)
  outputs/cesm2_test_20261019/surface_grids.npz     100's dense grids, 1964-2007 (old months copied)
  outputs/cesm2_test_20261019/lagged_surface.npz    93's lagged columns, leads 1/3/6 (old rows copied)

The normalisation of every profile, field and feature is the 2000-2003 fit that training used; the
new months only add rows. Checks before anything is written: the store's 1964-2005 months equal
the training store value for value; the climatology rebuilt from the store reproduces the cohort's
``CLIM_POS``; the feature, grid and lag code applied to old months reproduces the existing caches.
A forecast at lead L scored in 2006 reads inputs from 2005 back to 2005-07, which every model saw
only as validation inputs, never as a scored target of training.

  .venv/bin/python experiments/synthetic/110_cesm2_test_cohort.py --workers 8
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import xarray as xr

from ocean_tokenizer import synth_argo_eval as S
from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.synthetic_latent_experiment import position_features


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module              # worker processes unpickle functions by module name
    spec.loader.exec_module(module)
    return module


COHORT_87 = _module("long_cohort", "experiments/synthetic/87_cesm2_long_cohort.py")
SURFACE_88 = _module("long_surface", "experiments/synthetic/88_cesm2_long_surface.py")

OLD = ROOT / "data/synthetic_argo/cesm2_long_uniform.nc"
NEW = ROOT / "data/synthetic_argo/cesm2_long_test.nc"
OLD_STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr"
STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels_1964_2007.zarr"
LONG = ROOT / "outputs/long_record_20261017"
GRIDS = ROOT / "outputs/grid_ka_20261019/surface_grids.npz"
LAGGED = ROOT / "outputs/forecast_20261019/lagged_surface.npz"
OUTPUT = ROOT / "outputs/cesm2_test_20261019"
TEST_MONTHS = tuple(range(72, 96))          # 2006-01 ... 2007-12
LEADS = (1, 3, 6)


def store_fields(task):
    """88's surface fields of one store position, read from the 1964-2007 store."""
    SURFACE_88.STORE = STORE
    return SURFACE_88.month_fields(task)


def columns(z, absolute, lat, lon, grid_lat, grid_lon):
    """88's columns 44-50 from one month's z-scored fields and absolute SST."""
    sampled = SURFACE_88.sample_grid(z, lat, lon, grid_lat, grid_lon)
    sst = SURFACE_88.sample_grid(absolute[:1], lat, lon, grid_lat, grid_lon)
    out = np.zeros((len(lat), 7), dtype=np.float32)
    for k in range(3):
        out[:, 2 * k] = np.nan_to_num(sampled[:, k])
        out[:, 2 * k + 1] = np.isfinite(sampled[:, k])
    out[:, 6] = np.nan_to_num((sst[:, 0] - 14.) / 11.)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    started = time.time()
    if NEW.exists():
        raise SystemExit(f"{NEW} exists; the sealed cohort is built once")

    # 1. the 1964-2007 store: its 1964-2005 months are the training store's, value for value
    store = xr.open_zarr(STORE, chunks=None)
    old_store = xr.open_zarr(OLD_STORE, chunks=None)
    months = np.asarray(store.month_index.values, dtype=int)
    if not np.array_equal(months, np.arange(-432, 96)):
        raise SystemExit("the store must hold 1964-01 ... 2007-12")
    for i in (0, 215, 431, 432, 503):
        for v in ("TEMP", "SALT"):
            if not np.array_equal(np.asarray(store[v][i].values), np.asarray(old_store[v][i].values), equal_nan=True):
                raise SystemExit(f"store month {months[i]} {v} differs from the training store")
    old_store.close()
    print(f"store: 1964-2005 equal to the training store at the checked months ({time.time() - started:.0f}s)", flush=True)
    lat_c, lon_c = store.lat.values.astype(float), store.lon.values.astype(float)

    def month(t):
        i = int(np.searchsorted(months, t))
        raw = {v: np.asarray(store[v][i].values, dtype="float32") for v in ("TEMP", "SALT")}
        fill = (raw["TEMP"] == 0) & (raw["SALT"] == 0)
        return raw, {v: np.where(fill, np.nan, a) for v, a in raw.items()}

    # 2. 87's climatology from the store, which must reproduce the cohort's CLIM_POS
    train = [t for t in range(72) if 2000 <= 2000 + t // 12 <= 2003]
    clean_train = {v: np.stack([month(t)[1][v] for t in train]) for v in ("TEMP", "SALT")}
    cal_train = np.array(train) % 12
    clim = {v: np.stack([np.nanmean(a[cal_train == m], axis=0) for m in range(12)]).astype("float32")
            for v, a in clean_train.items()}
    del clean_train
    wet0 = np.isfinite(month(0)[1]["TEMP"][0])
    old = xr.open_dataset(OLD)
    check = np.sort(np.random.default_rng(1).choice(old.sizes["profile"], 20_000, replace=False))
    cal = np.asarray(old.month_index.values)[check] % 12
    for v in ("TEMP", "SALT"):
        rebuilt = np.full((check.size, old.level.size), np.nan, "float32")
        for m in range(12):
            sel = cal == m
            idx, w = COHORT_87.stencil(old.lat.values[check][sel], old.lon.values[check][sel], lat_c, lon_c)
            rebuilt[sel] = COHORT_87.interp(clim[v][m], idx, w)
        if not np.array_equal(rebuilt, np.asarray(old[f"CLIM_POS_{v}"].values)[check], equal_nan=True):
            raise SystemExit(f"the climatology does not reproduce CLIM_POS_{v} of the training cohort")
    print(f"climatology reproduces the training cohort ({time.time() - started:.0f}s)", flush=True)

    # 3. the new months, drawn as 87 draws a month
    fresh = {}
    for t in TEST_MONTHS:
        raw, clean = month(t)
        col, arr = COHORT_87.draw_month(t, raw, clean, clim, wet0, lat_c, lon_c)
        for k, v in {**col, **arr}.items():
            fresh.setdefault(k, []).append(v)
    store.close()
    fresh = {k: np.concatenate(v) for k, v in fresh.items()}
    n_new, n_levels = fresh["lat"].size, old.level.size
    data_vars = {}
    for name, var in old.data_vars.items():
        values = np.asarray(var.values)
        if name == "year":
            extra = 2000 + fresh["month_index"] // 12
        elif name == "year_split":
            extra = np.full(n_new, "unassigned")
        elif name in ("TEMP_ERR", "SALT_ERR"):
            extra = np.full((n_new, n_levels), np.nan, "float32")
        else:
            extra = fresh[name]
        if values.dtype.kind == "U" or np.asarray(extra).dtype.kind == "U":
            joined = np.concatenate([values.astype(str), np.asarray(extra).astype(str)])
        else:
            joined = np.concatenate([values, np.asarray(extra).astype(values.dtype)])
        data_vars[name] = (var.dims, joined)
    attrs = dict(old.attrs)
    attrs["test_extension"] = json.dumps({
        "months": "2006-01 to 2007-12", "seed": COHORT_87.SEED,
        "per_month_generator": "numpy default_rng([seed, month_index + 10000]) (87's)",
        "store": str(STORE.relative_to(ROOT)), "rows_before": int(old.sizes["profile"]),
        "climatology": "2000-2003, identical to cesm2_long_uniform.nc",
        "script": "experiments/synthetic/110_cesm2_test_cohort.py"})
    out = xr.Dataset(data_vars, coords={"level": old.level.values}, attrs=attrs)
    n_old = old.sizes["profile"]
    old.close()
    temporary = NEW.with_name(NEW.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, NEW)
    print(f"cohort: {n_old:,} profiles copied + {n_new:,} new -> {NEW.name} ({time.time() - started:.0f}s)", flush=True)

    # 4. the caches: old rows and months copied, new ones by 88/100/93's recipes
    c, norm, obs = S.load(str(ROOT), cohort="cesm2_long_test")
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels,
                                     np.stack([obs[ch] for ch in ("TEMP", "SALT")], -1)))
    if not np.array_equal(c.month_index[:n_old], np.asarray(xr.open_dataset(OLD).month_index.values)[
            np.argsort(np.asarray(xr.open_dataset(OLD).month_index.values), kind="stable")]):
        raise SystemExit("the old rows are not first in the new cohort's order")
    with np.load(LONG / "satellites.npz", allow_pickle=False) as cache:
        satellites = {k: cache[k] for k in cache.files}
    climate, mean, std = satellites["surface_climatology"], satellites["surface_mean"], satellites["surface_std"]
    grid_lat, grid_lon = satellites["grid_lat"], satellites["grid_lon"]
    position = {int(m): i for i, m in enumerate(months)}
    wanted = list(range(66, 96))              # the test months and their inputs back to 2005-07
    fields = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for where, f in pool.map(store_fields, ((position[m], np.asarray(c.levels)) for m in wanted)):
            fields[int(months[where])] = f
    z = {m: (f - climate[m % 12] - mean[:, None, None]) / std[:, None, None] for m, f in fields.items()}
    # the recipe reproduces the existing caches on old months
    with np.load(GRIDS, allow_pickle=False) as cache:
        grids, grid_month = cache["grids"], cache["grid_month"]
    for m in (66, 71):
        stored = grids[int(np.flatnonzero(grid_month == m)[0])].astype(float)
        if np.nanmax(np.abs(stored - z[m]) / np.maximum(1., np.abs(z[m]))) > 1e-3:      # float16 storage
            raise SystemExit(f"grid of month {m} differs from {GRIDS.name}")
        rows = c.month(m)
        if np.abs(columns(z[m], fields[m], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
                  - satellites["features"][rows, 44:51]).max() > 1e-5:
            raise SystemExit(f"features of month {m} differ from {LONG.name}/satellites.npz")
    print(f"grid and feature recipes reproduce the existing caches ({time.time() - started:.0f}s)", flush=True)

    new_rows = np.arange(n_old, len(c.lat))
    features = np.zeros((len(c.lat), 56), dtype=np.float32)
    features[:n_old] = satellites["features"]
    features[new_rows] = position_features(SimpleNamespace(lat=c.lat[new_rows], lon=c.lon[new_rows],
                                                           month_index=c.month_index[new_rows]))
    lagged = np.zeros((len(c.lat), len(LEADS), 7), dtype=np.float32)
    with np.load(LAGGED, allow_pickle=False) as cache:
        lagged[:n_old] = cache["features"]
    for m in TEST_MONTHS:
        rows = c.month(m)
        features[rows, 44:51] = columns(z[m], fields[m], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
        for i, lead in enumerate(LEADS):
            lagged[rows, i] = columns(z[m - lead], fields[m - lead], c.lat[rows], c.lon[rows], grid_lat, grid_lon)
    all_grids = np.concatenate([grids, np.stack([z[m] for m in TEST_MONTHS]).astype(np.float16)])
    all_months = np.r_[grid_month, np.array(TEST_MONTHS)]
    common = {"cohort_fingerprint": fingerprint, "cohort": "cesm2_long_test", "normalization_scope": "training years only",
              "normalization_years": [2000, 2003], "test_months": [TEST_MONTHS[0], TEST_MONTHS[-1]],
              "copied_from": {"satellites": str((LONG / "satellites.npz").relative_to(ROOT)),
                              "grids": str(GRIDS.relative_to(ROOT)), "lagged": str(LAGGED.relative_to(ROOT))},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / "satellites.npz", features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
             surface_climatology=climate, surface_mean=mean, surface_std=std, grid_month=all_months,
             grid_lat=grid_lat, grid_lon=grid_lon, normalization_mean=satellites["normalization_mean"],
             normalization_std=satellites["normalization_std"],
             metadata=json.dumps({**common, "feature_fingerprint": array_fingerprint((features,))}))
    np.savez(OUTPUT / "surface_grids.npz", grids=all_grids, grid_month=all_months, grid_lat=grid_lat,
             grid_lon=grid_lon, metadata=json.dumps(common))
    np.savez(OUTPUT / "lagged_surface.npz", features=lagged, leads=np.array(LEADS), month_index=c.month_index,
             lat=c.lat, lon=c.lon, metadata=json.dumps(common))
    print(json.dumps({"cohort": str(NEW), "profiles": int(len(c.lat)), "new": int(n_new), "caches": str(OUTPUT),
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
