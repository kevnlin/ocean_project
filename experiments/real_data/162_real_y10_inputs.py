"""Inputs of the 10-year training window (2011-2020): the satellite caches on the 2011-2024 cohort.

The paper's real-data runs train on 2016-2020 (``105_real_surface.py`` caches, ``data/real_obs_1deg.zarr``);
the sealed 2024 test reads the caches of ``132_real_2024_inputs.py`` on the 2016-2024 cohort
(``outputs/real_2024_test``). This stream trains on 2011-2020 with everything else unchanged, so
its caches must be the registered ones with five years ADDED in front:

  1. ``41_download_real_obs.py --start 2011 --end 2015 --align union`` built
     ``data/real_obs_1deg_2011_2015.zarr`` from the same three products and the same regrid; the
     monthly OISSS starts in 2011-09, so SSS is NaN (masked through the flag columns) for
     2011-01..2011-08;
  2. the three stores are stacked (2011-2015 new, 2016-2023 old, 2024 of the 2023-2024 store) on
     one grid and the caches of ``105`` are built on the 2011-2024 cohort
     (``load_real(years=(2011, 2024))``) with the 2016-2020 monthly climatology and spread;
  3. every 2016-2024 row and month of the new caches must equal the registered 2024-test caches
     value for value: the dense grids, the profile features and the lagged features. The lagged
     features of the 2016-2024 rows are therefore taken as registered (a source month before
     2016-01 gives zeros, as ``105`` built them); only the 2011-2015 rows read the 2011-2015 grids
     (zeros before 2011-01). The count of 2016 rows this convention keeps at zero is recorded;
  4. the satellite SST of the 2011-2015 profiles (training years) against their measured 5 m
     temperature, as ``105``.

Only the three cache files of the registered 2024 test are read (satellite inputs, no profile
values); nothing in its ``sealed`` folder is touched.

  .venv/bin/python experiments/real_data/162_real_y10_inputs.py
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

OLD_STORE = ROOT / "data/real_obs_1deg.zarr"                 # 2016-2023
STORE_2024 = ROOT / "data/real_obs_1deg_2023_2024.zarr"      # 2023-2024 (2024 used)
NEW_STORE = ROOT / "data/real_obs_1deg_2011_2015.zarr"       # 2011-2015 (union-aligned)
#: the caches the sealed 2024 test registered (ka-dev4); only these three files are read
REGISTERED = Path("/DATA2/linke/ocean_project_ka_dev4/outputs/real_2024_test")
OUTPUT = ROOT / "outputs/real_y10_20261021"
YEARS = (2011, 2024)
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
            ds.attrs.get("provenance"), ds.attrs.get("month_alignment")


def month_label(m):
    return f"{2000 + m // 12}-{m % 12 + 1:02d}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    started = time.time()
    if (args.output / "satellites.npz").exists():
        raise SystemExit(f"{args.output} holds caches: the 10-year inputs are built once")
    old_months, old_fields, grid_lat, grid_lon, old_prov, _ = read_store(OLD_STORE)
    m24, f24, lat24, lon24, prov24, _ = read_store(STORE_2024)
    new_months, new_fields, new_lat, new_lon, new_prov, alignment = read_store(NEW_STORE)
    for name, (la, lo) in (("2023-2024", (lat24, lon24)), ("2011-2015", (new_lat, new_lon))):
        if not (np.array_equal(la, grid_lat) and np.array_equal(lo, grid_lon)):
            raise SystemExit(f"the {name} store is on another grid")
    if alignment != "union":
        raise SystemExit("the 2011-2015 store must be union-aligned (SSS masked, not dropped, before 2011-08)")
    wanted_new = np.arange(12 * 11, 12 * 16)
    wanted_24 = np.arange(12 * 24, 12 * 25)
    if not np.array_equal(new_months, wanted_new):
        raise SystemExit(f"the 2011-2015 store must hold exactly the 60 months of 2011-2015, has {len(new_months)}")
    if not np.isin(wanted_24, m24).all():
        raise SystemExit("the 2023-2024 store must hold every month of 2024")
    if not np.array_equal(old_months, np.arange(12 * 16, 12 * 24)):
        raise SystemExit("the old store must hold every month of 2016-2023")
    months = np.r_[new_months, old_months, wanted_24]
    fields = np.concatenate([new_fields, old_fields, f24[np.isin(m24, wanted_24)]], 0)
    if not np.all(np.diff(months) == 1):
        raise SystemExit("the stacked months are not contiguous")

    # the new months: which are missing per variable, and the cell pattern against 2016's
    coverage = {}
    for k, name in enumerate(VARIABLES):
        finite = np.isfinite(new_fields[:, k]).mean(axis=(1, 2))
        missing = [month_label(m) for m, f in zip(new_months, finite) if f == 0]
        same_cells = [float(np.mean(np.isfinite(new_fields[i, k]) == np.isfinite(old_fields[i % 12, k])))
                      for i in range(len(new_months)) if finite[i] > 0]
        coverage[name] = {"months_with_data": int((finite > 0).sum()), "months_missing": missing,
                          "finite_fraction_mean": float(finite[finite > 0].mean()),
                          "cell_pattern_agreement_with_2016_min": float(min(same_cells))}
    print(json.dumps({"2011-2015 coverage": coverage}), flush=True)
    for name, r in coverage.items():
        # OISSS L4 v2 monthly starts in 2011-09 (the first monthly granule): the months before it are
        # the only ones a variable may lack, and they are missing in one block at the front
        allowed = [month_label(m) for m in new_months[:8]] if name == "SSS" else []
        if r["months_missing"] != allowed[:len(r["months_missing"])] or r["cell_pattern_agreement_with_2016_min"] < .97:
            raise SystemExit(f"{name}: unexpected coverage of the 2011-2015 store: {r}")

    c, norm, obs = load_real(ROOT, years=YEARS)
    n_pre = int(np.sum(c.year <= 2015))
    if not np.all(c.year[:n_pre] <= 2015) or not np.all(c.year[n_pre:] >= 2016):
        raise SystemExit("the 2011-2015 rows must come first, then the 2016-2024 rows")
    year = 2000 + months // 12
    train = (year >= REAL_SPLITS["train"][0]) & (year <= REAL_SPLITS["train"][1])
    z, climate, mean, spread = surface_z(fields, months, train)
    print(f"surface z: mean {np.round(mean, 4).tolist()}, spread {np.round(spread, 4).tolist()} "
          f"({time.time() - started:.0f}s)", flush=True)
    features = position_features(c).astype(np.float32)
    features[:, 44:51] = profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon,
                                                 c.month_index)
    lagged_all = np.stack([profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon,
                                                   c.month_index, lag=lead) for lead in LEADS], 1).astype(np.float32)
    # the registered convention for the 2016-2024 rows: grids from 2016-01 on only
    reg = months >= 12 * 16
    lagged_reg = np.stack([profile_surface_columns(z[reg], fields[reg, 0], months[reg], grid_lat, grid_lon, c.lat,
                                                   c.lon, c.month_index, lag=lead) for lead in LEADS], 1).astype(np.float32)
    lagged = np.where((np.arange(len(c.lat)) < n_pre)[:, None, None], lagged_all, lagged_reg)
    rows_kept_zero = int(np.sum(np.any(lagged_all[n_pre:] != lagged_reg[n_pre:], axis=(1, 2))))

    # 3. every 2016-2024 row and month equals the registered caches
    with np.load(REGISTERED / "surface_grids.npz") as g, np.load(REGISTERED / "satellites.npz") as s, \
            np.load(REGISTERED / "lagged_surface.npz") as l:
        reg_months, reg_grids = g["grid_month"], g["grids"]
        reg_meta = json.loads(str(s["metadata"].item()))
        if not np.array_equal(months[len(new_months):], reg_months):
            raise SystemExit("the 2016-2024 months differ from the registered grids")
        if s["features"].shape[0] != len(c.lat) - n_pre:
            raise SystemExit("the registered caches hold another number of 2016-2024 rows")
        for k in ("month_index", "lat", "lon"):
            if not np.array_equal(s[k], getattr(c, k)[n_pre:]):
                raise SystemExit(f"the registered {k} differ from the 2016-2024 rows of the cohort")
        checks = {"grids": bool(np.array_equal(z[len(new_months):].astype(np.float16), reg_grids, equal_nan=True)),
                  "surface_climatology": bool(np.array_equal(climate.astype(np.float32), g["surface_climatology"], equal_nan=True)),
                  "surface_mean": bool(np.array_equal(mean, g["surface_mean"])),
                  "surface_std": bool(np.array_equal(spread, g["surface_std"])),
                  "features": bool(np.array_equal(features[n_pre:], s["features"])),
                  "lagged": bool(np.array_equal(lagged[n_pre:], l["features"])),
                  "rows_compared": int(len(c.lat) - n_pre), "months_compared": int(len(reg_months)),
                  "rows_added_2011_2015": n_pre, "months_added": int(len(new_months)),
                  "lagged_2016_rows_kept_at_registered_zero": rows_kept_zero,
                  "registered_cohort_fingerprint": reg_meta["cohort_fingerprint"],
                  "registered_sha256": {name: hashlib.sha256((REGISTERED / f"{name}.npz").read_bytes()).hexdigest()
                                        for name in ("surface_grids", "satellites", "lagged_surface")}}
    print(json.dumps({"registered_rows_equal": checks}), flush=True)
    if not all(checks[k] for k in ("grids", "surface_climatology", "surface_mean", "surface_std", "features", "lagged")):
        raise SystemExit(f"the new caches do not reproduce the registered 2016-2024 inputs: {checks}")

    # 4. the satellite SST at the 2011-2015 profiles against their measured 5 m temperature
    with xr.open_dataset(cohort_path(str(ROOT), "global")) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        raw_year = np.asarray(raw.year.values)[order]
        rows = order[(raw_year >= YEARS[0]) & (raw_year <= YEARS[1])]
        if not np.array_equal(np.asarray(raw.lat.values)[rows], c.lat):
            raise SystemExit("the raw cohort rows differ from the loaded cohort")
        measured = np.asarray(raw.TEMP.values[rows[:n_pre], 0])
    sst = features[:n_pre, 50] * 11. + 14.
    ok = (features[:n_pre, 45] > 0) & np.isfinite(measured)
    corr = float(np.corrcoef(sst[ok], measured[ok])[0, 1])
    bias = float(np.mean(sst[ok] - measured[ok]))
    available = {name: float(features[:n_pre, col].mean()) for name, col in zip(VARIABLES, (45, 47, 49))}
    sst_check = {"r": corr, "mean_difference_C": bias, "profiles": int(ok.sum()), "available_2011_2015": available}
    print(json.dumps({"sst_vs_argo_5m_2011_2015": sst_check}), flush=True)
    if corr < .98 or abs(bias) > .5:
        raise SystemExit("the 2011-2015 satellite SST does not match the measured 5 m temperature; check the grids")

    args.output.mkdir(parents=True, exist_ok=True)
    common = {"cohort_fingerprint": cohort_fingerprint(c, obs), "cohort": f"real_global {YEARS[0]}-{YEARS[1]}",
              "splits": {k: list(v) for k, v in REAL_SPLITS.items()},
              "normalization_scope": "training years 2016-2020 only (the paper's); the 10-year training window is a "
                                     "runner option (--real-train-split 2011 2020)",
              "surface_variables": list(VARIABLES),
              "source": [str(NEW_STORE.relative_to(ROOT)) + " (2011-2015)", str(OLD_STORE.relative_to(ROOT)) + " (2016-2023)",
                         str(STORE_2024.relative_to(ROOT)) + " (2024)"],
              "store_provenance": [new_prov, old_prov, prov24], "coverage_2011_2015": coverage,
              "registered": str(REGISTERED), "registered_rows_equal": checks,
              "lagged_surface": "2016-2024 rows as registered (zeros for a source month before 2016-01); "
                                "2011-2015 rows from the 2011-2015 grids (zeros before 2011-01)",
              "sst_vs_argo_5m_2011_2015": sst_check,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(args.output / "surface_grids.npz", grids=z.astype(np.float16), grid_month=months, grid_lat=grid_lat,
         grid_lon=grid_lon, surface_climatology=climate.astype(np.float32), surface_mean=mean, surface_std=spread,
         metadata=json.dumps({**common, "units": "z-score of the anomaly against the 2016-2020 monthly climatology"}))
    save(args.output / "satellites.npz", features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
         metadata=json.dumps({**common, "feature_fingerprint": array_fingerprint((features,))}))
    save(args.output / "lagged_surface.npz", features=lagged, leads=np.array(LEADS), month_index=c.month_index,
         lat=c.lat, lon=c.lon, metadata=json.dumps(common))
    print(json.dumps({"output": str(args.output), "profiles": int(len(c.lat)), "months": int(len(months)),
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
