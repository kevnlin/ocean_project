"""Real satellite inputs of the gridded model, built as the CESM2 caches were.

From ``data/real_obs_1deg.zarr`` (monthly OISST, OISSS L4 and CMEMS DUACS SLA on the 1-degree
grid, 2016-2023) and the 2016-2023 real cohort (``ocean_tokenizer.real_global``):

  surface_grids.npz   [96, 3, 180, 360] SST, SSS, SLA z-scores (anomaly against the 2016-2020
                      monthly climatology, divided by the 2016-2020 spread; ``100``'s layout)
  satellites.npz      [P, 56] profile features: 44 position/season fields, then SST/SSS/SLA
                      z-scores with flags and (SST - 14) / 11 from the profile's own month
                      (``88``'s layout)
  lagged_surface.npz  [P, 3, 7] the same seven columns from the month ``lead`` months earlier,
                      for leads 1, 3, 6 (``93``'s layout); zeros before 2016-01

Check before writing: the sampled SST at every profile against its measured 5 m temperature
(a correlation below 0.98 or a mean difference above 0.5 degC stops the script).

  .venv/bin/python experiments/real_data/105_real_surface.py
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
from ocean_tokenizer.real_global import (REAL_SPLITS, YEARS, cohort_fingerprint, load_real,
                                         profile_surface_columns, surface_z)
from ocean_tokenizer.synthetic_latent_experiment import position_features
from ocean_tokenizer.latent_experiment import array_fingerprint

STORE = ROOT / "data/real_obs_1deg.zarr"
OUTPUT = ROOT / "outputs/real_global_20261019"
LEADS = (1, 3, 6)


def save(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    started = time.time()
    c, norm, obs = load_real(ROOT)
    fingerprint = cohort_fingerprint(c, obs)
    with xr.open_zarr(STORE) as ds:
        time_index = ds.time.values.astype("datetime64[M]")
        months = ((time_index.astype(int) // 12 + 1970 - 2000) * 12 + time_index.astype(int) % 12).astype(int)
        fields = np.stack([ds[v].values for v in ("SST", "SSS", "SLA")], 1).astype(np.float64)
        grid_lat, grid_lon = ds.lat.values.astype(np.float64), ds.lon.values.astype(np.float64)
        provenance = ds.attrs.get("provenance")
    if not (np.all(np.diff(months) == 1) and months[0] == 12 * (YEARS[0] - 2000) and months[-1] == 12 * (YEARS[1] - 2000) + 11):
        raise SystemExit("the satellite store must hold every month of 2016-2023")
    year = 2000 + months // 12
    train = (year >= REAL_SPLITS["train"][0]) & (year <= REAL_SPLITS["train"][1])
    z, climate, mean, spread = surface_z(fields, months, train)
    print(f"surface z: mean {np.round(mean, 4).tolist()}, spread {np.round(spread, 4).tolist()} "
          f"({time.time() - started:.0f}s)", flush=True)

    columns = profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon, c.month_index)
    features = position_features(c).astype(np.float32)
    features[:, 44:51] = columns
    lagged = np.stack([profile_surface_columns(z, fields[:, 0], months, grid_lat, grid_lon, c.lat, c.lon,
                                               c.month_index, lag=lead) for lead in LEADS], 1).astype(np.float32)

    # the satellite SST at each profile against the profile's own measured 5 m temperature
    with xr.open_dataset(cohort_path(str(ROOT), "global")) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        raw_year = np.asarray(raw.year.values)[order]
        keep = (raw_year >= YEARS[0]) & (raw_year <= YEARS[1])
        measured = np.asarray(raw.TEMP.values[:, 0])[order][keep]
        if not np.array_equal(np.asarray(raw.lat.values)[order][keep], c.lat):
            raise SystemExit("the raw cohort rows differ from the loaded cohort")
    sst = features[:, 50] * 11. + 14.
    ok = (features[:, 45] > 0) & np.isfinite(measured)
    corr = float(np.corrcoef(sst[ok], measured[ok])[0, 1])
    bias = float(np.mean(sst[ok] - measured[ok]))
    print(f"satellite SST vs Argo 5 m: r = {corr:.4f}, mean difference {bias:+.3f} degC, "
          f"{ok.sum()} profiles; SST/SSS/SLA available at {np.round(features[:, 45:50:2].mean(0), 3).tolist()} of profiles",
          flush=True)
    if corr < .98 or abs(bias) > .5:
        raise SystemExit("satellite SST does not match the measured 5 m temperature; check the grids")

    args.output.mkdir(parents=True, exist_ok=True)
    common = {"cohort_fingerprint": fingerprint, "cohort": "real_global 2016-2023",
              "splits": {k: list(v) for k, v in REAL_SPLITS.items()}, "normalization_scope": "training years only",
              "surface_variables": ["SST", "SSS", "SLA"], "source": str(STORE.relative_to(ROOT)),
              "store_provenance": provenance, "sst_vs_argo_5m": {"r": corr, "mean_difference_C": bias},
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
