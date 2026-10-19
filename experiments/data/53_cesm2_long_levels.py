"""CESM2-LE TEMP/SALT at the synthetic cohort's 20 depths, 1964-01 ... 2005-12.

Reads the raw 1-degree files from ``52_cesm2_le_download.py`` and writes the
months a 40-year training window ending in 2003 needs, plus the 2004 validation
and 2005 test years, at the 20 levels of ``data/synthetic_argo/cesm2_uniform.nc``
(5-985 m, all native levels):

    data/cesm2_long/cesm2_le_cohort_levels.zarr
        TEMP, SALT  (month, level, lat, lon) float32, as stored in the source
        month_index (month,) months since 2000-01, negative before 2000
        MASK, lat, lon, level

Values are the source's, cast to float32 exactly as
``experiments/data/standardize.py`` casts them. Before anything is written the
script checks that its 2000-2005 months equal ``data/cesm2_le_full_standard.zarr``
(the store every CESM2 result so far was built from) value for value.

  .venv/bin/python experiments/data/53_cesm2_long_levels.py      # ~10 min, one process
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/cesm2_le_raw"
REFERENCE = ROOT / "data/cesm2_le_full_standard.zarr"
COHORT = ROOT / "data/synthetic_argo/cesm2_uniform.nc"
NAME = "data_CESM_CESM2-LE_regrid_1x1deg_pop.h.{}.185001-210012.nc"


def month_numbers(times):
    """Months since 2000-01 of the source's (cftime) timestamps."""
    return np.array([(t.year - 2000) * 12 + t.month - 1 for t in times], dtype=int)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first-year", type=int, default=1964)
    p.add_argument("--last-year", type=int, default=2005)
    p.add_argument("--output", type=Path, default=ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr")
    args = p.parse_args()
    started = time.time()
    with xr.open_dataset(COHORT) as cohort:
        levels = np.asarray(cohort["level"].values, dtype=float)
    sources = {v: xr.open_dataset(RAW / NAME.format(v), use_cftime=True) for v in ("TEMP", "SALT")}
    temp = sources["TEMP"]
    depth = np.asarray(temp.z_t.values, dtype=float)
    di = [int(np.argmin(np.abs(depth - d))) for d in levels]
    if not np.allclose(depth[di], levels, atol=0.2):
        raise SystemExit(f"cohort depths are not levels of the source: {depth[di]}")
    months = month_numbers(temp.time.values)
    if not np.array_equal(months, month_numbers(sources["SALT"].time.values)):
        raise SystemExit("TEMP and SALT have different time axes")
    first, last = (args.first_year - 2000) * 12, (args.last_year - 2000) * 12 + 11
    keep = np.flatnonzero((months >= first) & (months <= last))
    if len(keep) != last - first + 1 or not np.array_equal(months[keep], np.arange(first, last + 1)):
        raise SystemExit("the source does not hold every wanted month exactly once")
    lat = temp.LAT.values[:, 0].astype("float32")
    lon = temp.LON.values[0, :].astype("float32")
    shape = (len(keep), len(di), len(lat), len(lon))
    data = {v: np.empty(shape, dtype="float32") for v in sources}
    for start in range(0, len(keep), 12):
        rows = keep[start:start + 12]
        for v, ds in sources.items():
            block = ds[v].isel(time=rows, z_t=di).values
            data[v][start:start + len(rows)] = block.astype("float32")
        print(f"read {args.first_year + start // 12} ({time.time() - started:.0f}s)", flush=True)
    mask = temp.MASK.values.astype("float32")
    for ds in sources.values():
        ds.close()

    # the 2000-2005 months must be the existing store's, value for value
    with xr.open_zarr(REFERENCE, chunks=None) as ref:
        rdi = [int(np.argmin(np.abs(ref.depth.values - d))) for d in levels]
        at = np.searchsorted(np.arange(first, last + 1), np.arange(72))
        for v in sources:
            expected = np.asarray(ref[v].isel(depth=rdi).values, dtype="float32")
            if not np.array_equal(data[v][at], expected, equal_nan=True):
                bad = ~((data[v][at] == expected) | (np.isnan(data[v][at]) & np.isnan(expected)))
                raise SystemExit(f"{v}: {int(bad.sum())} values of 2000-2005 differ from {REFERENCE.name}")
        if not (np.array_equal(ref.lat.values, lat) and np.array_equal(ref.lon.values, lon)):
            raise SystemExit("grid differs from the existing store")
    print("2000-2005 equal to the existing store, value for value", flush=True)

    out = xr.Dataset(
        {v: (("month", "level", "lat", "lon"), a) for v, a in data.items()}
        | {"MASK": (("lat", "lon"), mask)},
        coords={"month_index": ("month", np.arange(first, last + 1)), "level": levels,
                "lat": lat, "lon": lon},
        attrs={"source": f"{RAW.relative_to(ROOT)}/{NAME.format('{TEMP,SALT}')} "
                         "(SingleBicycle/cesm2-le-ocean-sample)",
               "months": f"{args.first_year}-01 to {args.last_year}-12",
               "month_index": "months since 2000-01",
               "levels": "the 20 depths of data/synthetic_argo/cesm2_uniform.nc",
               "check": "2000-2005 equal to data/cesm2_le_full_standard.zarr value for value",
               "manifest": json.dumps(json.loads((RAW / "manifest.json").read_text())["files"])})
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    out.to_zarr(temporary, mode="w", encoding={v: {"chunks": (12, len(di), len(lat), len(lon))}
                                               for v in data})
    if args.output.exists():
        shutil.rmtree(args.output)
    os.replace(temporary, args.output)
    print(f"wrote {args.output} {shape} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
