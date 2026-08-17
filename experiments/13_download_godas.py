"""GODAS regional subset download (mentor doc §4).

Source / acknowledgement: https://psl.noaa.gov/data/gridded/data.godas.html

Downloads the Gulf-Stream box used by the mentor's experiment — pottmp, salt
and sshg for 2000..2025 — and writes one NetCDF per (variable, year) plus a
manifest recording every file's size and SHA-256.

Subsetting is done **server-side over OPeNDAP**, so each file is a few hundred
kB rather than the ~100 MB global original.  That matters here: this host has
~14 GB free, and pulling the full globe would be ~5 GB of traffic and disk for
data that is immediately discarded.

Region (doc §4): 25–50 N, 280–330 E, every second depth level through 1000 m.

  Latitude  25–50 N on GODAS's 1/3-degree grid -> 75 cells -> stride 2 -> 38.
  Longitude on the 1-degree grid has cell centres at .5, so a literal
  280–330 E gives 50 cells -> stride 2 -> 25, but the doc states a 38 x 26
  grid.  26 requires including the 330.5 centre, i.e. slice(280, 331), which
  yields 51 cells.  This script matches the doc's stated GRID (38 x 26), because
  that is the shape the experiment actually runs on, and records the choice in
  the manifest as `lon_interval_note`.  Flag it if the intent was 25 columns.

  Depth: every second level, keeping those <= 1000 m -> 16 levels, 5–949 m,
  which reproduces the doc's stated range exactly.

Writes are atomic (tmp + os.replace) and cached files are validated before
being skipped, so an interrupted run resumes without corrupting anything.

Parallelism uses PROCESSES, not threads: netCDF4/HDF5 is not thread-safe and a
thread pool calling ``to_netcdf`` concurrently segfaults the interpreter
(observed here as "dumped core" before this was changed).

  .venv/bin/python experiments/13_download_godas.py --start-year 2000 --end-year 2025
"""
import sys, os, json, time, hashlib, argparse, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas"
VARS = ("pottmp", "salt", "sshg")
LAT = (25.0, 50.0)
LON = (280.0, 331.0)          # half-open in effect; see the module docstring
MAX_DEPTH_M = 1000.0
LEVEL_STRIDE = 2
SPACE_STRIDE = 2              # applied by the experiment, not stored here

ap = argparse.ArgumentParser()
ap.add_argument("--start-year", type=int, default=2000)
ap.add_argument("--end-year", type=int, default=2025)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default=None, help="default: <repo>/data/godas_gulfstream")
ap.add_argument("--smoke", action="store_true",
                help="one year only. NOT usable as training input — the "
                     "training split needs the full year range.")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = args.out or os.path.join(ROOT, "data", "godas_gulfstream")
os.makedirs(OUT, exist_ok=True)
years = ([args.start_year] if args.smoke
         else list(range(args.start_year, args.end_year + 1)))
print(f"godas download -> {OUT}\n  vars={VARS} years={years[0]}..{years[-1]} "
      f"region lat{LAT} lon{LON} workers={args.workers}", flush=True)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_reserved(ds: xr.Dataset, src: str) -> xr.Dataset:
    """Remove attributes netCDF4 refuses to write, or that the subset invalidates.

    ``_NCProperties`` is reserved: the library writes it itself and raises
    "NetCDF: String match to name in use" if it is set explicitly.  Any other
    leading-underscore attribute is reserved by the same convention.
    ``_ChunkSizes`` additionally still describes the ORIGINAL 418x360 global
    grid, which is meaningless once the region is cut out.  The OPeNDAP
    ``DODS_EXTRA.*`` keys are transport artefacts, not data provenance.
    """
    drop = lambda a: {k: v for k, v in a.items()
                      if not k.startswith("_") and not k.startswith("DODS_EXTRA")}
    ds.attrs = drop(ds.attrs)
    ds.attrs["subset_source"] = src
    ds.attrs["subset_by"] = "experiments/13_download_godas.py (mentor doc §4)"
    for name in ds.variables:
        ds[name].attrs = drop(ds[name].attrs)
        ds[name].encoding.pop("chunksizes", None)
    return ds


def subset(var: str, year: int) -> xr.Dataset:
    """Server-side region + level subset for one (variable, year)."""
    src = f"{BASE}/{var}.{year}.nc"
    ds = xr.open_dataset(src, decode_times=True)
    sub = ds.sel(lat=slice(*LAT), lon=slice(*LON))
    if "level" in sub.dims:
        lv = sub.level.values[::LEVEL_STRIDE]
        sub = sub.sel(level=lv[lv <= MAX_DEPTH_M])
    return _strip_reserved(sub.load(), src)


def valid_cached(path: str, var: str) -> bool:
    """A cached file counts only if it opens and carries the variable."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with xr.open_dataset(path) as d:
            return var in d and d.sizes.get("time", 0) == 12
    except Exception:
        return False


def fetch(var: str, year: int) -> dict:
    path = os.path.join(OUT, f"{var}.{year}.nc")
    if valid_cached(path, var):
        return dict(var=var, year=year, file=os.path.basename(path),
                    bytes=os.path.getsize(path), sha256=sha256(path),
                    cached=True)
    ds = subset(var, year)
    fd, tmp = tempfile.mkstemp(dir=OUT, suffix=".tmp")
    os.close(fd)
    try:
        ds.to_netcdf(tmp)
        ds.close()
        os.replace(tmp, path)                      # atomic
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dict(var=var, year=year, file=os.path.basename(path),
                bytes=os.path.getsize(path), sha256=sha256(path), cached=False)


def main() -> None:
    jobs = [(v, y) for v in VARS for y in years]
    files, t0, done = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, v, y): (v, y) for v, y in jobs}
        for f in as_completed(futs):
            v, y = futs[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  FAILED {v}.{y}: {type(e).__name__}: {e}", flush=True)
                continue
            files.append(rec)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files  ({time.time()-t0:.0f}s)",
                      flush=True)

    if not files:
        raise SystemExit("no files downloaded")
    files.sort(key=lambda r: (r["var"], r["year"]))
    probe = xr.open_dataset(os.path.join(OUT, files[0]["file"]))
    grid = dict(nlat=int(probe.sizes["lat"]), nlon=int(probe.sizes["lon"]),
                nlevel=int(probe.sizes.get("level", 0)),
                lat_min=float(probe.lat.min()), lat_max=float(probe.lat.max()),
                lon_min=float(probe.lon.min()), lon_max=float(probe.lon.max()),
                levels_m=([float(x) for x in probe.level.values]
                          if "level" in probe else []))
    probe.close()

    manifest = {
        "source": "NOAA PSL GODAS monthly (https://psl.noaa.gov/data/gridded/data.godas.html)",
        "acknowledgement": "NOAA PSL, Boulder, Colorado, USA",
        "role": "mentor dfs_d4rt_intern_plan.md \u00a74 regional subset",
        "access": "OPeNDAP server-side subset via " + BASE,
        "region": {"lat": list(LAT), "lon": list(LON)},
        "lon_interval_note": (
            "slice(280, 331) on 1-degree centres at .5 gives 51 raw cells -> the "
            "doc's 38 x 26 experiment grid after stride 2. A literal 280-330 gives "
            "50 raw -> 25 columns, one fewer than the doc states. Matched the "
            "stated GRID, not the stated interval; flag if 25 was intended."),
        "years": [years[0], years[-1]],
        "variables": list(VARS),
        "level_selection": f"every {LEVEL_STRIDE}nd level with depth <= {MAX_DEPTH_M} m",
        "experiment_space_stride": SPACE_STRIDE,
        "experiment_grid": [-(-grid["nlat"] // SPACE_STRIDE),
                            -(-grid["nlon"] // SPACE_STRIDE)],
        "grid": grid,
        "file_count": len(files),
        "total_bytes": sum(r["bytes"] for r in files),
        "files": files,
    }
    mpath = os.path.join(OUT, "manifest.json")
    with open(mpath + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(mpath + ".tmp", mpath)

    print(f"\n{len(files)} files, {manifest['total_bytes']:,} bytes")
    # len(range(n)[::k]) is ceil(n/k), NOT n//k — 75 and 51 stride 2 give
    # 38 and 26, which is the doc's stated grid.
    strided = lambda n: -(-n // SPACE_STRIDE)
    print(f"grid: lat {grid['nlat']} lon {grid['nlon']} level {grid['nlevel']} "
          f"-> stride {SPACE_STRIDE} gives {strided(grid['nlat'])} x "
          f"{strided(grid['nlon'])}")
    print(f"levels: {np.round(grid['levels_m'], 0)}")
    print(f"manifest: {mpath}")
    print(f"manifest sha256: {sha256(mpath)}")
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
