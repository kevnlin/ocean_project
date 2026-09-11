"""EN4 and ECCO — the external gridded reference products for plan P0/P4.

Both were already *cited* in this repo (`oi.py`, `reports/synthetic/oi_baseline.md`) as the
operational ancestors of our OI baseline.  Neither had ever been downloaded or
scored.  A citation is not a baseline; this script fetches the actual products so
the P0 table can carry a real EN4 and ECCO row.

EN4  (UK Met Office Hadley Centre, EN.4.2.2 objective analyses, g10 correction)
    Monthly 1 deg x 1 deg, 42 levels, 1900-present.  This is the closest thing to
    a like-for-like competitor: EN4 is *itself* an optimal-interpolation analysis
    of the same in-situ profiles we now hold in `data/argo/`, so it answers "does
    the learned method beat the operational OI product that ingests the same
    observations?" rather than merely "does it beat our own OI re-implementation?"

    Bias correction: **g10** (Gouretski & Reseghetti 2010) — the variant most
    used in the literature.  The choice is recorded in the manifest because
    c13/c14/l09 give materially different XBT-era numbers; ours is post-2000
    Argo-dominated, so the correction matters less here, but the run must still
    say which one it used.

ECCO (NASA/JPL ECCO Central Estimate V4r4, PO.DAAC)
    Monthly 0.5 deg, 50 levels, **1992-2017 only**.  That end date is the "valid
    overlap period" the plan's P0 refers to, and it is a hard constraint: ECCO
    V4r4 cannot be scored on the GODAS holdout year (2025) at all.  The honest
    treatment is to report ECCO on the overlap it has and to say plainly that it
    is absent from the holdout row -- not to quietly swap in a different ECCO
    release with different physics.

    Needs a free NASA Earthdata login.  `earthaccess` reads ~/.netrc.

Both are cut to the P1 region boxes on the way in.  Regridding onto the GODAS
analysis grid is deliberately NOT done here: it is a modelling decision that
belongs in the baseline adapter, where it can be tested, not buried in a
download script.

Output
------
``data/reference/en4/EN.4.2.2.g10.<region>.<year>.nc``
``data/reference/ecco/ECCO_V4r4.<region>.<year>.nc``
``data/reference/manifest.json``

Run:
    .venv/bin/python experiments/data/43_download_reference_products.py --products en4
    .venv/bin/python experiments/data/43_download_reference_products.py --products ecco
    .venv/bin/python experiments/data/43_download_reference_products.py --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import zipfile
# PROCESSES, not threads.  netCDF4/HDF5 is not thread-safe and a thread pool
# calling to_netcdf concurrently segfaults the interpreter -- the same failure
# 13_download_godas.py documents, reproduced here before this was changed.
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import requests
import xarray as xr

EN4_ROOT = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1"
EN4_FILE = "EN.4.2.2.analyses.{corr}.{year}.zip"
#: The Met Office changed the layout mid-series: years <= 2021 sit in an
#: EN.4.2.2/ subdirectory, 2022 onward sit flat in en4-2-1/ (2021 exists at
#: both).  Verified by HEAD against 2019-2026.  Trying both is not defensive
#: padding -- assuming either one silently loses the 2022-2025 years, which is
#: exactly the development and holdout era.
EN4_BASE = EN4_ROOT + "/EN.4.2.2"
ECCO_SHORT = "ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4"
ECCO_LAST_YEAR = 2017          # hard end of V4r4; see module docstring

REGIONS = {
    "gulfstream": dict(lat=(25.0, 50.0), lon=(280.0, 331.0)),
    "npac_gyre":  dict(lat=(20.0, 45.0), lon=(180.0, 231.0)),
    # Whole ocean: EN4 is the gridded reference for the GLOBAL reconstruction
    # map. It is a pure in-situ objective analysis on the same 1 deg grid we
    # analyse on, so no horizontal regridding is needed -- only a depth
    # interpolation onto the 20 protocol levels.
    "global":     dict(lat=(-90.0, 90.0), lon=(0.0, 360.0)),
}
MAX_DEPTH_M = 1000.0           # matches the GODAS subset; nothing below is used

ap = argparse.ArgumentParser()
ap.add_argument("--products", nargs="+", default=["en4", "ecco"],
                choices=["en4", "ecco"])
ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
ap.add_argument("--start-year", type=int, default=2000)
ap.add_argument("--end-year", type=int, default=2025)
ap.add_argument("--en4-correction", default="g10", choices=["g10", "c13", "c14", "l09"])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default=None)
ap.add_argument("--max-depth", type=float, default=MAX_DEPTH_M,
                help="deepest level to keep. The 700-1400 m band needs ~1450.")
ap.add_argument("--suffix", default="",
                help="appended to the region label, e.g. '_deep', so a deeper "
                     "cut lands in its own files and the existing 20-level "
                     "artifacts keep referring to unchanged data")
ap.add_argument("--smoke", action="store_true", help="one year only")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = args.out or os.path.join(ROOT, "data", "reference")
os.makedirs(OUT, exist_ok=True)
years = [args.start_year] if args.smoke else list(range(args.start_year, args.end_year + 1))
t0 = time.time()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(ds: xr.Dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp"); os.close(fd)
    try:
        for name in ds.variables:
            ds[name].encoding.pop("chunksizes", None)
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cut(ds: xr.Dataset, region: str, latname="lat", lonname="lon") -> xr.Dataset:
    """Cut to a region box in the 0-360 convention, and to <= MAX_DEPTH_M."""
    box = REGIONS[region]
    lon = ds[lonname]
    ds = ds.assign_coords({lonname: lon % 360.0}).sortby(lonname)
    ds = ds.sel({latname: slice(*box["lat"]), lonname: slice(*box["lon"])})
    for dn in ("depth", "Z", "z", "lev"):
        if dn in ds.dims:
            d = np.abs(ds[dn].values)
            ds = ds.isel({dn: np.flatnonzero(d <= args.max_depth)})
            break
    return ds


# ---------------------------------------------------------------------- EN4
def fetch_en4(year: int) -> list[dict]:
    """One EN4 year: download the zip, cut every month, keep no zip on disk.

    The zips are ~314 MB each and hold 12 global monthly files we immediately
    reduce to two small boxes, so streaming through memory beats writing 8 GB of
    archives we would delete anyway.
    """
    out = []
    want = {r: os.path.join(OUT, "en4",
                            f"EN.4.2.2.{args.en4_correction}.{r}{args.suffix}.{year}.nc")
            for r in args.regions}
    if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in want.values()):
        return [dict(product="en4", region=r, year=year, file=os.path.relpath(p, OUT),
                     bytes=os.path.getsize(p), sha256=sha256(p), cached=True)
                for r, p in want.items()]
    fname = EN4_FILE.format(corr=args.en4_correction, year=year)
    errs = []
    for url in (f"{EN4_ROOT}/EN.4.2.2/{fname}", f"{EN4_ROOT}/{fname}"):
        r = requests.get(url, timeout=1800)
        if r.status_code == 200:
            break
        errs.append(f"{r.status_code} {url}")
    else:
        raise requests.HTTPError("; ".join(errs))
    per_region: dict[str, list] = {k: [] for k in args.regions}
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".nc")):
            with z.open(name) as fh:
                ds = xr.open_dataset(io.BytesIO(fh.read()), decode_times=True)
            keep = [v for v in ("temperature", "salinity",
                                "temperature_uncertainty", "salinity_uncertainty")
                    if v in ds]
            ds = ds[keep]
            for region in args.regions:
                per_region[region].append(_cut(ds, region).load())
            ds.close()
    for region, months in per_region.items():
        ds = xr.concat(months, dim="time").sortby("time")
        ds.attrs.update(source="EN.4.2.2 objective analyses, UK Met Office Hadley Centre",
                        bias_correction=args.en4_correction, region=region,
                        subset_by="experiments/data/43_download_reference_products.py",
                        acknowledgement=("Good, Martin & Rayner (2013), JGR Oceans; "
                                         "EN4 data (c) British Crown Copyright, Met "
                                         "Office, provided under a Non-Commercial "
                                         "Government Licence"))
        _write_atomic(ds, want[region])
        p = want[region]
        out.append(dict(product="en4", region=region + args.suffix, year=year,
                        file=os.path.relpath(p, OUT), bytes=os.path.getsize(p),
                        sha256=sha256(p), cached=False))
    return out


# --------------------------------------------------------------------- ECCO
def fetch_ecco(years_: list[int]) -> list[dict]:
    import earthaccess
    earthaccess.login(strategy="netrc")
    out = []
    for year in years_:
        if year > ECCO_LAST_YEAR:
            continue
        want = {r: os.path.join(OUT, "ecco", f"ECCO_V4r4.{r}{args.suffix}.{year}.nc")
                for r in args.regions}
        if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in want.values()):
            out += [dict(product="ecco", region=r, year=year,
                         file=os.path.relpath(p, OUT), bytes=os.path.getsize(p),
                         sha256=sha256(p), cached=True) for r, p in want.items()]
            continue
        res = earthaccess.search_data(short_name=ECCO_SHORT,
                                      temporal=(f"{year}-01-01", f"{year}-12-31"))
        if not res:
            print(f"  ecco {year}: no granules", flush=True)
            continue
        with tempfile.TemporaryDirectory() as td:
            paths = earthaccess.download(res, td)
            ds = xr.open_mfdataset(sorted(paths), combine="by_coords")
            keep = [v for v in ("THETA", "SALT") if v in ds]
            ds = ds[keep]
            for region in args.regions:
                sub = _cut(ds, region, latname="latitude", lonname="longitude").load()
                sub.attrs.update(source=f"NASA/JPL ECCO Central Estimate V4r4 ({ECCO_SHORT})",
                                 region=region,
                                 subset_by="experiments/data/43_download_reference_products.py",
                                 note=f"V4r4 ends {ECCO_LAST_YEAR}; absent from later rows",
                                 acknowledgement="ECCO Consortium, Fukumori et al., PO.DAAC")
                _write_atomic(sub, want[region])
                p = want[region]
                out.append(dict(product="ecco", region=region + args.suffix, year=year,
                                file=os.path.relpath(p, OUT), bytes=os.path.getsize(p),
                                sha256=sha256(p), cached=False))
            ds.close()
        print(f"  ecco {year}: done ({time.time()-t0:.0f}s)", flush=True)
    return out


records, failures = [], []
if "en4" in args.products:
    print(f"EN4 {years[0]}..{years[-1]} corr={args.en4_correction} "
          f"regions={args.regions}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_en4, y): y for y in years}
        for i, f in enumerate(as_completed(futs), 1):
            y = futs[f]
            try:
                records += f.result()
            except Exception as e:
                failures.append(dict(product="en4", year=y, error=f"{type(e).__name__}: {e}"))
                print(f"  FAILED en4 {y}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"  en4 {i}/{len(years)} years ({time.time()-t0:.0f}s)", flush=True)

if "ecco" in args.products:
    skipped = [y for y in years if y > ECCO_LAST_YEAR]
    print(f"ECCO {years[0]}..{min(years[-1], ECCO_LAST_YEAR)} "
          f"(V4r4 ends {ECCO_LAST_YEAR}; skipping {len(skipped)} later years)",
          flush=True)
    try:
        records += fetch_ecco(years)
    except Exception as e:
        failures.append(dict(product="ecco", error=f"{type(e).__name__}: {e}"))
        print(f"  FAILED ecco: {type(e).__name__}: {e}", flush=True)

mpath = os.path.join(OUT, "manifest.json")
prev = json.load(open(mpath)) if os.path.exists(mpath) else {"files": []}
bykey = {(r["product"], r["region"], r["year"]): r for r in prev.get("files", [])}
bykey.update({(r["product"], r["region"], r["year"]): r for r in records})
manifest = {
    "role": "external gridded reference products for plan P0/P4",
    "en4": {"source": EN4_ROOT, "version": "EN.4.2.2",
            "bias_correction": args.en4_correction,
            "licence": "Non-Commercial Government Licence; (c) British Crown Copyright, Met Office"},
    "ecco": {"short_name": ECCO_SHORT, "version": "V4r4",
             "last_year": ECCO_LAST_YEAR,
             "limitation": "V4r4 ends 2017 — cannot be scored on the 2025 GODAS holdout"},
    "regions": {k: REGIONS[k] for k in args.regions},
    "max_depth_m": args.max_depth,
    "period": [args.start_year, args.end_year],
    "failures": failures,
    "file_count": len(bykey),
    "total_bytes": sum(r["bytes"] for r in bykey.values()),
    "files": sorted(bykey.values(), key=lambda r: (r["product"], r["region"], r["year"])),
}
with open(mpath + ".tmp", "w") as f:
    json.dump(manifest, f, indent=1)
os.replace(mpath + ".tmp", mpath)
print(f"\n{len(bykey)} files, {manifest['total_bytes']:,} bytes")
if failures:
    print(f"{len(failures)} FAILURES — rerun to resume")
print(f"manifest: {mpath}\ntotal {time.time()-t0:.0f}s")
