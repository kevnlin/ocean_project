"""Real Argo profile download — the float cohort the cross-check plan assumes.

Why this exists
---------------
`parallel_replication_crosscheck_plan.md` defines its primary inference unit as
**WMO/platform** (P7), clusters every confidence interval by WMO (P0), sizes the
layout experiment in "whole Argo profiles" (P1), and asks P2 to separate
*same-provenance* duplicates from *independent-provenance* observations.  None
of that is expressible against what the pipeline called a "profile" until now: a
uniformly random **column of the GODAS reanalysis** (`godas_obs.build_sample`),
which has no float identity, no trajectory, no instrument error and no QC flag.

This script downloads the real thing from the Argo GDAC so those quantities stop
being placeholders.

What it fetches
---------------
1. ``ar_index_global_prof.txt.gz`` — the GDAC's global profile directory
   (~3.4 M rows: file path, date, lat, lon, ocean, profiler type, institution).
   Used to decide *which floats to pull* without touching a single data file.
2. One ``<wmo>_prof.nc`` per float intersecting a requested region and period.
   The per-float multiprofile file is the right granularity: ~2 MB for a whole
   float's life, versus thousands of ~50 kB single-cycle files for the same
   bytes and 100x the requests.

A float is downloaded **whole** if *any* of its profiles falls in the box.  That
is deliberate — clipping a float to the box at download time would make the
held-out-float cohort depend on the box, and P1 compares two boxes.

Provenance that the plan actually needs, and that this preserves
---------------------------------------------------------------
``DATA_MODE``          R/A/D — real-time vs adjusted vs delayed-mode.
``DATA_CENTRE``        the DAC that processed it.
``PLATFORM_TYPE`` /
``WMO_INST_TYPE``      instrument family.
``*_ADJUSTED_ERROR``   the reported observation-error estimate, per level.

The last one matters most: the DFS estimator's ``lambda_i = noise_density_i *
|support_i|`` has so far been fed a **pilot constant** (``NOISE_DENSITY_POINT =
0.08``).  Argo ships a real per-observation error, so the noise model can stop
being a tuned constant.  The first four give P2 genuine provenance groups
instead of synthetic labels.

Output
------
``data/argo/ar_index_global_prof.txt.gz``   the index, with its SHA-256 recorded
``data/argo/dac/<dac>/<wmo>_prof.nc``       one file per float
``data/argo/manifest.json``                 index hash, region boxes, per-file
                                            SHA-256, float and profile counts

Writes are atomic (tmp + os.replace) and cached files are validated before being
skipped, so an interrupted run resumes without corrupting anything — same
contract as ``13_download_godas.py``.

Run:
    .venv/bin/python experiments/42_download_argo.py                 # both boxes
    .venv/bin/python experiments/42_download_argo.py --smoke         # 25 floats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

GDAC = "https://data-argo.ifremer.fr"
INDEX = "ar_index_global_prof.txt.gz"

#: The two P1 regions.  Spans are 25 deg lat x 51 deg lon in BOTH boxes so the
#: GODAS subsets land on an identical 38 x 26 grid and the model, token budget
#: and patch tiling are literally the same object across regions.  lon is 0-360.
REGIONS = {
    "gulfstream": dict(lat=(25.0, 50.0), lon=(280.0, 331.0),
                       note="western boundary current, high EKE"),
    "npac_gyre":  dict(lat=(20.0, 45.0), lon=(180.0, 231.0),
                       note="North Pacific subtropical gyre, quiet interior"),
    # The hardest water in the global reconstruction: the equatorial Pacific
    # cold tongue and its sharp, shallow thermocline. The global error maps put
    # the brightest band right here, so it is where extra observations should
    # pay off most -- and Argo samples it densely (128,850 profiles, 1,157
    # floats, 2000-2025). Same 25 deg x 51 deg span as the other two boxes.
    "eq_pacific": dict(lat=(-12.5, 12.5), lon=(180.0, 231.0),
                       note="central equatorial Pacific, cold tongue and "
                            "shallow thermocline"),
    # Whole-ocean cohort, for the global reconstruction map. Every float that
    # reported anywhere in the period; the equator weighting happens at
    # SAMPLING time, not by throwing away the rest of the array.
    "global": dict(lat=(-90.0, 90.0), lon=(0.0, 360.0),
                   note="global ocean"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
ap.add_argument("--start-year", type=int, default=2000)
ap.add_argument("--end-year", type=int, default=2025)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--out", default=None, help="default: <repo>/data/argo")
ap.add_argument("--refresh-index", action="store_true")
ap.add_argument("--smoke", action="store_true", help="25 floats per region")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = args.out or os.path.join(ROOT, "data", "argo")
DACDIR = os.path.join(OUT, "dac")
os.makedirs(DACDIR, exist_ok=True)
t0 = time.time()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- the index
idx_path = os.path.join(OUT, INDEX)
if args.refresh_index or not os.path.exists(idx_path):
    print(f"fetching {INDEX} ...", flush=True)
    r = requests.get(f"{GDAC}/{INDEX}", timeout=600)
    r.raise_for_status()
    fd, tmp = tempfile.mkstemp(dir=OUT, suffix=".tmp"); os.close(fd)
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, idx_path)
idx_sha = sha256(idx_path)
print(f"index {os.path.getsize(idx_path):,} bytes  sha256 {idx_sha[:16]}...",
      flush=True)

df = pd.read_csv(idx_path, comment="#", low_memory=False,
                 usecols=["file", "date", "latitude", "longitude", "ocean",
                          "profiler_type", "institution"])
df = df.dropna(subset=["file", "date", "latitude", "longitude"])
df["year"] = df["date"].astype("int64").astype(str).str[:4].astype(int)
# index paths look like "aoml/13857/profiles/D13857_001.nc"
parts = df["file"].str.split("/", n=3, expand=True)
df["dac"], df["wmo"] = parts[0], parts[1]
lat = df["latitude"].to_numpy()
lon = df["longitude"].to_numpy() % 360.0
in_period = (df["year"].to_numpy() >= args.start_year) & \
            (df["year"].to_numpy() <= args.end_year)

# ------------------------------------------------------- float selection
floats: dict[tuple[str, str], set[str]] = {}
region_stats = {}
for name in args.regions:
    box = REGIONS[name]
    m = in_period & (lat >= box["lat"][0]) & (lat <= box["lat"][1]) \
                  & (lon >= box["lon"][0]) & (lon <= box["lon"][1])
    sub = df[m]
    pairs = sub[["dac", "wmo"]].drop_duplicates()
    if args.smoke:
        pairs = pairs.head(25)
    for dac, wmo in pairs.itertuples(index=False):
        floats.setdefault((dac, wmo), set()).add(name)
    region_stats[name] = dict(
        box=box, in_box_profiles=int(m.sum()), floats=int(len(pairs)),
        first_year=int(sub["year"].min()) if len(sub) else None,
        last_year=int(sub["year"].max()) if len(sub) else None)
    print(f"  {name:12s} {int(m.sum()):>8,} in-box profiles  "
          f"{len(pairs):>5,} floats", flush=True)

jobs = sorted(floats)
print(f"{len(jobs):,} unique floats to fetch "
      f"({sum(len(v) > 1 for v in floats.values())} appear in both boxes)",
      flush=True)


def valid_cached(path: str) -> bool:
    """A cached float counts only if it opens and carries the core variables.

    Size alone is not enough: a truncated NetCDF has a plausible size and fails
    only when read, which would then surface days later inside a training run.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    try:
        import xarray as xr
        with xr.open_dataset(path) as d:
            return {"PRES", "TEMP", "JULD", "LATITUDE"} <= set(d.variables)
    except Exception:
        return False


def has_salinity(path: str) -> bool:
    """T-only floats are a real population, not a corrupt download.

    16 floats (mostly JMA) carry PRES/TEMP and no PSAL at all.  Treating that as
    a download failure retried them forever and reported a false error count;
    the honest handling is to keep the file, record it, and let the cohort
    builder drop it for want of salinity -- the reconstruction target is T AND S.
    """
    try:
        import xarray as xr
        with xr.open_dataset(path) as d:
            return "PSAL" in d.variables
    except Exception:
        return False


def fetch(job) -> dict:
    dac, wmo = job
    d = os.path.join(DACDIR, dac)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{wmo}_prof.nc")
    cached = valid_cached(path)
    if not cached:
        url = f"{GDAC}/dac/{dac}/{wmo}/{wmo}_prof.nc"
        r = requests.get(url, timeout=600)
        if r.status_code == 404:
            return dict(dac=dac, wmo=wmo, missing=True)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp"); os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(r.content)
            if not valid_cached(tmp):
                raise IOError(f"{wmo}: downloaded file does not open")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return dict(dac=dac, wmo=wmo, file=f"{dac}/{wmo}_prof.nc",
                bytes=os.path.getsize(path), sha256=sha256(path),
                regions=sorted(floats[job]), cached=cached, missing=False,
                has_salinity=has_salinity(path))


files, missing, failed, done = [], [], [], 0
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(fetch, j): j for j in jobs}
    for f in as_completed(futs):
        dac, wmo = futs[f]
        try:
            rec = f.result()
        except Exception as e:
            failed.append(dict(dac=dac, wmo=wmo, error=f"{type(e).__name__}: {e}"))
            print(f"  FAILED {dac}/{wmo}: {type(e).__name__}: {e}", flush=True)
            continue
        (missing if rec.get("missing") else files).append(rec)
        done += 1
        if done % 100 == 0 or done == len(jobs):
            mb = sum(r["bytes"] for r in files) / 1e6
            print(f"  {done}/{len(jobs)} floats  {mb:,.0f} MB  "
                  f"({time.time()-t0:.0f}s)", flush=True)

if not files:
    raise SystemExit("no float files downloaded")

# Merge into any existing manifest rather than replacing it.  Fetching one
# region alone otherwise drops every other region's floats from the record --
# the manifest hash is an exact-match quantity in the cross-check, so silently
# shrinking it looks like a protocol violation later.  Same fix the cohort
# builder needed.
prev_files, prev_regions = [], {}
if os.path.exists(os.path.join(OUT, "manifest.json")):
    try:
        _pm = json.load(open(os.path.join(OUT, "manifest.json")))
        prev_files = _pm.get("files", [])
        prev_regions = _pm.get("regions", {})
    except Exception:
        pass
by_key = {(r["dac"], r["wmo"]): r for r in prev_files}
by_key.update({(r["dac"], r["wmo"]): r for r in files})
files = sorted(by_key.values(), key=lambda r: (r["dac"], r["wmo"]))
region_stats = {**prev_regions, **region_stats}

manifest = {
    "source": f"Argo Global Data Assembly Centre, {GDAC}",
    "acknowledgement": (
        "These data were collected and made freely available by the "
        "International Argo Program and the national programs that contribute "
        "to it (https://argo.ucsd.edu). The Argo Program is part of the Global "
        "Ocean Observing System."),
    "role": "real in-situ profile cohort for the parallel cross-check plan",
    "index": {"file": INDEX, "sha256": idx_sha,
              "bytes": os.path.getsize(idx_path),
              "rows_parsed": int(len(df))},
    "period": [args.start_year, args.end_year],
    "regions": region_stats,
    "float_count": len(files),
    "temperature_only_count": sum(1 for r in files if not r.get("has_salinity")),
    "missing_count": len(missing),
    "failed_count": len(failed),
    "total_bytes": sum(r["bytes"] for r in files),
    "missing": missing,
    "failed": failed,
    "files": files,
}
mpath = os.path.join(OUT, "manifest.json")
with open(mpath + ".tmp", "w") as f:
    json.dump(manifest, f, indent=1)
os.replace(mpath + ".tmp", mpath)

print(f"\n{len(files):,} floats, {manifest['total_bytes']:,} bytes")
if missing:
    print(f"{len(missing)} floats had no _prof.nc on the GDAC (listed in manifest)")
if failed:
    print(f"{len(failed)} floats FAILED — rerun to resume")
print(f"manifest: {mpath}\nmanifest sha256: {sha256(mpath)}")
print(f"total {time.time()-t0:.0f}s")
