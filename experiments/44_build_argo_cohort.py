"""Turn raw Argo float files into the analysis cohort the plan's protocol needs.

`42_download_argo.py` fetched 2,243 whole floats.  This script applies Argo's
own quality control, puts every surviving profile on the experiment's depth
levels and grid, and defines the **two independent splits** the cross-check plan
depends on:

    by YEAR   train / validation / development / holdout, matching the GODAS
              year splits exactly, so an Argo row and a GODAS row in the same
              table were selected on the same era.

    by FLOAT  a WMO-disjoint held-out cohort.  This is the one that makes
              "WMO/platform is the primary inference unit" (P7) mean anything:
              a float held out only by year still had its earlier cycles in
              train, and its instrument bias, its water mass and its
              trajectory all leak across the boundary.  A float in the held-out
              cohort appears in NO training month.

Quality control — and why each flag is where it is
--------------------------------------------------
Argo ships QC flags per value and a ``DATA_MODE`` per profile; using the raw
``TEMP``/``PSAL`` and ignoring both is the classic way to put uncorrected
salinity drift into a "real data" result.

``DATA_MODE``      D (delayed) and A (adjusted) -> use ``*_ADJUSTED`` and
                   ``*_ADJUSTED_QC``.  R (real-time) -> use the raw fields and
                   their QC.  Mixing an adjusted value with a raw QC flag is a
                   real and easy mistake; the pairing is done in one place.
``POSITION_QC``    keep 1 (good) and 2 (probably good).  A profile whose
                   position is bad cannot be assigned to a grid cell at all,
                   which is fatal for a spatially-clustered analysis.
``JULD_QC``        same, for the same reason in time.
value-level QC     keep 1, 2 and — adjusted only — 5 and 8, which mean "value
                   changed" and "interpolated": both are legitimate delayed-mode
                   products.  3 (bad, correctable), 4 (bad), 9 (missing) go.
``PRES``           must be positive and increasing; a level is dropped, not the
                   profile, so one bad bottle does not discard 200 good ones.

Pressure is converted to depth with the full TEOS-10 relation (``gsw.z_from_p``),
which is latitude-dependent.  The 1 dbar ~ 1 m shortcut is off by ~1% at 1000 m
and varies with latitude — small, but the vertical support kernel is exactly
what Phase 1a was about, so the depth axis should not carry an avoidable bias.

Interpolation onto ``GODAS_LEVELS_M``
-------------------------------------
Linear in depth, and **never extrapolated**: a level outside the profile's own
sampled range is NaN, not the nearest value.  Argo floats routinely start at
5-10 m and stop at 1000 m or 2000 m; filling their top and bottom by
extrapolation would invent the surface and abyssal values the reconstruction is
supposed to predict.  A profile is kept if it retains at least
``--min-levels`` valid levels.

Output
------
``data/argo_cohort/<region>.nc``   profiles x levels, with WMO, DAC, platform
                                   type, data mode, per-level adjusted error,
                                   grid indices, month index and split labels
``data/argo_cohort/manifest.json`` counts, split definitions, QC tallies, and
                                   the SHA-256 of every input manifest it read

Run:
    .venv/bin/python experiments/44_build_argo_cohort.py
    .venv/bin/python experiments/44_build_argo_cohort.py --smoke   # 60 floats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.batched_dfs import GODAS_LEVELS_M

REGIONS = {
    "gulfstream": dict(lat=(25.0, 50.0), lon=(280.0, 331.0)),
    "npac_gyre":  dict(lat=(20.0, 45.0), lon=(180.0, 231.0)),
    "eq_pacific": dict(lat=(-12.5, 12.5), lon=(180.0, 231.0)),
    # Whole ocean, for the global reconstruction map. With --grid global the
    # index formula below reduces to gy = lat + 90, gx = lon, i.e. exactly the
    # 1 deg analysis cells.
    "global": dict(lat=(-90.0, 90.0), lon=(0.0, 360.0)),
}
GRID_NY, GRID_NX = 38, 26          # the stride-2 GODAS experiment grid
LEVELS = np.asarray(GODAS_LEVELS_M, dtype=np.float64)

#: protocol_v1's 20 target levels, 5 m to 985 m, on the 1 deg global analysis
#: grid.  Used by the global cohort so the real-Argo reconstruction lands on
#: exactly the axes the CESM2 reconstruction figure uses, and the two are
#: readable side by side.
PROTOCOL_LEVELS = np.array(
    [5., 15., 25., 35., 45., 55., 65., 85., 105., 125., 145., 165.1,
     186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7], dtype=np.float64)
GRIDS = {"regional": (38, 26), "global": (180, 360)}
LEVELSETS = {"godas": LEVELS, "protocol": PROTOCOL_LEVELS}

#: identical to the GODAS driver's SPLITS, deliberately -- see module docstring
YEAR_SPLITS = {"train": (2000, 2018), "validation": (2019, 2021),
               "development": (2022, 2024), "holdout": (2025, 2025)}

GOOD_QC_RAW = {b"1", b"2"}
GOOD_QC_ADJ = {b"1", b"2", b"5", b"8"}

ap = argparse.ArgumentParser()
ap.add_argument("--argo", default=None, help="default: <repo>/data/argo")
ap.add_argument("--out", default=None, help="default: <repo>/data/argo_cohort")
ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
ap.add_argument("--min-levels", type=int, default=8,
                help="a profile must retain this many of the 16 target levels")
ap.add_argument("--holdout-float-frac", type=float, default=0.20,
                help="fraction of floats reserved as the WMO-disjoint cohort")
ap.add_argument("--cohort-seed", type=int, default=20260905)
ap.add_argument("--grid", default="regional", choices=list(GRIDS),
                help="'global' puts profiles on the 1 deg 180x360 analysis grid")
ap.add_argument("--levels", default="godas", choices=list(LEVELSETS),
                help="'protocol' uses protocol_v1's 20 levels to 985 m")
ap.add_argument("--suffix", default="",
                help="appended to the output filename, e.g. '_global'")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

GRID_NY, GRID_NX = GRIDS[ap.parse_known_args()[0].grid]
LEVELS = LEVELSETS[ap.parse_known_args()[0].levels]
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARGO = args.argo or os.path.join(ROOT, "data", "argo")
OUT = args.out or os.path.join(ROOT, "data", "argo_cohort")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bytes(a) -> np.ndarray:
    """Argo char arrays arrive as bytes or str depending on the file; normalise."""
    a = np.asarray(a)
    if a.dtype.kind == "U":
        return np.char.encode(a.astype(str), "utf-8")
    return a


def _txt(v) -> str:
    b = v.tobytes() if hasattr(v, "tobytes") else bytes(v)
    return b.decode("utf-8", "ignore").strip()


tally = Counter()


def read_float(path: str) -> dict | None:
    """QC one float file down to (profile, level) arrays on the target depths."""
    try:
        ds = xr.open_dataset(path)
    except Exception:
        tally["file_unreadable"] += 1
        return None
    try:
        npro = ds.sizes.get("N_PROF", 0)
        if npro == 0 or "PRES" not in ds:
            tally["file_empty"] += 1
            return None
        if "PSAL" not in ds:
            # 16 floats (mostly JMA) carry PRES/TEMP and no salinity at all.
            # They are a real population, not a corrupt download -- but the
            # reconstruction target is T AND S, so they cannot contribute.
            tally["floats_without_salinity"] += 1
            return None
        juld = ds["JULD"].values
        lat = np.asarray(ds["LATITUDE"].values, float)
        lon = np.asarray(ds["LONGITUDE"].values, float) % 360.0
        mode = _bytes(ds["DATA_MODE"].values)
        pos_qc = _bytes(ds["POSITION_QC"].values)
        juld_qc = _bytes(ds["JULD_QC"].values) if "JULD_QC" in ds else \
            np.full(npro, b"1")
        wmo = _txt(_bytes(ds["PLATFORM_NUMBER"].values)[0])
        dac = _txt(_bytes(ds["DATA_CENTRE"].values)[0]) if "DATA_CENTRE" in ds else ""
        ptype = _txt(_bytes(ds["PLATFORM_TYPE"].values)[0]) if "PLATFORM_TYPE" in ds else ""

        adj = np.isin(mode, [b"D", b"A"])
        def pick(base):
            """Adjusted where the profile says so, raw otherwise — paired with
            the matching QC array, never crossed."""
            a = ds[f"{base}_ADJUSTED"].values if f"{base}_ADJUSTED" in ds else None
            r = ds[base].values
            q_a = _bytes(ds[f"{base}_ADJUSTED_QC"].values) if f"{base}_ADJUSTED_QC" in ds else None
            q_r = _bytes(ds[f"{base}_QC"].values) if f"{base}_QC" in ds else None
            val = np.array(r, float, copy=True)
            qc = np.full(val.shape, b"9", dtype="S1") if q_r is None else np.array(q_r, copy=True)
            good = np.zeros(val.shape, bool)
            if q_r is not None:
                good |= np.isin(qc, list(GOOD_QC_RAW))
            if a is not None and adj.any():
                val[adj] = np.asarray(a, float)[adj]
                if q_a is not None:
                    qc[adj] = np.array(q_a)[adj]
                    good[adj] = np.isin(np.array(q_a)[adj], list(GOOD_QC_ADJ))
            err = None
            if f"{base}_ADJUSTED_ERROR" in ds:
                err = np.asarray(ds[f"{base}_ADJUSTED_ERROR"].values, float)
            return val, good, err

        pres, pres_ok, _ = pick("PRES")
        temp, temp_ok, temp_err = pick("TEMP")
        psal, psal_ok, psal_err = pick("PSAL")

        keep_prof = (np.isin(pos_qc, list(GOOD_QC_RAW)) &
                     np.isin(juld_qc, list(GOOD_QC_RAW)) &
                     np.isfinite(lat) & np.isfinite(lon) &
                     ~np.isnat(np.asarray(juld, "datetime64[ns]")))
        tally["profiles_seen"] += int(npro)
        tally["profiles_rejected_position_or_time"] += int((~keep_prof).sum())
        if not keep_prof.any():
            return None

        import gsw
        out_T = np.full((npro, LEVELS.size), np.nan)
        out_S = np.full((npro, LEVELS.size), np.nan)
        out_Te = np.full((npro, LEVELS.size), np.nan)
        out_Se = np.full((npro, LEVELS.size), np.nan)
        nlev_kept = np.zeros(npro, int)

        for i in np.flatnonzero(keep_prof):
            ok = (pres_ok[i] & np.isfinite(pres[i]) & (pres[i] > 0) &
                  temp_ok[i] & np.isfinite(temp[i]) &
                  psal_ok[i] & np.isfinite(psal[i]))
            if ok.sum() < 2:
                continue
            p = pres[i][ok]
            # TEOS-10 depth: negative down, hence the sign flip
            z = -gsw.z_from_p(p, lat[i])
            order = np.argsort(z)
            z = z[order]
            uniq = np.concatenate([[True], np.diff(z) > 0])
            z = z[uniq]
            if z.size < 2:
                continue
            tv = temp[i][ok][order][uniq]
            sv = psal[i][ok][order][uniq]
            inside = (LEVELS >= z[0]) & (LEVELS <= z[-1])   # never extrapolate
            if not inside.any():
                continue
            out_T[i, inside] = np.interp(LEVELS[inside], z, tv)
            out_S[i, inside] = np.interp(LEVELS[inside], z, sv)
            if temp_err is not None:
                e = temp_err[i][ok][order][uniq]
                if np.isfinite(e).any():
                    out_Te[i, inside] = np.interp(LEVELS[inside], z,
                                                  np.nan_to_num(e, nan=np.nanmax(e)))
            if psal_err is not None:
                e = psal_err[i][ok][order][uniq]
                if np.isfinite(e).any():
                    out_Se[i, inside] = np.interp(LEVELS[inside], z,
                                                  np.nan_to_num(e, nan=np.nanmax(e)))
            nlev_kept[i] = int(inside.sum())

        keep = keep_prof & (nlev_kept >= args.min_levels)
        tally["profiles_rejected_too_few_levels"] += int(
            (keep_prof & (nlev_kept < args.min_levels)).sum())
        if not keep.any():
            return None
        k = np.flatnonzero(keep)
        return dict(wmo=wmo, dac=dac, platform_type=ptype,
                    juld=np.asarray(juld, "datetime64[ns]")[k],
                    lat=lat[k], lon=lon[k], data_mode=mode[k],
                    TEMP=out_T[k], SALT=out_S[k],
                    TEMP_ERR=out_Te[k], SALT_ERR=out_Se[k],
                    cycle=(np.asarray(ds["CYCLE_NUMBER"].values)[k]
                           if "CYCLE_NUMBER" in ds else np.arange(k.size)))
    finally:
        ds.close()


man = json.load(open(os.path.join(ARGO, "manifest.json")))
files = man["files"][:60] if args.smoke else man["files"]
print(f"QC-ing {len(files):,} floats ...", flush=True)

records = []
for i, rec in enumerate(files, 1):
    r = read_float(os.path.join(ARGO, "dac", rec["file"]))
    if r is not None:
        records.append(r)
    if i % 250 == 0 or i == len(files):
        print(f"  {i}/{len(files)} floats  {len(records):,} kept  "
              f"({time.time()-t0:.0f}s)", flush=True)

if not records:
    raise SystemExit("no float survived QC")

# ------------------------------------------------------- assemble + split
summary = {}
for region in args.regions:
    box = REGIONS[region]
    lat0, lat1 = box["lat"]; lon0, lon1 = box["lon"]
    rows = []
    for r in records:
        m = ((r["lat"] >= lat0) & (r["lat"] <= lat1) &
             (r["lon"] >= lon0) & (r["lon"] <= lon1))
        yr = r["juld"].astype("datetime64[Y]").astype(int) + 1970
        m &= (yr >= 2000) & (yr <= 2025)
        if not m.any():
            continue
        k = np.flatnonzero(m)
        # grid index on the stride-2 experiment grid: the box spans GRID_NY x
        # GRID_NX cells, so a cell is (lat1-lat0)/GRID_NY deg tall.  clip keeps
        # a profile exactly on the upper edge inside the last cell.
        gy = np.clip(((r["lat"][k] - lat0) / (lat1 - lat0) * GRID_NY).astype(int), 0, GRID_NY - 1)
        gx = np.clip(((r["lon"][k] - lon0) / (lon1 - lon0) * GRID_NX).astype(int), 0, GRID_NX - 1)
        rows.append(dict(
            wmo=np.full(k.size, r["wmo"]), dac=np.full(k.size, r["dac"]),
            platform_type=np.full(k.size, r["platform_type"] or "unknown"),
            juld=r["juld"][k], lat=r["lat"][k], lon=r["lon"][k],
            gy=gy, gx=gx, cycle=np.asarray(r["cycle"])[k],
            data_mode=np.array([d.decode() for d in r["data_mode"][k]]),
            TEMP=r["TEMP"][k], SALT=r["SALT"][k],
            TEMP_ERR=r["TEMP_ERR"][k], SALT_ERR=r["SALT_ERR"][k]))
    if not rows:
        print(f"  {region}: no profiles"); continue

    cat = lambda key: np.concatenate([x[key] for x in rows], axis=0)
    juld = cat("juld")
    order = np.argsort(juld)                       # chronological, stable
    P = {k: cat(k)[order] for k in
         ("wmo", "dac", "platform_type", "lat", "lon", "gy", "gx", "cycle",
          "data_mode", "TEMP", "SALT", "TEMP_ERR", "SALT_ERR")}
    juld = juld[order]
    year = juld.astype("datetime64[Y]").astype(int) + 1970
    month_idx = (juld.astype("datetime64[M]").astype(int) -
                 np.datetime64("2000-01", "M").astype(int))

    year_split = np.full(juld.size, "unassigned", dtype=object)
    for name, (a, b) in YEAR_SPLITS.items():
        year_split[(year >= a) & (year <= b)] = name

    # WMO-disjoint cohort.  Chosen from the float ids alone with a registered
    # seed, BEFORE any score is read, and independent of the year split -- so a
    # float is either in the cohort for its whole life or never.
    floats = np.unique(P["wmo"])
    rng = np.random.default_rng([args.cohort_seed, abs(hash(region)) % (2**31)])
    n_hold = max(1, int(round(args.holdout_float_frac * floats.size)))
    held = set(rng.choice(floats, size=n_hold, replace=False).tolist())
    float_split = np.where(np.isin(P["wmo"], list(held)), "heldout_float",
                           "cohort_float")

    ds = xr.Dataset(
        {"TEMP": (("profile", "level"), P["TEMP"].astype("float32")),
         "SALT": (("profile", "level"), P["SALT"].astype("float32")),
         "TEMP_ERR": (("profile", "level"), P["TEMP_ERR"].astype("float32")),
         "SALT_ERR": (("profile", "level"), P["SALT_ERR"].astype("float32")),
         "lat": ("profile", P["lat"].astype("float32")),
         "lon": ("profile", P["lon"].astype("float32")),
         "grid_y": ("profile", P["gy"].astype("int16")),
         "grid_x": ("profile", P["gx"].astype("int16")),
         "month_index": ("profile", month_idx.astype("int32")),
         "year": ("profile", year.astype("int16")),
         "cycle": ("profile", np.asarray(P["cycle"]).astype("int32")),
         "wmo": ("profile", P["wmo"].astype(str)),
         "dac": ("profile", P["dac"].astype(str)),
         "platform_type": ("profile", P["platform_type"].astype(str)),
         "data_mode": ("profile", P["data_mode"].astype(str)),
         "year_split": ("profile", year_split.astype(str)),
         "float_split": ("profile", float_split.astype(str))},
        coords={"level": LEVELS, "time": ("profile", juld)},
        attrs=dict(
            region=region, box_lat=list(box["lat"]), box_lon=list(box["lon"]),
            grid=[GRID_NY, GRID_NX], levels_set=args.levels,
            source="Argo GDAC; see data/argo/manifest.json",
            qc="POSITION_QC/JULD_QC in {1,2}; values QC {1,2} raw, {1,2,5,8} adjusted",
            depth="TEOS-10 gsw.z_from_p; linear interp, no extrapolation",
            min_levels=args.min_levels, cohort_seed=args.cohort_seed,
            acknowledgement=man["acknowledgement"]))
    path = os.path.join(OUT, f"{region}{args.suffix}.nc")
    fd, tmp = tempfile.mkstemp(dir=OUT, suffix=".tmp"); os.close(fd)
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    summary[region + args.suffix] = dict(
        file=os.path.basename(path), grid=[GRID_NY, GRID_NX],
        levels_set=args.levels, sha256=sha256(path),
        bytes=os.path.getsize(path),
        profiles=int(juld.size), floats=int(floats.size),
        heldout_floats=int(len(held)),
        heldout_profiles=int((float_split == "heldout_float").sum()),
        months=int(np.unique(month_idx).size),
        year_range=[int(year.min()), int(year.max())],
        by_year_split={k: int((year_split == k).sum()) for k in YEAR_SPLITS},
        floats_by_year_split={k: int(np.unique(P["wmo"][year_split == k]).size)
                              for k in YEAR_SPLITS},
        data_mode={k: int(v) for k, v in
                   Counter(P["data_mode"].tolist()).items()},
        level_coverage=[float(np.isfinite(P["TEMP"][:, j]).mean())
                        for j in range(LEVELS.size)],
        adjusted_error_available=float(np.isfinite(P["TEMP_ERR"]).any(axis=1).mean()))
    s = summary[region + args.suffix]
    print(f"  {region + args.suffix:12s} {s['profiles']:>7,} profiles  {s['floats']:>5,} floats  "
          f"{s['months']:>4} months  heldout {s['heldout_floats']} floats / "
          f"{s['heldout_profiles']:,} profiles", flush=True)

# Merge into any existing manifest rather than overwriting it.  Building one
# region alone otherwise drops the others from the record, which silently
# changes the manifest hash the cross-check treats as an exact-match quantity --
# the same trap the float downloader had.
mpath_prev = os.path.join(OUT, "manifest.json")
prev_regions = {}
if os.path.exists(mpath_prev):
    try:
        prev_regions = json.load(open(mpath_prev)).get("regions", {})
    except Exception:
        prev_regions = {}
merged = {**prev_regions, **summary}

manifest = {
    "role": "QC'd Argo profile cohort on the experiment grid and depth levels",
    "argo_manifest_sha256": sha256(os.path.join(ARGO, "manifest.json")),
    "argo_index_sha256": man["index"]["sha256"],
    "levels_m": LEVELS.tolist(),
    "grid": [GRID_NY, GRID_NX],
    "year_splits": YEAR_SPLITS,
    "float_cohort": {"heldout_fraction": args.holdout_float_frac,
                     "seed": args.cohort_seed,
                     "rule": "WMO-disjoint; drawn from float ids before any "
                             "score was read; independent of the year split"},
    "qc_tally": dict(tally),
    "min_levels": args.min_levels,
    "regions": merged,
    "regions_built_this_run": sorted(summary),
}
mpath = os.path.join(OUT, "manifest.json")
with open(mpath + ".tmp", "w") as f:
    json.dump(manifest, f, indent=1)
os.replace(mpath + ".tmp", mpath)
print(f"\nQC tally: {dict(tally)}")
print(f"manifest: {mpath}\nmanifest sha256: {sha256(mpath)}\ntotal {time.time()-t0:.0f}s")
