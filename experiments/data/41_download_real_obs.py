"""Download real satellite/in-situ observations and regrid to the 1-deg grid.

Replaces the CESM2-LE synthetic surface fields with genuine observations, per
the data-source review.  Everything is pulled at its NATIVE resolution and
stored, then block-mean regridded to the 1 deg analysis grid the model already
runs on.

Why download fine and coarsen here, rather than find a 1-deg product
-------------------------------------------------------------------
* There is no native 1-deg gridded altimetry product, so a regrid is required
  for SSH no matter what.  Doing it ourselves gives every variable the same
  treatment instead of inheriting four different upstream interpolations.
* The factors are exact -- 1 / 0.125 = 8 and 1 / 0.25 = 4 -- so the regrid is a
  pure block mean with no interpolation weights, no kernel choice, no edge
  cases.  It is conservative by construction within a latitude row.
* Averaging 16-64 cells cuts uncorrelated noise by 4-8x, which matters most for
  satellite SSS (RFI, poor cold-water sensitivity).
* The native files are kept, so a future 0.25 deg experiment needs no re-download.

Streams
-------
SSH   DUACS L4 reprocessed, ``cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-
      0.125deg_P1M-m`` (SEALEVEL_GLO_PHY_L4_MY_008_047).  0.125 deg monthly,
      1993-present.  The monthly product carries **sla** only, which is what we
      want: the model works on train-only monthly anomalies, so the static MDT
      inside ADT would be removed by the climatology anyway.  Needs a free
      Copernicus Marine account (``copernicusmarine login``).
SST   NOAA OISST v2.1, 0.25 deg.  Open, no credentials.
SSS   OISSS L4 multimission monthly v2 (PO.DAAC), 0.25 deg, Aug 2011-present.
      Needs a free NASA Earthdata login (``~/.netrc``).

Argo profiles are NOT downloaded here -- they are point observations, not a
gridded field, and belong in their own ingest.

Output: ``data/real_obs/<stream>_native/`` (as delivered) and
``data/real_obs_1deg.zarr`` (regridded, model-ready, same schema as the CESM2
store: dims time/lat/lon, lon 0-360).

Run:
    .venv/bin/python experiments/41_download_real_obs.py --start 2016 --end 2023
    .venv/bin/python experiments/41_download_real_obs.py --smoke     # 3 months
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer import config as C

DUACS_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m"
OISSS_SHORT = "OISSS_L4_multimission_monthly_v2"
OISST_MON = ("https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/"
             "sst.mon.mean.nc")

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=2016)
ap.add_argument("--end", type=int, default=2023)
ap.add_argument("--out", default=os.path.join(C.ROOT, "data", "real_obs"))
ap.add_argument("--zarr", default=os.path.join(C.ROOT, "data",
                                               "real_obs_1deg.zarr"))
ap.add_argument("--streams", nargs="+", default=["ssh", "sst", "sss"],
                choices=["ssh", "sst", "sss"])
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.start, args.end = 2016, 2016
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
Y0, Y1 = args.start, args.end
print(f"[real_obs] {Y0}-{Y1}  streams={args.streams}  out={args.out}", flush=True)


# ===================================================================== regrid
def block_mean_to_1deg(da: xr.DataArray, lat="latitude", lon="longitude"):
    """Exact block mean onto the 1 deg grid (lat -89.5..89.5, lon 0.5..359.5).

    Requires the native resolution to divide 1 deg exactly (0.125 -> 8,
    0.25 -> 4).  NaN-aware, so land does not bleed into ocean means.
    """
    ny, nx = da.sizes[lat], da.sizes[lon]
    fy, fx = ny // 180, nx // 360
    if fy * 180 != ny or fx * 360 != nx:
        raise ValueError(f"{ny}x{nx} is not an integer multiple of 180x360; "
                         "block mean would not be exact")
    # products differ in dim order -- OISSS ships (lat, lon, time) while DUACS
    # and OISST ship (time, lat, lon) -- so transpose explicitly rather than
    # trusting the file's layout
    tname = [d for d in da.dims if d not in (lat, lon)]
    da = da.transpose(*(tname + [lat, lon]))
    a = da.values
    if a.ndim == 2:
        a = a[None]
    a = a.reshape(a.shape[0], 180, fy, 360, fx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.nanmean(a, axis=(2, 4)).astype("float32")
    return out, fy, fx


def normalise_grid(ds, lat="latitude", lon="longitude"):
    """Put any provider's grid on the project convention before regridding.

    Providers disagree on both axes and the disagreements are SILENT: OISST
    ships latitude north->south while the project grid runs -89.5..89.5, and
    DUACS ships longitude -180..180 while the project uses 0..360.  Neither
    raises -- they just produce a flipped or rolled field that still looks
    plausible.  Normalise explicitly, then assert.
    """
    if float(ds[lon].min()) < 0:
        ds = ds.assign_coords({lon: (ds[lon] % 360)}).sortby(lon)
    if float(ds[lat][0]) > float(ds[lat][-1]):
        ds = ds.isel({lat: slice(None, None, -1)})
    la, lo = ds[lat].values, ds[lon].values
    assert la[0] < la[-1], "latitude must ascend after normalisation"
    assert lo[0] < lo[-1], "longitude must ascend after normalisation"
    assert -90 <= la[0] and la[-1] <= 90, f"latitude out of range: {la[0]}..{la[-1]}"
    assert 0 <= lo[0] and lo[-1] <= 360, f"longitude out of range: {lo[0]}..{lo[-1]}"
    return ds


def month_index(times):
    return np.asarray(times).astype("datetime64[M]")


fields, meta = {}, {}

# ======================================================================= SSH
if "ssh" in args.streams:
    print("\n[ssh] DUACS L4 monthly (Copernicus Marine)", flush=True)
    import copernicusmarine as cm
    raw = os.path.join(args.out, "ssh_native")
    os.makedirs(raw, exist_ok=True)
    nc = os.path.join(raw, f"duacs_sla_{Y0}_{Y1}.nc")
    if os.path.exists(nc) and not args.overwrite:
        print(f"  cached {nc}", flush=True)
        ds = xr.open_dataset(nc)
    else:
        ds = cm.open_dataset(dataset_id=DUACS_ID,
                             start_datetime=f"{Y0}-01-01",
                             end_datetime=f"{Y1}-12-31")
        ds = ds[["sla"]].load()
        ds.to_netcdf(nc)
        print(f"  saved native {ds.sizes['latitude']}x{ds.sizes['longitude']} "
              f"-> {nc} ({os.path.getsize(nc)/1e6:.0f} MB)", flush=True)
    ds = normalise_grid(ds)
    arr, fy, fx = block_mean_to_1deg(ds["sla"])
    fields["SLA"] = (arr, month_index(ds.time.values))
    meta["ssh"] = {"dataset_id": DUACS_ID, "variable": "sla",
                   "native": f"{ds.sizes['latitude']}x{ds.sizes['longitude']}",
                   "block": [fy, fx], "months": int(arr.shape[0])}
    times = month_index(ds.time.values)
    print(f"  regridded {fy}x{fx} block mean -> (180,360), "
          f"{arr.shape[0]} months, finite {100*np.isfinite(arr).mean():.1f}%",
          flush=True)

# ======================================================================= SST
if "sst" in args.streams:
    print("\n[sst] NOAA OISST v2.1 monthly (open)", flush=True)
    raw = os.path.join(args.out, "sst_native")
    os.makedirs(raw, exist_ok=True)
    nc = os.path.join(raw, "sst.mon.mean.nc")
    if not os.path.exists(nc) or args.overwrite:
        import urllib.request
        print(f"  downloading {OISST_MON}", flush=True)
        urllib.request.urlretrieve(OISST_MON, nc + ".tmp")
        os.replace(nc + ".tmp", nc)
        print(f"  saved ({os.path.getsize(nc)/1e6:.0f} MB)", flush=True)
    ds = xr.open_dataset(nc)
    ds = ds.sel(time=slice(f"{Y0}-01-01", f"{Y1}-12-31"))
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    ds = normalise_grid(ds)
    arr, fy, fx = block_mean_to_1deg(ds["sst"])
    fields["SST"] = (arr, month_index(ds.time.values))
    meta["sst"] = {"source": OISST_MON, "variable": "sst",
                   "native": f"{ds.sizes['latitude']}x{ds.sizes['longitude']}",
                   "block": [fy, fx], "months": int(arr.shape[0])}
    times = month_index(ds.time.values)
    print(f"  regridded {fy}x{fx} block mean -> (180,360), "
          f"{arr.shape[0]} months, finite {100*np.isfinite(arr).mean():.1f}%",
          flush=True)

# ======================================================================= SSS
if "sss" in args.streams:
    print("\n[sss] OISSS L4 multimission monthly v2 (PO.DAAC)", flush=True)
    import earthaccess
    earthaccess.login(strategy="netrc")
    raw = os.path.join(args.out, "sss_native")
    os.makedirs(raw, exist_ok=True)
    res = earthaccess.search_data(short_name=OISSS_SHORT,
                                  temporal=(f"{Y0}-01-01", f"{Y1}-12-31"))
    print(f"  {len(res)} granules", flush=True)
    have = set(os.listdir(raw))
    need = [g for g in res
            if os.path.basename(g.data_links()[0]) not in have]
    if need:
        print(f"  downloading {len(need)} ...", flush=True)
        earthaccess.download(need, raw)
    files = sorted(os.path.join(raw, f) for f in os.listdir(raw)
                   if f.endswith(".nc"))
    ds = xr.open_mfdataset(files, combine="by_coords").load()
    ren = {}
    if "lat" in ds.dims:
        ren.update(lat="latitude", lon="longitude")
    ds = ds.rename(ren) if ren else ds
    ds = normalise_grid(ds)
    var = "sss" if "sss" in ds else list(ds.data_vars)[0]
    arr, fy, fx = block_mean_to_1deg(ds[var])
    fields["SSS"] = (arr, month_index(ds.time.values))
    meta["sss"] = {"short_name": OISSS_SHORT, "variable": var,
                   "native": f"{ds.sizes['latitude']}x{ds.sizes['longitude']}",
                   "block": [fy, fx], "months": int(arr.shape[0])}
    times = month_index(ds.time.values)
    print(f"  regridded {fy}x{fx} block mean -> (180,360), "
          f"{arr.shape[0]} months, finite {100*np.isfinite(arr).mean():.1f}%",
          flush=True)

# ====================================================================== write
if not fields:
    raise SystemExit("no streams downloaded")
# Align on the INTERSECTION of the streams' real month axes.  Truncating each
# stream to a common LENGTH (v[:n]) silently misaligns them whenever one starts
# late or has a gap -- the arrays still stack, the file still writes, and every
# field is shifted against the others by however many months differ.  Join on
# the month labels instead, and fabricate nothing.
common = None
for k, (v, t) in fields.items():
    common = set(t.tolist()) if common is None else (common & set(t.tolist()))
tt = np.array(sorted(common), dtype="datetime64[M]")
if tt.size == 0:
    raise SystemExit("streams share no months in common")
for k, (v, t) in list(fields.items()):
    pos = {m: i for i, m in enumerate(t.tolist())}
    fields[k] = v[[pos[m] for m in tt.tolist()]]
    if v.shape[0] != tt.size:
        print(f"  note: {k} had {v.shape[0]} months, {tt.size} in common",
              flush=True)
months = int(tt.size)
# a monthly series with no gaps advances exactly one month per step
gaps = np.unique(np.diff(tt.astype("datetime64[M]")).astype(int))
if gaps.size and not (gaps == 1).all():
    print(f"  WARNING: common axis is not contiguous, month steps = {gaps}",
          flush=True)
lat = np.arange(-89.5, 90.0, 1.0, dtype="float32")
lon = np.arange(0.5, 360.0, 1.0, dtype="float32")
out = xr.Dataset({k: (("time", "lat", "lon"), v) for k, v in fields.items()},
                 coords={"time": tt, "lat": lat, "lon": lon},
                 attrs={"title": "Real satellite observations on the 1-deg "
                                 "analysis grid",
                        "regrid": "exact block mean from native resolution",
                        "lon_convention": "0-360",
                        "provenance": json.dumps(meta)})
if os.path.exists(args.zarr):
    import shutil
    shutil.rmtree(args.zarr)
enc = {k: {"chunks": (min(24, months), 90, 180)} for k in fields}
out.to_zarr(args.zarr, mode="w", encoding=enc, zarr_format=3, consolidated=True)
print(f"\nwrote {args.zarr}", flush=True)
print(f"  time {tt[0]} .. {tt[-1]}  ({months} months, from stream intersection)",
      flush=True)
for k, a in fields.items():
    print(f"  {k:4s} {a.shape}  finite {100*np.isfinite(a).mean():5.1f}%  "
          f"range [{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]", flush=True)
with open(os.path.join(args.out, "provenance.json"), "w") as f:
    json.dump({"years": [Y0, Y1], "streams": meta}, f, indent=2)
print(f"TOTAL {(time.time()-t0)/60:.1f} min", flush=True)
