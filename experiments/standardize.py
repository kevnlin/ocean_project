"""Standardize raw ocean datasets -> common zarr format (matches processed/).

Sources
-------
1. CESM2-LE single member sample : ocean_project/data/cesm2_lens/*.nc
2. WOA23 monthly climatology      : ocean_project/data/woa23/{temperature,salinity}/*.nc
3. CESM2-LE full (regridded 1deg) : cesm2-le-ocean-sample/*.{TEMP,SALT}.185001-210012.nc

Each is rewritten to a zarr store with
    dims      : (time, depth, lat, lon)
    variables : TEMP, SALT (4-D), SST, SSS (surface), [+ MASK for the LE full]
    units     : TEMP degC, SALT PSU, depth m
matching the schema/attrs/chunking of the existing processed/*_standard.zarr.

Usage
-----
    python experiments/standardize.py --datasets all            # all -> processed/
    python experiments/standardize.py --datasets cesm2 woa23    # subset
    python experiments/standardize.py --datasets cesm2_le_full --max-time 36   # quick test
    python experiments/standardize.py --datasets all --out processed_rebuilt --overwrite

By default existing stores are NOT overwritten (use --overwrite). The full LE
store is ~95 GB; --max-time N limits it to the first N months for validation.
"""
from __future__ import annotations
import os, argparse, shutil, glob
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/nvidia/ocean_project"
DATA = os.path.join(ROOT, "data")
LE_DIR = "/home/nvidia/cesm2-le-ocean-sample"

RAW = {
    "cesm2": {
        "TEMP": os.path.join(DATA, "cesm2_lens", "cesm2le_temp_sample.nc"),
        "SALT": os.path.join(DATA, "cesm2_lens", "cesm2le_salt_sample.nc"),
        "SST":  os.path.join(DATA, "cesm2_lens", "cesm2le_sst_sample.nc"),
        "SSS":  os.path.join(DATA, "cesm2_lens", "cesm2le_sss_sample.nc"),
    },
    "woa23": {
        "TEMP_dir": os.path.join(DATA, "woa23", "temperature"),
        "SALT_dir": os.path.join(DATA, "woa23", "salinity"),
    },
    "cesm2_le_full": {
        "TEMP": os.path.join(LE_DIR, "data_CESM_CESM2-LE_regrid_1x1deg_pop.h.TEMP.185001-210012.nc"),
        "SALT": os.path.join(LE_DIR, "data_CESM_CESM2-LE_regrid_1x1deg_pop.h.SALT.185001-210012.nc"),
    },
}


def _f32(a):
    return a.astype("float32")


# --------------------------------------------------------------------------
# 1. CESM2 single member sample (native curvilinear POP -> placeholder lat/lon)
# --------------------------------------------------------------------------
def build_cesm2():
    temp = xr.open_dataset(RAW["cesm2"]["TEMP"])      # TEMP (time,z_t,nlat,nlon) z_t in cm
    salt = xr.open_dataset(RAW["cesm2"]["SALT"])      # SALT (time,z_t,nlat,nlon)
    sst = xr.open_dataset(RAW["cesm2"]["SST"])        # TEMP (time,nlat,nlon)
    sss = xr.open_dataset(RAW["cesm2"]["SSS"])        # SALT (time,nlat,nlon)

    nlat, nlon = temp.sizes["nlat"], temp.sizes["nlon"]
    # placeholder regular coordinates (true grid is curvilinear POP)
    lat = np.linspace(-79.5, 89.5, nlat).astype("float32")
    lon = np.linspace(0.5, 359.5, nlon).astype("float32")
    depth = _f32(temp.z_t.values / 100.0)             # cm -> m

    TEMP = _f32(temp.TEMP.values)                     # (time,z_t,nlat,nlon)
    SALT = _f32(salt.SALT.values)
    SST = _f32(sst.TEMP.values)                       # (time,nlat,nlon)
    SSS = _f32(sss.SALT.values)
    time = temp.time.values

    ds = xr.Dataset(
        {
            "TEMP": (("time", "depth", "lat", "lon"), TEMP),
            "SALT": (("time", "depth", "lat", "lon"), SALT),
            "SST":  (("time", "lat", "lon"), SST),
            "SSS":  (("time", "lat", "lon"), SSS),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        attrs={
            "source": "CESM2 Large Ensemble – single member sample",
            "grid": "curvilinear POP (lat/lon are approximate placeholder)",
            "time_range": "2000-01 to 2009-12",
            "depth_units": "m (converted from cm)",
            "temp_units": "degC", "salt_units": "PSU",
        },
    )
    chunks = {"TEMP": (15, 4, 96, 80), "SALT": (15, 4, 96, 80),
              "SST": (30, 96, 80), "SSS": (30, 96, 80)}
    for d in (temp, salt, sst, sss):
        d.close()
    return ds, chunks


# --------------------------------------------------------------------------
# 2. WOA23 monthly climatology (12 monthly files per variable)
# --------------------------------------------------------------------------
def build_woa23():
    tfiles = sorted(glob.glob(os.path.join(RAW["woa23"]["TEMP_dir"], "*.nc")))
    sfiles = sorted(glob.glob(os.path.join(RAW["woa23"]["SALT_dir"], "*.nc")))
    assert len(tfiles) == 12 and len(sfiles) == 12, (len(tfiles), len(sfiles))

    def stack(files, var):
        months = []
        for fp in files:
            d = xr.open_dataset(fp, decode_times=False)
            months.append(_f32(d[var].isel(time=0).values))   # (depth,lat,lon)
            ref = d
        arr = np.stack(months, axis=0)                          # (12,depth,lat,lon)
        return arr, ref

    TEMP, ref = stack(tfiles, "t_an")
    SALT, _ = stack(sfiles, "s_an")
    lat = _f32(ref.lat.values); lon = _f32(ref.lon.values); depth = _f32(ref.depth.values)
    months = np.arange(1, 13, dtype="int32")

    ds = xr.Dataset(
        {
            "TEMP": (("time", "depth", "lat", "lon"), TEMP),
            "SALT": (("time", "depth", "lat", "lon"), SALT),
            "SST":  (("time", "lat", "lon"), TEMP[:, 0]),       # shallowest level (0 m)
            "SSS":  (("time", "lat", "lon"), SALT[:, 0]),
        },
        coords={"time": months, "depth": depth, "lat": lat, "lon": lon},
        attrs={
            "source": "World Ocean Atlas 2023",
            "variable": "t_an / s_an (objectively analysed climatological mean)",
            "time_meaning": "calendar month (1 = January … 12 = December)",
            "clim_period": "decav91C0 (1991–2020)",
            "depth_units": "m", "temp_units": "degC", "salt_units": "PSU",
        },
    )
    chunks = {"TEMP": (3, 15, 45, 180), "SALT": (3, 15, 45, 180),
              "SST": (6, 90, 180), "SSS": (6, 90, 180)}
    ref.close()
    return ds, chunks


# --------------------------------------------------------------------------
# 3. CESM2-LE full regridded 1deg (dask-streamed; ~95 GB at full length)
# --------------------------------------------------------------------------
def build_cesm2_le(max_time=None):
    tchunk = 189
    temp = xr.open_dataset(RAW["cesm2_le_full"]["TEMP"], chunks={"time": tchunk, "z_t": 8})
    salt = xr.open_dataset(RAW["cesm2_le_full"]["SALT"], chunks={"time": tchunk, "z_t": 8})
    if max_time:
        temp = temp.isel(time=slice(0, max_time))
        salt = salt.isel(time=slice(0, max_time))

    lat = _f32(temp.LAT.values[:, 0])                 # regular grid -> 1-D
    lon = _f32(temp.LON.values[0, :])                 # already 0-360
    depth = _f32(temp.z_t.values)                     # already m
    mask = _f32(temp.MASK.values)
    time = temp.time.values

    def prep(da):
        """Rename dims and STRIP all coords so the Dataset assigns coords
        positionally — avoids float32/float64 depth-coord misalignment that
        would NaN-fill the non-integer deep levels (165.1 m, 175.5 m, ...)."""
        da = da.rename({"z_t": "depth", "y_regr": "lat", "x_regr": "lon"}) \
            if "z_t" in da.dims else da.rename({"y_regr": "lat", "x_regr": "lon"})
        return da.drop_vars(list(da.coords), errors="ignore")

    TEMP = prep(temp.TEMP.astype("float32"))
    SALT = prep(salt.SALT.astype("float32"))
    SST = prep(temp.TEMP.isel(z_t=0).astype("float32"))   # shallowest (5 m)
    SSS = prep(salt.SALT.isel(z_t=0).astype("float32"))

    ds = xr.Dataset(
        {
            "TEMP": TEMP, "SALT": SALT, "SST": SST, "SSS": SSS,
            "MASK": (("lat", "lon"), mask),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        attrs={
            "source": "CESM2 Large Ensemble – 1°×1° regridded (jaisonk)",
            "grid": "regular 1°×1° (y_regr=180, x_regr=360)",
            "time_range": "1850-01 to 2100-12",
            "depth_units": "m", "temp_units": "degC", "salt_units": "PSU",
            "SST_note": "shallowest depth level (z_t=5 m)",
        },
    )
    nt = ds.sizes["time"]
    chunks = {"TEMP": (min(tchunk, nt), 8, 23, 45), "SALT": (min(tchunk, nt), 8, 23, 45),
              "SST": (min(377, nt), 23, 45), "SSS": (min(377, nt), 23, 45),
              "MASK": (90, 360)}
    return ds, chunks, (temp, salt)


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------
def write_zarr(ds, chunks, out_path, overwrite):
    if os.path.exists(out_path):
        if not overwrite:
            print(f"  SKIP (exists): {out_path}  [use --overwrite]")
            return False
        shutil.rmtree(out_path)
    enc = {}
    for v, ch in chunks.items():
        if v in ds:
            ds[v] = ds[v].chunk(dict(zip(ds[v].dims, ch)))
            enc[v] = {"chunks": ch}
    ds.to_zarr(out_path, mode="w", encoding=enc, zarr_format=3, consolidated=True)
    print(f"  wrote {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"],
                    choices=["all", "cesm2", "woa23", "cesm2_le_full"])
    ap.add_argument("--out", default=os.path.join(ROOT, "processed"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-time", type=int, default=None,
                    help="limit CESM2-LE full to first N months (validation)")
    args = ap.parse_args()
    sel = ["cesm2", "woa23", "cesm2_le_full"] if "all" in args.datasets else args.datasets
    os.makedirs(args.out, exist_ok=True)

    if "cesm2" in sel:
        print("[cesm2]"); ds, ch = build_cesm2()
        write_zarr(ds, ch, os.path.join(args.out, "cesm2_standard.zarr"), args.overwrite)
    if "woa23" in sel:
        print("[woa23]"); ds, ch = build_woa23()
        write_zarr(ds, ch, os.path.join(args.out, "woa23_standard.zarr"), args.overwrite)
    if "cesm2_le_full" in sel:
        print(f"[cesm2_le_full]{' (max_time=%d)' % args.max_time if args.max_time else ' (FULL ~95GB)'}")
        ds, ch, handles = build_cesm2_le(args.max_time)
        write_zarr(ds, ch, os.path.join(args.out, "cesm2_le_full_standard.zarr"), args.overwrite)
        for h in handles:
            h.close()
    print("done.")


if __name__ == "__main__":
    main()
