"""The synthetic CESM2 Argo cohort extended back to 1964: 504 months, 3.8 M profiles.

``data/synthetic_argo/cesm2_long_uniform.nc`` is ``cesm2_uniform.nc`` (2000-01 ...
2005-12, from ``41_synth_argo_cohort.py``) with 1964-01 ... 1999-12 added in
front, so a model can be trained on up to 40 years (1964-2003) and still be
selected on 2004 and scored on 2005:

* the 547,200 profiles of 2000-2005 are copied unchanged, every variable and
  their order, so the validation and test cells, the OI anchor and every
  normalisation are the ones all earlier CESM2 results used;
* each earlier month gets 7,600 new profiles drawn the way 41 draws them
  (area-uniform ocean positions whose bilinear stencil is wet at 5 m, 20 % of
  them queries), CESM2 TEMP/SALT from ``53_cesm2_long_levels.py`` bilinearly
  interpolated to each, with the regridding fill removed first;
* the climatology every anomaly is taken against stays the 2000-2003 one, the
  same array 41 subtracts, so only the number of training months changes
  between the training windows. Older months therefore carry the warming
  trend as part of their anomaly, as real data would against a recent
  climatology.

Month indices count from 2000-01 and are negative before 2000; their year split
is "unassigned", like any year outside the audit's splits. Float ids of the new
profiles start with "L" (41's start with "S").

  .venv/bin/python experiments/synthetic/87_cesm2_long_cohort.py     # a few minutes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "data/synthetic_argo/cesm2_uniform.nc"
LEVELS_STORE = ROOT / "data/cesm2_long/cesm2_le_cohort_levels.zarr"
OUT = ROOT / "data/synthetic_argo/cesm2_long_uniform.nc"
SEED = 20261017
N_PER_MONTH = 7600
QUERY_FRAC = 0.2
TRAIN = (2000, 2003)
ARRAYS = ("TEMP", "SALT", "TEMP_RAW", "SALT_RAW", "CLIM_POS_TEMP", "CLIM_POS_SALT",
          "CLIM_CELL_TEMP", "CLIM_CELL_SALT")


def stencil(lat, lon, lat_c, lon_c):
    """41's bilinear stencil on the cell-centre grid: indices and weights."""
    ny, nx = lat_c.size, lon_c.size
    fy = lat - lat_c[0]
    fx = (lon - lon_c[0]) % 360.0
    i0 = np.floor(fy).astype(int)
    j0 = np.floor(fx).astype(int) % nx
    wy, wx = fy - i0, fx - np.floor(fx)
    i1, j1 = np.minimum(i0 + 1, ny - 1), (j0 + 1) % nx
    idx = [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]
    w = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]
    return idx, w


def interp(field, idx, w):
    """field (L, NY, NX) -> (P, L); NaN where any stencil corner is dry."""
    out = 0.0
    for (i, j), wk in zip(idx, w):
        out = out + field[:, i, j].T * wk[:, None]
    return out


def draw_month(t, raw, clean, clim, wet0, lat_c, lon_c):
    """One month's profiles, as 41 draws them, from its own seeded generator."""
    rng = np.random.default_rng([SEED, t + 10_000])
    lat, lon = np.empty(0), np.empty(0)
    while lat.size < N_PER_MONTH:
        la = np.rad2deg(np.arcsin(rng.uniform(-1, 1, 4 * N_PER_MONTH)))
        lo = rng.uniform(0.0, 360.0, la.size)
        ok = np.abs(la) <= lat_c[-1]
        la, lo = la[ok], lo[ok]
        idx, _ = stencil(la, lo, lat_c, lon_c)
        wet = np.logical_and.reduce([wet0[i, j] for i, j in idx])
        lat, lon = np.r_[lat, la[wet]], np.r_[lon, lo[wet]]
    lat, lon = lat[:N_PER_MONTH], lon[:N_PER_MONTH]
    idx, w = stencil(lat, lon, lat_c, lon_c)
    iy = np.floor(lat + 90.0).astype(int)
    ix = np.floor(lon).astype(int) % lon_c.size
    cal = t % 12
    arrays = {}
    for v in ("TEMP", "SALT"):
        arrays[v] = interp(clean[v], idx, w)
        arrays[f"{v}_RAW"] = interp(raw[v], idx, w)
        arrays[f"CLIM_POS_{v}"] = interp(clim[v][cal], idx, w)
        cc = clim[v][cal][:, iy, ix].T
        arrays[f"CLIM_CELL_{v}"] = np.where(np.isfinite(arrays[v]), cc, np.nan)
    query = np.zeros(N_PER_MONTH, bool)
    query[rng.choice(N_PER_MONTH, int(round(QUERY_FRAC * N_PER_MONTH)), replace=False)] = True
    columns = {"lat": lat, "lon": lon, "month_index": np.full(N_PER_MONTH, t),
               "grid_y": iy, "grid_x": ix,
               "wmo": np.array([f"L{t + 432:03d}{i:04d}" for i in range(N_PER_MONTH)]),
               "float_split": np.where(query, "heldout_float", "cohort_float")}
    return columns, {k: a.astype("float32") for k, a in arrays.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first-year", type=int, default=1964)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    started = time.time()
    store = xr.open_zarr(LEVELS_STORE, chunks=None)
    months = np.asarray(store.month_index.values, dtype=int)
    lat_c, lon_c = store.lat.values.astype(float), store.lon.values.astype(float)
    short = xr.open_dataset(SHORT)
    if not np.allclose(store.level.values, short.level.values):
        raise SystemExit("store and cohort levels differ")

    def month(t):
        i = int(np.searchsorted(months, t))
        if months[i] != t:
            raise SystemExit(f"month {t} not in {LEVELS_STORE.name}")
        raw = {v: np.asarray(store[v][i].values, dtype="float32") for v in ("TEMP", "SALT")}
        fill = (raw["TEMP"] == 0) & (raw["SALT"] == 0)
        return raw, {v: np.where(fill, np.nan, a) for v, a in raw.items()}

    # 41's climatology: float32 nan-mean of the cleaned 2000-2003 months, by calendar month
    train = [t for t in range(72) if TRAIN[0] <= 2000 + t // 12 <= TRAIN[1]]
    clean_train = {v: np.stack([month(t)[1][v] for t in train]) for v in ("TEMP", "SALT")}
    cal_train = np.array(train) % 12
    clim = {v: np.stack([np.nanmean(a[cal_train == m], axis=0) for m in range(12)]).astype("float32")
            for v, a in clean_train.items()}
    del clean_train
    wet0 = np.isfinite(month(0)[1]["TEMP"][0])

    # the copied months must come out of this climatology and stencil unchanged
    check = np.sort(np.random.default_rng(0).choice(short.sizes["profile"], 20_000, replace=False))
    lat_k, lon_k = short.lat.values[check], short.lon.values[check]
    cal = np.asarray(short.month_index.values)[check] % 12
    for v in ("TEMP", "SALT"):
        rebuilt = np.full((check.size, short.level.size), np.nan, "float32")
        for m in range(12):
            sel = cal == m
            idx, w = stencil(lat_k[sel], lon_k[sel], lat_c, lon_c)
            rebuilt[sel] = interp(clim[v][m], idx, w)
        stored = np.asarray(short[f"CLIM_POS_{v}"].values)[check]
        if not np.array_equal(rebuilt, stored, equal_nan=True):
            raise SystemExit(f"CLIM_POS_{v} of the existing cohort is not reproduced: largest difference "
                             f"{np.nanmax(np.abs(rebuilt - stored)):.3g}")
    print(f"climatology and stencil reproduce cesm2_uniform.nc exactly ({time.time() - started:.0f}s)",
          flush=True)

    first = (args.first_year - 2000) * 12
    columns = {k: [] for k in ("lat", "lon", "month_index", "grid_y", "grid_x", "wmo", "float_split")}
    arrays = {k: [] for k in ARRAYS}
    for t in range(first, 0):
        raw, clean = month(t)
        col, arr = draw_month(t, raw, clean, clim, wet0, lat_c, lon_c)
        for k, v in col.items():
            columns[k].append(v)
        for k, v in arr.items():
            arrays[k].append(v)
        if (t - first + 1) % 60 == 0:
            print(f"drew {t - first + 1}/{-first} earlier months ({time.time() - started:.0f}s)", flush=True)
    store.close()

    new = {k: np.concatenate(v) for k, v in columns.items()}
    new.update({k: np.concatenate(v) for k, v in arrays.items()})
    n_new, n_levels = new["lat"].size, short.level.size
    year = 2000 + new["month_index"] // 12
    old = {k: np.asarray(short[k].values) for k in ("lat", "lon", "month_index", "grid_y", "grid_x",
                                                     "wmo", "float_split", "year", "year_split", *ARRAYS,
                                                     "TEMP_ERR", "SALT_ERR")}

    def joined(name, fresh):
        return np.concatenate([fresh.astype(old[name].dtype) if fresh.dtype.kind != "U" else fresh,
                               old[name]])

    profile_vars = {}
    for name in ("lat", "lon", "month_index", "grid_y", "grid_x"):
        profile_vars[name] = ("profile", joined(name, new[name]))
    profile_vars["year"] = ("profile", joined("year", year))
    profile_vars["wmo"] = ("profile", np.concatenate([new["wmo"], old["wmo"].astype(str)]))
    profile_vars["year_split"] = ("profile", np.concatenate([np.full(n_new, "unassigned"),
                                                             old["year_split"].astype(str)]))
    profile_vars["float_split"] = ("profile", np.concatenate([new["float_split"],
                                                              old["float_split"].astype(str)]))
    level_vars = {k: (("profile", "level"), np.concatenate([new[k], old[k]])) for k in ARRAYS}
    for k in ("TEMP_ERR", "SALT_ERR"):
        level_vars[k] = (("profile", "level"),
                         np.concatenate([np.full((n_new, n_levels), np.nan, "float32"), old[k]]))
    attrs = dict(short.attrs)
    attrs.update(source=f"{SHORT.name} for 2000-2005 (copied unchanged); {LEVELS_STORE.relative_to(ROOT)} "
                        "bilinear at uniform random ocean positions for earlier months",
                 extension=json.dumps({"first_month": f"{args.first_year}-01", "seed": SEED,
                                       "per_month_generator": "numpy default_rng([seed, month_index + 10000])",
                                       "climatology": "2000-2003, identical to cesm2_uniform.nc",
                                       "script": "experiments/synthetic/87_cesm2_long_cohort.py"}))
    out = xr.Dataset({**profile_vars, **level_vars}, coords={"level": short.level.values}, attrs=attrs)
    short.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    out.to_netcdf(temporary)
    os.replace(temporary, args.output)
    finite = np.isfinite(new["SALT"]).mean(axis=0)
    print(f"{out.sizes['profile']:,} profiles ({n_new:,} new over {-first} months); new finite fraction "
          f"by level {finite.min():.3f}-{finite.max():.3f}\n-> {args.output} ({time.time() - started:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
