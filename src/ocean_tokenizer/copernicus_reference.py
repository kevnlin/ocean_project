"""Operational Copernicus analyses (GLORYS12, ARMOR3D) read at the real held-out floats.

Both products assimilate every Argo profile, including the held-out floats the paper scores on,
so they are **reference analyses, never fair baselines**: they have seen the answer. The adapter
asks the only question that puts them on the paper's footing: what does the product say at the
exact place, depth and month a held-out float measured, in the paper's own anomaly and units?

Recipe (the paper's target, ``real_global.load_real``, step by step)

* the product's monthly field is read at the float's position: bilinear in latitude and
  longitude on the product's own grid (periodic in longitude), then linear in depth onto the
  cohort's 20 standard levels, never extrapolated; a missing corner or a level below the
  product's sea floor leaves the cell missing (``reference_adapters.GriddedReference``);
* the anomaly subtracts the same WOA23 climatology at the profile's position and month that the
  float's target anomaly subtracts (``audit_tools.exact_position_anomaly``; taken as the
  measured value minus the cohort's anomaly, so it is the stored climatology to the bit);
* the anomaly is z-scored with the same per-level 2016-2020 mean and spread as the target
  (``ArgoNorm.fit(c, "train")``), and the error is scored in physical units exactly as
  ``sealed_scoring.metrics`` does: ``(z_product - z_target) * std[level]``.

Algebraically the climatology and the normalisation cancel in the error, so the RMSE is that of
the product's absolute value against the float's measured value; the recipe is kept explicit so
the cells (the float's QC in z-space), the depth bands and the arithmetic are the paper's.

Downloads go through ``copernicusmarine.subset`` one global month at a time (0-1100 m, both
variables, ~1.2 GB for GLORYS12, ~0.8 GB for ARMOR3D) and are deleted once sampled.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .point_baselines import BANDS, CH, band_of_levels
from .reconstruction_metrics import regression_metrics

#: product -> Copernicus Marine dataset ids and variable names; ``nrt`` is the near-real-time
#: continuation used only for a month the multi-year dataset does not hold
PRODUCTS = {
    "glorys12": {"product_id": "GLOBAL_MULTIYEAR_PHY_001_030",
                 "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
                 "variables": {"TEMP": "thetao", "SALT": "so"},
                 "description": "Mercator Ocean GLORYS12V1 reanalysis, 1/12 deg, 50 levels, monthly means"},
    "armor3d": {"product_id": "MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012",
                "dataset_id": "cmems_obs-mob_glo_phy_my_0.125deg_P1M-m",
                "nrt_dataset_id": "cmems_obs-mob_glo_phy_nrt_0.125deg_P1M-m",
                "my_last_month": "2024-12",          # the multi-year dataset ends here (describe, 2026-10-21)
                "variables": {"TEMP": "to", "SALT": "so"},
                "description": "CLS ARMOR3D multi-observation analysis (REP), 1/8 deg, 50 levels, monthly means"},
}
MAX_DEPTH_M = 1100.0          # the deepest cohort level is 984.7 m; both products have a level at 1062-1100 m


def month_label(month_index: int) -> str:
    return str(np.datetime64("2000-01", "M") + int(month_index))


def dataset_for(product: str, month_index: int) -> str:
    """The dataset id that holds ``month_index`` (the NRT continuation past the multi-year end)."""
    spec = PRODUCTS[product]
    if "nrt_dataset_id" in spec and np.datetime64(month_label(month_index), "M") > np.datetime64(spec["my_last_month"], "M"):
        return spec["nrt_dataset_id"]
    return spec["dataset_id"]


def fetch_month(product: str, month_index: int, out_dir, max_depth=MAX_DEPTH_M) -> dict:
    """Download one global month of T and S to 0-``max_depth`` m; returns the file and sizes."""
    import copernicusmarine as cm
    spec = PRODUCTS[product]
    label = month_label(month_index)
    first = np.datetime64(label, "M").astype("datetime64[D]")
    last = (np.datetime64(label, "M") + 1).astype("datetime64[D]") - 1          # the month's last day
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{product}_{label}.nc"
    started = time.time()
    r = cm.subset(dataset_id=dataset_for(product, month_index), variables=list(spec["variables"].values()),
                  start_datetime=str(first), end_datetime=str(last), minimum_depth=0, maximum_depth=max_depth,
                  output_directory=str(out_dir), output_filename=name, overwrite=True, disable_progress_bar=True)
    path = out_dir / name
    return {"month": label, "dataset_id": dataset_for(product, month_index), "file": str(path),
            "file_bytes": int(os.path.getsize(path)), "transfer_mb": float(r.data_transfer_size or 0.),
            "seconds": time.time() - started}


class MonthField:
    """One downloaded month, queryable at float positions: ``(T, S)`` columns on ``levels``."""

    def __init__(self, path, product: str):
        import xarray as xr
        spec = PRODUCTS[product]
        with xr.open_dataset(path) as ds:
            if ds.sizes["time"] != 1:
                raise ValueError(f"{path}: expected one month, found {ds.sizes['time']}")
            self.time = str(np.asarray(ds["time"].values, "datetime64[M]")[0])
            self.lat = np.asarray(ds["latitude"].values, np.float64)
            self.lon = np.asarray(ds["longitude"].values, np.float64)
            self.depth = np.abs(np.asarray(ds["depth"].values, np.float64))
            self.fields = {ch: np.asarray(ds[var].isel(time=0).values, np.float32) for ch, var in spec["variables"].items()}
        for a in (self.lat, self.lon, self.depth):
            if not np.all(np.diff(a) > 0):
                raise ValueError(f"{path}: coordinates are not increasing")
        self.dlat, self.dlon = float(np.mean(np.diff(self.lat))), float(np.mean(np.diff(self.lon)))
        if abs(len(self.lon) * self.dlon - 360.) > 1e-3:
            raise ValueError(f"{path}: the longitude grid does not span the globe")

    def columns(self, lat, lon, levels) -> dict:
        """``{channel: (R, L)}`` bilinear in lat/lon (periodic in lon), linear in depth, no extrapolation;
        any missing corner leaves that depth missing, and a level outside the finite part of the
        column is missing."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        levels = np.asarray(levels, np.float64)
        R, L = lat.size, levels.size
        fy = (lat - self.lat[0]) / self.dlat
        fx = ((lon - self.lon[0]) % 360.) / self.dlon
        inside = (fy >= 0) & (fy <= len(self.lat) - 1)
        i0 = np.clip(np.floor(fy).astype(int), 0, len(self.lat) - 2)
        i1 = i0 + 1
        wy = np.clip(fy - i0, 0., 1.)
        j0 = np.floor(fx).astype(int) % len(self.lon)
        j1 = (j0 + 1) % len(self.lon)
        wx = fx - np.floor(fx)
        out = {}
        for ch, vol in self.fields.items():
            col = (vol[:, i0, j0] * ((1 - wy) * (1 - wx)) + vol[:, i0, j1] * ((1 - wy) * wx)
                   + vol[:, i1, j0] * (wy * (1 - wx)) + vol[:, i1, j1] * (wy * wx)).T.astype(np.float64)   # (R, D)
            col[~inside] = np.nan
            res = np.full((R, L), np.nan)
            for r in range(R):
                ok = np.isfinite(col[r])
                if ok.sum() < 2:
                    continue
                d, v = self.depth[ok], col[r, ok]
                res[r] = np.interp(levels, d, v, left=np.nan, right=np.nan)
                res[r, (levels < d[0]) | (levels > d[-1])] = np.nan
            out[ch] = res
        return out


def score_cells(pred_abs, climate, target_z, norm, level, levels_m, bands=BANDS) -> dict:
    """The paper's physical metrics of a product on held-out cells, per channel.

    ``pred_abs``/``climate``/``target_z``: ``{channel: (N,)}`` the product's absolute value, the
    WOA23 climatology at the cell (measured minus the cohort anomaly) and the target z-score;
    ``level``: ``(N,)`` level index. For each channel: the product on the cells where it is
    finite (``product``), the climatology (a zero anomaly) on those same cells
    (``climatology_same_cells``) and on every finite-target cell (``climatology_all_cells``),
    and the counts of finite-target cells and of the cells dropped because the product is
    missing there."""
    level = np.asarray(level, int)
    band = band_of_levels(np.asarray(levels_m), bands)[level]
    out = {}
    for ch in CH:
        std = np.asarray(norm.std[ch], np.float64)[level]
        mean = np.asarray(norm.mean[ch], np.float64)[level]
        tz = np.asarray(target_z[ch], np.float64)
        anomaly = np.asarray(pred_abs[ch], np.float64) - np.asarray(climate[ch], np.float64)
        pz = (anomaly - mean) / std
        target = tz * std                      # physical, as sealed_scoring.metrics
        pred = pz * std
        finite_target = np.isfinite(target)
        keep = finite_target & np.isfinite(pred)

        def block(p, t, k):
            b = regression_metrics(p[k], t[k])
            b["by_band"] = {name: regression_metrics(p[k & (band == name)], t[k & (band == name)])
                            for name, _, _ in bands}
            b["by_level"] = [float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if (m := k & (level == i)).any() else None
                             for i in range(len(levels_m))]
            return b

        out[ch] = {"product": block(pred, target, keep),
                   "climatology_same_cells": block(np.zeros_like(target), target, keep),
                   "climatology_all_cells": block(np.zeros_like(target), target, finite_target),
                   "cells_finite_target": int(finite_target.sum()),
                   "cells_dropped_product_missing": int((finite_target & ~np.isfinite(pred)).sum()),
                   "dropped_by_band": {name: int((finite_target & ~np.isfinite(pred) & (band == name)).sum())
                                       for name, _, _ in bands}}
    return out
