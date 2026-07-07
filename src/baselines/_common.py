"""Shared utilities for the three *reference* baselines.

These baselines (NeSPReSO PCA+MLP, OSnet-style MLP, Buongiorno-Nardelli stacked
LSTM) all reuse the EXISTING ocean_tokenizer pipeline so that the train/test
month split, the synthetic Argo profiles, the depth grid, the ocean mask and the
WOA23 prior are *bit-for-bit identical* to the baselines already reported in
``reports/baseline_table.md``.  This module centralises everything they share:

* reproducing the exact split + per-month synthetic Argo samples
* turning per-month samples into a flat per-profile training table
  (surface SST/SSS extracted AT the profile location -> no subsurface leakage)
* gap-filling profiles that hit bottom topography (NaN below the seafloor)
* running a per-profile model over every ocean cell of each test month to
  reconstruct a full (D,H,W) field comparable with the existing baselines
* depth-banded, valid-cell-weighted RMSE (identical recipe to 05_band_table.py)
* checkpoint / prediction / CSV IO under the paths the task asks for

Paradigm note
-------------
NeSPReSO / OSnet / Nardelli are *function-fitting* reconstructors: they map
(surface SST/SSS + location + time) -> (subsurface T/S profile).  Surface fields
are treated as DENSE observations (available on every ocean cell), while the
sparse Argo profiles are the training labels.  At test time we therefore feed the
dense surface field of each held-out month and predict the subsurface everywhere,
then score NaN-aware against the CESM2-LE ground truth.  The held-out subsurface
truth is never used as an input -> no data leakage.
"""
from __future__ import annotations
import os
import sys
import csv

import numpy as np

# make the ocean_tokenizer package importable regardless of CWD
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ocean_tokenizer import data, baselines as B, config as C, metrics  # noqa: E402

VARS = ("TEMP", "SALT")

# --------------------------------------------------------------------------
# Output locations (task: checkpoints/ and predictions/ under the project root)
# --------------------------------------------------------------------------
ROOT = C.ROOT
CKPT_DIR = os.path.join(ROOT, "checkpoints")
PRED_DIR = os.path.join(ROOT, "predictions")
CACHE = C.CACHE          # outputs/cache  (per-depth RMSE for the combined table)
REPORTS = C.REPORTS
for _d in (CKPT_DIR, PRED_DIR, CACHE, REPORTS):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------
# Time encoding: monthly snapshots -> representative day-of-year (15th of month)
# --------------------------------------------------------------------------
# The papers parameterise the seasonal cycle with day-of-year; our data are
# monthly means, so we map each calendar month to the day-of-year of its 15th.
_MID_MONTH_DOY = np.array([15, 46, 74, 105, 135, 166,
                           196, 227, 258, 288, 319, 349], dtype="float32")


def month_to_doy(month):
    """month: int/array in 1..12 -> day-of-year of the 15th (float32)."""
    return _MID_MONTH_DOY[np.asarray(month, dtype=int) - 1]


def doy_sincos(month):
    doy = month_to_doy(month)
    a = 2.0 * np.pi * doy / 365.0
    return np.sin(a).astype("float32"), np.cos(a).astype("float32")


# --------------------------------------------------------------------------
# Reproduce the EXACT split + synthetic Argo samples used by 03_baselines.py
# --------------------------------------------------------------------------
def build_split(smoke: bool = False) -> dict:
    """Rebuild grid, train/test months and per-month samples deterministically.

    The RNG call order mirrors experiments/03_baselines.py exactly
    (month draws -> train-month profile draws -> test-month profile draws), so
    the synthetic Argo profiles are identical to the existing baselines.
    """
    if smoke:
        C.N_TRAIN_MONTHS, C.N_TEST_MONTHS = 6, 3
        C.N_PROFILES = 600

    rng = np.random.default_rng(C.SEED)
    grid = data.CommonGrid()

    tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
    te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
    tr_idx = np.sort(rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size),
                                replace=False))
    te_idx = np.sort(rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size),
                                replace=False))

    ftrain = data.load_gt_fields(tr_idx, grid)
    ftest = data.load_gt_fields(te_idx, grid)
    woa = data.woa_prior(grid)
    norm = B.Norm(ftrain, ftrain)

    # same order as make_samples(ftrain) then make_samples(ftest)
    train_samples = [B.prepare_month(ftrain, ftrain, woa, grid, t, rng, C.N_PROFILES)
                     for t in range(len(ftrain["months"]))]
    test_samples = [B.prepare_month(ftest, ftest, woa, grid, t, rng, C.N_PROFILES)
                    for t in range(len(ftest["months"]))]

    return dict(grid=grid, ftrain=ftrain, ftest=ftest, woa=woa, norm=norm,
                train_samples=train_samples, test_samples=test_samples,
                tr_idx=tr_idx, te_idx=te_idx, rng=rng)


# --------------------------------------------------------------------------
# Per-profile training table
# --------------------------------------------------------------------------
def extract_profiles(samples, grid) -> dict:
    """Flatten per-month samples into a single per-profile table.

    Surface SST/SSS are read AT the profile grid cell (dense surface obs); the
    held-out subsurface target is never used as an input.

    Returns arrays with N = total profiles across the given samples:
        lat, lon, month, doy, sin_doy, cos_doy, sst, sss : (N,)
        i, j       : (N,)     grid indices (for looking up auxiliary fields)
        TEMP, SALT : (N, D)   raw profiles (NaN where below seafloor)
    """
    cols = {k: [] for k in ("lat", "lon", "month", "sst", "sss", "TEMP", "SALT",
                            "i", "j")}
    for s in samples:
        p = s["prof"]
        i, j = p["ij"][:, 0], p["ij"][:, 1]
        cols["i"].append(i.astype(int))
        cols["j"].append(j.astype(int))
        cols["lat"].append(p["lat"].astype("float32"))
        cols["lon"].append(p["lon"].astype("float32"))
        cols["month"].append(p["month"].astype(int))
        cols["sst"].append(s["surf"]["SST"][i, j].astype("float32"))
        cols["sss"].append(s["surf"]["SSS"][i, j].astype("float32"))
        cols["TEMP"].append(p["TEMP"].astype("float32"))
        cols["SALT"].append(p["SALT"].astype("float32"))
    out = {k: (np.concatenate(v, axis=0) if v[0].ndim == 1
               else np.concatenate(v, axis=0)) for k, v in cols.items()}
    out["doy"] = month_to_doy(out["month"])
    out["sin_doy"], out["cos_doy"] = doy_sincos(out["month"])
    return out


def fill_profile_matrix(P):
    """Fill NaNs in (N,D) profiles so PCA / dense targets see complete columns.

    Strategy (documented departure forced by CESM2 bottom topography): carry the
    last valid value downward (below-seafloor extension), then patch any leftover
    gaps with the per-depth mean.  Returns ``(filled (N,D), finite_mask (N,D))``;
    the mask flags ORIGINALLY-finite cells so profile-space losses can ignore the
    fabricated below-seafloor values.
    """
    P = np.array(P, dtype="float32", copy=True)
    mask = np.isfinite(P)
    N, D = P.shape
    for d in range(1, D):                       # downward (deepening) fill
        bad = ~np.isfinite(P[:, d])
        P[bad, d] = P[bad, d - 1]
    # leftover (e.g. a leading-NaN surface cell, or an all-land column): depth mean
    dmean = np.nanmean(np.where(mask, P, np.nan), axis=0)
    dmean = np.where(np.isfinite(dmean), dmean, 0.0).astype("float32")
    bad = ~np.isfinite(P)
    if bad.any():
        P[bad] = np.take(dmean, np.where(bad)[1])
    return P, mask


class DepthStandardizer:
    """Per-depth z-scoring of (N,D) profiles, fit on filled TRAINING profiles."""
    def __init__(self, profiles):
        filled, _ = fill_profile_matrix(profiles)
        self.mean = filled.mean(0).astype("float32")        # (D,)
        sd = filled.std(0).astype("float32")
        self.std = np.where(sd < 1e-6, 1.0, sd).astype("float32")

    def z(self, x):    return (x - self.mean) / self.std
    def unz(self, x):  return x * self.std + self.mean


def bathy_field(ftrain, grid):
    """Approximate bathymetry: deepest CESM2 level (m) with valid training data.

    Derived purely from where the model TEMP becomes NaN below the seafloor
    (topography / geometry), NOT from target values -> usable as an input
    everywhere without leakage.  Land cells -> NaN.
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)   # all-land columns
        clim = np.nanmean(ftrain["TEMP"], axis=0)            # (D,H,W); land -> NaN
    finite = np.isfinite(clim)
    # deepest finite level index per cell (-1 where none)
    depth_m = grid.depth.astype("float32")
    idx = np.where(finite.any(0),
                   (finite * np.arange(grid.ndepth)[:, None, None]).max(0), -1)
    bf = np.where(idx >= 0, depth_m[np.clip(idx, 0, grid.ndepth - 1)], np.nan)
    return bf.astype("float32")                              # (H,W)


def ocean_cell_feats(sample, grid):
    """Per-cell feature dict for every ocean cell of one (test) month sample.

    Provides the same surface/location/time quantities as ``extract_profiles``
    plus the grid indices (i, j) and the calendar month so a model closure can
    look up auxiliary fields (e.g. the WOA prior) at those cells.
    """
    oi, oj = sample["ocean_ij"]
    mo = int(sample["month"])
    sdoy, cdoy = doy_sincos(np.full(oi.size, mo))
    return dict(
        i=oi, j=oj, month=mo,
        lat=grid.lat[oi].astype("float32"),
        lon=grid.lon[oj].astype("float32"),
        doy=month_to_doy(np.full(oi.size, mo)),
        sin_doy=sdoy, cos_doy=cdoy,
        sst=sample["surf"]["SST"][oi, oj].astype("float32"),
        sss=sample["surf"]["SSS"][oi, oj].astype("float32"),
    )


# --------------------------------------------------------------------------
# Reconstruct full (D,H,W) test fields from a per-profile predictor
# --------------------------------------------------------------------------
def predict_test_fields(test_samples, grid, predict_cells, want_std=False):
    """Run ``predict_cells`` over every ocean cell of each test month.

    ``predict_cells(feats)`` receives the dict from :func:`ocean_cell_feats` and
    must return ``{"TEMP": (M,D), "SALT": (M,D)}`` (optionally ``*_std``).
    Returns ``preds`` (list of {var:(D,H,W)}) and ``stds`` (same, or None).
    """
    preds, stds = [], ([] if want_std else None)
    for s in test_samples:
        feats = ocean_cell_feats(s, grid)
        res = predict_cells(feats)
        oi, oj = feats["i"], feats["j"]
        pv, sv = {}, {}
        for v in VARS:
            arr = np.full((grid.ndepth, grid.nlat, grid.nlon), np.nan, "float32")
            arr[:, oi, oj] = res[v].T
            pv[v] = np.where(grid.ocean[None], arr, np.nan).astype("float32")
            if want_std:
                a2 = np.full((grid.ndepth, grid.nlat, grid.nlon), np.nan, "float32")
                a2[:, oi, oj] = res[f"{v}_std"].T
                sv[v] = np.where(grid.ocean[None], a2, np.nan).astype("float32")
        preds.append(pv)
        if want_std:
            stds.append(sv)
    return preds, stds


def true_stack(test_samples):
    return {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in VARS}


# --------------------------------------------------------------------------
# Depth-banded, valid-cell-weighted RMSE  (matches experiments/05_band_table.py)
# --------------------------------------------------------------------------
def _band_defs(depths):
    return [("surface_~5m", lambda d: np.isclose(d, depths[0])),
            ("0-50m", lambda d: d <= 50),
            ("50-200m", lambda d: (d > 50) & (d <= 200)),
            ("200m+", lambda d: d > 200)]


def per_depth_rmse(pred_stack, true_stack_):
    return metrics.rmse_by_depth(pred_stack, true_stack_)        # (D,)


def valid_weights(true_stack_):
    return np.isfinite(true_stack_).sum(axis=(0, 2, 3)).astype(float)  # (D,)


def band_rmse(rmse_d, w_d, depths):
    out = {}
    for name, sel in _band_defs(depths):
        m = sel(depths) & np.isfinite(rmse_d) & (w_d > 0)
        out[name] = (float(np.sqrt(np.sum(w_d[m] * rmse_d[m] ** 2) / np.sum(w_d[m])))
                     if m.any() else float("nan"))
    return out


def evaluate_fields(preds, test_samples, grid):
    """Return per-depth RMSE and depth-banded RMSE for TEMP and SALT."""
    TRUE = true_stack(test_samples)
    PRED = {v: np.stack([p[v] for p in preds], 0) for v in VARS}
    rmse_d, bands = {}, {}
    for v in VARS:
        w = valid_weights(TRUE[v])
        rd = per_depth_rmse(PRED[v], TRUE[v])
        rmse_d[v] = rd
        bands[v] = band_rmse(rd, w, grid.depth)
    return rmse_d, bands


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def print_band_table(name, bands):
    band_names = list(next(iter(bands.values())).keys())
    print(f"\n  RMSE by depth band -- {name}")
    print("    {:6s} | {}".format("var", " | ".join(f"{b:>11s}" for b in band_names)))
    for v in VARS:
        cells = " | ".join(f"{bands[v][b]:>11.4f}" for b in band_names)
        print(f"    {v:6s} | {cells}")


def write_band_csv(path, method_label, bands):
    band_names = list(next(iter(bands.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variable"] + band_names)
        for v in VARS:
            w.writerow([method_label, v] + [f"{bands[v][b]:.6f}" for b in band_names])
    print(f"  wrote {path}")


def save_predictions(name, preds, grid, te_idx, stds=None):
    payload = dict(depths=grid.depth.astype("float32"),
                   te_idx=np.asarray(te_idx))
    for v in VARS:
        payload[v] = np.stack([p[v] for p in preds], 0).astype("float32")
    if stds is not None:
        for v in VARS:
            payload[f"{v}_std"] = np.stack([s[v] for s in stds], 0).astype("float32")
    path = os.path.join(PRED_DIR, f"{name}.npz")
    np.savez_compressed(path, **payload)
    print(f"  wrote {path}")


def save_depth_rmse(name, rmse_d, grid):
    """Persist per-depth RMSE so build_comparison_table.py can re-band it."""
    path = os.path.join(CACHE, f"ref_{name}_depth.npz")
    np.savez(path, depths=grid.depth.astype("float32"),
             TEMP=np.asarray(rmse_d["TEMP"], "float64"),
             SALT=np.asarray(rmse_d["SALT"], "float64"))
    print(f"  wrote {path}")
