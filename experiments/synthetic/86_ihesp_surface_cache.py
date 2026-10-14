"""Surface inputs for the iHESP cohorts, in the layout of ``50_prepare_matched_satellites.py``.

The multi-modal arm of the OI-anchored model reads three surface fields at
every input profile and every query position. iHESP's files hold only
temperature and salinity, so, as in the CESM2 arm:

  SST, SSS    the model's 5 m temperature and salinity;
  sea level   steric height of the model's own T/S column down to 985 m
              (TEOS-10, reference 990 dbar), not an independent altimeter field.

All three are noise free and derived from the truth being reconstructed.

  --source lr   fields on the model's own 1-degree grid, read at each profile's
                model cell                         -> ihesp_lr_uniform
  --source hr   the 0.1-degree fields averaged to 1-degree boxes first, read at
                each profile's box: what a 1-degree satellite product would
                see, as in the real-data pipeline  -> ihesp_hr_uniform

The monthly climatology and the z-score statistics come from the training
years only. Writes ``outputs/radius_scale_20261008/<cohort>/satellites.npz``.

  .venv/bin/python experiments/synthetic/86_ihesp_surface_cache.py --source hr
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ocean_tokenizer.ihesp_sources import FILL, LEVELS, N_MONTHS, SOURCES, year_month

SPLITS = {"train": (2000, 2003), "validation": (2004, 2004), "development": (2005, 2005)}
N_BOXES = 180 * 360
SURFACE = ("SST", "SSS", "steric_SSH")


def one_degree_box(lat, lon):
    """Row-major index of the 1-degree box of each position (the cohort's ``grid_y``, ``grid_x``)."""
    iy = np.clip(np.floor(np.asarray(lat, dtype=float) + 90.0).astype(int), 0, 179)
    ix = np.floor(np.asarray(lon, dtype=float) % 360.0).astype(int) % 360
    return iy * 360 + ix


def box_mean(values, weight, box, n_boxes):
    """``(L, n)`` cell values -> ``(L, n_boxes)`` area-weighted means; NaN cells skipped, empty boxes NaN."""
    out = np.full((values.shape[0], n_boxes), np.nan)
    for level, v in enumerate(values):
        ok = np.isfinite(v)
        total = np.bincount(box[ok], weights=weight[ok] * v[ok], minlength=n_boxes)
        area = np.bincount(box[ok], weights=weight[ok], minlength=n_boxes)
        out[level] = np.where(area > 0, total / np.maximum(area, 1e-300), np.nan)
    return out


def anomaly_z(fields, month_of_year, train):
    """``(T, K, n)`` fields -> z-scored anomalies from the training months' monthly climatology.

    Returns ``(z, climatology (12, K, n), mean (K,), std (K,))``; the last two are
    the training-month statistics of the anomalies of each field.
    """
    month_of_year, train = np.asarray(month_of_year), np.asarray(train, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-missing cells (land) stay NaN
        clim = np.stack([np.nanmean(fields[train & (month_of_year == m)], axis=0) for m in range(12)])
        anomalies = fields - clim[month_of_year]
        mean = np.nanmean(anomalies[train], axis=(0, 2))
        std = np.nanstd(anomalies[train], axis=(0, 2))
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return (anomalies - mean[None, :, None]) / std[None, :, None], clim, mean, std


def surface_features(position, z, sst):
    """The model's 56 feature columns: 44 positional, then value and presence of each field.

    ``z`` is ``(P, 3)`` z-scored anomalies at the profiles (NaN where a field is
    undefined), ``sst`` the absolute surface temperature there.
    """
    x = np.array(position, dtype=np.float32, copy=True)
    for k in range(3):
        x[:, 44 + 2 * k] = np.nan_to_num(z[:, k])
        x[:, 45 + 2 * k] = np.isfinite(z[:, k])
    x[:, 50] = np.nan_to_num((np.asarray(sst, dtype=float) - 14.0) / 11.0)
    return x


def cached(path, compute):
    """``compute()`` once; later calls read the saved array.

    Reading the 0.1-degree files is heavy enough for the server's memory-pressure
    monitor to kill the job, so each finished month is kept and a rerun resumes.
    """
    path = Path(path)
    if path.exists():
        return np.load(path)
    value = compute()
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)
    return value


def month_fields(job):
    """``(3, cells)`` SST, SSS and steric height of one month on the grid the profiles read."""
    source_key, root, t, folder = job
    return cached(Path(folder) / f"t{t:02d}.npy", lambda: compute_month(source_key, root, t))


def compute_month(source_key, root, t):
    from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns
    source = SOURCES[source_key](root)
    level = np.asarray(source.full_level_index)[:, None, None]
    kmt = source.kmt[None]
    columns = []
    for variable in ("TEMP", "SALT"):
        v = source.field(variable, t).astype(np.float64)
        v[(np.abs(v) > FILL) | (level >= kmt) | (kmt <= 0)] = np.nan
        columns.append(v.reshape(len(LEVELS), -1))
    if source_key == "hr":
        wet = np.flatnonzero(source.kmt.ravel() > 0)
        box = one_degree_box(source.lat.ravel()[wet], source.lon.ravel()[wet])
        area = source.area.ravel()[wet].astype(np.float64)
        temp, salt = (box_mean(v[:, wet], area, box, N_BOXES) for v in columns)
        lat = np.repeat(np.arange(180) - 89.5, 360)
        lon = np.tile(np.arange(360) + 0.5, 180)
    else:
        temp, salt = columns
        lat, lon = source.lat.ravel().astype(float), source.lon.ravel().astype(float)
    full = np.isfinite(temp).all(axis=0) & np.isfinite(salt).all(axis=0)
    steric = np.full(temp.shape[1], np.nan)
    if full.any():
        steric[full] = steric_height_columns(temp[:, full], salt[:, full], LEVELS, lat[full], lon[full], P_REF_DBAR)
    return np.stack([temp[0], salt[0], steric]).astype(np.float32)


def main():
    import xarray as xr
    from ocean_tokenizer import synth_argo_eval as S
    from ocean_tokenizer.audit_tools import cohort_path
    from ocean_tokenizer.latent_experiment import array_fingerprint
    from ocean_tokenizer.synthetic_latent_experiment import position_features
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES), required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    started = time.time()
    source = SOURCES[args.source](ROOT)
    cohort = source.name
    output = args.output or ROOT / "outputs/radius_scale_20261008" / cohort / "satellites.npz"
    c, norm, obs = S.load(str(ROOT), cohort=cohort)
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    months = np.arange(N_MONTHS)
    if not np.array_equal(np.unique(c.month_index), months):
        raise SystemExit("the cohort does not hold the 72 months 2000-01 .. 2005-12")
    month_folder = output.parent / "surface_months"
    month_folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        fields = []
        for t, f in enumerate(pool.map(month_fields, [(args.source, str(ROOT), int(t), str(month_folder)) for t in months])):
            fields.append(f)
            if (t + 1) % 12 == 0:
                print(f"surface fields {t + 1}/{N_MONTHS} months ({time.time() - started:.0f}s)", flush=True)
    fields = np.stack(fields)                                  # (72, 3, cells)
    years = np.array([year_month(int(t))[0] for t in months])
    train = (years >= SPLITS["train"][0]) & (years <= SPLITS["train"][1])
    z, climate, mean, std = anomaly_z(fields.astype(np.float64), months % 12, train)

    cohort_file = Path(cohort_path(str(ROOT), "synthetic", cohort=cohort))
    with xr.open_dataset(cohort_file) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        if args.source == "hr":
            cell = one_degree_box(c.lat, c.lon)
        else:
            cell = (np.asarray(raw.model_j.values)[order] * source.lat.shape[1] + np.asarray(raw.model_i.values)[order])
        surface_truth = {ch: np.asarray(raw[ch].values)[order, 0] for ch in ("TEMP", "SALT")}
    at_profiles = z[c.month_index, :, cell]                    # (P, 3)
    absolute = fields[c.month_index, :, cell]
    features = surface_features(position_features(c), at_profiles, absolute[:, 0])
    if not np.isfinite(features).all():
        raise SystemExit("non-finite surface features")
    agreement = {}
    for k, ch in enumerate(("TEMP", "SALT")):
        ok = np.isfinite(absolute[:, k]) & np.isfinite(surface_truth[ch])
        agreement[ch] = {"correlation_with_profile_5m": float(np.corrcoef(absolute[ok, k], surface_truth[ch][ok])[0, 1]),
                         "rms_difference": float(np.sqrt(np.mean((absolute[ok, k] - surface_truth[ch][ok]) ** 2)))}
        # on the model's own grid the surface field IS the profile's 5 m value; a box mean only tracks it
        if args.source == "lr" and not np.allclose(absolute[ok, k], surface_truth[ch][ok], atol=2e-4, rtol=2e-5):
            raise SystemExit(f"{ch}: the surface field differs from the cohort's 5 m values")
        if agreement[ch]["correlation_with_profile_5m"] < 0.9:
            raise SystemExit(f"{ch}: the surface field does not track the cohort's 5 m values")
    present = {name: float(np.isfinite(at_profiles[:, k]).mean()) for k, name in enumerate(SURFACE)}
    metadata = {
        "cohort_fingerprint": fingerprint,
        "normalization_scope": "training years only",
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "recipe": "noise-free model TEMP/SALT at 5 m and TEOS-10 steric height of the 20-level column to 985 m",
        "resolution": ("0.1-degree fields averaged to 1-degree boxes (area-weighted), read at the profile's box"
                       if args.source == "hr" else "the model's own 1-degree grid, read at the profile's model cell"),
        "surface_variables": list(SURFACE), "surface_feature_offsets": [44, 46, 48],
        "surface_is_independent_of_truth": False, "satellite_noise": "none",
        "share_of_profiles_with_field": present, "agreement_with_profile_surface": agreement,
        "source": source.describe,
        "feature_fingerprint": array_fingerprint((features,)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
                        metadata=json.dumps(metadata), surface_climatology=climate.astype(np.float32),
                        surface_mean=mean, surface_std=std)
    os.replace(temporary, output)
    print(json.dumps({"cache": str(output), "profiles": int(len(c.lat)), "present": present,
                      "agreement": agreement, "anomaly_std": dict(zip(SURFACE, std.tolist())),
                      "runtime_s": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
