"""Surface inputs of the sealed iHESP test cohort (``ihesp_hr_test``), by 86's recipe.

For every test month (1997-07 ... 1999-12) the 0.1-degree SST, SSS and steric height of
``data/ihesp/hr_test_levels`` are averaged to 1-degree boxes as ``86_ihesp_surface_cache.py`` does
(``outputs/ihesp_20261021/test_surface_months/t<month>.npy``). The cohort's 56 profile features
are then 86's position features and the box z-scores under 86's 2000-2003 normalisation
(recomputed from 86's 2000-2003 box fields, as ``136`` does). Writes
``outputs/ihesp_20261021/ihesp_hr_test/satellites.npz``; ``136`` then builds the grids and the
lagged features from it. Check: every training row gives 86's own features back exactly.

  .venv/bin/python experiments/synthetic/139_ihesp_test_surface.py --workers 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S86 = _module("ihesp_surface86", "experiments/synthetic/86_ihesp_surface_cache.py")
G136 = _module("ihesp_grids136", "experiments/synthetic/136_ihesp_grid_caches.py")
T138 = _module("ihesp_test138", "experiments/synthetic/138_ihesp_test_cohort.py")
OUTPUT = ROOT / "outputs/ihesp_20261021"
MONTHS = OUTPUT / "test_surface_months"
TRAIN_SATELLITES = ROOT / "outputs/radius_scale_20261008/ihesp_hr_uniform/satellites.npz"


def compute_month(job):
    """86's ``compute_month`` (the 'hr' branch) on the test-year files, cached per month."""
    root, t = job
    path = MONTHS / f"t{t:02d}.npy"
    if path.exists():
        return t
    from ocean_tokenizer.ihesp_sources import FILL, LEVELS
    from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_columns
    source = T138.TestYears(root)
    level = np.asarray(source.full_level_index)[:, None, None]
    kmt = source.kmt[None]
    columns = []
    for variable in ("TEMP", "SALT"):
        v = source.field(variable, t).astype(np.float64)
        v[(np.abs(v) > FILL) | (level >= kmt) | (kmt <= 0)] = np.nan
        columns.append(v.reshape(len(LEVELS), -1))
    wet = np.flatnonzero(source.kmt.ravel() > 0)
    box = S86.one_degree_box(source.lat.ravel()[wet], source.lon.ravel()[wet])
    area = source.area.ravel()[wet].astype(np.float64)
    temp, salt = (S86.box_mean(v[:, wet], area, box, S86.N_BOXES) for v in columns)
    lat = np.repeat(np.arange(180) - 89.5, 360)
    lon = np.tile(np.arange(360) + 0.5, 180)
    full = np.isfinite(temp).all(axis=0) & np.isfinite(salt).all(axis=0)
    steric = np.full(temp.shape[1], np.nan)
    if full.any():
        steric[full] = steric_height_columns(temp[:, full], salt[:, full], LEVELS, lat[full], lon[full], P_REF_DBAR)
    out = np.stack([temp[0], salt[0], steric]).astype(np.float32)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, out)
    os.replace(temporary, path)
    return t


def main():
    from ocean_tokenizer import synth_argo_eval as S
    from ocean_tokenizer.latent_experiment import array_fingerprint
    from ocean_tokenizer.synthetic_latent_experiment import position_features
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    started = time.time()
    MONTHS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, t in enumerate(pool.map(compute_month, [(str(ROOT), t) for t in T138.TEST_MONTHS]), 1):
            if i % 6 == 0:
                print(f"test surface months {i}/{len(T138.TEST_MONTHS)} ({time.time() - started:.0f}s)", flush=True)
    c, norm, obs = S.load(str(ROOT), cohort="ihesp_hr_test")
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    train_months = np.arange(48)
    _, climate, mean, std = S86.anomaly_z(G136.box_fields(train_months, G136.TRAINING_MONTHS), train_months % 12,
                                          np.ones(48, bool))
    months = np.unique(c.month_index)
    fields = G136.box_fields(months, [MONTHS, G136.TRAINING_MONTHS])
    z = (fields - climate[months % 12] - mean[None, :, None]) / std[None, :, None]
    where = {int(m): i for i, m in enumerate(months)}
    row = np.array([where[int(m)] for m in c.month_index])
    box = S86.one_degree_box(c.lat, c.lon)
    features = position_features(c).astype(np.float32)
    features[:, 44:51] = G136.surface_columns(z[row, :, box], fields[row, 0, box], box)
    with np.load(TRAIN_SATELLITES, allow_pickle=False) as cache:
        old = cache["features"]
        old_meta = json.loads(str(cache["metadata"].item()))
    train_rows = np.flatnonzero(c.month_index >= 0)
    if len(train_rows) != len(old) or not np.array_equal(features[train_rows], old):
        raise SystemExit("the training rows do not give 86's features back")
    if not np.isfinite(features).all():
        raise SystemExit("non-finite surface features")
    out = OUTPUT / "ihesp_hr_test"
    out.mkdir(parents=True, exist_ok=True)
    meta = {**{k: old_meta[k] for k in ("normalization_scope", "splits", "recipe", "resolution", "surface_variables",
                                        "surface_feature_offsets", "surface_is_independent_of_truth", "satellite_noise")},
            "cohort_fingerprint": fingerprint, "cohort": "ihesp_hr_test",
            "training_rows_equal_86": True, "feature_fingerprint": array_fingerprint((features,)),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    temporary = out / "satellites.tmp.npz"
    np.savez(temporary, features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
             surface_climatology=climate.astype(np.float32), surface_mean=mean, surface_std=std, metadata=json.dumps(meta))
    os.replace(temporary, out / "satellites.npz")
    print(json.dumps({"output": str(out / "satellites.npz"), "rows": int(len(c.lat)), "test_rows": int((c.month_index < 0).sum()),
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
