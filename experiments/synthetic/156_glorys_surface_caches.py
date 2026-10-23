"""Surface inputs of a GLORYS12 cohort: 86's satellites cache and 136's grid and lagged caches.

Spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``. The 1-degree box fields of every
month come from ``experiments/data/155_glorys_months.py`` (SST and SSS = the 5 m level of the
box-mean column, steric height of the box-mean column to 985 m). Their monthly climatology,
mean and spread over the 240 training months (GLORYS 1997-2016) give the z-scores; each profile
reads its own box. Writes under ``outputs/glorys_20261021/<cohort>/``:

  satellites.npz       [P, 56] 86's profile features (44 positional + value/presence of the three
                       z-scores + absolute SST), ``normalization_scope = "training years only"``
  surface_grids.npz    [M, 3, 180, 360] float16 z-scores of the boxes for every cohort month
  lagged_surface.npz   [P, 3, 7] the surface columns of month m - lead at each profile's box
                       (leads 1, 3, 6), zeros where that month is not in the cohort

Checks: the surface fields track the cohort's own 5 m values (correlation above 0.9); every
feature is finite; lead 0 of the lagged recipe gives the satellites' columns 44-50 exactly; for
``glorys12_test`` the training rows give ``glorys12_uniform``'s features back exactly (139).

  .venv/bin/python experiments/synthetic/156_glorys_surface_caches.py glorys12_uniform
  .venv/bin/python experiments/synthetic/156_glorys_surface_caches.py glorys12_test
"""
from __future__ import annotations

import argparse
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

from ocean_tokenizer import glorys_sources as G

spec = importlib.util.spec_from_file_location("ihesp_surface86", ROOT / "experiments/synthetic/86_ihesp_surface_cache.py")
S86 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S86
spec.loader.exec_module(S86)

OUTPUT = ROOT / "outputs/glorys_20261021"
SURFACE = ("SST", "SSS", "steric_SSH")
LEADS = (1, 3, 6)
TRAINING = "glorys12_uniform"


def box_fields(folders, months):
    """``[M, 3, 64800]`` absolute SST, SSS and steric height of the boxes of ``months`` (float64)."""
    out = []
    for t in months:
        with np.load(folders.boxes_file(int(t)), allow_pickle=False) as z:
            if list(z["surface_names"][:3]) != list(SURFACE):
                raise SystemExit(f"{folders.boxes_file(int(t))}: unexpected surface fields")
            out.append(z["surface"][:3])
    return np.stack(out).astype(np.float64)


def surface_columns(z, absolute_sst):
    """The seven surface columns (44-50) from ``z [n, 3]`` and the absolute SST at the boxes."""
    out = np.zeros((len(z), 7), np.float32)
    for k in range(3):
        out[:, 2 * k] = np.nan_to_num(z[:, k])
        out[:, 2 * k + 1] = np.isfinite(z[:, k])
    out[:, 6] = np.nan_to_num((absolute_sst - 14.0) / 11.0)
    return out


def main():
    import xarray as xr
    from ocean_tokenizer import synth_argo_eval as S
    from ocean_tokenizer.audit_tools import cohort_path
    from ocean_tokenizer.latent_experiment import array_fingerprint
    from ocean_tokenizer.synthetic_latent_experiment import position_features
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cohort", choices=[TRAINING, "glorys12_test"])
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    started = time.time()
    folders = G.Folders(ROOT)
    output = args.output or OUTPUT / args.cohort
    c, norm, obs = S.load(str(ROOT), cohort=args.cohort)
    standardized = np.stack([obs[ch] for ch in ("TEMP", "SALT")], axis=-1)
    fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized))
    months = np.unique(c.month_index)
    expected = G.COHORT_MONTHS if args.cohort == TRAINING else G.ALL_MONTHS
    if not np.array_equal(months, np.array(expected)):
        raise SystemExit(f"{args.cohort} does not hold the months {expected[0]} .. {expected[-1]}")
    train_months = np.array(G.TRAIN_MONTHS)
    _, climate, mean, std = S86.anomaly_z(box_fields(folders, train_months), train_months % 12, np.ones(len(train_months), bool))
    fields = box_fields(folders, months)                                             # [M, 3, 64800]
    z = (fields - climate[months % 12] - mean[None, :, None]) / std[None, :, None]
    where = {int(m): i for i, m in enumerate(months)}
    row = np.array([where[int(m)] for m in c.month_index])
    box = G.one_degree_box(c.lat, c.lon)
    at_profiles = z[row, :, box]                                                      # [P, 3]
    absolute = fields[row, :, box]
    features = S86.surface_features(position_features(c), at_profiles, absolute[:, 0])
    if not np.isfinite(features).all():
        raise SystemExit("non-finite surface features")
    if not np.array_equal(surface_columns(at_profiles, absolute[:, 0]), features[:, 44:51]):
        raise SystemExit("the lagged recipe at lead 0 does not give 86's surface columns")
    with xr.open_dataset(Path(cohort_path(str(ROOT), "synthetic", cohort=args.cohort))) as raw:
        order = np.argsort(np.asarray(raw.month_index.values, int), kind="stable")
        surface_truth = {ch: np.asarray(raw[ch].values)[order, 0] for ch in ("TEMP", "SALT")}
    agreement = {}
    for k, ch in enumerate(("TEMP", "SALT")):
        ok = np.isfinite(absolute[:, k]) & np.isfinite(surface_truth[ch])
        agreement[ch] = {"correlation_with_profile_5m": float(np.corrcoef(absolute[ok, k], surface_truth[ch][ok])[0, 1]),
                         "rms_difference": float(np.sqrt(np.mean((absolute[ok, k] - surface_truth[ch][ok]) ** 2)))}
        if agreement[ch]["correlation_with_profile_5m"] < 0.9:
            raise SystemExit(f"{ch}: the surface field does not track the cohort's 5 m values")
    present = {name: float(np.isfinite(at_profiles[:, k]).mean()) for k, name in enumerate(SURFACE)}
    training_rows_equal = None
    if args.cohort != TRAINING:
        with np.load(OUTPUT / TRAINING / "satellites.npz", allow_pickle=False) as cache:
            old = cache["features"]
        train_rows = np.flatnonzero(c.month_index <= G.COHORT_MONTHS[-1])
        if len(train_rows) != len(old) or not np.array_equal(features[train_rows], old):
            raise SystemExit("the training rows do not give the training cohort's features back")
        training_rows_equal = True
    metadata = {
        "cohort_fingerprint": fingerprint, "cohort": args.cohort,
        "normalization_scope": "training years only",
        "splits": {k: list(v) for k, v in G.SPLITS.items()},
        "training_months": [int(train_months[0]), int(train_months[-1])],
        "recipe": "noise-free GLORYS12 thetao/so at 5 m and TEOS-10 steric height of the 20-level column to 985 m",
        "resolution": "1/12-degree fields averaged to 1-degree boxes (area-weighted), read at the profile's box",
        "surface_variables": list(SURFACE), "surface_feature_offsets": [44, 46, 48],
        "surface_is_independent_of_truth": False, "satellite_noise": "none",
        "share_of_profiles_with_field": present, "agreement_with_profile_surface": agreement,
        "source": G.DESCRIBE, "training_rows_equal_training_cohort": training_rows_equal,
        "feature_fingerprint": array_fingerprint((features,)),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "satellites.tmp.npz"
    np.savez_compressed(temporary, features=features, month_index=c.month_index, lat=c.lat, lon=c.lon,
                        metadata=json.dumps(metadata), surface_climatology=climate.astype(np.float32),
                        surface_mean=mean, surface_std=std)
    os.replace(temporary, output / "satellites.npz")

    lagged = np.zeros((len(c.lat), len(LEADS), 7), np.float32)
    for j, lead in enumerate(LEADS):
        source = c.month_index - lead
        have = np.isin(source, months)
        r = np.array([where[int(m)] for m in source[have]])
        lagged[have, j] = surface_columns(z[r, :, box[have]], fields[r, 0, box[have]])
    common = {"cohort_fingerprint": fingerprint, "cohort": args.cohort, "source": str(output / "satellites.npz"),
              "satellites_sha256": hashlib.sha256((output / "satellites.npz").read_bytes()).hexdigest(),
              "normalization": "monthly climatology, mean and spread of the box fields over GLORYS 1997-2016",
              "profiles_read": "the 1-degree box that holds the profile",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    grids = z.reshape(len(months), 3, 180, 360).astype(np.float16)
    for name, arrays in (("surface_grids.npz", dict(grids=grids, grid_month=months.astype(int),
                                                     grid_lat=np.arange(-89.5, 90., 1.), grid_lon=np.arange(.5, 360., 1.),
                                                     metadata=json.dumps({**common, "units": "z-score"}))),
                         ("lagged_surface.npz", dict(features=lagged, leads=np.array(LEADS), month_index=c.month_index,
                                                     lat=c.lat, lon=c.lon, metadata=json.dumps(common)))):
        temporary = output / (name + ".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, output / name)
    print(json.dumps({"output": str(output), "profiles": int(len(c.lat)), "months": int(len(months)), "present": present,
                      "agreement": agreement, "anomaly_std": dict(zip(SURFACE, std.tolist())),
                      "lagged_available": {str(l): float((lagged[:, j, 1] > 0).mean()) for j, l in enumerate(LEADS)},
                      "bytes": {n: (output / n).stat().st_size for n in ("satellites.npz", "surface_grids.npz", "lagged_surface.npz")},
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
