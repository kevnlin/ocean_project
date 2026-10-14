"""Local scale tables for the OI-anchored model, from the training years only.

A per-depth z-score gives a level one standard deviation for the whole ocean.
This writes four tables of how the spread varies inside a level
(``ocean_tokenizer.local_scale``), by 5-degree cell and, for two of them, by
calendar month:

  anom_cell, anom_month   spread of the at-position anomaly itself;
  res_cell,  res_month    spread of the frozen spherical OI's error, cross-fitted
                          on the training years: every month's 7,600 profiles
                          are split into five folds and each fold is analysed
                          from the other four (6,080 inputs, as at evaluation).

Only 2000-2003 profiles are read. ``81_radius_scale_reconstruction.py --scale``
looks a table up at each profile's position and month. ``--cohort NAME`` fits
the tables of another synthetic cohort, into ``<output folder>/NAME/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer import local_scale as LS
from ocean_tokenizer.point_baselines import CH
from ocean_tokenizer.radius_scale_scenes import RadiusScaleScenes, crossfit_residual

TABLES = ("anom_cell", "anom_month", "res_cell", "res_month")


def describe(scale, floor):
    """Spread of a table over the training profiles it was fitted on."""
    out = {}
    for j, ch in enumerate(CH):
        s = scale[..., j]
        out[ch] = {"percentiles_5_25_50_75_95": [float(v) for v in np.percentile(s, [5, 25, 50, 75, 95])],
                   "share_at_floor": float(np.mean(s <= floor)),
                   "max": float(s.max())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None,
                   help="default: outputs/radius_scale_20261008/[<cohort>/]scale_tables.npz")
    p.add_argument("--cohort", default=None, help="another synthetic cohort, by file name without .nc")
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--pseudo", type=float, default=20.)
    p.add_argument("--floor", type=float, default=.2)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--oi-summary", type=Path)
    args = p.parse_args()
    started = time.time()
    torch.set_num_threads(args.threads)
    folder = ROOT / "outputs/radius_scale_20261008"
    if args.cohort is not None:
        folder = folder / args.cohort
        if args.oi_summary is None:
            args.oi_summary = ROOT / "outputs/audit/synthetic" / f"fixed_baselines_{args.cohort}" / "summary.json"
    if args.output is None:
        args.output = folder / "scale_tables.npz"
    data = RadiusScaleScenes(ROOT, args.device, oi_summary=args.oi_summary, cohort=args.cohort)
    c = data.c
    train = np.asarray(c.year_split) == "train"
    months = [int(m) for m in c.months_in("train")]
    month_of_year = np.asarray(c.month_index) % 12
    anomaly = np.stack([data.obs[ch] for ch in CH], axis=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    residual_path = args.output.with_name("oi_residual_train.npz")
    key = {"cohort_fingerprint": data.metadata["cohort_fingerprint"],
           "oi_summary_sha256": data.metadata["oi_summary_sha256"],
           "folds": args.folds, "seed": args.seed, "months": months}
    residual = None
    if residual_path.exists():
        with np.load(residual_path, allow_pickle=False) as saved:
            if json.loads(str(saved["key"].item())) == key:
                residual = saved["residual"].astype(np.float32)
    if residual is None:
        print(f"cross-fitting the frozen OI on {len(months)} training months", flush=True)
        residual = np.full(anomaly.shape, np.nan, dtype=np.float32)
        for i, month in enumerate(months):
            rows = np.flatnonzero(np.asarray(c.month_index) == month)
            residual[rows] = crossfit_residual(data, [month], args.folds, args.seed, args.chunk)[rows]
            if (i + 1) % 6 == 0:
                print(f"  {i + 1}/{len(months)} months, {time.time() - started:.0f}s", flush=True)
        temporary = residual_path.with_name(residual_path.name + ".tmp.npz")
        np.savez_compressed(temporary, residual=residual, key=json.dumps(key))
        os.replace(temporary, residual_path)
    if np.isfinite(residual[~train]).any() or not np.array_equal(
            np.isfinite(residual[train]), np.isfinite(anomaly[train])):
        raise ValueError("cross-fitted residuals must cover the training profiles and nothing else")
    oi_rmse = {ch: float(np.sqrt(np.nanmean((residual[train][..., j] * data.norm.std[ch]) ** 2)))
               for j, ch in enumerate(CH)}
    print("cross-fitted OI error on the training years (physical RMSE): " + json.dumps(oi_rmse), flush=True)

    tables, description = {}, {}
    for name in TABLES:
        values = anomaly if name.startswith("anom") else residual
        tables[name] = LS.scale_table(values, c.lat, c.lon, month_of_year, train,
                                      by_month=name.endswith("month"), pseudo=args.pseudo,
                                      floor=args.floor).astype(np.float32)
        on_train = LS.lookup(tables[name], c.lat[train], c.lon[train], month_of_year[train])
        description[name] = describe(on_train, args.floor)
        print(name + " " + json.dumps(description[name]), flush=True)
    metadata = {
        "cohort_fingerprint": data.metadata["cohort_fingerprint"],
        "fit": "every profile of the training years 2000-2003; no later year is read",
        "definition": "sqrt(local second moment / the level's second moment), per variable and level",
        "cell_deg": LS.CELL_DEG, "month_window": "the calendar month and its two neighbours",
        "pseudo": args.pseudo, "floor": args.floor, "tables": list(TABLES),
        "residual": {"analysis": "frozen spherical OI of the matched campaign",
                     "oi_summary_sha256": data.metadata["oi_summary_sha256"],
                     "folds": args.folds, "seed": args.seed,
                     "inputs_per_fold": "6,080 of the month's 7,600 profiles",
                     "train_year_rmse_physical": oi_rmse},
        "scale_over_training_profiles": description,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "src/ocean_tokenizer/local_scale.py",
                                       ROOT / "src/ocean_tokenizer/radius_scale_scenes.py")},
        "runtime_s": time.time() - started,
    }
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(temporary, metadata=json.dumps(metadata), **tables)
    os.replace(temporary, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("wrote " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
