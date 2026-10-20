"""Spread calibration of finished gridded-model runs, fitted on their own selection years.

For every run folder given, reads the run's summary and the predictions it saved on every cell of
its selection years (``forecast_seed*.npz``: CESM2 2004-2005, or real 2021), fits one CRPS-optimal
factor per lead, variable and depth band (``ocean_tokenizer.calibration``) and writes
``outputs/calibration_20261020/<campaign>__<run>.json`` with the sha256 of the run's chosen
checkpoint; the sealed scorers apply a calibration to that checkpoint only. No sealed year is read:
every saved prediction must lie in the run's selection years, which may not include 2006-2007
(sealed CESM2 test) or 2022-2025 (sealed real test and holdout).

  .venv/bin/python experiments/synthetic/116_fit_calibration.py RUN_FOLDER ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import calibration as C
from ocean_tokenizer.point_baselines import CH

OUTPUT = ROOT / "outputs/calibration_20261020"
SEALED_YEARS = {2006, 2007, 2022, 2023, 2024, 2025}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def coverage_and_crps(mean, std, target):
    keep = np.isfinite(target)
    error = np.abs(target[keep] - mean[keep])
    return {"coverage_68": float(np.mean(error <= std[keep])), "coverage_95": float(np.mean(error <= 1.959964 * std[keep])),
            "crps": float(C.crps(mean[keep], std[keep], target[keep]).mean())}


def fit(folder):
    """The calibration record of one finished run."""
    folder = Path(folder)
    summaries = sorted(folder.glob("summary_seed*.json"))
    if len(summaries) != 1:
        raise SystemExit(f"{folder}: expected one finished run, found {len(summaries)} summaries")
    summary = json.loads(summaries[0].read_text())
    seed, years = summary["seed"], set(summary["val_years"])
    if years & SEALED_YEARS:
        raise SystemExit(f"{folder}: its selection years {sorted(years)} include a sealed year")
    checkpoint, predictions = folder / f"best_seed{seed}.pt", folder / f"forecast_seed{seed}.npz"
    with np.load(predictions, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in ("mean", "std", "target", "lead", "level", "month", "normalization_scale")}
    seen = set((2000 + arrays["month"] // 12).tolist())
    if not seen <= years:
        raise SystemExit(f"{folder}: saved predictions cover {sorted(seen)}, outside the selection years {sorted(years)}")
    levels_m = summary["data"]["levels_m"]
    scale = arrays["normalization_scale"][arrays["level"]]            # (n, 2): z to physical anomaly units
    mean, std, target = (arrays[k].astype(np.float64) * scale for k in ("mean", "std", "target"))
    lead, level = arrays["lead"].astype(int), arrays["level"]
    scales = C.fit_scales(mean, std, target, lead, level, levels_m)
    calibrated = C.apply_scales(std, lead, level, levels_m, scales)
    in_sample = {str(l): {ch: {"raw": coverage_and_crps(mean[lead == l, j], std[lead == l, j], target[lead == l, j]),
                               "calibrated": coverage_and_crps(mean[lead == l, j], calibrated[lead == l, j],
                                                               target[lead == l, j])}
                          for j, ch in enumerate(CH)} for l in sorted(set(lead.tolist()))}
    return {"run": str(folder), "tag": summary["tag"], "seed": seed, "best_step": summary["best_step"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "predictions": str(predictions), "predictions_sha256": sha256(predictions),
            "fit_years": sorted(years), "levels_m": levels_m, "units": "physical anomaly (degC, PSU)",
            "method": "per lead, variable and depth band: the factor of the predicted std minimising mean CRPS",
            "scales": scales, "in_sample": in_sample}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder in args.runs:
        target = OUTPUT / C.record_name(folder)
        record = fit(folder)
        if target.exists():
            if json.loads(target.read_text())["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise SystemExit(f"{target} was fitted to another checkpoint")
            print(f"kept {target.name} (same checkpoint)")
            continue
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=1) + "\n")
        os.replace(temporary, target)
        six = record["in_sample"].get("6", {}).get("TEMP")
        note = "" if six is None else (f"; lead 6 T coverage95 {six['raw']['coverage_95']:.3f} -> "
                                       f"{six['calibrated']['coverage_95']:.3f}, CRPS {six['raw']['crps']:.4f} -> "
                                       f"{six['calibrated']['crps']:.4f}")
        print(f"wrote {target.name}{note}")


if __name__ == "__main__":
    main()
