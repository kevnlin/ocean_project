"""The sealed real-data test: frozen checkpoints scored once on 2022-2023 of the real global cohort.

Real Argo profiles with OISST, OISSS L4 and DUACS SLA (``real_forecast_scenes``); every model was
trained on 2016-2020 and its checkpoint chosen on 2021 (``docs/paper/2026-10-19-main-tables.md``).
2025 is the Real-Data v1 sealed holdout and is not read here either. Each step refuses to run twice:

  --register CHECKPOINT ...   records every checkpoint (all gridded-model runs: scratch, fine-tuned,
                              zero shot, the ARMOR3D- and Aardvark-style baselines) and the cohort,
                              anomaly cache, satellite caches and OI selection by sha256, and each
                              checkpoint's spread calibration if ``116`` fitted one (on the run's 2021)
  --score                     verifies every recorded file, then scores each checkpoint on every
                              held-out cell of 2022-2023 at leads 0, 1, 3, 6 (``sealed_scoring``:
                              RMSE, MAE, bias, R², r, CRPS, NLL, 68/95 % coverage, by depth band and
                              level, with the persisted OI and the WOA23 climatology on the same
                              cells; Gaussian scores also under the registered calibration); writes
                              results.json and never overwrites it
  --check CHECKPOINT          opens nothing: scores the run's own 2021 cells and requires its
                              summary's scores back

  .venv/bin/python experiments/real_data/113_sealed_real_test.py --check PATH
  .venv/bin/python experiments/real_data/113_sealed_real_test.py --register PATH ...
  .venv/bin/python experiments/real_data/113_sealed_real_test.py --score
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import calibration as CAL
from ocean_tokenizer import sealed_scoring as SS

CACHES = ROOT / "outputs/real_global_20261019"
OUTPUT = CACHES / "sealed"
CALIBRATIONS = ROOT / "outputs/calibration_20261020"
REGISTRATION = OUTPUT / "registration.json"
RESULTS = OUTPUT / "results.json"
TEST_YEARS = (2022, 2023)
LEADS = (0, 1, 3, 6)
DATA = {"cohort": ROOT / "data/argo_cohort/global_global.nc", "anomalies": ROOT / "data/argo_cohort/global_anomx.npz",
        "satellites": CACHES / "satellites.npz", "surface_grids": CACHES / "surface_grids.npz",
        "lagged_surface": CACHES / "lagged_surface.npz",
        "oi_summary": ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json"}


def evaluate(path, years, device, cache, scales=None):
    """``sealed_scoring.summarize`` of one checkpoint on every held-out cell of ``years``."""
    import numpy as np
    import torch
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    if training.get("dataset") != "real_global":
        raise SystemExit(f"{path} was not trained on the real cohort")
    if "data" not in cache:
        cache["data"] = RealGridForecastScenes(ROOT, device, caches=CACHES, oi_summary=DATA["oi_summary"], leads=LEADS)
    data = cache["data"]
    for kind in ("mean", "std"):
        for ch in ("TEMP", "SALT"):
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the scenes' normalisation differs from training's")
    model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(data.device).eval()
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(years), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return checkpoint, SS.summarize(collected, data.norm, data.levels, years, scales)


def calibration_of(checkpoint):
    """The calibration ``116`` fitted to a checkpoint's run, if any."""
    path = CALIBRATIONS / CAL.record_name(Path(checkpoint).parent)
    return path if path.exists() else None


def check(path, device):
    """The run's own 2021 cells; its summary's scores must come back."""
    checkpoint, outcome = evaluate(path, (2021,), device, {})
    summary = json.loads((Path(path).parent / f"summary_seed{checkpoint['seed']}.json").read_text())
    if summary["best_step"] != checkpoint["step"]:
        raise SystemExit("the checkpoint is not the run's chosen one")
    worst = 0.
    for lead in map(str, LEADS):
        run = summary["forecast"]["best"]["by_lead"][lead]
        for ch in ("TEMP", "SALT"):
            mine = outcome[lead]["all"][ch]
            for name, theirs in (("model", run["scores"][ch]["rmse_physical"]),
                                 ("persisted_oi", run["baseline_scores"][ch]["rmse_physical"])):
                worst = max(worst, abs(mine[name]["rmse"] / theirs - 1))
            print(f"lead {lead} {ch}: model {mine['model']['rmse']:.4f} (run {run['scores'][ch]['rmse_physical']:.4f}), "
                  f"CRPS {mine['model']['crps']:.4f}, 95% coverage {mine['model']['coverage_95']:.3f}")
    if worst > 1e-3:
        raise SystemExit(f"the scorer does not reproduce the run's 2021 scores (relative {worst:.2g})")
    print(f"check passed: 2021 reproduced to a relative {worst:.2g}")


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    results, cache = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    for entry in record["entries"]:
        started = time.time()
        scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"] if "calibration" in entry else None
        checkpoint, outcome = evaluate(entry["checkpoint"], TEST_YEARS, device, cache, scales)
        results["entries"].append({**entry, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                                   "step": checkpoint["step"], "seconds": time.time() - started, "by_lead": outcome})
        print(json.dumps({"scored": checkpoint["training"]["tag"], "rmse": {
            lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in ("TEMP", "SALT")]
            for lead in map(str, LEADS)}}), flush=True)
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, RESULTS)
    print(f"wrote {RESULTS}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--register", nargs="+", metavar="CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.register, args.score, args.check))) != 1:
        raise SystemExit("pass one of --register, --score, --check")
    if args.check:
        check(args.check, args.device)
    elif args.register:
        record = SS.register(REGISTRATION, [("grid", Path(p), calibration_of(p)) for p in args.register], DATA,
                             {"dataset": "real_global 2016-2023", "test_years": list(TEST_YEARS),
                              "leads": list(LEADS), "cells": "every held-out-float cell"})
        print(f"registered {len(record['entries'])} checkpoints ({sum('calibration' in e for e in record['entries'])} "
              f"with a calibration) -> {REGISTRATION}")
    else:
        score(args.device)


if __name__ == "__main__":
    main()
