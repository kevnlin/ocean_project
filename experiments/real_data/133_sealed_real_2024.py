"""The sealed real test of 2024: every frozen real-data checkpoint scored once on 2024.

No model or choice of this paper used 2024: the gridded models read the 2016-2023 cohort (trained
on 2016-2020, checkpoints chosen on 2021), the satellite store ended in 2023, and the 2024 inputs
were built after the models were frozen (``132_real_2024_inputs.py``). The models are every
checkpoint of the 2022-2023 test's registration, with the same spread calibrations, plus the runs
added with ``--add`` (the CLOINet-style read-out of ``131``); damped persistence uses the factors
``121`` fitted on 2021. 2025 is the Real-Data v1 sealed holdout and is never read.

  --check CHECKPOINT  scores the run's own 2021 cells from the 2016-2024 scenes; its summary's
                      scores must come back (the 2024 rows and caches change nothing before 2024)
  --register [--add CHECKPOINT ...]
                      records the checkpoints, their calibrations, the cohort, the anomaly cache,
                      the 2016-2024 caches, the frozen OI selection and the damped-persistence
                      factors by sha256; once
  --score             verifies every file, scores every checkpoint on every held-out cell of 2024
                      at leads 0, 1, 3, 6 (``sealed_scoring``), adds damped persistence on the
                      same cells, and writes results.json once

  .venv/bin/python experiments/real_data/133_sealed_real_2024.py --check PATH
  .venv/bin/python experiments/real_data/133_sealed_real_2024.py --register --add PATH ...
  .venv/bin/python experiments/real_data/133_sealed_real_2024.py --score
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

CACHES = ROOT / "outputs/real_2024_test"
OUTPUT = CACHES / "sealed"
CALIBRATIONS = ROOT / "outputs/calibration_20261020"
REGISTRATION = OUTPUT / "registration.json"
RESULTS = OUTPUT / "results.json"
PREVIOUS = ROOT / "outputs/real_global_20261019/sealed/registration.json"
ALPHAS = ROOT / "outputs/statistical_baselines_20261020/alphas.json"
TEST_YEARS = (2024,)
LEADS = (0, 1, 3, 6)
#: the 2016-2023 cohort on which the frozen OI selection (106) was made
OI_COHORT = "6366666ef8ad4ec33388fd8cc7d7f1666236d8f29158b28bee4818f16584e3a1"
DATA = {"cohort": ROOT / "data/argo_cohort/global_global.nc", "anomalies": ROOT / "data/argo_cohort/global_anomx.npz",
        "satellites": CACHES / "satellites.npz", "surface_grids": CACHES / "surface_grids.npz",
        "lagged_surface": CACHES / "lagged_surface.npz",
        "oi_summary": ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json",
        "damped_persistence_factors": ALPHAS}


def scenes(device):
    from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
    return RealGridForecastScenes(ROOT, device, caches=CACHES, oi_summary=DATA["oi_summary"], leads=LEADS,
                                  years=(2016, 2024), oi_cohort_fingerprint=OI_COHORT)


def evaluate(path, years, device, cache, scales=None):
    """``sealed_scoring.summarize`` of one checkpoint on every held-out cell of ``years``, and the
    collected cells."""
    import numpy as np
    import torch
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    if training.get("dataset") != "real_global":
        raise SystemExit(f"{path} was not trained on the real cohort")
    if "data" not in cache:
        cache["data"] = scenes(device)
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
    return checkpoint, SS.summarize(collected, data.norm, data.levels, years, scales), collected


def calibration_of(checkpoint):
    """The calibration ``116`` fitted to a checkpoint's run on its 2021, if any."""
    path = CALIBRATIONS / CAL.record_name(Path(checkpoint).parent)
    return path if path.exists() else None


def frozen_before_2024(path):
    """The run read only the 2016-2023 cohort and chose its checkpoint on 2021."""
    summary = json.loads(next(Path(path).parent.glob("summary_seed*.json")).read_text())
    if summary["data"].get("cohort") != "real_global 2016-2023" or set(summary["val_years"]) != {2021}:
        raise SystemExit(f"{path}: not a run on the 2016-2023 cohort chosen on 2021")


def check(path, device):
    """The run's own 2021 cells from the 2016-2024 scenes; its summary's scores must come back."""
    checkpoint, outcome, _ = evaluate(path, (2021,), device, {})
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
    if worst > 1e-3:
        raise SystemExit(f"the 2016-2024 scenes do not reproduce the run's 2021 scores (relative {worst:.2g})")
    print(f"check passed: 2021 reproduced to a relative {worst:.2g} ({path})")


def register(added):
    previous = json.loads(PREVIOUS.read_text())
    paths = [e["checkpoint"] for e in previous["entries"]] + [str(Path(p)) for p in added]
    if len(set(paths)) != len(paths):
        raise SystemExit("a checkpoint is listed twice")
    for p in paths:
        frozen_before_2024(p)
    record = SS.register(REGISTRATION, [("grid", Path(p), calibration_of(p)) for p in paths], DATA,
                         {"dataset": "real_global 2016-2024", "test_years": list(TEST_YEARS), "leads": list(LEADS),
                          "cells": "every held-out-float cell", "models": "the 2022-2023 registration plus --add",
                          "previous_registration": {"path": str(PREVIOUS), "sha256": SS.sha256(PREVIOUS)}})
    print(f"registered {len(record['entries'])} checkpoints ({sum('calibration' in e for e in record['entries'])} "
          f"with a calibration) -> {REGISTRATION}")


def damped_persistence(collected, data):
    """Damped persistence (121's factors, fitted on 2021) on the collected cells."""
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("statistical121", ROOT / "experiments/synthetic/121_statistical_baselines.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    record = json.loads(ALPHAS.read_text())["tests"]["real"]
    out = {"alphas_sha256": SS.sha256(ALPHAS), "method": "damped persistence", "by_lead": {}}
    for lead, a in sorted(collected.items()):
        mean = S.damped(a["base"], np.full(len(a["level"]), lead), a["level"], record["levels_m"], record["alphas"])
        out["by_lead"][str(lead)] = {"all": SS.metrics(mean, None, a["base"], a["target"], a["level"], data.norm,
                                                       data.levels), "cells": int(len(a["level"]))}
    return out


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    results, cache = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    for i, entry in enumerate(record["entries"]):
        started = time.time()
        scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"] if "calibration" in entry else None
        checkpoint, outcome, collected = evaluate(entry["checkpoint"], TEST_YEARS, device, cache, scales)
        if i == 0:
            results["damped_persistence"] = damped_persistence(collected, cache["data"])
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
    p.add_argument("--register", action="store_true")
    p.add_argument("--add", nargs="+", default=[], metavar="CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.register, args.score, args.check))) != 1:
        raise SystemExit("pass one of --register, --score, --check")
    if args.check:
        check(args.check, args.device)
    elif args.register:
        register(args.add)
    else:
        score(args.device)


if __name__ == "__main__":
    main()
