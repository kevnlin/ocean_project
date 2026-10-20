"""The sealed CESM2 test: frozen checkpoints scored once on 2006-2007 of ``cesm2_long_test``.

No model was trained or selected on 2006-2007; ``110_cesm2_test_cohort.py`` drew those months after
every run had been set up (``docs/superpowers/specs/2026-10-19-sealed-cesm2-test.md``). Each step
refuses to run twice:

  --register FAMILY:CHECKPOINT ...   records every checkpoint and data file (sha256), the test years
                                     and leads. FAMILY is "grid" (runner 101, any backbone) or
                                     "reference" (runner 94, the OI-anchored local transformer with
                                     its local OI-error scale table)
  --score                            verifies every recorded file, then scores each checkpoint on
                                     every cell of 2006-2007 at leads 0, 1, 3, 6
                                     (``sealed_scoring``: RMSE, MAE, bias, R², r, CRPS, NLL,
                                     68/95 % coverage, by depth band and by level, with the
                                     persisted OI and the climatology on the same cells); writes
                                     results.json and never overwrites it
  --check FAMILY:CHECKPOINT          opens nothing: scores 2004-2005 of the test cohort (rows
                                     identical to training's) and requires the run's own scores back

  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --check grid:PATH
  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --register grid:PATH reference:PATH ...
  .venv/bin/python experiments/synthetic/109_sealed_cesm2_test.py --score
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

from ocean_tokenizer import sealed_scoring as SS

OUTPUT = ROOT / "outputs/cesm2_test_20261019"
REGISTRATION = OUTPUT / "registration.json"
RESULTS = OUTPUT / "results.json"
COHORT = "cesm2_long_test"
TEST_YEARS = (2006, 2007)
LEADS = (0, 1, 3, 6)
DATA = {"cohort": ROOT / f"data/synthetic_argo/{COHORT}.nc",
        "satellites": OUTPUT / "satellites.npz", "surface_grids": OUTPUT / "surface_grids.npz",
        "lagged_surface": OUTPUT / "lagged_surface.npz",
        "oi_summary": ROOT / f"outputs/audit/synthetic/fixed_baselines_{COHORT}/summary.json",
        "scale_tables": ROOT / "outputs/long_record_20261017/scale_tables.npz"}


def scenes(family, training, device):
    """The test cohort's forecast scenes as the checkpoint's runner built them."""
    import numpy as np
    from ocean_tokenizer import local_scale
    from ocean_tokenizer.forecast_scenes import ForecastScenes
    from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
    common = dict(leads=LEADS, lagged_surface=DATA["lagged_surface"], train_years=training["train_years"],
                  surface=True, satellite_cache=DATA["satellites"], oi_summary=DATA["oi_summary"], cohort=COHORT)
    if family == "grid":
        return GridForecastScenes(ROOT, device, surface_grids=DATA["surface_grids"], **common)
    data = ForecastScenes(ROOT, device, **common)
    if training["scale"] is not None:
        # the table is a function of position and calendar month fitted on 2000-2003; test rows only add lookups
        with np.load(DATA["scale_tables"], allow_pickle=False) as tables:
            scale = local_scale.lookup(tables[training["scale"]], data.c.lat, data.c.lon, data.c.month_index % 12)
        data.configure(None, scale, "query")
    return data


def model_of(family, checkpoint, device):
    if family == "grid":
        from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
        model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
    else:
        from ocean_tokenizer.innovation_ocean import InnovationOceanConfig
        from ocean_tokenizer.query_scaled_innovation import QueryScaledInnovationModel
        model = QueryScaledInnovationModel(InnovationOceanConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def evaluate(family, path, years, device, cache):
    """``sealed_scoring.summarize`` of one checkpoint on every cell of ``years``."""
    import numpy as np
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    key = (family, training["train_years"], training.get("scale"))
    if key not in cache:
        cache.clear()
        cache[key] = scenes(family, training, device)
    data = cache[key]
    for kind in ("mean", "std"):
        for ch in ("TEMP", "SALT"):
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the test cohort's normalisation differs from training's")
    model = model_of(family, checkpoint, data.device)
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(years), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return checkpoint, SS.summarize(collected, data.norm, data.levels, years)


def check(family, path, device):
    """Score a checkpoint on 2004-2005 of the test cohort; its run's own scores must come back."""
    checkpoint, outcome = evaluate(family, path, (2004, 2005), device, {})
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
            print(f"lead {lead} {ch}: model {mine['model']['rmse']:.5f} (run {run['scores'][ch]['rmse_physical']:.5f}), "
                  f"CRPS {mine['model']['crps']:.5f}, 95% coverage {mine['model']['coverage_95']:.3f}")
    if worst > 1e-3:
        raise SystemExit(f"the test cohort does not reproduce the run's 2004-2005 scores (relative {worst:.2g})")
    print(f"check passed: 2004-2005 reproduced to a relative {worst:.2g}")


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    results, cache = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    for entry in record["entries"]:
        started = time.time()
        checkpoint, outcome = evaluate(entry["family"], entry["checkpoint"], TEST_YEARS, device, cache)
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
    p.add_argument("--register", nargs="+", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--check", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.register, args.score, args.check))) != 1:
        raise SystemExit("pass one of --register, --score, --check")
    if args.check:
        family, path = args.check.split(":", 1)
        check(family, path, args.device)
    elif args.register:
        entries = [(item.split(":", 1)[0], Path(item.split(":", 1)[1])) for item in args.register]
        record = SS.register(REGISTRATION, entries, DATA, {"cohort": COHORT, "test_years": list(TEST_YEARS),
                                                             "leads": list(LEADS), "cells": "every held-out cell"})
        print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")
    else:
        score(args.device)


if __name__ == "__main__":
    main()
