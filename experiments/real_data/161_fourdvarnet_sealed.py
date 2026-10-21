"""The official 4DVarNet on the three opened tests, as one declared additional pass (2026-10-21).

Added after every test was opened, at Zihao's request for a published method run from its authors'
code (``docs/superpowers/specs/2026-10-21-official-4dvarnet-baseline.md``); nothing in it is chosen
on test data. The runs of ``160`` were trained and chosen like KA-Ocean (CESM2 1964-2003 / 2004-2005,
real 2016-2020 / 2021). Checkpoints and test files are registered by sha256 before any scoring; each
test is then scored once, through the main pass's own scenes (``109`` CESM2 2006-2007, ``113`` real
2022-2023, ``133`` real 2024) on the main pass's cells: the persisted OI on them must come back
unchanged. 4DVarNet is deterministic, so only the regression metrics are reported (its CRPS equals
its MAE); at a forecast lead its reconstruction of the input month is persisted.

  --check CHECKPOINT   opens nothing: the run's own selection cells through the tests' scenes (CESM2:
                       2004-2005 of the test cohort; real: 2021 of the 2016-2023 and the 2016-2024
                       scenes); its summary's scores must come back
  --register           the six chosen checkpoints and every test file, once
  --score TEST         cesm2 | real | real2024, once each

  .venv/bin/python experiments/real_data/161_fourdvarnet_sealed.py --check PATH
  .venv/bin/python experiments/real_data/161_fourdvarnet_sealed.py --register
  .venv/bin/python experiments/real_data/161_fourdvarnet_sealed.py --score cesm2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import sealed_scoring as SS
from ocean_tokenizer.point_baselines import CH

RUNS = ROOT / "outputs/fourdvarnet_20261021"
OUTPUT = RUNS / "sealed"
REGISTRATION = OUTPUT / "registration.json"
SEEDS = (1234, 1235, 1236)
#: the runs of each dataset and the steps they must have completed (KA-Ocean's)
ARMS = {"cesm2": ("fv_cesm2", 70000), "real_global": ("fv_real", 30000)}
TESTS = {"cesm2": ("cesm2", ROOT / "experiments/synthetic/109_sealed_cesm2_test.py"),
         "real": ("real_global", ROOT / "experiments/real_data/113_sealed_real_test.py"),
         "real2024": ("real_global", ROOT / "experiments/real_data/133_sealed_real_2024.py")}


def scorer(test):
    spec = importlib.util.spec_from_file_location(f"scorer_{test}", TESTS[test][1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenes(test, training, device):
    """The scenes the main pass of ``test`` scored."""
    S = scorer(test)
    if test == "cesm2":
        return S.scenes("grid", training, device)
    if test == "real":
        from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
        return RealGridForecastScenes(ROOT, device, caches=S.CACHES, oi_summary=S.DATA["oi_summary"], leads=S.LEADS)
    return S.scenes(device)


def summarize(collected, norm, levels_m, years):
    """``sealed_scoring.summarize`` of a deterministic model: no spread, so no Gaussian scores."""
    result = {}
    for lead, a in sorted(collected.items()):
        year = 2000 + a["month"] // 12
        result[str(lead)] = {"all": SS.metrics(a["mean"], None, a["base"], a["target"], a["level"], norm, levels_m),
                             "cells": int(len(a["level"]))}
        for y in years:
            keep = year == y
            if keep.any():
                result[str(lead)][str(y)] = SS.metrics(a["mean"][keep], None, a["base"][keep], a["target"][keep],
                                                       a["level"][keep], norm, levels_m)
    return result


def evaluate(test, path, years, device, cache):
    """The regression metrics of one checkpoint on every cell of ``years`` of ``test``'s scenes."""
    import numpy as np
    import torch
    from ocean_tokenizer.fourdvar_grid import FourDVarGridModel
    from ocean_tokenizer.fourdvar_synthetic import FourDVarConfig
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    if training["dataset"] != TESTS[test][0]:
        raise SystemExit(f"{path} was trained on {training['dataset']}, not for the {test} test")
    if test not in cache:
        cache.clear()
        cache[test] = test_scenes(test, training, device)
    data = cache[test]
    for kind in ("mean", "std"):
        for ch in CH:
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the test scenes' normalisation differs from training's")
    model = FourDVarGridModel(training["official_repository"], FourDVarConfig(**checkpoint["config"]),
                              checkpoint["output"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(data.device).eval()
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(years), 0),
                           training["context_profiles"], False)
    return checkpoint, summarize(collected, data.norm, data.levels, years)


def check(path, device):
    """The run's own selection cells through every test scene of its dataset; its scores must come back."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    summary = json.loads((Path(path).parent / f"summary_seed{checkpoint['seed']}.json").read_text())
    if summary["best_step"] != checkpoint["step"] or summary["output"] != checkpoint["output"]:
        raise SystemExit("the checkpoint is not the run's chosen one")
    years = tuple(summary["val_years"])
    for test, (dataset, _) in TESTS.items():
        if dataset != checkpoint["training"]["dataset"]:
            continue
        _, outcome = evaluate(test, path, years, device, {})
        worst = 0.
        for lead in ("0", "1", "3", "6"):
            run = summary["forecast"]["best"]["by_lead"][lead]
            for ch in CH:
                mine = outcome[lead]["all"][ch]
                for name, theirs in (("model", run["scores"][ch]["rmse_physical"]),
                                     ("persisted_oi", run["baseline_scores"][ch]["rmse_physical"])):
                    worst = max(worst, abs(mine[name]["rmse"] / theirs - 1))
                print(f"{test} lead {lead} {ch}: model {mine['model']['rmse']:.5f} "
                      f"(run {run['scores'][ch]['rmse_physical']:.5f})")
        if worst > 1e-3:
            raise SystemExit(f"{test}: the scenes do not reproduce the run's {years} scores (relative {worst:.2g})")
        print(f"check passed ({test}): {years} reproduced to a relative {worst:.2g}")


def chosen(dataset, seed):
    arm, steps = ARMS[dataset]
    path = RUNS / f"{arm}_s{seed}" / f"best_seed{seed}.pt"
    summary = json.loads((path.parent / f"summary_seed{seed}.json").read_text())
    if summary["dataset"] != dataset or summary["completed_steps"] != steps:
        raise SystemExit(f"{path}: not a finished {dataset} run of {steps} steps")
    return path


def register():
    entries = [("fourdvarnet", chosen(dataset, seed)) for dataset in ARMS for seed in SEEDS]
    data = {}
    for test in TESTS:
        S = scorer(test)
        data.update({f"{test}_{name}": path for name, path in S.DATA.items()})
        data[f"{test}_main_pass_results"] = S.RESULTS
    record = SS.register(REGISTRATION, entries, data, {
        "tests": {"cesm2": [2006, 2007], "real": [2022, 2023], "real2024": [2024]}, "leads": [0, 1, 3, 6],
        "pass": "official 4DVarNet (4dvarnet-starter, pinned commit), added after every test was opened"})
    print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")


def score(test, device):
    results_path = OUTPUT / f"results_{test}.json"
    if results_path.exists():
        raise SystemExit(f"{results_path} exists: each test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    S = scorer(test)
    main_pass = json.loads(S.RESULTS.read_text())["entries"][0]["by_lead"]
    arm = ARMS[TESTS[test][0]][0]
    out, cache = {"registration_sha256": SS.sha256(REGISTRATION), "test": test,
                  "years": list(S.TEST_YEARS), "entries": []}, {}
    for entry in record["entries"]:
        if f"/{arm}_s" not in entry["checkpoint"]:
            continue
        checkpoint, outcome = evaluate(test, entry["checkpoint"], S.TEST_YEARS, device, cache)
        for lead, block in outcome.items():
            for ch in CH:
                mine, theirs = block["all"][ch]["persisted_oi"]["rmse"], main_pass[lead]["all"][ch]["persisted_oi"]["rmse"]
                if abs(mine / theirs - 1) > 1e-4 or block["cells"] != main_pass[lead]["cells"]:
                    raise SystemExit(f"lead {lead} {ch}: not the cells of the main pass")
        out["entries"].append({**entry, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                               "step": checkpoint["step"], "output": checkpoint["output"], "by_lead": outcome})
        print(json.dumps({"scored": checkpoint["training"]["tag"], "rmse": {
            lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in CH] for lead in outcome}}), flush=True)
    temporary = results_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(out, indent=1, default=float) + "\n")
    os.replace(temporary, results_path)
    print(f"wrote {results_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--register", action="store_true")
    p.add_argument("--score", choices=list(TESTS))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.check, args.register, args.score))) != 1:
        raise SystemExit("pass one of --check, --register, --score TEST")
    if args.check:
        check(args.check, args.device)
    elif args.register:
        register()
    else:
        score(args.score, args.device)


if __name__ == "__main__":
    main()
