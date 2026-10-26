"""The GLORYS12 transfer test: CESM2-trained checkpoints scored once on the sealed GLORYS 2019-2020 cells.

Spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``. A declared extra pass on the
test ``159`` scored on 2026-10-22 (cohort ``glorys12_test``, cohort years 2006-2007 = GLORYS 2019-2020, every
held-out cell, leads 0, 1, 3, 6). The checkpoints were trained and chosen on CESM2 only. Everything
statistical is GLORYS's own (the climatology, the per-depth standardisation, the surface z-scores, the OI of
the persisted-OI row), built as ``159`` built them for ``g_ka`` (``train_years`` 20), so the persisted OI
and the climatology of every entry must equal ``159``'s main pass, or nothing is written. Each step refuses
to run twice:

  --check PATH          scores a checkpoint on its own CESM2 2004-2005 (the caches its run recorded); the
                        run's scores must come back (``109``'s check): this code still makes the model it trained
  --register PATH ...   the checkpoints (CESM2-trained, each its run's chosen one, on the GLORYS grid), the GLORYS
                        data files (as 159 registered them) and the main pass's results, by sha256
  --score               verifies every file, scores every checkpoint, writes results.json once

  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --check PATH
  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --register PATH ...
  .venv/bin/python experiments/synthetic/173_glorys_transfer.py --score
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ocean_tokenizer.point_baselines import CH, Scores      # noqa: E402


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = _module("glorys159", "experiments/synthetic/159_sealed_glorys_test.py")
T, SS = G.T, G.SS
SPEC = "docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md"
COHORT, TEST_YEARS, LEADS = T.COHORT, T.TEST_YEARS, T.LEADS
GLORYS = {name: path for name, path in T.DATA.items() if name != "damped_persistence_factors"}
MAIN = T.RESULTS                                  # 159's main pass: read, never written
GLORYS_TRAIN_YEARS = 20                           # g_ka's --train-years: the scenes 159 built
OUTPUT = ROOT / "outputs/transfer_20261026"
REGISTRATION, RESULTS = OUTPUT / "registration.json", OUTPUT / "results.json"


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def own_data(training):
    """``109``'s DATA for the cohort a checkpoint trained on: the caches its run recorded."""
    return {"cohort": ROOT / f"data/synthetic_argo/{training['cohort']}.nc",
            "satellites": Path(training["satellite_cache"]), "surface_grids": Path(training["surface_grids"]),
            "lagged_surface": Path(training["lagged_surface"]), "oi_summary": Path(training["oi_summary"])}


def check(path, device):
    """``109``'s check on the checkpoint's own cohort: its run's 2004-2005 scores must come back."""
    training = load(path)["training"]
    T.DATA, T.COHORT = own_data(training), training["cohort"]
    T.check("grid", path, device)


def glorys_scenes(device, cache):
    """The GLORYS test cohort's scenes as ``159`` built them for ``g_ka`` (built once)."""
    if "glorys" not in cache:
        T.DATA, T.COHORT = GLORYS, COHORT
        cache["glorys"] = T.scenes("grid", {"train_years": GLORYS_TRAIN_YEARS, "scale": None}, device)
    return cache["glorys"]


def compatible(checkpoint, data):
    """The checkpoint's grid, depth levels and query feature count must be the scenes'."""
    grid = (float(data.grid_lat[0]), float(data.grid_lat[1] - data.grid_lat[0]), float(data.grid_lon[0]),
            float(data.grid_lon[1] - data.grid_lon[0]), len(data.grid_lat), len(data.grid_lon))
    config, features = checkpoint["config"], data.x.shape[1] + 2        # + the two first-guess columns
    if (not np.allclose(grid, config["grid"]) or config["n_query_features"] != features
            or not np.allclose(checkpoint["data"]["levels_m"], data.levels)):
        raise SystemExit(f"{checkpoint['training']['tag']}: grid {config['grid']}, {config['n_query_features']} query "
                         f"features, levels {checkpoint['data']['levels_m']}; the GLORYS scenes have {grid}, {features} "
                         f"query features, levels {list(data.levels)}")


def macro_z(collected, data):
    """{lead: the mean over T and S of the RMSE in the scored cohort's per-depth z units} (``101``'s score)."""
    out = {}
    for lead, a in sorted(collected.items()):
        scores = Scores(data.levels, data.norm.std)
        for j, ch in enumerate(CH):
            scores.add(ch, a["mean"][:, j], a["target"][:, j], a["level"])
        out[str(lead)] = scores.result()["macro_z"]
    return out


def evaluate(path, device, cache):
    """One CESM2-trained checkpoint on every cell of the GLORYS test years, with GLORYS's own statistics."""
    checkpoint = load(path)
    training = checkpoint["training"]
    data = glorys_scenes(device, cache)
    compatible(checkpoint, data)
    model = T.model_of("grid", checkpoint, data.device)
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(TEST_YEARS), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return (checkpoint, SS.summarize(collected, data.norm, data.levels, TEST_YEARS), macro_z(collected, data),
            sum(p.numel() for p in model.parameters()))


def register(paths):
    """Once: every check that could refuse a checkpoint runs here, before anything is recorded or scored."""
    if not MAIN.exists():
        raise SystemExit(f"{MAIN} missing: the extra pass needs 159's main pass")
    if not paths:
        raise SystemExit("nothing to register")
    with np.load(GLORYS["surface_grids"]) as grids:
        lat, lon = grids["grid_lat"], grids["grid_lon"]
    grid = (float(lat[0]), float(lat[1] - lat[0]), float(lon[0]), float(lon[1] - lon[0]), len(lat), len(lon))
    for path in paths:
        checkpoint = load(path)
        training = checkpoint["training"]
        if training.get("dataset", "cesm2") != "cesm2" or not training["cohort"].startswith("cesm2"):
            raise SystemExit(f"{path}: trained on {training['cohort']}; the transfer test scores CESM2-trained models only")
        summary = json.loads((Path(path).parent / f"summary_seed{checkpoint['seed']}.json").read_text())
        if summary["best_step"] != checkpoint["step"]:
            raise SystemExit(f"{path}: not its run's chosen checkpoint (step {checkpoint['step']}, chosen {summary['best_step']})")
        if not np.allclose(grid, checkpoint["config"]["grid"]):
            raise SystemExit(f"{path}: grid {checkpoint['config']['grid']}; the GLORYS grids are {grid}")
    sealed = json.loads(T.REGISTRATION.read_text())["data"]
    for name, path in GLORYS.items():
        if SS.sha256(path) != sealed[name]["sha256"]:
            raise SystemExit(f"{path}: not the file 159 registered as {name}")
    record = SS.register(REGISTRATION, [("grid", Path(p).resolve()) for p in paths], {**GLORYS, "main_results": MAIN},
                         {"cohort": COHORT, "test_years": list(TEST_YEARS), "test_years_glorys": [2019, 2020],
                          "leads": list(LEADS), "cells": "every held-out cell", "spec": SPEC,
                          "declared_extra_pass": "the test 159 scored on 2026-10-22; nothing is selected on it",
                          "statistics": "the scored cohort's own (GLORYS); the checkpoints' CESM2 normalisation is recorded, not used"})
    print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")


def same_baselines(entries, main):
    """Every entry's persisted OI and climatology must be the main pass's: the same cells, the same OI."""
    first = main["entries"][0]["by_lead"]
    for e in entries:
        for lead in map(str, LEADS):
            if e["by_lead"][lead]["cells"] != first[lead]["cells"]:
                raise SystemExit(f"{e['tag']} lead {lead}: {e['by_lead'][lead]['cells']} cells, the main pass's "
                                 f"{first[lead]['cells']}")
            for ch in CH:
                for row in ("persisted_oi", "climatology"):
                    a, b = e["by_lead"][lead]["all"][ch][row]["rmse"], first[lead]["all"][ch][row]["rmse"]
                    if abs(a - b) > 1e-9 * max(1., abs(b)):
                        raise SystemExit(f"{e['tag']} {row} lead {lead} {ch}: {a} differs from the main pass {b}")


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the transfer test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    main, cache = json.loads(MAIN.read_text()), {}
    data = glorys_scenes(device, cache)
    for entry in record["entries"]:                  # every refusal before the first sealed cell is scored
        compatible(load(entry["checkpoint"]), data)
    results = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}
    for entry in record["entries"]:
        started = time.time()
        checkpoint, outcome, macro, params = evaluate(entry["checkpoint"], device, cache)
        training = checkpoint["training"]
        results["entries"].append({**entry, "tag": training["tag"], "seed": checkpoint["seed"], "step": checkpoint["step"],
            "trained_on": training["cohort"], "backbone": checkpoint["config"]["backbone"],
            "kriging": checkpoint["config"]["kriging"], "params": int(params),
            "normalization_trained": {k: {ch: checkpoint["normalization"][k][ch].tolist() for ch in CH} for k in ("mean", "std")},
            "seconds": time.time() - started, "by_lead": outcome, "macro_z": macro})
        same_baselines(results["entries"][-1:], main)
        # scores are printed only once they are recorded: a refusal leaves no unrecorded look at the test
        print(json.dumps({"scored": training["tag"], "seconds": round(time.time() - started)}), flush=True)
    norm = cache["glorys"].norm
    results["normalization_used"] = {k: {ch: np.asarray(getattr(norm, k)[ch]).tolist() for ch in CH} for k in ("mean", "std")}
    results["declared_extra_pass"] = {"main_results": str(MAIN), "main_results_sha256": SS.sha256(MAIN),
                                      "identity": "persisted OI, climatology and cells of every entry equal the main pass's"}
    results["test_years_glorys"] = [2019, 2020]
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, RESULTS)
    for e in results["entries"]:
        print(json.dumps({"tag": e["tag"], "macro_z": {k: round(v, 4) for k, v in e["macro_z"].items()}, "rmse": {
            lead: [round(e["by_lead"][lead]["all"][ch]["model"]["rmse"], 4) for ch in CH] for lead in map(str, LEADS)}}), flush=True)
    print(f"wrote {RESULTS}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--register", nargs="+", metavar="CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.check, args.register, args.score))) != 1:
        raise SystemExit("pass one of --check, --register, --score")
    if args.check:
        check(args.check, args.device)
    elif args.register:
        register(args.register)
    else:
        score(args.device)


if __name__ == "__main__":
    main()
