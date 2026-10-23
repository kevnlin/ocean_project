"""The sealed CESM2 test v2: frozen checkpoints scored once on 2008-2010 of the two test cohorts.

Spec ``docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md``. The scorer of ``109`` / ``140``
with two test cohorts (``167``): a checkpoint is scored on the test cohort of its own training
cohort, found by its ``data.cohort_fingerprint`` (``c_flow100`` -> ``cesm2_long_test2``, ``c_y154`` ->
``cesm2_vlong_test2``). No model was trained or selected on 2006-2010. Each step refuses to run twice:

  --fit-damped        damped-persistence factors (121's least squares per lead, variable and depth
                      band) from the persisted-OI and target arrays c_flow100_s1234 saved on
                      2004-2005; c_y154_s1234's arrays must give the same factors
  --check grid:PATH   scores 2004-2005 of the checkpoint's test cohort; the run's own scores must
                      come back (opens nothing: those rows are the training cohort's)
  --register grid:PATH ...   checkpoints, their calibrations (116) if any, both cohorts' data files
                      and the factors, by sha256
  --score             verifies every file, scores every checkpoint on every held-out cell of
                      2008-2010 at leads 0, 1, 3, 6 with the persisted OI and the climatology of
                      the same cells, adds damped persistence per cohort, writes results.json once

  .venv/bin/python experiments/synthetic/168_sealed_cesm2_test2.py --fit-damped
  .venv/bin/python experiments/synthetic/168_sealed_cesm2_test2.py --check grid:PATH
  .venv/bin/python experiments/synthetic/168_sealed_cesm2_test2.py --register grid:PATH ...
  .venv/bin/python experiments/synthetic/168_sealed_cesm2_test2.py --score
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


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


T = _module("scorer109", "experiments/synthetic/109_sealed_cesm2_test.py")
S121 = _module("statistical121", "experiments/synthetic/121_statistical_baselines.py")
SS = T.SS
SPEC = "docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md"
OUTPUT = ROOT / "outputs/cesm2_test2_20261022"
SEALED = OUTPUT / "sealed"
REGISTRATION, RESULTS = SEALED / "registration.json", SEALED / "results.json"
ALPHAS = OUTPUT / "damped_persistence.json"
TEST_YEARS = (2008, 2009, 2010)
SELECTION = (2004, 2005)
LEADS = T.LEADS
#: the runs whose saved 2004-2005 arrays fit the damped-persistence factors (the first is recorded as
#: the source; the others must agree, as the persisted OI and the targets do not depend on the model)
DAMPED_SOURCES = [ROOT / "outputs/wang_20261021/c_flow100_s1234", ROOT / "outputs/wang_20261021/c_y154_s1234"]
COHORTS = ("cesm2_long_test2", "cesm2_vlong_test2")


def data_of(cohort):
    """109's DATA for one test cohort."""
    caches = OUTPUT / cohort
    return {"cohort": ROOT / f"data/synthetic_argo/{cohort}.nc", "satellites": caches / "satellites.npz",
            "surface_grids": caches / "surface_grids.npz", "lagged_surface": caches / "lagged_surface.npz",
            # the test cohort's own OI file (44 --cohort): tuned on its 2004 rows, which are the training cohort's,
            # so the selection equals the training cohort's exactly (167 checks it)
            "oi_summary": ROOT / f"outputs/audit/synthetic/fixed_baselines_{cohort}/summary.json",
            "scale_tables": T.DATA["scale_tables"]}


def cohort_table():
    """{test cohort: (training cohort, training cohort fingerprint)} from the caches' metadata."""
    table = {}
    for cohort in COHORTS:
        with np.load(data_of(cohort)["satellites"], allow_pickle=False) as cache:
            meta = json.loads(str(cache["metadata"].item()))
        table[cohort] = meta["training_cohort"], meta["training_cohort_fingerprint"]
    return table


def cohort_of(checkpoint, path, table):
    """The test cohort of a checkpoint: the one built on its training cohort."""
    fingerprint, name = checkpoint["data"]["cohort_fingerprint"], checkpoint["training"]["cohort"]
    matches = [c for c, (train, fp) in table.items() if fp == fingerprint]
    if len(matches) != 1 or table[matches[0]][0] != name:
        raise SystemExit(f"{path}: trained on {name} ({fingerprint[:12]}), which no test cohort was built on")
    return matches[0]


def evaluate(family, path, years, device, cache, scales=None, table=None):
    """``sealed_scoring.summarize`` of one checkpoint on every cell of ``years`` of its test cohort."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    cohort = cohort_of(checkpoint, path, table or cohort_table())
    key = (cohort, family, training["train_years"], training.get("scale"))
    if key not in cache:
        cache.clear()
        T.DATA, T.COHORT = data_of(cohort), cohort
        cache[key] = T.scenes(family, training, device)
    data = cache[key]
    for kind in ("mean", "std"):
        for ch in ("TEMP", "SALT"):
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the test cohort's normalisation differs from training's")
    model = T.model_of(family, checkpoint, data.device)
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(years), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return checkpoint, cohort, SS.summarize(collected, data.norm, data.levels, years, scales)


def fit_damped():
    if ALPHAS.exists():
        raise SystemExit(f"{ALPHAS} exists: the factors are fitted once")
    fits = []
    for run in DAMPED_SOURCES:
        summary = json.loads(next(run.glob("summary_seed*.json")).read_text())
        if set(summary["val_years"]) != set(SELECTION):
            raise SystemExit(f"{run} was not chosen on 2004-2005")
        path = run / f"forecast_seed{summary['seed']}.npz"
        with np.load(path, allow_pickle=False) as z:
            if not set((2000 + z["month"] // 12).tolist()) <= set(SELECTION):
                raise SystemExit(f"{path}: arrays outside 2004-2005")
            scale = z["normalization_scale"][z["level"]]
            alphas = S121.fit_alphas(z["baseline"] * scale, z["target"] * scale, z["lead"].astype(int), z["level"],
                                     summary["data"]["levels_m"])
            cells = int(len(z["level"]))
        fits.append({"source": str(path), "source_sha256": S121.sha256(path), "run": summary["tag"], "cells": cells,
                     "levels_m": summary["data"]["levels_m"], "alphas": alphas})
    first = fits[0]
    worst = 0.
    for other in fits[1:]:
        if other["levels_m"] != first["levels_m"] or other["cells"] != first["cells"]:
            raise SystemExit("the source runs saved different cells")
        for lead in first["alphas"]:
            for ch in first["alphas"][lead]:
                for band, a in first["alphas"][lead][ch].items():
                    worst = max(worst, abs(other["alphas"][lead][ch][band] - a))
    if worst > 1e-9:
        raise SystemExit(f"the source runs give different factors (largest difference {worst:.3g})")
    ALPHAS.write_text(json.dumps({"created_unix": time.time(), "method": "damped persistence: alpha per (lead, variable, "
                                  "depth band), least squares of the target on the persisted OI in physical units (121)",
                                  "fit_years": list(SELECTION), **first,
                                  "agreeing_sources": [{k: f[k] for k in ("source", "source_sha256", "run")} for f in fits[1:]],
                                  "largest_difference_between_sources": worst}, indent=1) + "\n")
    print(json.dumps({lead: {ch: {b: round(a, 4) for b, a in bands.items()} for ch, bands in by_ch.items()}
                      for lead, by_ch in first["alphas"].items()}, indent=1))
    print(f"wrote {ALPHAS} (sources agree to {worst:.2g})")


def check(family, path, device):
    """Score a checkpoint on 2004-2005 of its test cohort; its run's own scores must come back."""
    checkpoint, cohort, outcome = evaluate(family, path, SELECTION, device, {})
    summary = json.loads((Path(path).parent / f"summary_seed{checkpoint['seed']}.json").read_text())
    if summary["best_step"] != checkpoint["step"]:
        raise SystemExit("the checkpoint is not the run's chosen one")
    print(f"{checkpoint['training']['tag']} (step {checkpoint['step']}) on {cohort}")
    worst = 0.
    for lead in map(str, LEADS):
        run = summary["forecast"]["best"]["by_lead"][lead]
        for ch in ("TEMP", "SALT"):
            mine = outcome[lead]["all"][ch]
            for name, theirs in (("model", run["scores"][ch]["rmse_physical"]),
                                 ("persisted_oi", run["baseline_scores"][ch]["rmse_physical"])):
                worst = max(worst, abs(mine[name]["rmse"] / theirs - 1))
            print(f"lead {lead} {ch}: model {mine['model']['rmse']:.5f} (run {run['scores'][ch]['rmse_physical']:.5f}), "
                  f"persisted OI {mine['persisted_oi']['rmse']:.5f} (run {run['baseline_scores'][ch]['rmse_physical']:.5f}), "
                  f"CRPS {mine['model']['crps']:.5f}, 95% coverage {mine['model']['coverage_95']:.3f}, "
                  f"cells {outcome[lead]['cells']}")
    if worst > 1e-3:
        raise SystemExit(f"the test cohort does not reproduce the run's 2004-2005 scores (relative {worst:.2g})")
    print(f"check passed: 2004-2005 reproduced to a relative {worst:.2g}")


def register(items):
    import torch
    if REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} exists: a test is registered once")
    if not ALPHAS.exists():
        raise SystemExit(f"{ALPHAS} missing: run --fit-damped first")
    table = cohort_table()
    entries, cohorts = [], {}
    for item in items:
        family, path = item.split(":", 1)
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cohorts[str(path)] = cohort_of(checkpoint, path, table)
        entries.append((family, path, T.calibration_of(path)))
    data = {f"{cohort}/{name}": p for cohort in COHORTS for name, p in data_of(cohort).items()}
    data["damped_persistence_factors"] = ALPHAS
    meta = {"test_years": list(TEST_YEARS), "leads": list(LEADS), "cells": "every held-out cell", "spec": SPEC,
            "cohorts": {cohort: {"training_cohort": train, "training_cohort_fingerprint": fp} for cohort, (train, fp) in table.items()},
            "checkpoint_cohorts": cohorts}
    record = SS.register(REGISTRATION, entries, data, meta)
    print(f"registered {len(record['entries'])} checkpoints ({sum('calibration' in e for e in record['entries'])} "
          f"with a calibration) on {sorted(set(cohorts.values()))} -> {REGISTRATION}")


def score(device):
    """109's scoring per checkpoint on its cohort, plus damped persistence on each cohort's cells."""
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    factors = json.loads(ALPHAS.read_text())
    table = cohort_table()
    results, cache = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    original_collect = SS.collect
    kept, current = {}, {}

    def collect(*args, **kwargs):           # keep each cohort's first checkpoint's cells for damped persistence
        out = original_collect(*args, **kwargs)
        kept.setdefault(current["cohort"], out)
        return out

    SS.collect = collect
    try:
        for entry in record["entries"]:
            started = time.time()
            current["cohort"] = record["checkpoint_cohorts"][entry["checkpoint"]]
            scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"] if "calibration" in entry else None
            checkpoint, cohort, outcome = evaluate(entry["family"], entry["checkpoint"], TEST_YEARS, device, cache, scales, table)
            if cohort != current["cohort"]:
                raise SystemExit(f"{entry['checkpoint']}: cohort differs from the registration")
            results["entries"].append({**entry, "cohort": cohort, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                                       "step": checkpoint["step"], "seconds": time.time() - started, "by_lead": outcome})
            print(json.dumps({"scored": checkpoint["training"]["tag"], "cohort": cohort, "rmse": {
                lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in ("TEMP", "SALT")]
                for lead in map(str, LEADS)}}), flush=True)
            if cohort not in results.setdefault("norms", {}):
                data = next(iter(cache.values()))
                results["norms"][cohort] = (data.norm, data.levels)
    finally:
        SS.collect = original_collect
    norms = results.pop("norms")
    results["damped_persistence"] = {"alphas_sha256": SS.sha256(ALPHAS), "method": "damped persistence", "by_cohort": {}}
    for cohort, cells in kept.items():
        norm, levels = norms[cohort]
        by_lead = {}
        for lead, a in sorted(cells.items()):
            mean = S121.damped(a["base"], np.full(len(a["level"]), lead), a["level"], factors["levels_m"], factors["alphas"])
            year = 2000 + a["month"] // 12
            by_lead[str(lead)] = {"all": SS.metrics(mean, None, a["base"], a["target"], a["level"], norm, levels),
                                  "cells": int(len(a["level"]))}
            for y in TEST_YEARS:
                keep = year == y
                by_lead[str(lead)][str(y)] = SS.metrics(mean[keep], None, a["base"][keep], a["target"][keep],
                                                        a["level"][keep], norm, levels)
        results["damped_persistence"]["by_cohort"][cohort] = {"by_lead": by_lead}
    temporary = RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, RESULTS)
    print(f"wrote {RESULTS}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-damped", action="store_true")
    p.add_argument("--register", nargs="+", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--score", action="store_true")
    p.add_argument("--check", metavar="FAMILY:CHECKPOINT")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.fit_damped, args.register, args.score, args.check))) != 1:
        raise SystemExit("pass one of --fit-damped, --register, --score, --check")
    if args.fit_damped:
        fit_damped()
    elif args.check:
        family, path = args.check.split(":", 1)
        check(family, path, args.device)
    elif args.register:
        register(args.register)
    else:
        score(args.device)


if __name__ == "__main__":
    main()
