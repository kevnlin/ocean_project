"""The sealed GLORYS12 test: frozen checkpoints scored once on GLORYS 2019-2020 of ``glorys12_test``.

Spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``. ``109``'s scorer (as ``140`` used it for
iHESP) with the GLORYS test cohort and caches (``157``, ``156``) and the test cohort's own OI
file (``44 --cohort glorys12_test``: tuned on the cohort's 2004 rows, which are the training
cohort's). The test years are the cohort's 2006-2007, GLORYS 2019-2020 (month index 72-95); no
model was trained or selected on them. Each step refuses to run twice:

  --fit-damped        damped-persistence factors (121's least squares per lead, variable and depth
                      band) from a run's saved predictions of 2004-2005 (GLORYS 2017-2018)
  --check grid:PATH   scores 2004-2005 of the test cohort; the run's own scores must come back
  --register grid:PATH ...   checkpoints, their calibrations (116), the data files and the damped-
                      persistence factors, by sha256
  --score             verifies every file, scores every checkpoint on every held-out cell of
                      2006-2007 at leads 0, 1, 3, 6, adds damped persistence on the same cells,
                      writes results.json once

  .venv/bin/python experiments/synthetic/159_sealed_glorys_test.py --fit-damped
  .venv/bin/python experiments/synthetic/159_sealed_glorys_test.py --check grid:PATH
  .venv/bin/python experiments/synthetic/159_sealed_glorys_test.py --register grid:PATH ...
  .venv/bin/python experiments/synthetic/159_sealed_glorys_test.py --score
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
BASE = ROOT / "outputs/glorys_20261021"
CACHES = BASE / "glorys12_test"
T.OUTPUT = CACHES / "sealed"
T.REGISTRATION, T.RESULTS = T.OUTPUT / "registration.json", T.OUTPUT / "results.json"
T.COHORT = "glorys12_test"
T.TEST_YEARS = (2006, 2007)                   # cohort calendar: GLORYS 2019-2020
SPEC = "docs/superpowers/specs/2026-10-21-glorys-osse.md"
ALPHAS = BASE / "damped_persistence.json"
#: the run whose saved 2004-2005 predictions fit the damped-persistence factors
DAMPED_SOURCE = BASE / "runs/g_ka_s1234"
T.DATA = {"cohort": ROOT / "data/synthetic_argo/glorys12_test.nc", "satellites": CACHES / "satellites.npz",
          "surface_grids": CACHES / "surface_grids.npz", "lagged_surface": CACHES / "lagged_surface.npz",
          # the test cohort's own OI file (44 --cohort glorys12_test): OI tuned on its 2004 rows, which are the training
          # cohort's, so the selection equals fixed_baselines_glorys12_uniform's
          "oi_summary": ROOT / "outputs/audit/synthetic/fixed_baselines_glorys12_test/summary.json",
          "damped_persistence_factors": ALPHAS}
SELECTION = (2004, 2005)


def fit_damped():
    if ALPHAS.exists():
        raise SystemExit(f"{ALPHAS} exists: the factors are fitted once")
    summary = json.loads(next(DAMPED_SOURCE.glob("summary_seed*.json")).read_text())
    if set(summary["val_years"]) != set(SELECTION):
        raise SystemExit("the source run was not chosen on 2004-2005")
    path = DAMPED_SOURCE / f"forecast_seed{summary['seed']}.npz"
    with np.load(path, allow_pickle=False) as z:
        if not set((2000 + z["month"] // 12).tolist()) <= set(SELECTION):
            raise SystemExit(f"{path}: arrays outside 2004-2005")
        scale = z["normalization_scale"][z["level"]]
        alphas = S121.fit_alphas(z["baseline"] * scale, z["target"] * scale, z["lead"].astype(int), z["level"],
                                 summary["data"]["levels_m"])
    ALPHAS.write_text(json.dumps({"created_unix": time.time(), "source": str(path), "source_sha256": S121.sha256(path),
                                  "fit_years": list(SELECTION), "fit_years_glorys": [2017, 2018],
                                  "levels_m": summary["data"]["levels_m"], "alphas": alphas}, indent=1) + "\n")
    print(f"wrote {ALPHAS}")


def score(device):
    """109's scoring, plus damped persistence on the first checkpoint's cells."""
    if T.RESULTS.exists():
        raise SystemExit(f"{T.RESULTS} exists: the test is scored once")
    record = json.loads(T.REGISTRATION.read_text())
    SS.verify(record)
    factors = json.loads(ALPHAS.read_text())
    results, cache = {"registration_sha256": SS.sha256(T.REGISTRATION), "started_unix": time.time(), "entries": []}, {}
    original_collect = SS.collect
    kept = {}

    def collect(*args, **kwargs):           # keep the first checkpoint's cells for damped persistence
        out = original_collect(*args, **kwargs)
        kept.setdefault("cells", out)
        return out

    SS.collect = collect
    try:
        for entry in record["entries"]:
            started = time.time()
            scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"] if "calibration" in entry else None
            checkpoint, outcome = T.evaluate(entry["family"], entry["checkpoint"], T.TEST_YEARS, device, cache, scales)
            results["entries"].append({**entry, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                                       "step": checkpoint["step"], "seconds": time.time() - started, "by_lead": outcome})
            print(json.dumps({"scored": checkpoint["training"]["tag"], "rmse": {
                lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in ("TEMP", "SALT")]
                for lead in map(str, T.LEADS)}}), flush=True)
    finally:
        SS.collect = original_collect
    data = next(iter(cache.values()))
    damped = {"alphas_sha256": SS.sha256(ALPHAS), "method": "damped persistence", "by_lead": {}}
    for lead, a in sorted(kept["cells"].items()):
        mean = S121.damped(a["base"], np.full(len(a["level"]), lead), a["level"], factors["levels_m"], factors["alphas"])
        damped["by_lead"][str(lead)] = {"all": SS.metrics(mean, None, a["base"], a["target"], a["level"], data.norm,
                                                          data.levels), "cells": int(len(a["level"]))}
    results["damped_persistence"] = damped
    results["test_years_glorys"] = [2019, 2020]
    temporary = T.RESULTS.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1, default=float) + "\n")
    os.replace(temporary, T.RESULTS)
    print(f"wrote {T.RESULTS}", flush=True)


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
        T.check(family, path, args.device)
    elif args.register:
        entries = [(item.split(":", 1)[0], Path(item.split(":", 1)[1])) for item in args.register]
        record = SS.register(T.REGISTRATION, [(family, path, T.calibration_of(path)) for family, path in entries], T.DATA,
                             {"cohort": T.COHORT, "test_years": list(T.TEST_YEARS), "test_years_glorys": [2019, 2020],
                              "leads": list(T.LEADS), "cells": "every held-out cell", "spec": SPEC})
        print(f"registered {len(record['entries'])} checkpoints ({sum('calibration' in e for e in record['entries'])} "
              f"with a calibration) -> {T.REGISTRATION}")
    else:
        score(args.device)


if __name__ == "__main__":
    main()
