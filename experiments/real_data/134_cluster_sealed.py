"""The CLOINet-style read-out on the two opened tests, as one declared additional pass.

Added after both tests were opened (2026-10-21, at Zihao's request for a comparison with CLOINet,
the closest ocean precedent); nothing in it is chosen on test data. The runs of ``131`` use method
A's setups with the cluster covariance; their checkpoints were chosen on the selection years and
their spread calibrations fitted there (``116``). Checkpoints, calibrations and test files are
registered by sha256 before any scoring; each test is then scored once with the scorers of ``109``
and ``113``, on the same cells as the main pass (the persisted OI must come back unchanged).

  .venv/bin/python experiments/real_data/134_cluster_sealed.py --register
  .venv/bin/python experiments/real_data/134_cluster_sealed.py --score cesm2|real
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

from ocean_tokenizer import calibration as CAL
from ocean_tokenizer import sealed_scoring as SS

RUNS = ROOT / "outputs/cluster_20261021"
OUTPUT = RUNS / "sealed"
REGISTRATION = OUTPUT / "registration.json"
CALIBRATIONS = ROOT / "outputs/calibration_20261020"
SEEDS = (1234, 1235, 1236)
TESTS = {"cesm2": ("cl_cesm2", ROOT / "experiments/synthetic/109_sealed_cesm2_test.py"),
         "real": ("cl_real", ROOT / "experiments/real_data/113_sealed_real_test.py")}


def scorer(test):
    spec = importlib.util.spec_from_file_location(f"scorer_{test}", TESTS[test][1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chosen(arm, seed):
    path = RUNS / f"{arm}_s{seed}" / f"best_seed{seed}.pt"
    summary = json.loads((path.parent / f"summary_seed{seed}.json").read_text())
    if summary["kriging"] != "ka_cluster" or summary["completed_steps"] not in (30000, 70000):
        raise SystemExit(f"{path}: not a finished cluster run")
    calibration = CALIBRATIONS / CAL.record_name(path.parent)
    if not calibration.exists():
        raise SystemExit(f"{path}: fit its calibration first (116)")
    return path, calibration


def register():
    entries, data = [], {}
    for test, (arm, _) in TESTS.items():
        entries += [("grid", *chosen(arm, seed)) for seed in SEEDS]
        data.update({f"{test}_{name}": path for name, path in scorer(test).DATA.items()})
    record = SS.register(REGISTRATION, entries, data, {"tests": {"cesm2": [2006, 2007], "real": [2022, 2023]},
                         "leads": [0, 1, 3, 6], "pass": "CLOINet-style read-out, added after both tests were opened"})
    print(f"registered {len(record['entries'])} checkpoints -> {REGISTRATION}")


def score(test, device):
    results_path = OUTPUT / f"results_{test}.json"
    if results_path.exists():
        raise SystemExit(f"{results_path} exists: each test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    S = scorer(test)
    arm = TESTS[test][0]
    main_pass = json.loads(S.RESULTS.read_text())["entries"][0]["by_lead"]
    out, cache = {"registration_sha256": SS.sha256(REGISTRATION), "entries": []}, {}
    for entry in record["entries"]:
        if f"/{arm}_s" not in entry["checkpoint"]:
            continue
        scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"]
        if test == "cesm2":
            checkpoint, outcome = S.evaluate("grid", entry["checkpoint"], S.TEST_YEARS, device, cache, scales)
        else:
            checkpoint, outcome = S.evaluate(entry["checkpoint"], S.TEST_YEARS, device, cache, scales)
        for lead, block in outcome.items():
            for ch in ("TEMP", "SALT"):
                mine, theirs = block["all"][ch]["persisted_oi"]["rmse"], main_pass[lead]["all"][ch]["persisted_oi"]["rmse"]
                if abs(mine / theirs - 1) > 1e-4 or block["cells"] != main_pass[lead]["cells"]:
                    raise SystemExit(f"lead {lead} {ch}: not the cells of the main pass")
        out["entries"].append({**entry, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                               "step": checkpoint["step"], "by_lead": outcome})
        print(json.dumps({"scored": checkpoint["training"]["tag"], "rmse": {
            lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in ("TEMP", "SALT")] for lead in outcome}}),
              flush=True)
    temporary = results_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(out, indent=1, default=float) + "\n")
    os.replace(temporary, results_path)
    print(f"wrote {results_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--register", action="store_true")
    p.add_argument("--score", choices=list(TESTS))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.register == bool(args.score):
        raise SystemExit("pass --register or --score TEST")
    register() if args.register else score(args.score, args.device)


if __name__ == "__main__":
    main()
