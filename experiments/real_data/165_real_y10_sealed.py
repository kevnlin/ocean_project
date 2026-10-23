"""The 10-year training window (``r_y10``, trained on real 2011-2020) on the opened 2024 real test,
as ONE declared additional pass.

Added after the 2024 test was opened (``133_sealed_real_2024.py``, 2026-10-21); nothing in it is
chosen on test data: the three runs trained on 2011-2020 with the paper's 2016-2020 normalisation,
targets, 2021 validation cells and frozen OI selection (``164`` checked them identical), their
checkpoints were chosen on 2021 and their spread calibrations are fitted on 2021 (``116``). The
pattern is ``134_cluster_sealed.py``: own registration and results files, every checkpoint,
calibration and data file registered by sha256 before any scoring, one scoring of every held-out
cell of 2024 at leads 0, 1, 3, 6 with the scorers of ``133`` (``sealed_scoring``), and the persisted
OI and the climatology of the main pass must come back on the same cells. 2025 is never read
(the scenes refuse any year after 2024). The main pass's files are only read.

  --calibrate         fits 116's spread calibration of each run on its 2021 predictions into
                      outputs/real_y10_20261021/calibration/ (the paper's calibrations are untouched)
  --check CHECKPOINT  scores the run's own 2021 cells from the 2011-2024 test-cohort scenes; the
                      summary's scores must come back (133's relative 1e-3: under bf16 autocast another
                      process reproduces the model's RMSE to about 1e-5, the persisted OI to the bit;
                      the worst relative difference is recorded in check_s<seed>.json)
  --register          records the three checkpoints, their calibrations, the caches, the cohort,
                      the OI selection and the main pass's files by sha256; once
  --score             verifies every file, scores every checkpoint on every held-out cell of 2024,
                      checks the persisted OI and the climatology against the main pass, writes
                      results.json (with the mean and sd over seeds) and extra_pass.md; once

  .venv/bin/python experiments/real_data/165_real_y10_sealed.py --calibrate
  .venv/bin/python experiments/real_data/165_real_y10_sealed.py --check RUN/best_seedN.pt
  .venv/bin/python experiments/real_data/165_real_y10_sealed.py --register
  .venv/bin/python experiments/real_data/165_real_y10_sealed.py --score
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import calibration as CAL
from ocean_tokenizer import sealed_scoring as SS

CACHES = ROOT / "outputs/real_y10_20261021"
RUNS = CACHES / "runs"
OUTPUT = CACHES / "sealed_2024"
CALIBRATIONS = CACHES / "calibration"
REGISTRATION = OUTPUT / "registration.json"
RESULTS = OUTPUT / "results.json"
REPORT = OUTPUT / "extra_pass.md"
CHECKS_164 = CACHES / "checks.json"
FIT_SCRIPT = ROOT / "experiments/synthetic/116_fit_calibration.py"
#: the paper's 2024 test (ka_dev4): read only, never re-scored
MAIN = Path("/DATA2/linke/ocean_project_ka_dev4/outputs/real_2024_test/sealed")
MAIN_REGISTRATION, MAIN_RESULTS = MAIN / "registration.json", MAIN / "results.json"
SEEDS = (1234, 1235, 1236)
TAG = "r_y10"
YEARS, TRAIN_SPLIT = (2011, 2024), (2011, 2020)
TEST_YEARS, SELECTION_YEARS = (2024,), (2021,)
LEADS = (0, 1, 3, 6)
CH = ("TEMP", "SALT")
#: the 2016-2023 cohort on which the frozen OI selection (106) was made
OI_COHORT = "6366666ef8ad4ec33388fd8cc7d7f1666236d8f29158b28bee4818f16584e3a1"
DATA = {"cohort": ROOT / "data/argo_cohort/global_global.nc", "anomalies": ROOT / "data/argo_cohort/global_anomx.npz",
        "satellites": CACHES / "satellites.npz", "surface_grids": CACHES / "surface_grids.npz",
        "lagged_surface": CACHES / "lagged_surface.npz",
        "oi_summary": ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json"}
#: the paper's 2024 rows (RMSE by lead 0/1/3/6), for the report only
PAPER = {"KA-Ocean, trained 2016-2020 (paper, mean of 3 seeds)": {"TEMP": (0.849, 1.009, 1.095, 1.131),
                                                                   "SALT": (0.1399, 0.1633, 0.1762, 0.1830)},
         "ARMOR3D-style OI (paper)": {"TEMP": (0.868, 1.018, 1.102, 1.139), "SALT": (0.1431, 0.1648, 0.1777, 0.1837)}}
CHECK_RTOL = 1e-3   # 133's; the model's RMSE comes back to ~1e-5 (bf16 autocast), the OI exactly
IDENTITY_RTOL = 1e-4


def run_folder(seed):
    return RUNS / f"{TAG}_s{seed}"


def summary_of(folder):
    summaries = sorted(Path(folder).glob("summary_seed*.json"))
    if len(summaries) != 1:
        raise SystemExit(f"{folder}: expected one finished run, found {len(summaries)} summaries")
    return json.loads(summaries[0].read_text())


def frozen_run(folder):
    """The run trained on 2011-2020 of the 2011-2024 cohort with the paper's normalisation, chose
    its checkpoint on 2021 and finished; the chosen checkpoint and its calibration."""
    folder = Path(folder)
    s = summary_of(folder)
    t = s["training"]
    problems = []
    if s["completed_steps"] != t["steps"]:
        problems.append("not finished")
    if s["data"].get("cohort") != "real_global 2011-2024" or list(t.get("real_years") or []) != list(YEARS):
        problems.append("not the 2011-2024 cohort")
    if list(s.get("real_train_split") or []) != list(TRAIN_SPLIT) or s["data"].get("splits", {}).get("train") != list(TRAIN_SPLIT):
        problems.append("not trained on 2011-2020")
    if s["data"].get("normalization_splits", {}).get("train") != [2016, 2020]:
        problems.append("not the 2016-2020 normalisation")
    if set(s["val_years"]) != set(SELECTION_YEARS):
        problems.append("not chosen on 2021")
    if t.get("real_oi_cohort") != OI_COHORT:
        problems.append("not the frozen OI selection of the 2016-2023 cohort")
    if s["tag"] != folder.name:
        problems.append(f"tag {s['tag']!r} is not the folder's")
    if problems:
        raise SystemExit(f"{folder}: " + "; ".join(problems))
    checkpoint = folder / f"best_seed{s['seed']}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} is missing")
    return s, checkpoint, CALIBRATIONS / CAL.record_name(folder)


def scenes(device):
    from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
    return RealGridForecastScenes(ROOT, device, caches=CACHES, oi_summary=DATA["oi_summary"], leads=LEADS,
                                  years=YEARS, train_split=TRAIN_SPLIT, oi_cohort_fingerprint=OI_COHORT)


def evaluate(path, years, device, cache, scales=None):
    """``sealed_scoring.summarize`` of one checkpoint on every held-out cell of ``years`` (as 133),
    and the collected cells."""
    import torch
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    if training.get("dataset") != "real_global" or list(training.get("real_years") or []) != list(YEARS) \
            or list(training.get("real_train_split") or []) != list(TRAIN_SPLIT):
        raise SystemExit(f"{path} was not trained on real 2011-2020 of the 2011-2024 cohort")
    if "data" not in cache:
        cache["data"] = scenes(device)
    data = cache["data"]
    for kind in ("mean", "std"):
        for ch in CH:
            if not np.array_equal(np.asarray(checkpoint["normalization"][kind][ch]), getattr(data.norm, kind)[ch]):
                raise SystemExit(f"{path}: the scenes' normalisation differs from training's")
    model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(data.device).eval()
    collected = SS.collect(model, data, data.analysis(), data.forecast_months(list(years), 0),
                           training.get("eval_context_profiles") or training["context_profiles"], training["amp"])
    return checkpoint, SS.summarize(collected, data.norm, data.levels, years, scales), collected


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1, default=float) + "\n")
    os.replace(temporary, path)


def calibrate():
    """116's CRPS-optimal factors per lead, variable and depth band on each run's saved 2021
    predictions, written into this stream's own folder (116 refuses any sealed year)."""
    spec = importlib.util.spec_from_file_location("fit116", FIT_SCRIPT)
    fit116 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit116)
    CALIBRATIONS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        folder = run_folder(seed)
        _, checkpoint, target = frozen_run(folder)
        record = fit116.fit(folder)
        if record["fit_years"] != list(SELECTION_YEARS) or record["checkpoint"] != str(checkpoint):
            raise SystemExit(f"{folder}: the calibration is not of the chosen checkpoint on 2021")
        record["fit_script"] = {"path": str(FIT_SCRIPT), "sha256": SS.sha256(FIT_SCRIPT)}
        if target.exists():
            if json.loads(target.read_text())["checkpoint_sha256"] != record["checkpoint_sha256"]:
                raise SystemExit(f"{target} was fitted to another checkpoint")
            print(f"kept {target} (same checkpoint)")
            continue
        write_json(target, record)
        six = record["in_sample"]["6"]["TEMP"]
        print(f"wrote {target}; lead 6 T coverage95 {six['raw']['coverage_95']:.3f} -> {six['calibrated']['coverage_95']:.3f}, "
              f"CRPS {six['raw']['crps']:.4f} -> {six['calibrated']['crps']:.4f}")


def check(path, device):
    """The run's own 2021 cells from the 2011-2024 scenes; its summary's scores must come back."""
    path = Path(path).resolve()
    summary, chosen, _ = frozen_run(path.parent)
    if path != chosen.resolve():
        raise SystemExit(f"{path} is not the run's chosen checkpoint {chosen}")
    checkpoint, outcome, _ = evaluate(path, SELECTION_YEARS, device, {})
    if summary["best_step"] != checkpoint["step"] or summary["seed"] != checkpoint["seed"]:
        raise SystemExit("the checkpoint is not the run's chosen one")
    worst, rows = 0., {}
    for lead in map(str, LEADS):
        run = summary["forecast"]["best"]["by_lead"][lead]
        rows[lead] = {"cells": outcome[lead]["cells"]}
        for ch in CH:
            mine = outcome[lead]["all"][ch]
            rows[lead][ch] = {}
            for name, theirs in (("model", run["scores"][ch]["rmse_physical"]),
                                 ("persisted_oi", run["baseline_scores"][ch]["rmse_physical"])):
                relative = abs(mine[name]["rmse"] / theirs - 1)
                worst = max(worst, relative)
                rows[lead][ch][name] = {"rmse": mine[name]["rmse"], "summary_rmse": theirs, "n": mine[name]["n"],
                                        "summary_n": run["scores" if name == "model" else "baseline_scores"][ch]["n"],
                                        "relative": relative}
    passed = worst <= CHECK_RTOL
    write_json(OUTPUT / f"check_s{checkpoint['seed']}.json",
               {"checkpoint": str(path), "sha256": SS.sha256(path), "step": checkpoint["step"], "seed": checkpoint["seed"],
                "years": list(SELECTION_YEARS), "rtol": CHECK_RTOL, "worst_relative": worst, "passed": passed, "by_lead": rows,
                "checked_unix": time.time()})
    if not passed:
        raise SystemExit(f"the 2011-2024 scenes do not reproduce the run's 2021 scores (relative {worst:.2g} > {CHECK_RTOL})")
    print(f"check passed: 2021 reproduced to a relative {worst:.2g} ({path})")


def register():
    entries = []
    for seed in SEEDS:
        _, checkpoint, calibration = frozen_run(run_folder(seed))
        if not calibration.exists():
            raise SystemExit(f"{checkpoint}: fit its calibration first (--calibrate)")
        verdict = OUTPUT / f"check_s{seed}.json"
        if not verdict.exists() or not json.loads(verdict.read_text())["passed"] \
                or json.loads(verdict.read_text())["sha256"] != SS.sha256(checkpoint):
            raise SystemExit(f"{checkpoint}: run --check on it first")
        entries.append(("grid", checkpoint, calibration))
    file = lambda p: {"path": str(p), "sha256": SS.sha256(p)}
    record = SS.register(REGISTRATION, entries, DATA, {
        "dataset": "real_global 2011-2024", "test_years": list(TEST_YEARS), "leads": list(LEADS),
        "cells": "every held-out-float cell",
        "pass": "10-year training window (2011-2020, the paper's 2016-2020 normalisation, targets, 2021 selection and "
                "OI selection), one declared extra pass added after the 2024 test was opened",
        "training_split": list(TRAIN_SPLIT), "normalization_split": [2016, 2020], "selection_years": list(SELECTION_YEARS),
        "oi_cohort_fingerprint": OI_COHORT,
        "main_pass": {"registration": file(MAIN_REGISTRATION), "results": file(MAIN_RESULTS)},
        "data_checks_164": file(CHECKS_164),
        "checks_2021": {str(seed): file(OUTPUT / f"check_s{seed}.json") for seed in SEEDS},
        "fit_script_116": file(FIT_SCRIPT)})
    print(f"registered {len(record['entries'])} checkpoints ({sum('calibration' in e for e in record['entries'])} "
          f"with a calibration) -> {REGISTRATION}")


def identity(outcome, main_pass):
    """The persisted OI and the climatology must be the main pass's on every lead, variable and
    band, and the cells must be as many; the largest relative difference seen."""
    worst = 0.
    for lead, block in outcome.items():
        theirs = main_pass[lead]
        if block["cells"] != theirs["cells"]:
            raise SystemExit(f"lead {lead}: {block['cells']} cells, the main pass scored {theirs['cells']}")
        for ch in CH:
            for row in ("persisted_oi", "climatology"):
                mine, ref = block["all"][ch][row], theirs["all"][ch][row]
                pairs = [(mine["n"], ref["n"]), (mine["rmse"], ref["rmse"]), (mine["mae"], ref["mae"])]
                pairs += [(mine["by_band"][b]["rmse"], ref["by_band"][b]["rmse"]) for b in ref["by_band"]]
                for a, b in pairs:
                    relative = abs(a / b - 1) if b else abs(a - b)
                    worst = max(worst, relative)
                    if relative > IDENTITY_RTOL or mine["n"] != ref["n"]:
                        raise SystemExit(f"lead {lead} {ch} {row}: not the cells of the main pass ({a} vs {b})")
    return worst


def aggregate(entries):
    """Mean and sd (over seeds) of every scalar metric of every row, overall and by depth band."""
    def stats(values):
        values = np.asarray(values, np.float64)
        return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "seeds": values.tolist()}
    out = {}
    for lead in map(str, LEADS):
        out[lead] = {"cells": entries[0]["by_lead"][lead]["cells"]}
        for ch in CH:
            out[lead][ch] = {}
            first = entries[0]["by_lead"][lead]["all"][ch]
            for row in first:
                blocks = [e["by_lead"][lead]["all"][ch][row] for e in entries]
                out[lead][ch][row] = {k: stats([b[k] for b in blocks]) for k, v in first[row].items()
                                      if isinstance(v, (int, float)) and v is not None}
                out[lead][ch][row]["by_band"] = {
                    band: {k: stats([b["by_band"][band][k] for b in blocks]) for k, v in first[row]["by_band"][band].items()
                           if isinstance(v, (int, float)) and v is not None} for band in first[row]["by_band"]}
    return out


def report(results):
    """``extra_pass.md``: the 2024 table next to the paper's rows."""
    S = results["summary"]
    seeds = [e["seed"] for e in results["entries"]]
    fmt = {"TEMP": "{:.4f}", "SALT": "{:.4f}"}
    lines = ["# The 10-year training window on the 2024 real test (declared extra pass)", "",
             f"Three `r_y10` runs (seeds {', '.join(map(str, seeds))}; real 2011-2020 training, the paper's 2016-2020 "
             "normalisation, targets, 2021 checkpoint selection and OI selection), scored once on every held-out cell "
             f"of 2024 ({S['0']['cells']:,} cells a lead) with the scorers of the paper's 2024 test (`133`), as "
             "`134` scored the CLOINet-style runs. The persisted OI and the climatology on the same cells equal the "
             f"main pass's (largest relative difference {results['main_pass']['worst_relative_difference']:.1e}).", "",
             "## RMSE of the anomaly by lead (months), 2024", "",
             "| row | T 0 | T 1 | T 3 | T 6 | S 0 | S 1 | S 3 | S 6 |", "|---|---|---|---|---|---|---|---|---|"]
    def row(name, t, s):
        lines.append(f"| {name} | " + " | ".join(fmt["TEMP"].format(v) for v in t) + " | " + " | ".join(fmt["SALT"].format(v) for v in s) + " |")
    for name, values in PAPER.items():
        row(name, values["TEMP"], values["SALT"])
    pick = lambda ch, r, k: [S[l][ch][r][k]["mean"] for l in map(str, LEADS)]
    row("persisted OI (this pass = main pass)", pick("TEMP", "persisted_oi", "rmse"), pick("SALT", "persisted_oi", "rmse"))
    row("climatology (this pass = main pass)", pick("TEMP", "climatology", "rmse"), pick("SALT", "climatology", "rmse"))
    row("**KA-Ocean, trained 2011-2020 (mean of 3 seeds)**", pick("TEMP", "model", "rmse"), pick("SALT", "model", "rmse"))
    lines.append("| sd over seeds | " + " | ".join(f"{S[l]['TEMP']['model']['rmse']['sd']:.4f}" for l in map(str, LEADS))
                 + " | " + " | ".join(f"{S[l]['SALT']['model']['rmse']['sd']:.4f}" for l in map(str, LEADS)) + " |")
    for e in results["entries"]:
        row(f"seed {e['seed']} (step {e['step']})", [e["by_lead"][l]["all"]["TEMP"]["model"]["rmse"] for l in map(str, LEADS)],
            [e["by_lead"][l]["all"]["SALT"]["model"]["rmse"] for l in map(str, LEADS)])
    lines += ["", "## Other scores of the 10-year model (mean over seeds), 2024", "",
              "| lead | var | MAE | bias | CRPS raw | CRPS cal. | cov68 raw | cov68 cal. | cov95 raw | cov95 cal. |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for l in map(str, LEADS):
        for ch in CH:
            m, c = S[l][ch]["model"], S[l][ch]["model_calibrated"]
            lines.append(f"| {l} | {ch[0]} | {m['mae']['mean']:.4f} | {m['mean_bias']['mean']:+.4f} | {m['crps']['mean']:.4f} | "
                         f"{c['crps']['mean']:.4f} | {m['coverage_68']['mean']:.3f} | {c['coverage_68']['mean']:.3f} | "
                         f"{m['coverage_95']['mean']:.3f} | {c['coverage_95']['mean']:.3f} |")
    lines += ["", "## RMSE by depth band (mean over seeds), 2024: model / persisted OI / climatology", "",
              "| lead | var | " + " | ".join(S["0"]["TEMP"]["model"]["by_band"]) + " |",
              "|---|---|" + "---|" * len(S["0"]["TEMP"]["model"]["by_band"])]
    for l in map(str, LEADS):
        for ch in CH:
            cells = [f"{S[l][ch]['model']['by_band'][b]['rmse']['mean']:.4f} / {S[l][ch]['persisted_oi']['by_band'][b]['rmse']['mean']:.4f}"
                     f" / {S[l][ch]['climatology']['by_band'][b]['rmse']['mean']:.4f}" for b in S[l][ch]["model"]["by_band"]]
            lines.append(f"| {l} | {ch[0]} | " + " | ".join(cells) + " |")
    lines += ["", "The calibrated spread is 116's factor per lead, variable and depth band fitted on 2021; the mean "
              "(and every RMSE) is untouched. Sources: `registration.json` (sha256 of the checkpoints, calibrations, "
              "caches, cohort and the main pass's files) and `results.json` (every seed, every metric, by lead, year "
              "and band) in this folder; the paper's rows are quoted from its 2024 test and were not re-scored.", ""]
    REPORT.write_text("\n".join(lines))


def score(device):
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} exists: the test is scored once")
    record = json.loads(REGISTRATION.read_text())
    SS.verify(record)
    for name in ("registration", "results"):
        item = record["main_pass"][name]
        if SS.sha256(item["path"]) != item["sha256"]:
            raise SystemExit(f"{item['path']} changed since the registration")
    main_pass = json.loads(MAIN_RESULTS.read_text())
    reference = main_pass["entries"][0]["by_lead"]
    results, cache, worst = {"registration_sha256": SS.sha256(REGISTRATION), "started_unix": time.time(),
                             "main_pass": {"results": record["main_pass"]["results"],
                                           "reference_entry": main_pass["entries"][0]["tag"]}, "entries": []}, {}, 0.
    for entry in record["entries"]:
        started = time.time()
        scales = json.loads(Path(entry["calibration"]["path"]).read_text())["scales"]
        checkpoint, outcome, _ = evaluate(entry["checkpoint"], TEST_YEARS, device, cache, scales)
        worst = max(worst, identity(outcome, reference))
        results["entries"].append({**entry, "tag": checkpoint["training"]["tag"], "seed": checkpoint["seed"],
                                   "step": checkpoint["step"], "seconds": time.time() - started, "by_lead": outcome})
        print(json.dumps({"scored": checkpoint["training"]["tag"], "rmse": {
            lead: [round(outcome[lead]["all"][ch]["model"]["rmse"], 4) for ch in CH] for lead in map(str, LEADS)}}), flush=True)
    results["main_pass"]["worst_relative_difference"] = worst
    results["main_pass"]["identity"] = "persisted OI and climatology (n, RMSE, MAE, band RMSE) equal to the main pass's on every lead"
    results["summary"] = aggregate(results["entries"])
    results["finished_unix"] = time.time()
    write_json(RESULTS, results)
    report(results)
    print(f"wrote {RESULTS} and {REPORT}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--check", metavar="CHECKPOINT")
    p.add_argument("--register", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if sum(map(bool, (args.calibrate, args.check, args.register, args.score))) != 1:
        raise SystemExit("pass one of --calibrate, --check, --register, --score")
    if args.calibrate:
        calibrate()
    elif args.check:
        check(args.check, args.device)
    elif args.register:
        register()
    else:
        score(args.device)


if __name__ == "__main__":
    main()
