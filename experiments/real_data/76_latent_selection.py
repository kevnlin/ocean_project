"""Choose the final three-seed ocean model using validation predictions only.

The rule is registered before this campaign opens its previously used 2022--2023
development years. This module never opens their arrays or accesses their score
fields. All seven registered candidates must finish before a choice is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/latent_ocean"
REPORT_PATH = Path(__file__).with_name("71_latent_report.py")
_spec = importlib.util.spec_from_file_location("ocean_validation_report_helpers", REPORT_PATH)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

SEEDS = (1234, 1235, 1236)
FINGERPRINTS = ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")
FLAGS = ("analysis_only", "freeze_analysis", "latent_off", "local_off")
TRAINING_KEYS = ("steps", "queries", "eval_context_profiles", "lr", "analysis_lr",
                 "weight_decay", "nll_weight", "amp", "val_cells", "val_every", "eval_chunk")
RULE = {
    "selection_input_split": "validation_2021",
    "selection_metric": "macro_z of the mean prediction of seeds 1234,1235,1236",
    "definition": "equal mean of TEMP and SALT standardized RMSE, computed from validation arrays",
    "criterion": "minimum validation ensemble macro_z across all seven preregistered candidates",
    "tie_break": "lexicographic candidate tag only for exactly equal numerical scores",
    "required_seeds": list(SEEDS),
    "required_training_steps": 6000,
    "candidate_families": ["dense", "soft_moe", "local_transformer", "analysis_only",
                           "fixed_oi_full", "fixed_oi_latent_off", "fixed_oi_local_off"],
    "parameter_cost": "recorded for interpretation; never a selection threshold",
    "declaration": "Fixed on 2026-10-07 before this campaign opens its 2022-2023 development evaluation.",
}
UNCERTAINTY_LAW = "ensemble variance = mean(seed_std^2 + seed_mean^2) - ensemble_mean^2"


class PendingSelection(ValueError):
    """Registered training or required artifacts are not yet complete."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_hashes() -> dict[str, str]:
    helpers = "\n".join(inspect.getsource(getattr(report, name)) for name in
                        ("load_predictions", "assert_identical", "ensemble_predictions", "z_scores",
                         "match_job", "same_configuration"))
    return {"selector_sha256": sha256(Path(__file__)),
            "report_validation_helpers_sha256": hashlib.sha256(helpers.encode()).hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def register_rule(path: Path) -> dict:
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("rule") != RULE or stored.get("method_hashes") != method_hashes():
            raise ValueError("registered validation rule or selection code changed; refusing replacement")
        return stored
    value = {"version": 1, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
             "rule": RULE, "method_hashes": method_hashes(), "status": "rule registered; no model chosen"}
    write_json(path, value)
    return value


def tag_prefix(job: dict) -> str:
    suffix = f"_s{job['seed']}"
    if not job["tag"].endswith(suffix):
        raise ValueError(f"tag lacks its registered seed suffix: {job['tag']}")
    return job["tag"][:-len(suffix)]


def candidate_jobs(selected_manifest: dict, ablation_manifest: dict) -> list[tuple[str, list[dict]]]:
    selected = selected_manifest.get("selected", [])
    if len(selected) != 4:
        raise ValueError("final selection requires all three frozen architectures and OI-only")
    families = ["analysis_only" if j.get("analysis_only") else j["variant"] for j in selected]
    if set(families) != {"dense", "soft_moe", "local_transformer", "analysis_only"}:
        raise ValueError("frozen architecture families are missing or duplicated")
    groups = []
    for job in selected:
        if job.get("seed") != SEEDS[0] or job.get("steps") != 6000:
            raise ValueError("frozen configuration must come from its full seed-1234 search")
        if any(job.get(flag, False) for flag in ("freeze_analysis", "latent_off", "local_off")):
            raise ValueError("a control cannot replace an ordinary frozen architecture")
        prefix = tag_prefix(job)
        groups.append((prefix, [{**job, "seed": seed, "tag": f"{prefix}_s{seed}"} for seed in SEEDS]))
    winner = next((j for j in selected if j["tag"] == ablation_manifest.get("selected_neural")), None)
    if winner is None or winner.get("analysis_only"):
        raise ValueError("component controls must reference a frozen neural configuration")
    controls: dict[str, list[dict]] = {}
    for job in ablation_manifest.get("jobs", []):
        if not job.get("freeze_analysis") or job.get("analysis_only"):
            raise ValueError("all component controls must freeze OI and use the neural architecture")
        for key in ("variant", "width", "latents", "blocks", "context_profiles", "experts", "steps"):
            if job.get(key) != winner.get(key):
                raise ValueError(f"control differs from frozen winner in {key}")
        controls.setdefault(tag_prefix(job), []).append(job)
    if len(controls) != 3:
        raise ValueError("exactly three complete frozen-OI component controls are required")
    control_flags = set()
    for prefix, jobs in sorted(controls.items()):
        if len(jobs) != 3 or {j["seed"] for j in jobs} != set(SEEDS):
            raise ValueError(f"control manifest lacks exactly three registered seeds: {prefix}")
        flags = {(bool(j.get("latent_off")), bool(j.get("local_off"))) for j in jobs}
        if len(flags) != 1:
            raise ValueError(f"component flags differ within a control: {prefix}")
        control_flags.update(flags)
        groups.append((prefix, sorted(jobs, key=lambda j: j["seed"])))
    if control_flags != {(False, False), (True, False), (False, True)}:
        raise ValueError("controls must contain full, latent-off and local-off independently")
    if len({prefix for prefix, _ in groups}) != 7:
        raise ValueError("candidate tags are duplicated")
    return groups


def validate_summary(job: dict, stored: dict) -> dict:
    # Access a whitelist rather than passing a complete run report to selection.
    required = ("tag", "seed", "config", "training", "data", "source_sha256", "params", "best_step", "history")
    summary = {key: stored[key] for key in required}
    summary["validation"] = {"scores": stored["validation"]["scores"]}
    report.match_job(job, summary)
    if job["steps"] != 6000 or summary["training"]["steps"] != 6000:
        raise ValueError(f"incomplete training budget: {job['tag']}")
    history = summary["history"]
    if not history or max(h["step"] for h in history) != 6000:
        raise PendingSelection(f"training has not reached step 6000: {job['tag']}")
    if not 0 <= summary["best_step"] <= 6000:
        raise ValueError(f"invalid selected checkpoint step: {job['tag']}")
    for flag in FLAGS:
        if bool(summary["training"].get(flag, False)) != bool(job.get(flag, False)):
            raise ValueError(f"unregistered {flag}: {job['tag']}")
    if summary["config"].get("use_latent") != (not bool(job.get("latent_off"))):
        raise ValueError(f"latent configuration does not match control: {job['tag']}")
    if summary["config"].get("use_local") != (not bool(job.get("local_off"))):
        raise ValueError(f"local configuration does not match control: {job['tag']}")
    if any(not isinstance(summary["data"].get(key), str) or not summary["data"][key] for key in FINGERPRINTS):
        raise ValueError(f"missing input fingerprint: {job['tag']}")
    if not summary["source_sha256"]:
        raise ValueError(f"missing training source hashes: {job['tag']}")
    return summary


def build_selection(selected_path: Path, ablation_path: Path, output_root: Path,
                    registration: dict) -> dict:
    if registration.get("rule") != RULE or registration.get("method_hashes") != method_hashes():
        raise ValueError("selection differs from the registered validation-only rule/code")
    if not ablation_path.exists():
        raise PendingSelection("component-control manifest is not frozen yet")
    selected = json.loads(selected_path.read_text())
    ablations = json.loads(ablation_path.read_text())
    groups = candidate_jobs(selected, ablations)
    missing = []
    for _, jobs in groups:
        for job in jobs:
            folder = output_root / job["tag"]
            for name in (f"summary_seed{job['seed']}.json", f"validation_seed{job['seed']}.npz",
                         f"best_seed{job['seed']}.pt"):
                if not (folder / name).is_file():
                    missing.append(str(folder / name))
    if missing:
        raise PendingSelection("required complete three-seed artifacts are missing: " + ", ".join(missing))
    reference_identity = reference_summary = None
    candidates = []
    for prefix, jobs in groups:
        paths, checkpoints, summaries, array_hashes = [], [], [], {}
        for job in jobs:
            folder = output_root / job["tag"]
            raw = json.loads((folder / f"summary_seed{job['seed']}.json").read_text())
            summary = validate_summary(job, raw)
            del raw
            path = folder / f"validation_seed{job['seed']}.npz"
            arrays = report.load_predictions(path)
            if np.any((arrays["month"] < 252) | (arrays["month"] > 263)):
                raise ValueError(f"prediction identity is outside validation year 2021: {path}")
            if reference_identity is None:
                reference_identity = {key: arrays[key] for key in report.IDENTITY_KEYS}
                reference_summary = summary
            else:
                report.assert_identical(reference_identity, arrays)
                if any(summary["data"][key] != reference_summary["data"][key] for key in FINGERPRINTS):
                    raise ValueError(f"candidate input fingerprints differ: {job['tag']}")
                if summary["source_sha256"] != reference_summary["source_sha256"]:
                    raise ValueError(f"candidate training source hashes differ: {job['tag']}")
                if any(summary["training"].get(key) != reference_summary["training"].get(key) for key in TRAINING_KEYS):
                    raise ValueError(f"candidate validation/training protocol differs: {job['tag']}")
            if summaries and not report.same_configuration(summaries[0], summary):
                raise ValueError(f"seed configuration differs within candidate: {prefix}")
            score = report.z_scores(arrays["mean"], arrays["target"])
            for key, actual in (("macro_z", score["macro_z"]),
                                ("TEMP", score["TEMP"]["rmse_z"]), ("SALT", score["SALT"]["rmse_z"])):
                saved = summary["validation"]["scores"][key]
                expected = saved["rmse_z"] if isinstance(saved, dict) else saved
                if actual is None or not np.isclose(actual, expected, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"validation arrays differ from saved score ({key}): {job['tag']}")
            array_hashes[str(path)] = sha256(path)
            checkpoint = folder / f"best_seed{job['seed']}.pt"
            checkpoints.append({"tag": job["tag"], "seed": job["seed"], "step": summary["best_step"],
                                "path": str(checkpoint), "sha256": sha256(checkpoint)})
            paths.append(path)
            summaries.append(summary)
            del arrays
        ensemble = report.ensemble_predictions(paths)
        scores = report.z_scores(ensemble["mean"], ensemble["target"])
        if scores["macro_z"] is None or not np.isfinite(scores["macro_z"]):
            raise ValueError(f"candidate has no finite validation score: {prefix}")
        first = summaries[0]
        architecture = ("OI-only" if first["training"].get("analysis_only") else
                        "query MLP + exact local innovations; no shared latent" if not first["config"]["use_latent"] else
                        first["config"]["variant"] + " shared ocean latent")
        candidates.append({"tag": prefix, "architecture": architecture, "seeds": list(SEEDS),
                           "scores": scores, "neural_parameters": first["params"],
                           "config": first["config"], "flags": {flag: bool(first["training"].get(flag)) for flag in FLAGS},
                           "seed_validation_scores": [s["validation"]["scores"] for s in summaries],
                           "checkpoints": checkpoints, "inputarray_sha256": array_hashes})
    chosen = min(candidates, key=lambda candidate: (candidate["scores"]["macro_z"], candidate["tag"]))
    return {"version": 1, "status": "selected", "rule": RULE,
            "rule_registered_at_utc": registration["registered_at_utc"],
            "selected_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates": candidates, "selectedtag": chosen["tag"], "scores": chosen["scores"],
            "selected_architecture": chosen["architecture"], "seeds": list(SEEDS),
            "checkpoints": chosen["checkpoints"], "uncertaintyensemblelaw": UNCERTAINTY_LAW,
            "code_sha256": {**method_hashes(), "report_file_sha256": sha256(REPORT_PATH)},
            "manifests": {str(selected_path): sha256(selected_path), str(ablation_path): sha256(ablation_path)},
            "data_fingerprints": {key: reference_summary["data"][key] for key in FINGERPRINTS},
            "training_source_sha256": reference_summary["source_sha256"],
            "inputarray_sha256": {path: digest for candidate in candidates for path, digest in candidate["inputarray_sha256"].items()},
            "interpretation": "Recommendation follows validation only; a non-latent winner is retained without forcing the shared backbone."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", type=Path, default=OUTPUT / "selected_20261007.json")
    parser.add_argument("--ablation-manifest", type=Path, default=OUTPUT / "ablations_20261007.json")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT / "final_selection_20261007.json")
    parser.add_argument("--rule-record", type=Path, default=OUTPUT / "final_selection_rule_20261007.json")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        registration = register_rule(args.rule_record)
        print(json.dumps({"status": registration["status"], "rule_record": str(args.rule_record)}), flush=True)
        return
    if not args.rule_record.exists():
        raise SystemExit("register the validation-only rule with --register-only before final selection")
    registration = json.loads(args.rule_record.read_text())
    try:
        result = build_selection(args.selected_manifest, args.ablation_manifest, args.output_root, registration)
    except PendingSelection as error:
        print(json.dumps({"status": "pending", "reason": str(error), "chosen": None}), flush=True)
        return
    if args.output.exists() and json.loads(args.output.read_text()).get("selectedtag") != result["selectedtag"]:
        raise SystemExit("refusing to replace a different frozen recommendation")
    write_json(args.output, result)
    print(json.dumps({"status": "selected", "tag": result["selectedtag"],
                      "validation_macro_z": result["scores"]["macro_z"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
