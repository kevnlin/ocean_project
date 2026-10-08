"""Verify final artifacts after the independently running matched GPU campaign.

This guard never starts training or opens model test predictions before the
validation selection. It repairs only report generation after all evaluations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/synthetic_matched_20261007"
REPORT = ROOT / "reports/synthetic/matched_reconstruction_20261007.md"


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(metrics, manifest):
    jobs = manifest["jobs"]
    if len(jobs) != 39 or len({job["tag"] for job in jobs}) != 39:
        raise ValueError("Exactly 39 distinct registered runs are required")
    groups = {}
    for job in jobs:
        if job["steps"] != 15000:
            raise ValueError("Each registered run requires 15000 steps")
        key = ("surface" if job["surface"] else "argo", job["family"])
        groups.setdefault(key, []).append(job["seed"])
    if len(groups) != 13 or any(sorted(seeds) != [1234, 1235, 1236] for seeds in groups.values()):
        raise ValueError("All 13 architecture settings require three distinct seeds")
    if metrics.get("test_n_per_variable") != {"TEMP": 351895, "SALT": 351895}:
        raise ValueError("Final report must use every matched 2005 scored value")
    if set(metrics.get("selection", {}).get("runs", {})) != {j["tag"] for j in manifest["jobs"]}:
        raise ValueError("Final report must include all 39 registered training runs")
    if not any(row["family"] == "prior_learned_oi" for row in metrics.get("rows", [])):
        raise ValueError("Final comparison must include the strongest prior learned OI")
    if not any(row["family"] == "fixed_pointwise_mlp" for row in metrics.get("rows", [])):
        raise ValueError("Final comparison must include the fixed pointwise MLP")
    neural_rows = {(row["mode"], row["family"]): row for row in metrics["rows"]
                   if row["kind"] != "classical"}
    if set(neural_rows) != set(groups):
        raise ValueError("The final report must contain all 13 architecture settings")
    for key, row in neural_rows.items():
        if sorted(row["seeds"]) != [1234, 1235, 1236] or sorted(
                result["seed"] for result in row["seed_metrics"]) != [1234, 1235, 1236]:
            raise ValueError(f"Missing seed results for {key}")
    for job in jobs:
        frozen = metrics["selection"]["runs"][job["tag"]]
        contract = frozen["training_contract"]
        nominal = contract.get("training", {}).get("steps", contract.get("steps"))
        if nominal != 15000:
            raise ValueError(f"Frozen training budget differs for {job['tag']}")
        complete = contract.get("completed_steps")
        if complete is None:
            summary = json.loads(Path(frozen["summary_path"]).read_text())
            complete = max((h["step"] for h in summary.get("history", [])), default=0)
        if complete != 15000:
            raise ValueError(f"Actual training is incomplete for {job['tag']}")


def evaluated(manifest):
    for job in manifest["jobs"]:
        folder, seed = Path(job["output"]), job["seed"]
        if not any((folder / name).exists() for name in
                   (f"development_seed{seed}.npz", f"development_predictions_seed{seed}.npz")):
            return False
    return (OUTPUT / "fixed_pointwise_mlp/development_seed1234.npz").exists()


def main():
    manifest_path = OUTPUT / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    status_path = OUTPUT / "finalization_status.json"
    try:
        while True:
            state = json.loads((OUTPUT / "status.json").read_text())
            atomic_json(status_path, {"phase": "waiting_for_training_and_evaluation",
                "main_phase": state["phase"], "updated_unix": time.time()})
            # Avoid concurrently writing the same report as the main process.
            terminal = state["phase"] == "complete" or bool(state.get("failure"))
            if terminal and state.get("failure") and state["phase"] != "report":
                raise RuntimeError(f"The main campaign failed before reports: {state['failure']}")
            if terminal and (OUTPUT / "selection.json").exists() and evaluated(manifest):
                break
            time.sleep(10)
        metrics_path = OUTPUT / "metrics.json"
        if not metrics_path.exists() or not REPORT.exists():
            command = [sys.executable, str(ROOT / "experiments/synthetic/53_matched_reconstruction_report.py"),
                "--campaign", str(manifest_path), "--selection", str(OUTPUT / "selection.json"),
                "--report", str(REPORT), "--output", str(metrics_path)]
            subprocess.run(command, cwd=ROOT, check=True)
        if not metrics_path.exists() or not REPORT.exists():
            raise RuntimeError("Final report is still pending; do not mark the campaign complete")
        metrics = json.loads(metrics_path.read_text())
        verify(metrics, manifest)
        atomic_json(OUTPUT / "completion.json", {"phase": "complete", "training_runs": 39,
            "steps_per_run": 15000, "metrics_sha256": sha256(metrics_path),
            "report_sha256": sha256(REPORT), "selection_sha256": sha256(OUTPUT / "selection.json"),
            "selected": metrics["selection"]["selected"], "verified_unix": time.time()})
        atomic_json(status_path, {"phase": "complete", "verified_unix": time.time()})
        print("All 39 runs, full predictions, strong baselines and final reports verified", flush=True)
    except Exception as error:
        atomic_json(status_path, {"phase": "failed", "failure": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
