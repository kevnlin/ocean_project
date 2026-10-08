"""Reproduce paired, fixed-OI component contrasts from frozen predictions.

This is a retrospective development analysis, not a model-selection tool.
All identities must match before averaging seeds or comparing components.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/latent_ocean"
SEEDS = (1234, 1235, 1236)
CONDITIONS = ("full", "latent_off", "local_off")

_spec = importlib.util.spec_from_file_location(
    "ocean_latent_component_report", Path(__file__).with_name("71_latent_report.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_contrasts(manifest_path: Path, output_root: Path,
                    draws: int = 4000) -> dict:
    manifest = R.read_json(manifest_path)
    completed, pending = R.load_campaign(manifest, output_root)
    if pending:
        raise ValueError("component training is incomplete")
    backbone = manifest["selected_neural"].rsplit("_s", 1)[0]
    expected = {f"{backbone}_fixed_oi_{condition}_s{seed}"
                for condition in CONDITIONS for seed in SEEDS}
    if {entry["job"]["tag"] for entry in completed} != expected:
        raise ValueError("expected three complete seeds for each fixed-OI component condition")
    entries = {entry["job"]["tag"]: entry for entry in completed}
    reference_summary = None
    arrays, sources = {}, []
    for condition in CONDITIONS:
        paths = []
        for seed in SEEDS:
            entry = entries[f"{backbone}_fixed_oi_{condition}_s{seed}"]
            summary, folder = entry["summary"], entry["folder"]
            training, config = summary["training"], summary["config"]
            if (training["steps"] != 6000 or not summary["history"]
                    or summary["history"][-1]["step"] != 6000
                    or not training.get("freeze_analysis")
                    or training.get("analysis_only", False)
                    or config["use_latent"] != (condition != "latent_off")
                    or config["use_local"] != (condition != "local_off")
                    or "development" not in summary):
                raise ValueError("incomplete or incorrect fixed-OI condition: " + summary["tag"])
            if reference_summary is None:
                reference_summary = summary
            elif (summary["data"] != reference_summary["data"]
                  or summary["source_sha256"] != reference_summary["source_sha256"]):
                raise ValueError("component evidence or training source differs")
            path = folder / f"development_seed{seed}.npz"
            paths.append(path)
            summary_path = folder / f"summary_seed{seed}.json"
            sources.extend({"path": str(p), "sha256": sha256(p)}
                           for p in (summary_path, path))
        arrays[condition] = R.ensemble_predictions(paths)
    for condition in CONDITIONS[1:]:
        R.assert_identical(arrays["full"], arrays[condition])
    scores = {condition: {
        "scores": R.z_scores(value["mean"], value["target"]),
        "uncertainty": R.gaussian_metrics(value["mean"], value["std"], value["target"]),
    } for condition, value in arrays.items()}
    contrasts = {}
    for condition in CONDITIONS[1:]:
        paired = dict(arrays["full"], baseline=arrays[condition]["mean"])
        result = R.paired_month_bootstrap(paired, draws=draws)
        result["sign"] = "negative means full has lower error than the removed-component ensemble"
        contrasts[f"full_minus_{condition}"] = result
    return {
        "scores": scores,
        "paired_contrasts": contrasts,
        "interpretation": "Contrast uses the validation-selected single-model "
            f"{backbone} architecture; controls retain frozen OI, inputs and scoring cells. "
            "Negative contrast means full improves over the removed component.",
        "task": "retrospective reconstruction on previously used 2022-2023 development",
        "seeds": list(SEEDS),
        "ensemble_variance": "mean(sigma^2 + mean^2) - ensemble_mean^2",
        "source_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "source_artifacts": sources,
        "analysis_source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                                   for p in (Path(__file__), Path(__file__).with_name("71_latent_report.py"))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUTPUT / "ablations_20261007.json")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT / "component_contrasts_20261007.json")
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap draws must be positive")
    result = build_contrasts(args.manifest, args.output_root, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "conditions": list(result["scores"]),
                      "seeds": result["seeds"]}))


if __name__ == "__main__":
    main()
