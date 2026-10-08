"""Audit and replay the pre-existing learned anisotropic OI on canonical cells.

No model is trained or changed. This prior 120-parameter, single-seed baseline
already has historical 2005 results. It is an off-the-shelf validation candidate
and comparator, separately identified from the new 3-seed/15k-step campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
LEGACY = ROOT / "experiments/real_data/67_anchored_fusion.py"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from ocean_tokenizer import anchored as A, synth_argo_eval as E
from ocean_tokenizer.audit_tools import cohort_path
from ocean_tokenizer.point_baselines import CH, Scores, band_of_levels
from ocean_tokenizer.reconstruction_metrics import evaluate_reconstruction, paired_rmse_bootstrap


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def audit_historical(summary, state, depths):
    expected = {"region": "synthetic", "seed": 1234, "anomaly": "exact", "weights": "kriging",
                "k": 32, "use_time": False, "use_state": False, "params": 120,
                "steps": 1500, "lr": .02, "frozen": False, "input_qc_z": None}
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"historical learned OI setting differs in {key}")
    if summary["splits"] != {key: list(value) for key, value in E.SPLITS.items()} or summary["first_guess"] != {"kind": "none"}:
        raise ValueError("historical split/zero-first-guess setting differs")
    fields = 2 * len(depths)
    if len(depths) != 20 or set(state) != {"log_ell_x", "log_ell_y", "log_gamma"}:
        raise ValueError("checkpoint is not the 20-depth, 120-parameter OI")
    if any(value.shape != (fields,) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint fields/scales are invalid")
    history = summary["history"]
    winner = min(history, key=lambda entry: entry["validation/macro_z"])
    if winner["step"] != summary["best_step"] or not np.isclose(
            winner["validation/macro_z"], summary["scores"]["validation"]["macro_z"], rtol=1e-12, atol=1e-12):
        raise ValueError("historical selected checkpoint does not follow recorded validation minimum")
    band = band_of_levels(depths)
    for key, value in state.items():
        physical = value.exp().numpy()
        name = key.removeprefix("log_")
        for ci, ch in enumerate(CH):
            for label in np.unique(band):
                recorded = summary["learned"][name][ch][str(label)]
                reproduced = float(physical[ci * len(depths):(ci + 1) * len(depths)][band == label].mean())
                if not np.isclose(recorded, reproduced, rtol=2e-6, atol=1e-7):
                    raise ValueError("checkpoint scales do not match the historical summary")
    return {"selection_metric": "mean temperature/salinity standardized RMSE on 2004 validation only",
            "selected_step": int(winner["step"]), "recorded_validation_mean_standardized_rmse": float(winner["validation/macro_z"]),
            "source_recipe": "67_anchored_fusion.py synthetic/no first guess/no time/no state/kriging",
            "parameterization": "separate east/north length scale and noise ratio for every variable and depth; 20×2×3=120 parameters",
            "summary_depth_bands": "scale values in the old summary are band averages; checkpoint parameters are not tied by band",
            "training_steps": 1500, "seed": 1234, "optimizer": "Adam, lr0.02, cosine schedule, clip1.0 (source-derived)",
            "training_query_budget": {"source_default_profiles_per_step": 512, "depths_per_profile": 20,
                 "actual_original_cli_queries": "not saved in historical summary; default is source-derived, not independently recorded"},
            "historical_development_use": "2005 has already been scored and inspected in prior work; replay is not independent confirmation",
            "historical_provenance_limit": "old summary lacks original cohort/source hashes; present-day parity is established by canonical identity, checkpoint scales and historical RMSE replay"}


def compare_historical(scores, expected, split):
    for ch in CH:
        if scores[ch]["n"] != expected[ch]["n"]:
            raise ValueError(f"historical {split}/{ch} count differs")
        for key in ("rmse_z", "rmse_physical", "climatology_z"):
            if not np.isclose(scores[ch][key], expected[ch][key], rtol=2e-5, atol=1e-8):
                raise ValueError(f"historical {split}/{ch}/{key} does not replay")


def export_split(c, norm, obs, model, split, baseline_dir, output, device, historical, bootstrap_draws):
    evaluations = E.eval_set(c, obs, split, E.EVAL_CELLS if split == "validation" else 0)
    scores, zero_scores = Scores(c.levels, norm.std), Scores(c.levels, norm.std)
    collected = {name: [] for name in ("month", "profile", "level", "target", "mean")}
    source_counts = {}
    model.eval()
    for ev in evaluations:
        source, query = ev["src"], ev["tgt"]
        if len(source) != 6080 or len(query) != 1520 or np.intersect1d(c.wmo[source], c.wmo[query]).size:
            raise ValueError("historical/current input-profile or heldout-float parity differs")
        tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
        innovation = np.concatenate([obs[ch][source] for ch in CH], axis=1)
        # This is the exact existing model invocation used by 67, including its
        # float32 coordinates/innovations and unchanged per-field missing masks.
        with torch.no_grad():
            full = model(tensor(c.lat[query]), tensor(c.lon[query] % 360.),
                         tensor(c.lat[source]), tensor(c.lon[source] % 360.), tensor(innovation)).double().cpu().numpy()
        mean = np.stack([full[ev["prof"], ci * len(c.levels) + ev["lev"]] for ci in range(2)], axis=-1)
        target = np.stack([ev["target"][ch] for ch in CH], axis=-1)
        if np.any(~np.isfinite(mean[np.isfinite(target)])):
            raise ValueError("nonfinite learned OI prediction for a scored target")
        source_counts[str(ev["month"])] = {"input_profiles": len(source), "query_profiles": len(query)}
        collected["month"].append(np.full(len(ev["lev"]), ev["month"], dtype=np.int64))
        collected["profile"].append(query[ev["prof"]].astype(np.int64))
        collected["level"].append(ev["lev"].astype(np.int64))
        collected["target"].append(target); collected["mean"].append(mean)
        for ci, ch in enumerate(CH):
            scores.add(ch, mean[:, ci], target[:, ci], ev["lev"])
            zero_scores.add(ch, np.zeros(len(target)), target[:, ci], ev["lev"])
        print(f"replayed learned OI {split} month{ev['month']}", flush=True)
    identity = E.identity_check(str(ROOT), split, zero_scores.result())
    result = scores.result()
    compare_historical(result, historical["scores"][split], split)
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    canonical_path = baseline_dir / f"oi_{split}.npz"
    with np.load(canonical_path, allow_pickle=False) as reference:
        for key in ("month", "profile", "level", "target"):
            if not np.array_equal(arrays[key], reference[key], equal_nan=True):
                raise ValueError(f"prior learned OI and canonical scoring identity differ in {key}")
        arrays["baseline"] = reference["mean"].copy()
        for key in ("climatology_physical", "normalization_mean", "normalization_std"):
            arrays[key] = reference[key].copy()
    offset = np.column_stack([norm.mean[ch] for ch in CH])
    scale = np.column_stack([norm.std[ch] for ch in CH])
    if not np.array_equal(offset, arrays["normalization_mean"]) or not np.array_equal(scale, arrays["normalization_std"]):
        raise ValueError("prior learned OI and canonical per-depth training normalization differ")
    sd, mu = scale[arrays["level"]], offset[arrays["level"]]
    arrays["levels"] = c.levels
    for key in ("mean", "target"):
        arrays[f"{key}_physical"] = arrays[key] * sd + mu
        arrays[f"{key}_absolute_physical"] = arrays[f"{key}_physical"] + arrays["climatology_physical"]
    fingerprint = hashlib.sha256()
    for key in ("month", "profile", "level", "target"):
        a = np.ascontiguousarray(arrays[key])
        fingerprint.update(key.encode()); fingerprint.update(str(a.dtype).encode())
        fingerprint.update(str(a.shape).encode()); fingerprint.update(a.tobytes())
    metadata = {"method": "prior_learned_oi", "seed": 1234, "split": split,
                "training_steps": 1500, "parameters": 120, "uncertainty_available": False,
                "physical_arrays": "anomalies", "scoring_fingerprint": fingerprint.hexdigest(),
                "input_profiles_per_month": 6080, "input_qc": None, "first_guess": "zero standardized anomaly",
                "prior_development_results": "previously scored; not a newly opened neural candidate"}
    path = output / f"{split}_seed1234.npz"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, path)
    standardized = evaluate_reconstruction(arrays["mean"], arrays["target"])
    physical = evaluate_reconstruction(arrays["mean_physical"], arrays["target_physical"])
    return {"prediction_path": str(path), "prediction_sha256": sha256(path),
            "scoring_fingerprint": fingerprint.hexdigest(), "source_counts": source_counts,
            "identity_check": identity, "historical_rmse_replay_verified": True,
            "historical_rmse": {ch: historical["scores"][split][ch]["rmse_physical"] for ch in CH},
            "standardized_rmse": {ch: standardized[ch]["rmse"] for ch in CH},
            "mean_standardized_rmse": float(np.mean([standardized[ch]["rmse"] for ch in CH])),
            "physical_metrics": physical,
            "absolute_physical_metrics": evaluate_reconstruction(arrays["mean_absolute_physical"], arrays["target_absolute_physical"]),
            "by_depth_physical_metrics": {str(float(depth)): evaluate_reconstruction(
                arrays["mean_physical"][arrays["level"] == i], arrays["target_physical"][arrays["level"] == i])
                for i, depth in enumerate(c.levels)},
            "paired_month_rmse_bootstrap_against_classic_oi": {
                ch: paired_rmse_bootstrap(arrays["mean_physical"][:, ci], arrays["baseline"][:, ci] * sd[:, ci] + mu[:, ci],
                                         arrays["target_physical"][:, ci], arrays["month"], draws=bootstrap_draws)
                for ci, ch in enumerate(CH)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-dir", type=Path, default=ROOT / "outputs/audit/synthetic/anc_kriging")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/prior_learned_oi")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/previous_baselines")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use the authorized native runtime or CPU")
    args.out.mkdir(parents=True, exist_ok=True)
    registry_path = args.out / "aux_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        artifacts = [registry["source_checkpoint"], registry["historical_summary"], registry["summary"]] + list(registry["arrays"].values())
        if not all(sha256(item["path"]) == item["sha256"] for item in artifacts):
            raise ValueError("immutable learned-OI registry artifacts changed")
        if any(sha256(ROOT / path) != recorded for path, recorded in registry["source_hashes"].items()):
            raise ValueError("immutable learned-OI replay source changed")
        print(json.dumps({"existing_immutable_registry": str(registry_path)}), flush=True)
        return
    started = time.time()
    source_summary_path = args.prior_dir / "summary_seed1234.json"
    checkpoint_path = args.prior_dir / "model_seed1234.pt"
    historical = json.loads(source_summary_path.read_text())
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    c, norm, obs = E.load(str(ROOT))
    parity = audit_historical(historical, state, c.levels)
    dataset = Path(cohort_path(str(ROOT), "synthetic"))
    dataset_hash = sha256(dataset)
    baseline_inventory = json.loads((args.baseline_dir / "replay_manifest.json").read_text())
    if baseline_inventory["cohort_sha256"] != dataset_hash:
        raise ValueError("prior/canonical cohort file hashes differ")
    model = A.InnovationAnalysis(40, mode="kriging", k=32, use_time=False, use_state=False).to(args.device)
    model.load_state_dict(state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 120:
        raise ValueError("prior checkpoint parameter budget differs")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sources = (RUNNER, LEGACY, ROOT / "src/ocean_tokenizer/anchored.py",
               ROOT / "src/ocean_tokenizer/synth_argo_eval.py", ROOT / "src/ocean_tokenizer/point_baselines.py",
               ROOT / "src/ocean_tokenizer/argo_obs.py", ROOT / "src/ocean_tokenizer/audit_tools.py",
               ROOT / "src/ocean_tokenizer/reconstruction_metrics.py")
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    summary = {"tag": "prior_learned_oi", "seed": 1234, "steps": 1500, "completed_steps": 1500,
        "params": 120, "best_step": historical["best_step"], "architecture": "learned anisotropic per-depth optimal interpolation",
        "checkpoint_path": str(checkpoint_path), "source_checkpoint_sha256": sha256(checkpoint_path),
        "historical_summary_path": str(source_summary_path), "historical_summary_sha256": sha256(source_summary_path),
        "source_sha256": source_hashes, "source_hashes": source_hashes,
        "data": {"cohort_sha256": dataset_hash, "splits": historical["splits"], "levels_m": c.levels.tolist(),
                 "anomaly_target": "at Argo profile position", "input_profiles_per_evaluation_month": 6080,
                 "query_profiles_per_evaluation_month": 1520, "normalization_scope": "all training-year profiles, inherited protocol",
                 "input_qc_z": None, "satellite": False, "task": "retrospective same-month reconstruction"},
        "audit": parity, "training_recipe": {"steps": 1500, "seed": 1234, "lr": .02,
                  "parameter_fields": 40, "kernel": "query-centred azimuthal east/north anisotropic Gaussian",
                  "neighbors": "32 nearest profiles by horizontal position; per-field missing observations masked, not replaced by further finite neighbors",
                  "first_guess": "zero standardized anomaly", "time": False, "state": False},
        "budget_status": "historical fixed single-seed 1500-step model, separately labelled from new 3-seed/15000-step models"}
    for split in ("validation", "development"):
        summary[split] = export_split(c, norm, obs, model, split, args.baseline_dir, args.out,
                                     args.device, historical, args.bootstrap_draws)
    summary["runtime_s"] = time.time() - started
    summary_path = args.out / "summary_seed1234.json"
    atomic_json(summary_path, summary)
    registry = {"schema_version": 1, "method": "prior_learned_oi", "eligible_validation_candidate": True,
        "parity_verified": True, "eligible_allowed_input_settings": ["argo", "surface"],
        "input_setting_interpretation": "Argo-only model can be used in either task; it ignores satellite fields in the surface setting",
        "candidate_scope": "existing single-seed off-the-shelf model; not a new campaign family or three-seed 15k run",
        "cohort_sha256": dataset_hash, "source_checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
        "historical_summary": {"path": str(source_summary_path), "sha256": sha256(source_summary_path)},
        "summary": {"path": str(summary_path), "sha256": sha256(summary_path)}, "seed": 1234,
        "training_steps": 1500, "parameters": 120, "source_hashes": source_hashes,
        "arrays": {split: {"path": summary[split]["prediction_path"], "sha256": summary[split]["prediction_sha256"]}
                   for split in ("validation", "development")},
        "verification": {"exact_canonical_identity": True, "exact_per_depth_normalization": True,
                         "full_6080_input_profiles_every_month": True, "historical_rmse_replay": True,
                         "checkpoint_scales_match_historical_summary": True, "checkpoint_is_validation_selected": True},
        "historical_development_status": parity["historical_development_use"],
        "historical_provenance_limit": parity["historical_provenance_limit"]}
    atomic_json(registry_path, registry)
    print(json.dumps({"summary": str(summary_path), "registry": str(registry_path),
        "rmse": {split: {ch: summary[split]["physical_metrics"][ch]["rmse"] for ch in CH}
                 for split in ("validation", "development")}, "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
