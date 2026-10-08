"""Fixed 2004 source-only robustness diagnostics on a selected checkpoint.

These capped diagnostics do not select checkpoints or replace full validation.
Argo evidence is perturbed; held-out query measurements and satellites remain
unchanged.  The OI anchor is recomputed for each condition.  All metrics use
canonical float64 truth, with identical query support across conditions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer.innovation_ocean import InnovationOceanConfig, InnovationOceanModel
from ocean_tokenizer.innovation_scenes import InnovationOceanScenes
from ocean_tokenizer.latent_ocean import LatentOceanConfig, LatentOceanModel
from ocean_tokenizer.synthetic_latent_experiment import EVAL_SEED
from ocean_tokenizer.reconstruction_metrics import evaluate_reconstruction, inverse_standardize


CONDITIONS = ("standard", "thin_profiles_25pct", "thin_profiles_50pct",
              "mask_depths_30pct", "profile_common_bias", "independent_depth_noise")
NOISE_STD = np.array([0.05, 0.01], dtype=np.float64)


def perturbation_plan(source, n_levels, condition, month, seed):
    """Create deterministic, physical-unit perturbations without reading labels."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    source = np.asarray(source, dtype=np.int64)
    rng = np.random.default_rng([seed, int(month), CONDITIONS.index(condition)])
    retained = source.copy()
    keep = np.ones((len(source), n_levels, 2), dtype=bool)
    noise = np.zeros((len(source), n_levels, 2), dtype=np.float64)
    if condition.startswith("thin_profiles"):
        fraction = 0.75 if "25pct" in condition else 0.5
        # Preserve canonical source order after random selection.
        selected = rng.permutation(len(source))[:max(1, int(np.floor(len(source) * fraction)))]
        retained = source[np.sort(selected)]
    elif condition == "mask_depths_30pct":
        keep = np.broadcast_to(rng.random((len(source), n_levels, 1)) >= 0.30, keep.shape).copy()
    elif condition == "profile_common_bias":
        noise = np.broadcast_to(rng.normal(size=(len(source), 1, 2)) * NOISE_STD,
                                noise.shape).copy()
    elif condition == "independent_depth_noise":
        noise = rng.normal(size=noise.shape) * NOISE_STD
    return retained, keep, noise


@contextmanager
def perturbed_source(data, source, query_rows, condition, month, seed):
    """Modify only source evidence, then restore it even after inference errors."""
    source = np.asarray(source, dtype=np.int64)
    if np.intersect1d(source, np.asarray(query_rows)).size:
        raise ValueError("Stress sources overlap held-out target rows")
    retained, keep, noise = perturbation_plan(source, data.n_levels, condition, month, seed)
    index = data.indices(source)
    names = ("innovation", "valid", "profile_physical", "profile_absolute")
    saved = {name: getattr(data, name)[index].clone() for name in names}
    saved_np = data.valid_np[source].copy()
    try:
        mask = torch.as_tensor(keep, dtype=torch.bool, device=data.device)
        delta = torch.as_tensor(noise, dtype=data.profile_physical.dtype, device=data.device)
        data.valid[index] = saved["valid"] & mask
        data.valid_np[source] = saved_np & keep
        data.innovation[index] = torch.where(data.valid[index], saved["innovation"] + delta / data.profile_std, 0.0)
        data.profile_physical[index] = saved["profile_physical"] + delta
        data.profile_absolute[index] = saved["profile_absolute"] + delta
        yield retained, {"n_source_profiles_original": int(len(source)),
                         "n_source_profiles_used": int(len(retained)),
                         "n_observed_source_values_original": saved_np.sum(0).sum(0).tolist(),
                         "n_observed_source_values_used": data.valid_np[retained].sum(0).sum(0).tolist(),
                         "physical_noise_std": NOISE_STD.tolist() if "noise" in condition or "bias" in condition else [0.0, 0.0]}
    finally:
        for name, value in saved.items():
            getattr(data, name)[index] = value
        data.valid_np[source] = saved_np


def reference_module():
    path = ROOT / "experiments/synthetic/53_matched_reconstruction_report.py"
    spec = importlib.util.spec_from_file_location("stress_matched_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_queries(data, reference, queries_per_month, max_cells):
    """Select a prefix of registered capped scoring cells, preserving their order."""
    months = []
    for ev in data.eval_months("validation", max_cells):
        registered = np.flatnonzero(reference["month"] == ev.month)
        rows = ev.target[ev.profile]
        if not np.array_equal(rows, reference["profile"][registered]) or not np.array_equal(ev.level, reference["level"][registered]):
            raise ValueError(f"Canonical scoring coordinates differ in month {ev.month}")
        observed = data.y[data.indices(rows), data.indices(ev.level)].cpu().numpy()
        if not np.array_equal(observed.astype(np.float32), reference["target32"][registered], equal_nan=True):
            raise ValueError(f"Canonical scoring labels differ in month {ev.month}")
        selected = registered[:queries_per_month]
        n = len(selected)
        months.append((ev.month, ev.source, rows[:n], ev.level[:n], selected))
    if not months:
        raise ValueError("No registered validation months")
    return months


@torch.no_grad()
def covariance_duplicate_probe(model, scene):
    component = getattr(model, "covariance", None)
    if component is None:
        return None
    width = model.config.width
    h = scene["baseline"].new_zeros((len(scene["baseline"]), width))
    if getattr(model, "alignment", None) is not None:
        scene, _ = model.alignment(scene, h)
        scene = {**scene, "local_profile_values": scene["aligned_profile_values"],
                 "local_profile_valid": scene["aligned_profile_valid"]}
    local_keys = ("local_profile_values", "local_profile_valid", "local_profile_ids",
                  "local_offsets", "local_profile_noise_variance")
    duplicated = {key: torch.cat((value, value), 1) if key in local_keys else value
                  for key, value in scene.items()}
    before, variance_before, _ = component(scene, h)
    after, variance_after, _ = component(duplicated, h)
    return {"scope": "covariance component only, fixed zero query context; not whole-system duplication",
            "n_queries": len(h), "max_abs_mean_difference_z": float((after-before).abs().max()),
            "max_abs_variance_difference_z2": float((variance_after-variance_before).abs().max())}


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    surface = bool(training.get("surface", checkpoint["config"].get("n_sat_features", 0) > 0))
    satellite = args.satellite_cache or training.get("satellite_cache") or ROOT / "outputs/synthetic_matched_20261007/satellites.npz"
    operator = args.operator_cache or training.get("operator_cache") or ROOT / "outputs/innovation_20261007/operators.npz"
    data = InnovationOceanScenes(ROOT, args.device, surface=surface,
        satellite_cache=satellite if surface else None, operator_cache=operator if surface else None,
        oi_summary=training.get("oi_summary"))
    for key in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint"):
        if checkpoint["data"][key] != data.metadata[key]:
            raise ValueError(f"Checkpoint {key} differs from stress data")
    innovative = "recipe" in checkpoint["config"]
    config = (InnovationOceanConfig if innovative else LatentOceanConfig)(**checkpoint["config"])
    model = (InnovationOceanModel if innovative else LatentOceanModel)(config).to(data.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    analysis = data.analysis().requires_grad_(False)
    analysis.load_state_dict(checkpoint["analysis"], strict=True)
    matched = reference_module()
    reference = matched.load_reference(args.baseline_dir, "validation")
    scale = np.stack([data.norm.std[ch] for ch in ("TEMP", "SALT")], -1)
    offset = np.stack([data.norm.mean[ch] for ch in ("TEMP", "SALT")], -1)
    if not np.array_equal(scale, reference["scale"]) or not np.array_equal(offset, reference["offset"]):
        raise ValueError("Canonical training normalization differs")
    months = select_queries(data, reference, args.queries_per_month, training.get("val_cells", 8000))
    selection = np.concatenate([entry[4] for entry in months])
    level = reference["level"][selection]
    target = reference["target"][selection]
    query_valid = np.concatenate([data.valid[data.indices(rows), data.indices(lev)].cpu().numpy()
                                  for _, _, rows, lev, _ in months])
    fixed_support = np.isfinite(target) & query_valid
    truth = inverse_standardize(target, level, reference["scale"], reference["offset"])
    amp = bool(training.get("amp", False)) and data.device.type == "cuda"
    context = training.get("eval_context_profiles") or training.get("context_profiles", 6080)
    results, probe = {}, None
    with torch.no_grad():
        for condition in CONDITIONS:
            predictions, uncertainty, anchors, counts = [], [], [], []
            for month, source, rows, levels, _ in months:
                labels_before = data.y[data.indices(rows), data.indices(levels)].clone()
                with perturbed_source(data, source, rows, condition, month, args.seed) as (retained, count):
                    scene = data.scene(retained, rows, levels, analysis, context, [EVAL_SEED, month])
                    if args.supply_noise_variance and condition == "independent_depth_noise":
                        scene["local_profile_noise_variance"] = scene["local_profile_values"].new_tensor(NOISE_STD ** 2).expand_as(scene["local_profile_values"])
                    if condition == "standard" and probe is None and getattr(model, "covariance", None) is not None:
                        probe = covariance_duplicate_probe(model, scene)
                    with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16, enabled=amp):
                        output = model(scene)
                    predictions.append(output["mean"].float().cpu().numpy())
                    uncertainty.append(output["std"].float().cpu().numpy())
                    anchors.append(scene["baseline"].float().cpu().numpy())
                    counts.append({"month": int(month), "n_queries": len(rows), **count})
                if not torch.allclose(data.y[data.indices(rows), data.indices(levels)], labels_before, equal_nan=True, rtol=0, atol=0):
                    raise ValueError("Stress altered a held-out query label")
            pred = inverse_standardize(np.concatenate(predictions), level, reference["scale"], reference["offset"])
            std = inverse_standardize(np.concatenate(uncertainty), level, reference["scale"], uncertainty=True)
            anchor = inverse_standardize(np.concatenate(anchors), level, reference["scale"], reference["offset"])
            results[condition] = {"metrics": evaluate_reconstruction(pred, truth, std, fixed_support),
                                  "recomputed_oi_metrics": evaluate_reconstruction(anchor, truth, mask=fixed_support),
                                  "months": counts}
            print(json.dumps({"condition": condition, "metrics": results[condition]["metrics"]}), flush=True)
    result = {"status": "fixed-checkpoint 2004 capped stress diagnostics; not checkpoint selection",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": matched.sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]), "model_config": checkpoint["config"],
        "seed": args.seed, "queries_per_month_cap": args.queries_per_month,
        "n_queries": len(selection), "n_scored_per_channel": fixed_support.sum(0).tolist(),
        "query_identity_sha256": matched.content_hash({key: reference[key][selection] for key in ("month", "profile", "level", "target32")}, ("month", "profile", "level", "target32")),
        "target_precision": "canonical float64 standardized truth, inverse training normalization",
        "protocol": "same registered first depth queries per month and fixed canonical support for every condition; source Argo only perturbed; target labels/satellites fixed; OI recomputed",
        "noise_variance_supplied": args.supply_noise_variance,
        "duplicate_component_probe": probe, "conditions": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--queries-per-month", type=int, default=256)
    p.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/previous_baselines")
    p.add_argument("--satellite-cache", type=Path)
    p.add_argument("--operator-cache", type=Path)
    p.add_argument("--supply-noise-variance", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.queries_per_month < 1 or args.seed < 0:
        p.error("query cap must be positive and seed nonnegative")
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
