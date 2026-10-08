"""Freeze validation-only recommendations and report matched physical metrics.

All comparisons use the exact earlier CESM2 profile-position target and scoring
cells.  Checkpoint/family selection never reads 2005 prediction arrays.  The
previously used test year remains development evidence, not a new independent
confirmation.  Float32 target identity bridges torch and classical exports;
all reported metrics use the same original float64 truth from classical replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from ocean_tokenizer.reconstruction_metrics import (
    evaluate_reconstruction, inverse_standardize, paired_rmse_bootstrap)

SEEDS = (1234, 1235, 1236)
CHANNELS = ("TEMP", "SALT")
EXPECTED_COUNTS = {"validation": 92517, "development": 351895}
IDENTITY = ("month", "profile", "level")
RULE = {"version": 1,
        "criterion": "mean temperature/salinity standardized RMSE of the three-seed ensemble mean; per-depth training standard deviations",
        "splits": {"train": [2000, 2003], "validation": [2004, 2004], "development": [2005, 2005]},
        "selection_split": "validation", "seeds": list(SEEDS), "steps": 15000,
        "candidate_policy": "all complete three-seed registered families plus registered classical reference methods and the parity-verified existing learned OI, separately for Argo-only and surface-input settings; required auxiliary seed-1234 fixed MLP is comparison-only",
        "tie_break": "lexicographic candidate identifier", "identity_precision": "canonical float32 standardized target",
        "metric_target_precision": "original float64 standardized truth from canonical 47 replay",
        "recommendation_policy": "a classical method may win; a latent or MoE is never forced"}


class PendingReport(RuntimeError):
    """An authorized campaign has not yet produced all required artifacts."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def content_hash(arrays, names):
    h = hashlib.sha256()
    for name in names:
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def canonical_target(value):
    a = np.asarray(value, dtype=np.float32).copy()
    a[np.isnan(a)] = np.nan
    if a.ndim != 2 or a.shape[1] != 2 or np.isinf(a).any():
        raise ValueError("target must be [sample,2] with finite values or NaNs")
    return a


def normalization(arrays):
    scale = arrays.get("normalization_std", arrays.get("normalization_scale"))
    offset = arrays.get("normalization_mean", arrays.get("normalization_offset"))
    if scale is None or offset is None:
        raise ValueError("saved per-depth training normalization is required")
    scale, offset = np.asarray(scale, np.float64), np.asarray(offset, np.float64)
    if scale.ndim != 2 or scale.shape[1] != 2 or offset.shape != scale.shape:
        raise ValueError("normalization must have [depth,2] shape")
    if not np.isfinite(scale).all() or not (scale > 0).all() or not np.isfinite(offset).all():
        raise ValueError("normalization must be finite with positive standard deviations")
    if "normalization_scale" in arrays and not np.array_equal(scale, np.asarray(arrays["normalization_scale"], np.float64)):
        raise ValueError("duplicate normalization standard deviations disagree")
    if "normalization_offset" in arrays and not np.array_equal(offset, np.asarray(arrays["normalization_offset"], np.float64)):
        raise ValueError("duplicate normalization means disagree")
    return scale, offset


def load_arrays(path):
    path = Path(path)
    if not path.is_file():
        raise PendingReport(f"prediction artifact missing: {path}")
    with np.load(path, allow_pickle=False) as f:
        result = {name: f[name].copy() for name in f.files if name != "metadata_json"}
        metadata = json.loads(str(f["metadata_json"].item())) if "metadata_json" in f else {}
    result["metadata"] = metadata
    return result


def load_reference(baseline_dir, split, *, expected_counts=EXPECTED_COUNTS):
    ref = load_arrays(Path(baseline_dir)/f"oi_{split}.npz")
    ref["target"] = np.asarray(ref["target"], np.float64)
    ref["target32"] = canonical_target(ref["target"])
    scale, offset = normalization(ref)
    ref["scale"], ref["offset"] = scale, offset
    n = len(ref["target"])
    for name in IDENTITY:
        if ref[name].shape != (n,) or ref[name].dtype.kind not in "iu":
            raise ValueError(f"reference {name} must be a matching integer vector")
    counts = np.isfinite(ref["target"]).sum(0)
    if any(int(count) != expected_counts[split] for count in counts):
        raise ValueError(f"{split} support differs: {counts.tolist()} expected {expected_counts[split]} per variable")
    if np.any(ref["level"] < 0) or np.any(ref["level"] >= len(scale)):
        raise ValueError("reference depth indices outside saved normalization")
    if np.asarray(ref["levels"]).shape != (len(scale),):
        raise ValueError("reference depth coordinates differ from normalization")
    if not ref["metadata"].get("cohort_sha256"):
        raise ValueError("classical reference must identify its raw cohort SHA256")
    ref["identity_fingerprint"] = content_hash({**ref,"target":ref["target32"]}, (*IDENTITY, "target"))
    return ref


def assert_identity(arrays, ref, name="method"):
    for key in IDENTITY:
        if not np.array_equal(arrays[key], ref[key]):
            raise ValueError(f"{name}: {key} scoring identity differs")
    if not np.array_equal(canonical_target(arrays["target"]), ref["target32"], equal_nan=True):
        raise ValueError(f"{name}: canonical float32 target/support differs")
    scale, offset = normalization(arrays)
    for key, value, expected in (("normalization_std", scale, ref["scale"]),
                                 ("normalization_mean", offset, ref["offset"])):
        if not np.array_equal(value, expected):
            raise ValueError(f"{name}: {key} differs")
    if "levels" in arrays and not np.array_equal(arrays["levels"], ref["levels"]):
        raise ValueError(f"{name}: depth coordinates differ")
    cohort_sha = arrays.get("metadata", {}).get("cohort_sha256")
    if cohort_sha is not None and cohort_sha != ref["metadata"]["cohort_sha256"]:
        raise ValueError(f"{name}: raw cohort SHA256 differs")
    mean = np.asarray(arrays["mean"], np.float64)
    if mean.shape != ref["target"].shape or not np.isfinite(mean[np.isfinite(ref["target"])]).all():
        raise ValueError(f"{name}: predictions must cover every scored target")
    if "std" in arrays:
        std = np.asarray(arrays["std"], np.float64)
        valid = np.isfinite(ref["target"])
        if std.shape != mean.shape or not np.isfinite(std[valid]).all() or not (std[valid] > 0).all():
            raise ValueError(f"{name}: uncertainty must be finite and positive at every scored target")
    if "baseline" in arrays:
        valid = np.isfinite(ref["target"])
        baseline = np.asarray(arrays["baseline"], np.float64)
        if baseline.shape != mean.shape or not np.allclose(baseline[valid], ref["mean"][valid],
                                                           rtol=5e-5, atol=5e-5):
            raise ValueError(f"{name}: fixed OI baseline differs")


def prediction_path(job, split):
    folder, seed = Path(job["output"]), job["seed"]
    candidates = [folder/f"{split}_seed{seed}.npz", folder/f"{split}_predictions_seed{seed}.npz"]
    explicit = job.get(f"{split}_array")
    if explicit:
        candidates.insert(0, Path(explicit))
    for path in candidates:
        if path.is_file():
            return path
    raise PendingReport(f"{split} prediction missing for {job['tag']}: {candidates}")


def summary_contract(job):
    path = Path(job["output"])/f"summary_seed{job['seed']}.json"
    if not path.is_file():
        raise PendingReport(f"summary missing: {path}")
    summary = json.loads(path.read_text())
    if int(summary.get("seed", -1)) != job["seed"]:
        raise ValueError(f"{job['tag']}: seed differs")
    if int(job["steps"]) != 15000:
        raise ValueError("registered campaign requires 15000 steps per learned run")
    history = summary.get("history", [])
    complete = summary.get("completed_steps", history[-1].get("step", -1) if history else -1)
    if complete != 15000:
        raise PendingReport(f"{job['tag']}: completed {complete}/15000 steps")
    nominal = summary.get("training", {}).get("steps", summary.get("steps"))
    if nominal != 15000:
        raise ValueError(f"{job['tag']}: saved training budget differs")
    hashes = summary.get("source_sha256", summary.get("source_hashes"))
    if not hashes or any(len(value) != 64 for value in hashes.values()):
        raise ValueError(f"{job['tag']}: training source hashes missing")
    if job["kind"] == "latent":
        training, config = summary["training"], summary["config"]
        if training.get("freeze_analysis") is not True:
            raise ValueError(f"{job['tag']}: exact OI must remain frozen")
        if training.get("context_profiles", 0) < 6080 or (training.get("eval_context_profiles") or 6080) < 6080:
            raise ValueError(f"{job['tag']}: latent context must retain the complete source pool")
        if config.get("use_latent") != (not job["family"].endswith("latent_off")):
            raise ValueError(f"{job['tag']}: latent ablation flag differs")
        if config.get("use_local") != (not job["family"].endswith("local_off")):
            raise ValueError(f"{job['tag']}: local ablation flag differs")
        if bool(config.get("n_sat_features", 0)) != job["surface"]:
            raise ValueError(f"{job['tag']}: surface input configuration differs")
    # Exclude evaluation results and wall-clock timing: adding development
    # evaluation may rewrite a summary without changing its frozen training.
    names = ("seed", "best_step", "params", "config", "training", "steps", "completed_steps",
             "n_latent", "n_slots", "backbone", "mass_mode", "refiner_km", "refiner_gate",
             "lr", "weight_decay", "batch", "n_profiles", "ablations", "historical_setting",
             "source_sha256", "source_hashes", "data")
    contract = {name: summary[name] for name in names if name in summary}
    candidates = [Path(job["output"])/f"best_seed{job['seed']}.pt",
                  Path(job["output"])/f"model_seed{job['seed']}.pt"]
    checkpoint = next((p for p in candidates if p.is_file()), None)
    if checkpoint is None:
        raise PendingReport(f"best checkpoint missing for {job['tag']}")
    return summary, {"summary_path": str(path), "summary_sha256_at_freeze": sha256(path),
                     "training_contract": contract, "training_contract_sha256": json_hash(contract),
                     "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}


def standardized_rmse(mean, ref):
    metrics = evaluate_reconstruction(mean, ref["target"])
    return float(np.mean([metrics[ch]["rmse"] for ch in CHANNELS]))


def validation_prediction(job, ref):
    summary, contract = summary_contract(job)
    path = prediction_path(job, "validation")
    a = load_arrays(path)
    assert_identity(a, ref, job["tag"])
    mean = np.asarray(a["mean"], np.float64)
    saved = summary.get("validation", {}).get("scores", summary.get("scores", {}).get("validation", {}))
    saved_score = saved.get("macro_z", summary.get("validation", {}).get("mean_standardized_rmse"))
    if saved_score is not None and not np.isclose(saved_score, standardized_rmse(mean, ref), rtol=2e-5, atol=2e-7):
        raise ValueError(f"{job['tag']}: stored validation standardized RMSE differs from predictions")
    payload = {**{k: ref[k] for k in IDENTITY}, "target": ref["target32"], "mean": mean}
    if "std" in a:
        payload["std"] = np.asarray(a["std"], np.float64)
    contract.update(array_path=str(path), array_sha256_at_freeze=sha256(path),
                    validation_content_sha256=content_hash(payload, (*IDENTITY, "target", "mean", *(('std',) if "std" in payload else ()))))
    return payload, contract


def fixed_mlp_validation(campaign_path, campaign, ref):
    """Required fixed seed-1234 comparison, outside architecture selection."""
    directory=Path(campaign.get("fixed_mlp_dir",Path(campaign_path).parent/"fixed_pointwise_mlp"))
    job={"output":str(directory),"seed":1234,"tag":"fixed_pointwise_mlp"}
    summary_path=directory/"summary_seed1234.json"
    if not summary_path.is_file():
        raise PendingReport(f"required fixed pointwise MLP summary missing: {summary_path}")
    summary=json.loads(summary_path.read_text())
    if summary.get("seed")!=1234:
        raise ValueError("fixed pointwise MLP must use original seed 1234")
    if summary.get("completed_epochs",summary.get("epochs_completed"))!=30:
        raise PendingReport("fixed pointwise MLP has not completed the original 30 epochs")
    sources=summary.get("source_sha256",summary.get("source_hashes"))
    if not sources or any(len(value)!=64 for value in sources.values()):
        raise ValueError("fixed pointwise MLP source hashes missing")
    checkpoint=directory/"model_seed1234.pt"
    if not checkpoint.is_file():
        raise PendingReport("fixed pointwise MLP model checkpoint missing")
    path=prediction_path(job,"validation")
    arrays=load_arrays(path);assert_identity(arrays,ref,"fixed_pointwise_mlp")
    mean=np.asarray(arrays["mean"],np.float64)
    stored=summary.get("validation",{}).get("mean_standardized_rmse")
    if stored is not None and not np.isclose(stored,standardized_rmse(mean,ref),rtol=2e-5,atol=2e-7):
        raise ValueError("fixed MLP validation score differs from saved predictions")
    names=("seed","completed_epochs","epochs_completed","training_recipe","data","params",
           "source_sha256","source_hashes","architecture","features","hidden")
    training={name:summary[name] for name in names if name in summary}
    payload={**{key:ref[key] for key in IDENTITY},"target":ref["target32"],"mean":mean}
    return {"directory":str(directory),"summary_path":str(summary_path),
            "summary_sha256_at_freeze":sha256(summary_path),"training_contract":training,
            "training_contract_sha256":json_hash(training),"checkpoint_path":str(checkpoint),
            "checkpoint_sha256":sha256(checkpoint),"validation_array":str(path),
            "validation_array_sha256_at_freeze":sha256(path),
            "validation_content_sha256":content_hash(payload,(*IDENTITY,"target","mean")),
            "included_in_architecture_selection":False,
            "reason":"original fixed 30-epoch seed-1234 baseline; comparison-only, not a three-seed candidate"}


def prior_learned_oi_validation(campaign_path,campaign,ref):
    """Stronger existing 67 baseline, replayed without retraining or searching."""
    directory=Path(campaign.get("prior_learned_oi_dir",Path(campaign_path).parent/"prior_learned_oi"))
    registry_path=directory/"aux_registry.json"
    if not registry_path.is_file():
        raise PendingReport(f"required prior learned-OI parity registry missing: {registry_path}")
    registry=json.loads(registry_path.read_text())
    if registry.get("parity_verified") is not True or registry.get("eligible_validation_candidate") is not True:
        raise ValueError("prior learned OI eligibility/parity audit unresolved; do not silently drop the strongest earlier comparator")
    if registry.get("cohort_sha256")!=ref["metadata"]["cohort_sha256"]:
        raise ValueError("prior learned OI raw cohort differs")
    if registry.get("seed")!=1234 or registry.get("training_steps")!=1500:
        raise ValueError("prior learned OI must identify its existing seed-1234 1500-step recipe")
    checkpoint_info=registry["source_checkpoint"]
    checkpoint=Path(checkpoint_info["path"])
    if not checkpoint.is_file() or sha256(checkpoint)!=checkpoint_info["sha256"]:
        raise ValueError("prior learned OI checkpoint changed since parity audit")
    historical_info=registry["historical_summary"]
    historical_path=Path(historical_info["path"])
    if sha256(historical_path)!=historical_info["sha256"]:
        raise ValueError("prior learned OI historical summary changed since parity audit")
    historical=json.loads(historical_path.read_text())
    if (historical.get("anomaly")!="exact" or historical.get("steps")!=1500
            or historical.get("first_guess",{}).get("kind")!="none"
            or historical.get("input_qc_z") is not None):
        raise ValueError("prior learned OI historical target/background/QC recipe differs")
    path=Path(registry["arrays"]["validation"]["path"])
    if sha256(path)!=registry["arrays"]["validation"]["sha256"]:
        raise ValueError("prior learned OI validation artifact changed since parity audit")
    arrays=load_arrays(path);assert_identity(arrays,ref,"prior_learned_oi")
    mean=np.asarray(arrays["mean"],np.float64)
    stored=historical["scores"]["validation"]["macro_z"]
    if not np.isclose(stored,standardized_rmse(mean,ref),rtol=2e-5,atol=2e-7):
        raise ValueError("prior learned OI prediction differs from its historical validation checkpoint")
    payload={**{key:ref[key] for key in IDENTITY},"target":ref["target32"],"mean":mean}
    return {"directory":str(directory),"registry_path":str(registry_path),"registry_sha256":sha256(registry_path),
            "registry":registry,"checkpoint_sha256":sha256(checkpoint),"historical_summary_sha256":sha256(historical_path),
            "validation_content_sha256":content_hash(payload,(*IDENTITY,"target","mean")),
            "validation_mean_standardized_rmse":standardized_rmse(mean,ref),"seed":1234,"training_steps":1500,
            "parameters":historical.get("params"),"included_in_architecture_selection":True,
            "role":"previously trained off-the-shelf baseline; fixed existing checkpoint, not a new 15000-step three-seed family"}


def grouped_jobs(campaign):
    groups, tags = {}, set()
    for job in campaign["jobs"]:
        if job["tag"] in tags:
            raise ValueError("campaign tags must be unique")
        tags.add(job["tag"])
        key = ("surface" if job["surface"] else "argo", job["family"])
        groups.setdefault(key, []).append(job)
    for key, jobs in groups.items():
        if sorted(job["seed"] for job in jobs) != list(SEEDS):
            raise PendingReport(f"{key}: exactly seeds {SEEDS} are required")
        if len({job["kind"] for job in jobs}) != 1:
            raise ValueError(f"{key}: run kinds differ within family")
    return groups


def baseline_candidates(baseline_dir, split):
    manifest_path = Path(baseline_dir)/"replay_manifest.json"
    if not manifest_path.is_file():
        raise PendingReport(f"baseline replay manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return {name: Path(item["path"]) for name, item in manifest["exports"][split].items()}


def ensemble(predictions):
    means = np.stack([p["mean"] for p in predictions])
    result = {"mean": means.mean(0)}
    available = ["std" in p for p in predictions]
    if any(available) and not all(available):
        raise ValueError("uncertainty availability differs within seed family")
    if all(available):
        variance = np.mean(np.stack([p["std"]**2 for p in predictions]) + means**2, axis=0)-result["mean"]**2
        result["std"] = np.sqrt(np.maximum(variance, 1e-12))
    return result


def build_selection(campaign_path, baseline_dir, *, cohort_path=None, expected_counts=EXPECTED_COUNTS):
    campaign_path, baseline_dir = Path(campaign_path), Path(baseline_dir)
    campaign = json.loads(campaign_path.read_text())
    ref = load_reference(baseline_dir, "validation", expected_counts=expected_counts)
    cohort_path = Path(cohort_path) if cohort_path else ROOT/"data/synthetic_argo/cesm2_uniform.nc"
    if sha256(cohort_path) != ref["metadata"]["cohort_sha256"]:
        raise ValueError("current raw cohort differs from classical replay cohort SHA256")
    fixed_mlp=fixed_mlp_validation(campaign_path,campaign,ref)
    prior_oi=prior_learned_oi_validation(campaign_path,campaign,ref)
    groups = grouped_jobs(campaign)
    candidates, contracts = {"argo": [], "surface": []}, {}
    for mode in candidates:
        candidates[mode].append({"id":"existing:prior_learned_oi","family":"prior_learned_oi",
            "kind":"existing_learned_oi","seeds":[1234],"training_steps":1500,
            "validation_mean_standardized_rmse":prior_oi["validation_mean_standardized_rmse"],
            "checkpoint_sha256":prior_oi["checkpoint_sha256"],"registry_sha256":prior_oi["registry_sha256"]})
    for name, path in baseline_candidates(baseline_dir, "validation").items():
        a = load_arrays(path); assert_identity(a, ref, name)
        item = {"id": f"classic:{name}", "family": name, "kind": "classical", "seeds": [],
                "validation_mean_standardized_rmse": standardized_rmse(a["mean"], ref),
                "validation_array": str(path), "validation_array_sha256": sha256(path),
                "validation_content_sha256": content_hash({**{k:ref[k] for k in IDENTITY}, "target":ref["target32"],
                    "mean": np.asarray(a["mean"], np.float64)}, (*IDENTITY,"target","mean"))}
        for mode in candidates:
            candidates[mode].append(dict(item))
    for (mode, family), jobs in sorted(groups.items()):
        values = []
        for job in sorted(jobs, key=lambda j:j["seed"]):
            a, contract = validation_prediction(job, ref)
            values.append(a); contracts[job["tag"]] = contract
        e = ensemble(values)
        candidates[mode].append({"id": f"learned:{family}", "family": family,
            "kind": jobs[0]["kind"], "seeds": list(SEEDS), "tags": [j["tag"] for j in sorted(jobs,key=lambda j:j["seed"])],
            "validation_mean_standardized_rmse": standardized_rmse(e["mean"], ref)})
    for mode in ("argo", "surface"):
        latent_contracts = [contract["training_contract"] for tag,contract in contracts.items()
                            if contract["training_contract"].get("config", {}).get("n_obs_features") is not None
                            and bool(contract["training_contract"]["config"].get("n_sat_features")) == (mode == "surface")]
        for key in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint"):
            fingerprints = {c.get("data", {}).get(key) for c in latent_contracts}
            if len(fingerprints) > 1:
                raise ValueError(f"{mode}: latent input {key} differs across families/seeds")
    selected = {mode: min(items, key=lambda a:(a["validation_mean_standardized_rmse"], a["id"]))
                for mode,items in candidates.items() if items}
    return {"format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "rule": RULE, "expected_counts": dict(expected_counts),
            "campaign_path": str(campaign_path), "campaign_sha256": sha256(campaign_path),
            "baseline_dir": str(baseline_dir), "raw_cohort_path": str(cohort_path),
            "raw_cohort_sha256": sha256(cohort_path), "validation_identity": ref["identity_fingerprint"],
            "reporter_source_sha256": sha256(Path(__file__)),
            "metric_source_sha256": sha256(ROOT/"src/ocean_tokenizer/reconstruction_metrics.py"),
            "candidates": candidates, "selected": selected, "runs": contracts,
            "auxiliary_fixed_mlp":fixed_mlp,"auxiliary_prior_learned_oi":prior_oi}


def freeze_selection(campaign_path, baseline_dir, destination, **kwargs):
    result = build_selection(campaign_path, baseline_dir, **kwargs)
    destination = Path(destination)
    if destination.exists():
        previous = json.loads(destination.read_text())
        for key in ("rule", "expected_counts", "campaign_sha256", "raw_cohort_sha256", "validation_identity", "selected"):
            if previous.get(key) != result[key]:
                raise ValueError(f"existing frozen selection differs: {key}")
        def candidate_contracts(candidates):
            return {mode:[{k:v for k,v in item.items() if k!="validation_array_sha256"}
                          for item in items] for mode,items in candidates.items()}
        if candidate_contracts(previous["candidates"]) != candidate_contracts(result["candidates"]):
            raise ValueError("frozen validation candidate set or scientific content changed")
        for tag, contract in result["runs"].items():
            for key in ("training_contract_sha256", "checkpoint_sha256", "validation_content_sha256"):
                if previous["runs"][tag][key] != contract[key]:
                    raise ValueError(f"frozen run changed: {tag} {key}")
        for key in ("training_contract_sha256","checkpoint_sha256","validation_content_sha256"):
            if previous["auxiliary_fixed_mlp"][key]!=result["auxiliary_fixed_mlp"][key]:
                raise ValueError(f"frozen fixed pointwise MLP changed: {key}")
        for key in ("registry_sha256","checkpoint_sha256","historical_summary_sha256","validation_content_sha256"):
            if previous["auxiliary_prior_learned_oi"][key]!=result["auxiliary_prior_learned_oi"][key]:
                raise ValueError(f"frozen prior learned OI changed: {key}")
        return previous
    atomic_json(destination, result)
    return result


def calibrated_oi_std(reference):
    """Per-depth/channel validation residual RMS; frozen before test use."""
    sigma = np.empty_like(reference["scale"])
    for level in range(len(sigma)):
        selected = reference["level"] == level
        for channel in range(2):
            y, p = reference["target"][selected, channel], reference["mean"][selected, channel]
            valid = np.isfinite(y)
            if not valid.any():
                raise ValueError(f"no validation observations to calibrate OI at depth {level}")
            sigma[level, channel] = max(float(np.sqrt(np.mean((p[valid]-y[valid])**2))), 1e-6)
    return sigma


def physical_metrics(prediction, reference, *, by_depth=False):
    levels, scale, offset = reference["level"], reference["scale"], reference["offset"]
    p = inverse_standardize(prediction["mean"], levels, scale, offset)
    y = inverse_standardize(reference["target"], levels, scale, offset)
    std = inverse_standardize(prediction["std"], levels, scale, uncertainty=True) if "std" in prediction else None
    metrics = evaluate_reconstruction(p, y, std)
    if "climatology_physical" in reference:
        absolute = evaluate_reconstruction(p+reference["climatology_physical"], y+reference["climatology_physical"])
        for ch in CHANNELS:
            metrics[ch].update(absolute_r2=absolute[ch]["r2"], absolute_pearson_r=absolute[ch]["pearson_r"])
    result = {"physical_metrics": metrics, "mean_standardized_rmse": standardized_rmse(prediction["mean"], reference)}
    if by_depth:
        result["by_depth_physical_metrics"] = {str(float(depth)): evaluate_reconstruction(
            p[levels==i], y[levels==i], None if std is None else std[levels==i])
            for i,depth in enumerate(reference["levels"])}
    return result


def summarize_seeds(metrics):
    result = {}
    for ch in CHANNELS:
        keys = metrics[0]["physical_metrics"][ch].keys()
        result[ch] = {}
        for key in keys:
            values = [m["physical_metrics"][ch][key] for m in metrics]
            if key == "n":
                if len(set(values)) != 1:
                    raise ValueError("seed support counts differ")
                result[ch][key] = values[0]
            elif all(v is None for v in values):
                result[ch][key] = {"mean": None, "sd": None}
            elif any(v is None for v in values):
                raise ValueError(f"seed metric availability differs: {ch} {key}")
            else:
                result[ch][key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
    return result


def paired_contrast(prediction, baseline, reference, *, draws=4000):
    levels, scale, offset = reference["level"], reference["scale"], reference["offset"]
    p = inverse_standardize(prediction["mean"], levels, scale, offset)
    b = inverse_standardize(baseline["mean"], levels, scale, offset)
    y = inverse_standardize(reference["target"], levels, scale, offset)
    return {ch: paired_rmse_bootstrap(p[:, i], b[:, i], y[:, i], reference["month"], draws=draws)
            for i,ch in enumerate(CHANNELS)}


def paired_seed_rmse(row, baseline_row):
    """Pair equal training seeds; report their physical RMSE differences."""
    actual={m["seed"]:m["physical_metrics"] for m in row["seed_metrics"]}
    base={m["seed"]:m["physical_metrics"] for m in baseline_row["seed_metrics"]}
    if actual.keys()!=base.keys():
        raise ValueError("paired comparisons require identical training seed sets")
    return {ch:{"seeds":list(SEEDS),
                "per_seed_delta_rmse":[actual[seed][ch]["rmse"]-base[seed][ch]["rmse"] for seed in SEEDS],
                "mean_delta_rmse":float(np.mean([actual[seed][ch]["rmse"]-base[seed][ch]["rmse"] for seed in SEEDS])),
                "sd_delta_rmse":float(np.std([actual[seed][ch]["rmse"]-base[seed][ch]["rmse"] for seed in SEEDS],ddof=1))}
            for ch in CHANNELS}


def build_report(campaign_path, baseline_dir, selection_path, *, bootstrap_draws=4000):
    selection_path = Path(selection_path)
    if not selection_path.is_file():
        raise PendingReport("freeze validation selection before reading any new-model development arrays")
    choice = json.loads(selection_path.read_text())
    if choice.get("rule") != RULE:
        raise ValueError("frozen selection rule differs")
    # Revalidate all training/checkpoints/validation contents first.  Summary
    # and ZIP timestamps may change during evaluation; scientific content cannot.
    confirmed = freeze_selection(campaign_path, baseline_dir, selection_path,
        cohort_path=choice["raw_cohort_path"], expected_counts=choice["expected_counts"])
    campaign = json.loads(Path(campaign_path).read_text())
    groups = grouped_jobs(campaign)
    validation = load_reference(baseline_dir, "validation", expected_counts=choice["expected_counts"])
    reference = load_reference(baseline_dir, "development", expected_counts=choice["expected_counts"])
    if not np.array_equal(validation["scale"], reference["scale"]) or not np.array_equal(validation["offset"], reference["offset"]):
        raise ValueError("validation/development normalization differs")
    if reference["metadata"]["cohort_sha256"] != choice["raw_cohort_sha256"]:
        raise ValueError("development raw cohort differs from frozen choice")
    sigma = calibrated_oi_std(validation)
    rows, predictions, file_hashes = [], {}, {}
    for name,path in baseline_candidates(baseline_dir, "development").items():
        a = load_arrays(path); assert_identity(a, reference, name)
        prediction = {"mean": np.asarray(a["mean"], np.float64)}
        if name == "oi":
            prediction["std"] = sigma[reference["level"]]
        row = {"family": name, "mode": "argo", "kind": "classical", "seeds": [],
               "uncertainty_method": "per-depth residual RMS fitted on validation 2004 only" if name=="oi" else "unavailable",
               **physical_metrics(prediction, reference, by_depth=True)}
        rows.append(row); predictions[("argo",name)] = prediction
        file_hashes[str(path)] = sha256(path)
    mlp_path=prediction_path({"output":confirmed["auxiliary_fixed_mlp"]["directory"],
                             "seed":1234,"tag":"fixed_pointwise_mlp"},"development")
    mlp_arrays=load_arrays(mlp_path);assert_identity(mlp_arrays,reference,"fixed_pointwise_mlp")
    mlp_prediction={"mean":np.asarray(mlp_arrays["mean"],np.float64)}
    rows.append({"family":"fixed_pointwise_mlp","mode":"argo","kind":"classical","seeds":[1234],
                 "statistical_role":"fixed single-seed learning baseline, comparison-only",
                 "uncertainty_method":"unavailable",**physical_metrics(mlp_prediction,reference,by_depth=True)})
    predictions[("argo","fixed_pointwise_mlp")]=mlp_prediction
    file_hashes[str(mlp_path)]=sha256(mlp_path)
    prior_contract=confirmed["auxiliary_prior_learned_oi"]
    prior_registry=prior_contract["registry"]
    prior_validation_path=Path(prior_registry["arrays"]["validation"]["path"])
    prior_validation=load_arrays(prior_validation_path)
    assert_identity(prior_validation,validation,"prior_learned_oi validation")
    prior_scale=calibrated_oi_std({**validation,"mean":np.asarray(prior_validation["mean"],np.float64)})
    prior_path=Path(prior_registry["arrays"]["development"]["path"])
    if sha256(prior_path)!=prior_registry["arrays"]["development"]["sha256"]:
        raise ValueError("prior learned OI development artifact changed since parity audit")
    prior_arrays=load_arrays(prior_path)
    assert_identity(prior_arrays,reference,"prior_learned_oi development")
    prior_prediction={"mean":np.asarray(prior_arrays["mean"],np.float64),"std":prior_scale[reference["level"]]}
    rows.append({"family":"prior_learned_oi","mode":"argo","kind":"classical","seeds":[1234],
                 "training_steps":1500,"parameters":prior_contract["parameters"],
                 "statistical_role":"fixed existing 1500-step single-seed learned OI, parity verified",
                 "uncertainty_method":"per-depth residual RMS fitted on validation 2004 only",
                 **physical_metrics(prior_prediction,reference,by_depth=True)})
    predictions[("argo","prior_learned_oi")]=prior_prediction
    file_hashes[str(prior_path)]=sha256(prior_path)
    for (mode,family),jobs in sorted(groups.items()):
        values, seed_metrics = [], []
        for job in sorted(jobs,key=lambda j:j["seed"]):
            path = prediction_path(job,"development")
            a = load_arrays(path); assert_identity(a,reference,job["tag"])
            prediction = {"mean": np.asarray(a["mean"],np.float64)}
            if "std" in a:
                prediction["std"] = np.asarray(a["std"],np.float64)
            values.append(prediction)
            seed_metrics.append({"seed":job["seed"],**physical_metrics(prediction,reference,by_depth=True)})
            file_hashes[str(path)] = sha256(path)
        e = ensemble(values)
        result = {"family":family,"mode":mode,"kind":jobs[0]["kind"],"seeds":list(SEEDS),
                  "seed_metrics":seed_metrics,"seed_mean_sd":summarize_seeds(seed_metrics),
                  "ensemble":physical_metrics(e,reference,by_depth=True),
                  "params_per_seed":[confirmed["runs"][job["tag"]]["training_contract"].get("params")
                                     for job in sorted(jobs,key=lambda j:j["seed"])],
                  "uncertainty_method":"Gaussian learned scales plus ensemble total variance" if "std" in e else "unavailable"}
        rows.append(result); predictions[(mode,family)] = e
    oi = predictions[("argo","oi")]
    prior=predictions[("argo","prior_learned_oi")]
    learned_rows={(row["mode"],row["family"]):row for row in rows if row["kind"]!="classical"}
    contrasts = {}
    for (mode,family),prediction in predictions.items():
        if (mode,family) == ("argo","oi"):
            continue
        key = f"{mode}:{family}"
        contrasts[key] = {"against_oi":paired_contrast(prediction,oi,reference,draws=bootstrap_draws)}
        if (mode,family)!=("argo","prior_learned_oi"):
            contrasts[key]["against_prior_learned_oi"]=paired_contrast(prediction,prior,reference,draws=bootstrap_draws)
        oldkey = (mode,"previous_token64")
        if oldkey in predictions and oldkey != (mode,family):
            contrasts[key]["against_previous_token64"] = paired_contrast(prediction,predictions[oldkey],reference,draws=bootstrap_draws)
            if (mode,family) in learned_rows:
                contrasts[key]["paired_seeds_against_previous_token64"] = paired_seed_rmse(
                    learned_rows[(mode,family)],learned_rows[oldkey])
    component_contrasts = {}
    for mode in ("argo","surface"):
        full = (mode,"soft_moe192")
        for component in ("latent_off","local_off"):
            removed = (mode,f"soft_moe192_{component}")
            if full in predictions and removed in predictions:
                component_contrasts[f"{mode}:full_minus_{component}"] = paired_contrast(
                    predictions[full],predictions[removed],reference,draws=bootstrap_draws)
                component_contrasts[f"{mode}:full_minus_{component}"]["paired_seed_rmse"] = paired_seed_rmse(
                    learned_rows[full],learned_rows[removed])
    multimodal_contrasts = {}
    for mode,family in predictions:
        if mode=="surface" and ("argo",family) in predictions:
            multimodal_contrasts[family] = paired_contrast(predictions[(mode,family)],predictions[("argo",family)],reference,draws=bootstrap_draws)
            multimodal_contrasts[family]["paired_seed_rmse"] = paired_seed_rmse(
                learned_rows[(mode,family)],learned_rows[("argo",family)])
    selected_against_prior={}
    for mode,choice_item in confirmed["selected"].items():
        selected_key=("argo",choice_item["family"]) if choice_item["kind"] in ("classical","existing_learned_oi") else (mode,choice_item["family"])
        delta=paired_contrast(predictions[selected_key],prior,reference,draws=bootstrap_draws)
        selected_against_prior[mode]={"selected_family":choice_item["family"],"against_prior_learned_oi":delta,
            "lower_test_rmse_in_both_variables":all(delta[ch]["delta_rmse"]<0 for ch in CHANNELS),
            "note":"development comparison only; never changes the frozen validation choice"}
    return {"format_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "task":"CESM2 matched retrospective reconstruction", "evidence":"previously used 2005 test year; development evidence",
        "selection":confirmed,"selection_file_sha256":sha256(selection_path),"metrics":"physical RMSE/MAE/bias; R2 and correlation separately on anomalies and absolute states",
        "test_identity":reference["identity_fingerprint"],"test_n_per_variable":dict(zip(CHANNELS,np.isfinite(reference["target"]).sum(0).tolist())),
        "normalization_mean":reference["offset"].tolist(),"normalization_std":reference["scale"].tolist(),
        "levels_m":reference["levels"].tolist(),"oi_calibration_std_z":sigma.tolist(),
        "prior_learned_oi_calibration_std_z":prior_scale.tolist(),"selected_against_previous_best":selected_against_prior,
        "rows":rows,"contrasts":contrasts,"component_contrasts":component_contrasts,"multimodal_contrasts":multimodal_contrasts,
        "development_array_sha256":file_hashes,"reporter_source_sha256":sha256(Path(__file__)),
        "metric_source_sha256":sha256(ROOT/"src/ocean_tokenizer/reconstruction_metrics.py"),
        "limits":["20 fixed depths; arbitrary-depth continuous inference not validated",
                  "same-month reconstruction; no causal forecasting",
                  "2005 has been inspected previously; not an independent new test",
                  "synthetic SST/SSS are noise-free 5m truth and steric sea level derives from the reconstructed T/S",
                  "new system changes OI anchoring, evidence representation, optimizer settings and sometimes capacity; not a pure backbone-only comparison",
                  "deterministic methods other than OI have no supplied predictive uncertainty",
                  "parameter counts include registered inactive modules in latent-off and old surface variants; not an effective-compute comparison",
                  "whole-month bootstrap covers 12 test months, not training-seed or long-term climate uncertainty"]}


def fmt(value,digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def metric_rows(result,mode,ensemble_values):
    lines = []
    for row in result["rows"]:
        if row["mode"] != mode and row["kind"] != "classical":
            continue
        label = row["family"]
        if row["kind"]=="classical":
            if mode=="surface":
                label += "（仅 Argo）"
            m = row["physical_metrics"]
            values = [fmt(m[ch][key]) for ch in CHANNELS for key in ("rmse","mae","mean_bias")]
            count = "固定种子 1234" if row["seeds"] else "固定方法"
        elif ensemble_values:
            m = row["ensemble"]["physical_metrics"]
            values = [fmt(m[ch][key]) for ch in CHANNELS for key in ("rmse","mae","mean_bias")]
            count = "三种子集成"
        else:
            m = row["seed_mean_sd"]
            values = [f"{fmt(m[ch][key]['mean'])} ± {fmt(m[ch][key]['sd'])}"
                      for ch in CHANNELS for key in ("rmse","mae","mean_bias")]
            count = "三种子均值 ± 样本标准差"
        lines.append("| "+" | ".join([label,count,*values])+" |")
    return lines


def chinese_report(result):
    n=result["test_n_per_variable"]; selections=result["selection"]["selected"]
    lines=["# CESM2 同协议重建：旧模型、新架构与经典基线\n",
        f"训练 2000–2003，验证 2004，评分 2005。每月 6080 个输入剖面、1520 个查询剖面；每个变量评分 **{n['TEMP']:,} 个有效值**。本轮注册 **{len(result['selection']['runs'])} 次架构训练**，各 **15000 步**，每个架构三个种子；固定 MLP 与既有 learned OI 的不同预算另列。",
        "异常目标取每个 Argo 剖面自身位置的气候态；所有方法共享评分顺序、有效值掩码和训练期逐深度归一化。2005 是以前看过的测试年份，本轮属于开发证据。\n",
        "主指标为温度 RMSE（°C）与盐度 RMSE（PSU），同时给出 MAE 和平均偏差（预测减真值）。模型选择使用验证集温盐平均标准化 RMSE：新架构按三种子集成均值评分，既有单检查点对照按其固定预测评分；经典或既有 learned OI 可以获选。\n",
        "## 仅由验证集确定的最终方案\n",
        "| 输入设置 | 验证集选中方法 | 温盐平均标准化 RMSE |",
        "|---|---|---:|"]
    for mode,title in (("argo","Argo"),("surface","Argo + SST/SSS/steric sea level")):
        selected=selections.get(mode)
        if selected:
            lines.append(f"| {title} | {selected['family']} | {selected['validation_mean_standardized_rmse']:.6f} |")
    lines += ["\n集成选择被冻结后才读取新增模型的 2005 预测。下表分开列出单次训练的三种子统计和集成结果，不混用两种口径。\n"]
    lines += ["## 是否超过此前最好方案\n",
              "对照为已核验重放的 67 learned OI；差值为本次验证集选中方案减去该既有方案。只超过 44 OI 不足以支持超过此前最好结果。\n",
              "| 输入设置 | 验证集选中方法 | 温度 ΔRMSE °C | 盐度 ΔRMSE PSU | 2005 两变量均改善 |",
              "|---|---|---:|---:|---|"]
    for mode,item in result["selected_against_previous_best"].items():
        delta=item["against_prior_learned_oi"]
        verdict="是（开发证据）" if item["lower_test_rmse_in_both_variables"] else "未同时改善两个变量"
        if item["selected_family"]=="prior_learned_oi":
            verdict="保留既有 learned OI"
        lines.append(f"| {mode} | {item['selected_family']} | {delta['TEMP']['delta_rmse']:+.6f} | {delta['SALT']['delta_rmse']:+.6f} | {verdict} |")
    lines.append("")
    lines.append("参数量按模型注册总数统计。关闭 latent 的模型仍可能保留未使用的模块，旧 surface 配置也含未使用编码器；不能把这些总数解释为有效计算量。共享 GPU 上的墙钟时间不用于宣称计算效率优势。\n")
    lines += ["## 架构与容量\n",
              "| 方法 / 输入 | 每个种子的参数量 | Latent 数 | 宽度 | 共享状态块 | AdamW 峰值 LR |",
              "|---|---:|---:|---:|---:|---:|"]
    prior=result["selection"]["auxiliary_prior_learned_oi"]
    lines.append(f"| prior_learned_oi / argo（既有 1500 步） | {prior['parameters']} | — | — | — | 历史学习设置，见原登记 |")
    for row in result["rows"]:
        if row["kind"]=="classical":
            continue
        contract=next(c["training_contract"] for tag,c in result["selection"]["runs"].items()
                      if tag in next(item["tags"] for item in result["selection"]["candidates"][row["mode"]]
                                     if item["family"]==row["family"] and item["kind"]!="classical"))
        cfg,train=contract.get("config",{}),contract.get("training",{})
        values=[str(contract.get("params","—")),str(cfg.get("n_latents",contract.get("n_latent","—"))),
                str(cfg.get("width",64 if row["family"]=="previous_token64" else "—")),
                str(cfg.get("n_blocks",2 if row["family"]=="previous_token64" else "—")),
                str(train.get("lr",contract.get("lr","—")))]
        lines.append("| "+" | ".join([f"{row['family']} / {row['mode']}",*values])+" |")
    lines.append("")
    for mode,title in (("argo","Argo 输入"),("surface","多模态输入")):
        if not any(row["mode"]==mode and row["kind"]!="classical" for row in result["rows"]):
            continue
        for ensembles,label in ((False,"三种子训练结果"),(True,"三种子集成结果")):
            lines += [f"## {title}：{label}\n",
                "| 方法 | 统计口径 | 温度 RMSE °C | 温度 MAE °C | 温度偏差 °C | 盐度 RMSE PSU | 盐度 MAE PSU | 盐度偏差 PSU |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
            lines += metric_rows(result,mode,ensembles)
            lines.append("")
    lines += ["## 不确定性与相关性\n",
        "44 OI 和此前的 learned OI 均仅用 2004 各深度残差 RMS 拟合区间尺度。它是验证集校准的误差模型，不是精确 OI 后验。其他确定性方法未提供预测不确定性，相关列留空。学习模型集成标准差按总方差公式合并，并以高斯矩匹配计算 NLL 与 CRPS。\n",
        "| 方法 / 输入 | 温度 R²（异常） | 温度 r（异常） | 盐度 R²（异常） | 盐度 r（异常） | 温度 NLL | 盐度 NLL | 温度 CRPS °C | 盐度 CRPS PSU | 温度 95% 覆盖 | 盐度 95% 覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        m=row.get("physical_metrics",row.get("ensemble",{}).get("physical_metrics"))
        values=[fmt(m[ch].get(k)) for k in ("r2","pearson_r","nll","crps","coverage_95") for ch in CHANNELS]
        # The table groups correlation columns by variable, then paired uncertainty.
        ordered=[fmt(m[ch].get(k)) for ch in CHANNELS for k in ("r2","pearson_r")]+values[4:]
        lines.append("| "+" | ".join([f"{row['family']} / {row['mode']}",*ordered])+" |")
    lines += ["\n完整 JSON 另含 68% 覆盖率、绝对温盐状态的 R²/相关系数、20 个深度指标以及逐种子结果。RMSE、MAE、偏差在恢复气候态前后相同；R² 和相关性必须注明计算对象。\n",
        "## 与旧实现相比改变了什么\n",
        "| 部分 | 旧 64-slot 模型 | 本轮新系统 |",
        "|---|---|---|",
        "| Argo 表示 | 深度带聚合 token | 保留每个剖面的 20 个深度温盐数值与有效标记 |",
        "| 初始预测 | 潜在网络预测，接局部 refiner | 验证期调优、冻结的球面 OI；新增网络初始输出严格等于 OI |",
        "| 共享状态 | Perceiver 风格 latent | 地理/深度锚定 latent Transformer，可替换 Soft MoE |",
        "| 查询 | 独立位置/深度查询 | 独立查询解码，加逐层数值新息与独立温盐输出头 |",
        "| 不确定性 | 未提供 | 有界高斯标准差头与校准评估 |",
        "| 输入池 | 每月完整 6080 个剖面 | 同一完整来源池，OI、latent 与局部通路全部保留 |",
        "| 优化设置 | 旧模型注册 AdamW LR=0.001，20% 步骤随机隐藏一个变量的目标 | 新模型 LR=0.0003、双变量监督并加入高斯 NLL；容量见 JSON |",
        "\n因此旧→新差值是完整系统的改进。冻结 OI 后关闭 latent 或新增局部通路的实验，才用于区分两个新组件的贡献。大模型也改变参数容量，不能把其差值全部归因于某一种 attention。\n",
        "此前 `67_anchored_fusion.py` 的 learned OI 使用同一位置异常目标、归一化、6080 个输入剖面和评分点，120 个参数、1500 步、固定种子 1234。本轮重放其既有检查点并核验历史误差，作为更强的既有对照，同时纳入仅验证集选型。它的训练预算与新三种子 15000 步系统不同，不能隐藏这一差别。只超过 44 的 OI 不等于超过此前最好方案。\n",
        "## 配对误差差异\n",
        "下表用相同评分点上的集成均值或固定方法计算，按完整月份配对 bootstrap。负差值表示本行模型 RMSE 更低；区间覆盖 12 个评分月份的变化，不包含训练种子不确定性。\n",
        "| 方法 / 输入 | 参照 | 温度 ΔRMSE °C [95% CI] | 盐度 ΔRMSE PSU [95% CI] |",
        "|---|---|---:|---:|"]
    for label,contrasts in result["contrasts"].items():
        for against,values in contrasts.items():
            if against.startswith("paired_seeds"):
                continue
            cells=[]
            for ch in CHANNELS:
                v=values[ch];lo,hi=v["ci95_delta_rmse"]
                cells.append(f"{v['delta_rmse']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            lines.append("| "+" | ".join([label,against,*cells])+" |")
    lines += ["\n## 组件贡献\n"]
    if not result["component_contrasts"]:
        lines.append("本批没有完整的对应组件对照。")
    else:
        for label,values in result["component_contrasts"].items():
            cells=[]
            for ch in CHANNELS:
                v=values[ch];lo,hi=v["ci95_delta_rmse"]
                cells.append(f"{ch}: {v['delta_rmse']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            lines.append(f"- **{label}**："+"；".join(cells)+"。负值才支持完整架构精度更高。")
    lines += ["\n## 适用范围\n",
        "当前是 20 个固定深度的同月回顾性重建。尚未证明任意深度连续推理、严格因果预报或独立测试上的 SOTA。合成 SST/SSS 是无噪声 5 m 真值，海平面为同一温盐真值导出的 steric 分量；它们比真实独立卫星观测更有利。",
        "用户粘贴的历史 4DVarNet/多模态数字缺少可重放产物，作为历史引用保留；本表只包含本轮保存预测并核验身份的实测行。作者代码的本任务适配也不等于完整论文原始协议。\n",
        f"冻结方案时间：`{result['selection']['created_at_utc']}`。原始 cohort SHA256：`{result['selection']['raw_cohort_sha256']}`。预测、检查点、代码与规则指纹见配套 JSON。"]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,default=ROOT/"outputs/synthetic_matched_20261007/campaign.json")
    p.add_argument("--baseline-dir",type=Path)
    p.add_argument("--freeze-selection",type=Path,help="write immutable validation-only recommendation, then exit")
    p.add_argument("--selection",type=Path)
    p.add_argument("--report",type=Path,default=ROOT/"reports/synthetic/matched_reconstruction_20261007.md")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/synthetic_matched_20261007/comparison.json")
    p.add_argument("--bootstrap-draws",type=int,default=4000)
    args=p.parse_args()
    baseline_dir=args.baseline_dir or args.campaign.parent/"previous_baselines"
    try:
        if args.freeze_selection:
            result=freeze_selection(args.campaign,baseline_dir,args.freeze_selection)
            print(json.dumps({"status":"frozen","selection":str(args.freeze_selection),"selected":result["selected"]},ensure_ascii=False))
            return
        selection=args.selection or args.campaign.parent/"final_selection.json"
        result=build_report(args.campaign,baseline_dir,selection,bootstrap_draws=args.bootstrap_draws)
        atomic_json(args.output,result)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(chinese_report(result))
        print(json.dumps({"status":"complete","report":str(args.report),"output":str(args.output)},ensure_ascii=False))
    except PendingReport as exc:
        print(json.dumps({"status":"pending","reason":str(exc)},ensure_ascii=False))


if __name__=="__main__":
    main()
