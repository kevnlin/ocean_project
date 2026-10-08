"""Recover the fixed seed-1234 pointwise MLP with the original 45 recipe.

Original feature construction, training-set draws and optimizer/epoch loop run
from unchanged AST nodes. Added code defers development, saves complete epoch
state and exports the common reconstruction arrays. Last-epoch weights remain
the fixed baseline; there is no architecture/checkpoint selection.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "experiments/synthetic/45_synth_argo_mlp.py"
RUNNER = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from ocean_tokenizer.reconstruction_metrics import evaluate_reconstruction


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def atomic_save(state, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def atomic_json(value, path):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def random_state(rng):
    legacy = np.random.get_state()
    return {"numpy_generator": copy.deepcopy(rng.bit_generator.state),
            "numpy_global": {"kind": legacy[0], "state": torch.as_tensor(legacy[1].astype(np.int64)),
                             "position": legacy[2], "has_gauss": legacy[3], "cached_gauss": legacy[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(rng, state):
    rng.bit_generator.state = state["numpy_generator"]
    legacy = state["numpy_global"]
    np.random.set_state((legacy["kind"], legacy["state"].numpy().astype(np.uint32), legacy["position"],
                         legacy["has_gauss"], legacy["cached_gauss"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(LEGACY), "exec"), namespace)


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def original_parts(source):
    """Return original setup/draw/loop nodes with only evaluation setup deferred."""
    tree = ast.parse(source, filename=str(LEGACY))
    loop_index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For)
                      and isinstance(node.target, ast.Name) and node.target.id == "ep")
    draw_index = next(i for i, node in enumerate(tree.body) if assigned(node, "rng"))
    setup, draw = [], tree.body[draw_index:loop_index]
    for node in tree.body[:draw_index]:
        if assigned(node, "test"):
            deferred = copy.deepcopy(node)
            deferred.value = ast.List(elts=[], ctx=ast.Load())
            setup.append(deferred)
        elif isinstance(node, ast.If) and any(assigned(statement, "ident") for statement in node.body):
            setup += ast.parse('ident = {"validation": identity_check(ROOT, "validation", score(val))}\n').body
        else:
            setup.append(node)
    return setup, draw, tree.body[loop_index]


def source_contract():
    names = ("baselines.py", "point_baselines.py", "argo_obs.py", "synth_argo_eval.py",
             "audit_tools.py", "config.py", "reconstruction_metrics.py", "data.py", "argo.py", "oi.py")
    paths = [LEGACY, RUNNER] + [ROOT / "src/ocean_tokenizer" / name for name in names]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def export_predictions(ns, split, baseline_dir, output):
    c, norm, model, features = (ns[key] for key in ("c", "norm", "model", "features"))
    channels = tuple(ns["CH"])
    evaluations = ns["synth_argo_eval"].eval_set(c, ns["OBS"], split, ns["EVAL_CELLS"] if split == "validation" else 0)
    identity = ns["identity_check"](str(ROOT), split, ns["score"](evaluations))
    model.eval()
    collected = {name: [] for name in ("month", "profile", "level", "target", "mean")}
    for ev in evaluations:
        inputs = torch.from_numpy(features(ev["src"], ev["tgt"], ev["month"])).to(ns["dev"])
        with torch.no_grad():
            full = model(inputs).cpu().numpy().astype(np.float64)
        selected = full[ev["prof"] * len(c.levels) + ev["lev"]]
        collected["mean"].append(selected)
        collected["target"].append(np.stack([ev["target"][ch] for ch in channels], axis=-1))
        collected["month"].append(np.full(len(ev["lev"]), ev["month"], dtype=np.int64))
        collected["profile"].append(ev["tgt"][ev["prof"]].astype(np.int64))
        collected["level"].append(ev["lev"].astype(np.int64))
    arrays = {key: np.concatenate(parts) for key, parts in collected.items()}
    reference_path = baseline_dir / f"oi_{split}.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        for key in ("month", "profile", "level", "target"):
            if not np.array_equal(arrays[key], reference[key], equal_nan=True):
                raise ValueError(f"fixed MLP scoring identity differs from canonical OI in {key}")
        arrays["baseline"] = reference["mean"].copy()
        arrays["climatology_physical"] = reference["climatology_physical"].copy()
        arrays["normalization_mean"] = reference["normalization_mean"].copy()
        arrays["normalization_std"] = reference["normalization_std"].copy()
    mean = np.column_stack([norm.mean[ch] for ch in channels])
    std = np.column_stack([norm.std[ch] for ch in channels])
    if not np.array_equal(mean, arrays["normalization_mean"]) or not np.array_equal(std, arrays["normalization_std"]):
        raise ValueError("fixed MLP and canonical OI training normalization differ")
    arrays["levels"] = c.levels
    sd, mu = std[arrays["level"]], mean[arrays["level"]]
    for key in ("mean", "target"):
        arrays[f"{key}_physical"] = arrays[key] * sd + mu
        arrays[f"{key}_absolute_physical"] = arrays[f"{key}_physical"] + arrays["climatology_physical"]
    fingerprint = hashlib.sha256()
    for key in ("month", "profile", "level", "target"):
        a = np.ascontiguousarray(arrays[key])
        fingerprint.update(key.encode()); fingerprint.update(str(a.dtype).encode())
        fingerprint.update(str(a.shape).encode()); fingerprint.update(a.tobytes())
    metadata = {"method": "fixed_pointwise_mlp", "seed": int(ns["args"].seed), "split": split,
                "training": "original 45 recipe; retrained because original weight file was absent",
                "epochs_completed": int(ns["C"].MLP_EPOCHS), "model_selection": "last epoch; no checkpoint tuning",
                "scoring_fingerprint": fingerprint.hexdigest(), "physical_arrays": "anomalies",
                "uncertainty_available": False, "legacy_source_sha256": sha256(LEGACY),
                "canonical_oi_sha256": sha256(reference_path)}
    artifact = output / f"{split}_seed{ns['args'].seed}.npz"
    temporary = artifact.with_name(artifact.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, artifact)
    metrics = evaluate_reconstruction(arrays["mean_physical"], arrays["target_physical"])
    baseline = evaluate_reconstruction(arrays["baseline"] * sd + mu, arrays["target_physical"])
    standardized = evaluate_reconstruction(arrays["mean"], arrays["target"])
    return {"prediction_path": str(artifact), "prediction_sha256": sha256(artifact),
            "scoring_fingerprint": fingerprint.hexdigest(), "identity_check": identity,
            "standardized_rmse": {ch: standardized[ch]["rmse"] for ch in channels},
            "mean_standardized_rmse": float(np.mean([standardized[ch]["rmse"] for ch in channels])),
            "physical_metrics": metrics, "baseline_physical_metrics": baseline,
            "absolute_physical_metrics": evaluate_reconstruction(arrays["mean_absolute_physical"], arrays["target_absolute_physical"]),
            "by_depth_physical_metrics": {str(float(depth)): evaluate_reconstruction(
                arrays["mean_physical"][arrays["level"] == level], arrays["target_physical"][arrays["level"] == level])
                for level, depth in enumerate(c.levels)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--no-final-development", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/previous_baselines")
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/synthetic_matched_20261007/selection.json")
    extra, legacy_args = parser.parse_known_args()
    if extra.evaluation_only and extra.resume_training:
        raise ValueError("evaluation-only and continuation are separate operations")
    if extra.development and (not extra.evaluation_only or extra.no_final_development or not extra.selection.exists()):
        raise ValueError("development export requires --evaluation-only and frozen selection.json")
    if "--smoke" in legacy_args:
        raise ValueError("fixed MLP baseline requires the full original setting; stop-after-epoch saves resumable work without results")
    if not any(arg == "--out" or arg.startswith("--out=") for arg in legacy_args):
        legacy_args += ["--out", str(ROOT / "outputs/synthetic_matched_20261007/fixed_pointwise_mlp")]
    torch.set_num_threads(extra.threads)
    source = LEGACY.read_text()
    setup, draw, original_loop = original_parts(source)
    ns = {"__file__": str(LEGACY), "__name__": "matched_fixed_pointwise_setup"}
    argv = sys.argv
    sys.argv = [str(LEGACY), *legacy_args]
    try:
        execute(setup, ns)
    finally:
        sys.argv = argv
    args, config = ns["args"], ns["C"]
    if args.seed != 1234 or config.MLP_HIDDEN != [256, 256, 256] or config.MLP_EPOCHS != 30 or config.MLP_BATCH != 65536 or config.MLP_LR != .001 or config.MLP_POINTS_PER_MONTH != 120000:
        raise ValueError("fixed comparison requires original seed1234/30epochs/256³/65536batch/1e-3/120000points-per-month recipe")
    if len(ns["MLP_FEATURES"]) != 9 or ns["TARGET_FRAC"] != .3 or ns["DRAWS"] != 4:
        raise ValueError("original nine-input feature and four training-target draw recipe changed")
    output = Path(ns["OUT"])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    contract = source_contract()
    last_path = output / f"last_training_seed{args.seed}.pt"
    summary_path = output / f"summary_seed{args.seed}.json"
    training_recipe = {"seed": args.seed, "hidden": config.MLP_HIDDEN, "epochs": config.MLP_EPOCHS,
        "batch": config.MLP_BATCH, "lr": config.MLP_LR, "points_per_month": config.MLP_POINTS_PER_MONTH,
        "target_fraction": ns["TARGET_FRAC"], "target_draws": ns["DRAWS"], "features": list(ns["MLP_FEATURES"]),
        "model_selection": "fixed final epoch; no validation selection"}
    if extra.evaluation_only:
        previous = json.loads(summary_path.read_text())
        if previous["source_hashes"] != contract or previous["epochs_completed"] != 30 or previous["training_recipe"] != training_recipe:
            raise ValueError("fixed completed model source/recipe differs")
        ns["model"] = ns["MLP"](9, config.MLP_HIDDEN).to(ns["dev"])
        ns["model"].load_state_dict(torch.load(output / f"model_seed{args.seed}.pt", map_location=ns["dev"], weights_only=True))
        splits = ("validation", "development") if extra.development else ("validation",)
        for split in splits:
            previous[split] = export_predictions(ns, split, extra.baseline_dir, output)
        previous["evaluation"] = {"runtime_s": time.time() - ns["t0"],
                                  "selection_path": str(extra.selection) if extra.development else None,
                                  "selection_sha256": sha256(extra.selection) if extra.development else None}
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        temporary.write_text(json.dumps(previous, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, summary_path)
        print(json.dumps({"summary": str(summary_path), "exports": splits}), flush=True)
        return
    if summary_path.exists():
        raise FileExistsError(f"Completed fixed baseline already exists: {summary_path}")
    atomic_json({"phase": "preparing_original_training_set", "planned_epochs": 30, "seed": args.seed}, status_path)
    execute(draw, ns)
    ns["model"].train()
    data_contract = {"X_shape": list(ns["Xtr"].shape), "Y_shape": list(ns["Ytr"].shape),
                     "X_sha256": array_sha256(ns["Xtr"]), "Y_sha256": array_sha256(ns["Ytr"])}
    start_epoch, history, prior_runtime = 0, [], 0.
    if extra.resume_training:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["source_hashes"] != contract or saved["training_recipe"] != training_recipe or saved["training_data"] != data_contract:
            raise ValueError("original source/recipe/training draw differs from saved continuation")
        ns["model"].load_state_dict(saved["model"])
        ns["opt"].load_state_dict(saved["optimizer"])
        restore_random_state(ns["rng"], saved["random"])
        start_epoch, history, prior_runtime = saved["epochs_completed"], saved["history"], saved["runtime_s"]
        print(f"resumed exact full training state at epoch {start_epoch}", flush=True)
    elif last_path.exists():
        raise FileExistsError(f"Existing epoch state requires --resume-training: {last_path}")
    end_epoch = min(config.MLP_EPOCHS, extra.stop_after_epoch if extra.stop_after_epoch is not None else config.MLP_EPOCHS)
    if end_epoch < start_epoch or end_epoch < 1:
        raise ValueError("stop-after-epoch must be positive and at least the saved completed epoch")

    def save_state(completed, train_mse):
        history.append({"epoch": completed, "train_mse": float(train_mse),
                        "elapsed_s": prior_runtime + time.time() - ns["t0"]})
        state = {"format_version": 1, "epochs_completed": completed,
                 "model": cpu_copy(ns["model"].state_dict()), "optimizer": cpu_copy(ns["opt"].state_dict()),
                 "random": random_state(ns["rng"]), "training_recipe": training_recipe,
                 "training_data": data_contract, "source_hashes": contract,
                 "history": history, "runtime_s": prior_runtime + time.time() - ns["t0"]}
        atomic_save(state, last_path)
        atomic_json({"phase": "training", "epochs_completed": completed, "planned_epochs": 30,
                     "train_mse": float(train_mse), "runtime_s": state["runtime_s"],
                     "last_training_checkpoint": str(last_path)}, status_path)

    loop = copy.deepcopy(original_loop)
    loop.iter = ast.parse("range(_start_epoch, _end_epoch)", mode="eval").body
    loop.body += ast.parse("_save_epoch(ep + 1, train_mse)\n").body
    ns.update(_start_epoch=start_epoch, _end_epoch=end_epoch, _save_epoch=save_state)
    execute([loop], ns)
    if end_epoch < config.MLP_EPOCHS:
        print(f"saved complete epoch {end_epoch}/{config.MLP_EPOCHS}; no comparison results exported", flush=True)
        return
    atomic_save(cpu_copy(ns["model"].state_dict()), output / f"model_seed{args.seed}.pt")
    validation = export_predictions(ns, "validation", extra.baseline_dir, output)
    summary = {"tag": "fixed_pointwise_mlp", "seed": args.seed, "architecture": "pointwise MLP",
               "training_recipe": training_recipe, "epochs_completed": end_epoch, "completed_epochs": end_epoch,
               "params": ns["n_par"], "history": history, "final_train_mse": history[-1]["train_mse"],
               "training_data": data_contract, "source_hashes": contract, "source_sha256": contract,
               "legacy_training_loop_ast_sha256": hashlib.sha256(ast.dump(original_loop, include_attributes=False).encode()).hexdigest(),
               "legacy_recipe_source": str(LEGACY), "legacy_recipe_sha256": sha256(LEGACY),
               "status": "exact recipe retraining; original checkpoint missing; fixed single-seed baseline",
               "validation": validation, "full_state_checkpoint": str(last_path),
               "model_path": str(output / f"model_seed{args.seed}.pt"),
               "data": {"cohort_sha256": sha256(ns["cohort_path"](str(ROOT), "synthetic")),
                        "splits": {key: list(value) for key, value in ns["SPLITS"].items()},
                        "levels_m": ns["LEV"].tolist(), "anomaly_target": "at Argo profile position",
                        "normalization_scope": "training years only; inherited exact synthetic protocol",
                        "input_profiles_per_evaluation_month": ns["N_INPUT"],
                        "query_profiles_per_evaluation_month": ns["N_QUERY"]},
               "runtime_s": prior_runtime + time.time() - ns["t0"],
               "development": None}
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, summary_path)
    atomic_json({"phase": "training_complete_validation_exported", "epochs_completed": end_epoch,
                 "development": "deferred until frozen selection", "summary": str(summary_path),
                 "validation_rmse": {ch: validation["physical_metrics"][ch]["rmse"] for ch in ns["CH"]}}, status_path)
    print("finished " + json.dumps({"summary": str(summary_path), "epochs_completed": end_epoch,
          "validation_rmse": {ch: validation["physical_metrics"][ch]["rmse"] for ch in ns["CH"]},
          "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
