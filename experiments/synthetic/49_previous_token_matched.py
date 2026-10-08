"""Recover the original 64-slot, 15k-step CESM2 token baseline with full state.

The model, optimizer, schedule, sampling and loss execute verbatim AST nodes
from 62_sanity_train.py.  Added code only saves/restores complete training state
and exports predictions.  The original runner remains unmodified.  A recorded
source hash makes this dependence explicit and refuses incompatible resumes.
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
RUNNER = Path(__file__).resolve()
LEGACY = ROOT / "experiments/real_data/62_sanity_train.py"
sys.path.insert(0, str(ROOT / "src"))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(torch, payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def to_cpu(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(torch, v) for v in value)
    return copy.deepcopy(value)


def execute_nodes(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(LEGACY), "exec"), namespace)


def source_contract():
    paths = [LEGACY, RUNNER]
    paths += sorted({Path(module.__file__).resolve()
                     for name, module in sys.modules.items()
                     if name.startswith("ocean_tokenizer")
                     and getattr(module, "__file__", None)
                     and Path(module.__file__).suffix == ".py"})
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def export_predictions(ns, baseline_dir, output):
    import numpy as np
    import torch
    import xarray as xr
    from ocean_tokenizer import synth_argo_eval as E
    from ocean_tokenizer.audit_tools import cohort_path
    from ocean_tokenizer.point_baselines import CH

    c, norm, model = ns["c"], ns["norm"], ns["model"]
    obs = {ch: norm.z(ch, getattr(c, ch)) for ch in CH}
    norm_mean = np.stack([norm.mean[ch] for ch in CH], axis=-1)
    norm_std = np.stack([norm.std[ch] for ch in CH], axis=-1)
    cpath = Path(cohort_path(str(ROOT), "synthetic"))
    with xr.open_dataset(cpath) as ds:
        order = np.argsort(np.asarray(ds.month_index.values, int), kind="stable")
        climatology = np.stack([np.asarray(ds[f"CLIM_POS_{ch}"].values, float)[order]
                                for ch in CH], axis=-1)
    model.eval()
    for split in ("validation", "development"):
        if split not in ns["eval_sets"]:
            continue
        arrays = {name: [] for name in ("month", "profile", "level", "target", "mean", "climatology_physical")}
        evs = E.eval_set(c, obs, split, ns["args"].eval_cells if split == "validation" else 0)
        samples = ns["eval_sets"][split]
        if len(evs) != len(samples):
            raise ValueError("Legacy evaluation month count differs from common protocol")
        for ev, sample in zip(evs, samples):
            with torch.no_grad():
                mean = model(sample).cpu().numpy().astype(np.float64)
            target = np.stack([ev["target"][ch] for ch in CH], axis=-1)
            sample_mask = sample["target_mask"].cpu().numpy()
            if not np.array_equal(sample["level_index"].cpu().numpy(), ev["lev"]):
                raise ValueError("Legacy query depth ordering differs")
            if not np.array_equal(sample_mask, np.isfinite(target)):
                raise ValueError("Legacy scored-cell mask differs")
            if not np.allclose(sample["target"].cpu().numpy()[sample_mask], target[sample_mask], rtol=1e-6, atol=1e-6):
                raise ValueError("Legacy target values differ from common anomaly target")
            rows = ev["tgt"][ev["prof"]]
            arrays["month"].append(np.full(ev["lev"].size, ev["month"], dtype=np.int64))
            arrays["profile"].append(rows.astype(np.int64))
            arrays["level"].append(ev["lev"].astype(np.int64))
            arrays["target"].append(target)
            arrays["mean"].append(mean)
            arrays["climatology_physical"].append(climatology[rows, ev["lev"]])
        arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
        baseline_path = baseline_dir / f"oi_{split}.npz"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Replay fixed baselines with 47 before exporting: {baseline_path}")
        with np.load(baseline_path, allow_pickle=False) as reference:
            for name in ("month", "profile", "level", "target"):
                if not np.array_equal(arrays[name], reference[name], equal_nan=True):
                    raise ValueError(f"Legacy/Common OI {name} identity differs")
            arrays["baseline"] = reference["mean"].copy()
        sd, mu = norm_std[arrays["level"]], norm_mean[arrays["level"]]
        arrays.update(levels=c.levels, normalization_mean=norm_mean, normalization_std=norm_std,
                      mean_physical=arrays["mean"] * sd + mu,
                      target_physical=arrays["target"] * sd + mu)
        for kind in ("mean", "target"):
            arrays[f"{kind}_absolute_physical"] = arrays[f"{kind}_physical"] + arrays["climatology_physical"]
        identity = hashlib.sha256()
        for name in ("month", "profile", "level", "target"):
            value = np.ascontiguousarray(arrays[name])
            identity.update(name.encode()); identity.update(str(value.dtype).encode())
            identity.update(str(value.shape).encode()); identity.update(value.tobytes())
        metadata = {"method": "previous_64_slot_token", "seed": ns["args"].seed,
                    "split": split, "steps": ns["args"].steps,
                    "best_step": ns["best"]["step"], "parameters": ns["n_par"],
                    "scoring_fingerprint": identity.hexdigest(), "physical_arrays": "anomalies",
                    "uncertainty_available": False, "cohort_sha256": file_sha256(cpath),
                    "legacy_source_sha256": file_sha256(LEGACY)}
        artifact = output / f"{split}_predictions_seed{ns['args'].seed}.npz"
        temporary = Path(str(artifact) + ".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        temporary.replace(artifact)
        print(f"exported {artifact} ({metadata['scoring_fingerprint']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--no-final-development", action="store_true")
    parser.add_argument("--baseline-dir", type=Path,
                        default=ROOT / "outputs/synthetic_matched_20261007/previous_baselines")
    extra, legacy_args = parser.parse_known_args()
    if extra.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be positive")
    # Only wrapper switches are consumed. Every legacy model/training switch is
    # parsed by the original parser, retaining its exact semantics and defaults.
    tree = ast.parse(LEGACY.read_text(), filename=str(LEGACY))
    run_index = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "best"
                                                            for t in node.targets))
    loop_index = next(i for i in range(run_index, len(tree.body))
                      if isinstance(tree.body[i], ast.For) and isinstance(tree.body[i].target, ast.Name)
                      and tree.body[i].target.id == "step")
    prefix = tree.body[:run_index]
    # Legacy --eval-only exits before export and has a narrower purpose. The
    # wrapper offers --evaluation-only while always executing setup first.
    if "--eval-only" in legacy_args:
        raise ValueError("Use --evaluation-only for prediction export")
    ns = {"__file__": str(LEGACY), "__name__": "matched_previous_token_setup"}
    argv = sys.argv
    sys.argv = [str(LEGACY), *legacy_args]
    try:
        execute_nodes(prefix, ns)
    finally:
        sys.argv = argv
    import numpy as np
    import torch
    args, model, opt, sched, rng = [ns[k] for k in ("args", "model", "opt", "sched", "rng")]
    output = Path(ns["OUT"])
    if not extra.diagnostic:
        required = {
            "region": "synthetic", "mode": "train", "backbone": "d4rt", "mass_mode": "dfs",
            "steps": 15000, "n_latent": 64, "d_model": 64, "n_self_blocks": 2,
            "n_dec_blocks": 2, "n_heads": 4, "target_dropout": 0.2, "queries": 1024,
            "batch": 1, "n_profiles": 0, "eval_cells": 8000, "lr": 0.001,
            "warmup": 300, "weight_decay": 0.01, "refiner_km": 500.0,
            "refiner_gate": 1.0, "refiner_dz": 100.0,
        }
        for name, value in required.items():
            if getattr(args, name) != value:
                raise ValueError(f"Registered historical setting requires {name}={value}; got {getattr(args,name)}")
        if ns["ABL"] != {"anomaly_exact"} or args.surface or ns["LEADS"] != [0]:
            raise ValueError("Historical setting requires anomaly_exact, profiles only, reconstruction")
        if ns["n_par"] != 407111:
            raise ValueError("Historical model parameter count changed")
    contract = source_contract()
    last_path = output / f"last_training_seed{args.seed}.pt"
    execute_nodes(tree.body[run_index:loop_index], ns)
    start_step = 0
    if extra.resume_training:
        state = torch.load(last_path, map_location=ns["dev"], weights_only=False)
        if state["source_hashes"] != contract or state["legacy_arguments"] != vars(args):
            raise ValueError("Training source or arguments changed; cannot replay exact continuation")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        ns["best"] = state["best"]; ns["loss_run"] = state["loss_run"]
        rng.bit_generator.state = state["numpy_generator"]
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if ns["dev"] == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        start_step = state["step"]
        print(f"resumed complete optimizer/RNG state at step {start_step}", flush=True)
    elif last_path.exists() and not extra.evaluation_only:
        raise FileExistsError(f"Existing training state requires --resume-training: {last_path}")
    if extra.evaluation_only:
        previous = json.loads((output / f"summary_seed{args.seed}.json").read_text())
        if previous.get("source_hashes") != contract:
            raise ValueError("Saved best model source contract differs from this evaluation runner")
        for name in ("seed", "steps", "n_latent", "refiner_km", "refiner_gate", "params"):
            current = ns["n_par"] if name == "params" else getattr(args, name)
            if previous.get(name) != current:
                raise ValueError(f"Saved best model {name} differs from this evaluation setting")
        model.load_state_dict(torch.load(output / f"model_seed{args.seed}.pt", map_location=ns["dev"], weights_only=True))
        ns["best"]["step"] = previous["best_step"]
    else:
        end_step = min(args.steps, extra.stop_after if extra.stop_after is not None else args.steps)
        if end_step < start_step:
            raise ValueError("stop-after precedes the loaded continuation step")

        def save_state(step):
            state = {
                "format_version": 1, "step": step, "source_hashes": contract,
                "legacy_arguments": vars(args), "model": to_cpu(torch, model.state_dict()),
                "optimizer": to_cpu(torch, opt.state_dict()), "scheduler": sched.state_dict(),
                "numpy_generator": copy.deepcopy(rng.bit_generator.state),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if ns["dev"] == "cuda" else [],
                "best": to_cpu(torch, ns["best"]), "loss_run": list(ns["loss_run"]),
            }
            atomic_torch_save(torch, state, last_path)

        ns.update(_matched_start=start_step, _matched_end=end_step,
                  _matched_every=extra.checkpoint_every, _matched_save=save_state)
        loop = copy.deepcopy(tree.body[loop_index])
        loop.iter = ast.parse("range(_matched_start, _matched_end)", mode="eval").body
        loop.body += ast.parse(
            "if (step + 1) % _matched_every == 0 or step + 1 == _matched_end:\n"
            "    _matched_save(step + 1)\n").body
        execute_nodes([loop], ns)
        if end_step < args.steps:
            ns["hist"].close()
            print(f"continuation checkpoint saved at {end_step}/{args.steps}; no final claims exported", flush=True)
            return
        if extra.no_final_development:
            ns["eval_sets"] = {"validation": ns["eval_sets"]["validation"]}
        execute_nodes(tree.body[loop_index + 1:], ns)
        summary_path = output / f"summary_seed{args.seed}.json"
        summary = json.loads(summary_path.read_text())
        summary.update(source_hashes=contract, full_state_checkpoint=str(last_path),
                       rerun_of="k15_l64_syn_r500_g1", precision="float32",
                       completed_steps=args.steps, historical_setting=not extra.diagnostic)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    export_predictions(ns, extra.baseline_dir, output)


if __name__ == "__main__":
    main()
