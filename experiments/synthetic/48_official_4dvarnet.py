"""Train the pinned official 4DVarNet solver on the matched CESM2 profile task.

The official three solver classes are unchanged. Depth/variable channels,
continuous scatter observations and heldout-query supervision are task adapters,
so these results are an official-code adaptation, not the paper's original task.
Default training opens only train/validation. Frozen 2005 evaluation is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer.fourdvar_synthetic import (
    FourDVarConfig, OfficialFourDVarNet, heldout_query_loss, make_batch,
)
from ocean_tokenizer.point_baselines import CH, Scores
from ocean_tokenizer.reconstruction_metrics import evaluate_reconstruction, inverse_standardize
from ocean_tokenizer.synthetic_latent_experiment import SyntheticOceanScenes


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--official-repository", type=Path, default=ROOT / "external/4dvarnet-starter")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--solver-lr", type=float, default=1000.)
    parser.add_argument("--prior-hidden", type=int, default=32)
    parser.add_argument("--gradient-hidden", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-cells", type=int, default=8000)
    parser.add_argument("--eval-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--satellite-cache", type=Path)
    parser.add_argument("--oi-summary", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--evaluate-only", "--evaluation-only", dest="evaluate_only", action="store_true")
    parser.add_argument("--development", action="store_true", help="frozen previously used 2005 test; evaluation-only required")
    parser.add_argument("--no-final-development", action="store_true", help="explicit train/validation-only spelling of the default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def atomic_save(state, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def random_state(rng):
    legacy = np.random.get_state()
    return {"numpy_generator": rng.bit_generator.state,
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


def load_surface_cache(path, data, config):
    """Read allowed synthetic satellite fields, without using dense target T/S."""
    if not config.surface:
        return None, {}
    if path is None:
        raise ValueError("surface adaptation requires --satellite-cache")
    with np.load(path, allow_pickle=False) as stored:
        months = stored["grid_month"].copy()
        fields = stored["surface_z"].copy()
        latitude, longitude = stored["grid_lat"].copy(), stored["grid_lon"].copy()
        metadata = json.loads(str(stored["metadata"].item()))
    if fields.shape != (len(months), 3, config.height, config.width) or len(set(months.tolist())) != len(months):
        raise ValueError("dense surface cache dimensions/month identities differ")
    if not np.allclose(latitude, -90 + (np.arange(config.height) + .5) * 180 / config.height, atol=1e-9) or not np.allclose(
            longitude, (np.arange(config.width) + .5) * 360 / config.width, atol=1e-9):
        raise ValueError("dense surface grid centres differ from observation operator")
    if metadata.get("cohort_fingerprint") != data.metadata["cohort_fingerprint"] or metadata.get("normalization_scope") != "training years only":
        raise ValueError("surface cache target/split normalization differs")
    if not set(data.months_available).issubset(set(months.tolist())):
        raise ValueError("surface cache lacks a cohort month")
    tensor = torch.as_tensor(fields, dtype=torch.float32, device=data.device)
    return tensor, {int(month): i for i, month in enumerate(months)}


def formal_scores(score):
    """Expose conventional names; retain no legacy ratio-name fields."""
    return {ch: {"n": score[ch]["n"], "rmse": score[ch]["rmse_physical"],
                 "unit": score[ch]["unit"], "standardized_rmse": score[ch]["rmse_z"],
                 "climatology_standardized_rmse": score[ch]["climatology_z"],
                 "rmse_by_depth_band": score[ch]["by_band_physical"]} for ch in CH}


def main():
    args = arguments()
    if args.steps < 1 or args.queries < 1 or args.val_every < 1 or args.checkpoint_every < 1 or args.eval_chunk < 1:
        raise ValueError("training/evaluation counts must be positive")
    if args.val_cells != 8000:
        raise ValueError("matched synthetic protocol requires the old 8000-cell validation cap")
    if args.development and (not args.evaluate_only or args.no_final_development):
        raise ValueError("2005 evaluation requires a frozen checkpoint and --evaluate-only --development")
    if (args.resume_training or args.evaluate_only) and args.load_checkpoint is None:
        raise ValueError("resume/evaluate-only require --load-checkpoint")
    if args.resume_training and args.evaluate_only:
        raise ValueError("resume and evaluation-only are separate operations")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the authorized native GPU runtime or explicitly choose --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    started = time.time()
    checkpoint = None
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        if args.evaluate_only and "continuation" in checkpoint:
            checkpoint = checkpoint["continuation"]["best"]["state"]
        args.seed = int(checkpoint["seed"])
        cfg = FourDVarConfig(**checkpoint["config"])
        args.surface = cfg.surface
        if cfg.surface and args.satellite_cache is None:
            args.satellite_cache = Path(checkpoint["training"]["satellite_cache"])
    else:
        cfg = FourDVarConfig(surface=args.surface, n_step=args.solver_steps, lr_grad=args.solver_lr,
                            prior_hidden=args.prior_hidden, gradient_hidden=args.gradient_hidden)
    output = args.output or ROOT / "outputs/synthetic_matched_20261007/official_4dvarnet" / args.tag
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"summary_seed{args.seed}.json"
    previous = json.loads(summary_path.read_text()) if args.evaluate_only and summary_path.exists() else None
    if args.development and previous is None:
        raise ValueError("development evaluation requires a completed training summary at the matched output path")
    if summary_path.exists() and not args.evaluate_only:
        raise SystemExit(f"refusing to overwrite completed run: {summary_path}")
    data = SyntheticOceanScenes(ROOT, args.device, surface=args.surface,
                                satellite_cache=args.satellite_cache, oi_summary=args.oi_summary)
    if data.n_levels != cfg.levels:
        raise ValueError("cohort depths differ from official adapter state")
    dense_surface, surface_month = load_surface_cache(args.satellite_cache, data, cfg)
    model = OfficialFourDVarNet(args.official_repository, cfg).to(data.device)
    analysis = data.analysis()
    sources = (Path(__file__), ROOT / "src/ocean_tokenizer/fourdvar_synthetic.py",
               ROOT / "src/ocean_tokenizer/synthetic_latent_experiment.py",
               ROOT / "src/ocean_tokenizer/latent_experiment.py",
               ROOT / "src/ocean_tokenizer/reconstruction_metrics.py")
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    training = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    adaptation = {"name": "official 4dvarnet-starter solver adapted to scattered 3D ocean profiles",
        "unchanged": ["GradSolver", "ConvLstmGradModel", "BilinAEPriorCost", "official final inference autoencoder projection"],
        "state": "global 180×360 grid; TEMP 20 depths then SALT 20 depths; original time-channel role replaced by variable/depth",
        "observation_operator": "manual bilinear gather at each exact input profile position; original values and per-depth finite masks",
        "initialization": "source-only nearest-cell means; missing cells converted to zero by original init_state",
        "supervision": "1024 sampled heldout depth queries/step, equal TEMP/SALT standardized MSE; no dense target truth or Sobel supervision",
        "training_difference": "Adam and cosine step schedule; 15000 optimizer steps replace original epoch count; clipping 0.5",
        "auxiliary_surface": "joint 43-channel prior and dense SST/SSS/stericSSH observation constraints" if cfg.surface else None,
        "comparison_scope": "faithful official-code task adaptation, not the original paper benchmark",
        "predictive_uncertainty": "deterministic solver; no invented predictive standard deviation"}
    if checkpoint:
        if checkpoint["config"] != cfg.to_dict() or checkpoint["source_sha256"] != source_hashes:
            raise ValueError("checkpoint architecture/training source differs; refusing nonidentical continuation/evaluation")
        if any(checkpoint["data"][name] != data.metadata[name]
               for name in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")):
            raise ValueError("checkpoint cohort/features/scoring protocol differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.resume_training:
            if "continuation" not in checkpoint:
                raise ValueError("full continuation state absent; a best checkpoint cannot resume training")
            for key in ("steps", "queries", "lr", "weight_decay", "val_every", "val_cells", "surface", "satellite_cache"):
                if checkpoint["training"].get(key) != training.get(key):
                    raise ValueError(f"continuation scientific setting differs in {key}")
    validation = data.eval_months("validation", args.val_cells)
    identity = {"validation": data.identity_check("validation", validation)}
    fixed_baselines = {}
    normalization_scale = np.column_stack([data.norm.std[ch] for ch in CH])
    normalization_offset = np.column_stack([data.norm.mean[ch] for ch in CH])

    def batch(month, source):
        src = data.indices(source)
        fields = dense_surface[surface_month[int(month)]][None] if dense_surface is not None else None
        return make_batch(data.y[src], data.lat[src], data.lon[src], cfg, fields)

    @torch.no_grad()
    def evaluate(evaluations, arrays_path=None):
        model.eval()
        scores, baseline_scores = Scores(data.levels, data.norm.std), Scores(data.levels, data.norm.std)
        all_mean, all_target, all_baseline = [], [], []
        for ev in evaluations:
            state = model(batch(ev.month, ev.source)).detach()
            means, references = [], []
            key = (ev.month, len(ev.level))
            for begin in range(0, len(ev.level), args.eval_chunk):
                end = min(begin + args.eval_chunk, len(ev.level))
                rows, levels = ev.target[ev.profile[begin:end]], ev.level[begin:end]
                q = data.indices(rows)
                prediction = model.query(state, data.lat[q], data.lon[q], levels)[0]
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite official solver prediction")
                means.append(prediction.cpu().numpy())
                if key not in fixed_baselines:
                    geometry = data.geometry(ev.source, rows, 32, levels)
                    references.append(analysis(geometry, data.indices(levels), data.innovation, data.xyz).cpu().numpy())
            p = np.concatenate(means)
            if key not in fixed_baselines:
                fixed_baselines[key] = np.concatenate(references)
            b = fixed_baselines[key]
            # Canonical float64 targets preserve exact equality with classical
            # saved baselines; training tensors remain float32.
            y = np.stack([data.obs[ch][ev.target][ev.profile, ev.level] for ch in CH], axis=-1)
            for j, ch in enumerate(CH):
                scores.add(ch, p[:, j], y[:, j], ev.level)
                baseline_scores.add(ch, b[:, j], y[:, j], ev.level)
            all_mean.append(p); all_target.append(y); all_baseline.append(b)
        mean, target, baseline = np.concatenate(all_mean), np.concatenate(all_target), np.concatenate(all_baseline)
        levels = np.concatenate([ev.level for ev in evaluations])
        mean_physical = inverse_standardize(mean, levels, normalization_scale, normalization_offset)
        target_physical = inverse_standardize(target, levels, normalization_scale, normalization_offset)
        baseline_physical = inverse_standardize(baseline, levels, normalization_scale, normalization_offset)
        score, base_score = scores.result(), baseline_scores.result()
        split = str(data.c.year_split[evaluations[0].target[0]])
        baseline_identity = data.baseline_identity_check(split, base_score)
        if arrays_path is not None:
            np.savez_compressed(arrays_path, mean=mean, target=target, baseline=baseline,
                month=np.concatenate([np.full(len(ev.level), ev.month, dtype=np.int64) for ev in evaluations]),
                profile=np.concatenate([ev.target[ev.profile] for ev in evaluations]), level=levels,
                normalization_mean=normalization_offset, normalization_std=normalization_scale,
                normalization_offset=normalization_offset, normalization_scale=normalization_scale,
                mean_physical=mean_physical, target_physical=target_physical)
        model.train()
        return {"scores": formal_scores(score), "baseline_scores": formal_scores(base_score),
                "mean_standardized_rmse": score["macro_z"],
                "physical_metrics": evaluate_reconstruction(mean_physical, target_physical),
                "baseline_physical_metrics": evaluate_reconstruction(baseline_physical, target_physical),
                "by_depth_physical_metrics": {str(float(depth)): evaluate_reconstruction(
                    mean_physical[levels == i], target_physical[levels == i]) for i, depth in enumerate(data.levels)},
                "oi_identity_check": baseline_identity,
                "metric_target": "at-position physical anomalies; RMSE/MAE/bias also equal absolute T/S errors; R²/r describe anomalies"}

    print(json.dumps({"tag": args.tag, "seed": args.seed, "config": cfg.to_dict(),
                      "params": sum(p.numel() for p in model.parameters()), "official_source": model.official_source,
                      "data_preparation_s": time.time() - started}, ensure_ascii=False), flush=True)
    history, initial, runtime_before = [], None, 0.
    best = {"score": float("inf"), "step": 0, "state": None}

    def model_state(step, validation_result):
        return {"format_version": 1, "model": cpu_state(model), "config": cfg.to_dict(),
                "seed": args.seed, "step": step, "data": data.metadata, "training": training,
                "normalization": {"mean": normalization_offset.tolist(), "std": normalization_scale.tolist()},
                "official_source": model.official_source, "adaptation": adaptation,
                "source_sha256": source_hashes, "validation": validation_result}

    if not args.evaluate_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup = min(300, max(1, args.steps // 10))
        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, args.steps - warmup)
            return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        rng = np.random.default_rng(args.seed)
        training_losses, start_step = [], 1
        if args.resume_training:
            saved = checkpoint["continuation"]
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            restore_random_state(rng, saved["random"])
            best, history, initial = saved["best"], saved["history"], saved["initial"]
            training_losses, runtime_before = saved["training_losses"], saved["runtime_s"]
            start_step = checkpoint["step"] + 1
            print(f"resumed complete state at step {start_step-1}", flush=True)
        else:
            initial = evaluate(validation)
            initial_checkpoint = model_state(0, initial)
            best = {"score": initial["mean_standardized_rmse"], "step": 0, "state": initial_checkpoint}
            atomic_save(initial_checkpoint, output / f"best_seed{args.seed}.pt")
        def save_last(step):
            state = model_state(step, history[-1] if history else initial)
            state["continuation"] = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "random": random_state(rng), "best": best, "history": history, "initial": initial,
                "training_losses": training_losses[-args.val_every:], "runtime_s": runtime_before + time.time() - started}
            atomic_save(state, output / f"last_seed{args.seed}.pt")
        if not args.resume_training:
            save_last(0)
        optimizer_started = time.time()
        for step in range(start_step, args.steps + 1):
            month, source, rows, levels = data.training_draw(rng, args.queries)
            q = data.indices(rows)
            optimizer.zero_grad(set_to_none=True)
            state = model(batch(month, source))
            prediction = model.query(state, data.lat[q], data.lon[q], levels)[0]
            target = data.targets(rows, levels)[0]
            loss = heldout_query_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step(); scheduler.step()
            training_losses.append(float(loss.detach()))
            if step <= 3 or step % 100 == 0:
                print(f"step {step} train_mse {training_losses[-1]:.6f} elapsed {time.time()-optimizer_started:.1f}s", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                result = evaluate(validation)
                record = {"step": step, "training_mse": float(np.mean(training_losses[-args.val_every:])),
                          "validation_mean_standardized_rmse": result["mean_standardized_rmse"],
                          "validation_temperature_rmse_C": result["physical_metrics"]["TEMP"]["rmse"],
                          "validation_salinity_rmse_PSU": result["physical_metrics"]["SALT"]["rmse"],
                          "elapsed_s": runtime_before + time.time() - started}
                history.append(record)
                improved = result["mean_standardized_rmse"] < best["score"]
                if improved:
                    best = {"score": result["mean_standardized_rmse"], "step": step, "state": model_state(step, result)}
                    atomic_save(best["state"], output / f"best_seed{args.seed}.pt")
                print("validation " + json.dumps(record) + (" *" if improved else ""), flush=True)
            if step % args.checkpoint_every == 0 or step % args.val_every == 0 or step == args.steps:
                save_last(step)
        checkpoint = best["state"]
        model.load_state_dict(checkpoint["model"], strict=True)
    final_validation = evaluate(validation, output / f"validation_seed{args.seed}.npz")
    summary = {"tag": args.tag, "seed": args.seed, "config": cfg.to_dict(), "data": data.metadata,
               "training": training, "params": sum(p.numel() for p in model.parameters()),
               "completed_steps": args.steps if not args.evaluate_only else (previous["completed_steps"] if previous else checkpoint["step"]),
               "architecture": "official_4dvarnet_scatter_adaptation", "official_source": model.official_source,
               "adaptation": adaptation, "best_step": checkpoint["step"], "initial": initial,
               "selection_metric": "mean TEMP/SALT standardized RMSE on 2004 validation only",
               "normalization": {"mean": normalization_offset.tolist(), "std": normalization_scale.tolist()},
               "validation": final_validation, "history": history,
               "runtime_s": runtime_before + time.time() - started, "source_sha256": source_hashes}
    if args.development:
        development = data.eval_months("development", 0)
        identity["development"] = data.identity_check("development", development)
        summary["development"] = evaluate(development, output / f"development_seed{args.seed}.npz")
        summary["runtime_s"] = runtime_before + time.time() - started
    if previous:
        if previous["config"] != cfg.to_dict() or previous["best_step"] != checkpoint["step"] or previous["source_sha256"] != source_hashes:
            raise ValueError("frozen evaluation differs from completed training summary")
        summary["evaluation"] = {"runtime_s": time.time() - started, "load_checkpoint": str(args.load_checkpoint),
                                 "development": args.development}
        for key in ("initial", "history", "training", "source_sha256"):
            summary[key] = previous[key]
        summary["runtime_s"] = previous["runtime_s"] + time.time() - started
        if not args.development and "development" in previous:
            summary["development"] = previous["development"]
            if "development" in previous.get("identity_check", {}):
                identity["development"] = previous["identity_check"]["development"]
    summary["identity_check"] = identity
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, summary_path)
    print("finished " + json.dumps({"summary": str(summary_path), "best_step": checkpoint["step"],
          "validation_rmse": {ch: final_validation["physical_metrics"][ch]["rmse"] for ch in CH},
          "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
