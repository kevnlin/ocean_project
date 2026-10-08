"""Full multimodal ocean latent/Soft-MoE/local-Transformer training and evaluation.

Architecture search uses 2021 validation only.  Development is evaluated only
after a checkpoint/configuration is frozen; 2022-2023 is previously used
development data and is never described as independent confirmation.
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

from ocean_tokenizer.latent_experiment import EVAL_SEED, OceanScenes, masked_losses
from ocean_tokenizer.latent_ocean import LatentOceanConfig, LatentOceanModel
from ocean_tokenizer.point_baselines import CH, Scores


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["dense", "soft_moe", "local_transformer"], default="dense")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--latents", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--query-blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--slots-per-expert", type=int, default=2)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--queries", type=int, default=1024, help="individual depth queries per optimizer step")
    p.add_argument("--context-profiles", type=int, default=512)
    p.add_argument("--eval-context-profiles", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--analysis-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--nll-weight", type=float, default=.02)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-cells", type=int, default=8000)
    p.add_argument("--eval-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--freeze-analysis", action="store_true")
    p.add_argument("--load-checkpoint", type=Path, default=None)
    p.add_argument("--warm-start", action="store_true", help="load model weights and restart the optimizer/schedule")
    p.add_argument("--resume-training", action="store_true", help="restore optimizer, scheduler, sampling and torch RNG state")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--analysis-only", action="store_true", help="matched jointly-retrained OI control, without neural correction")
    p.add_argument("--latent-off", action="store_true", help="local and query-MLP control without shared observation latent")
    p.add_argument("--local-off", action="store_true", help="retain original OI, disable added learned local evidence path")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--development", action="store_true", help="open earlier development years only after freezing")
    p.add_argument("--amp", action="store_true", help="bf16 neural model; exact kriging remains float64")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def random_state(rng):
    return {"numpy_generator": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(rng, state):
    rng.bit_generator.state = state["numpy_generator"]
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def uncertainty_metrics(pred, std, target, mask, levels):
    out = {}
    for j, ch in enumerate(CH):
        ok = mask[:, j]
        e, sigma = pred[ok, j]-target[ok, j], std[ok, j]
        out[ch] = {"n": int(ok.sum()), "nll": float(np.mean(.5*(e/sigma)**2 + np.log(sigma) + .5*np.log(2*np.pi))),
                   "coverage_68": float(np.mean(np.abs(e) <= sigma)),
                   "coverage_95": float(np.mean(np.abs(e) <= 1.95996398454*sigma)),
                   "mean_std_z": float(sigma.mean())}
    return out


def main():
    args = arguments()
    if args.resume_training and (args.load_checkpoint is None or args.warm_start or args.evaluate_only):
        raise ValueError("--resume-training requires --load-checkpoint and excludes warm-start/evaluate-only")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.load_checkpoint and (args.resume_training or args.evaluate_only):
        saved = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        args.seed = saved["seed"]
        del saved
    started = time.time()
    out = args.output or ROOT / "outputs/latent_ocean" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"summary_seed{args.seed}.json"
    previous_summary = json.loads(done.read_text()) if args.evaluate_only and done.exists() else None
    if done.exists() and not (args.evaluate_only or args.warm_start):
        raise SystemExit(f"refusing to overwrite finished run: {done}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run outside the device-restricted sandbox or explicitly choose --device cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = OceanScenes(ROOT, args.device)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT/"src/ocean_tokenizer/latent_ocean.py", ROOT/"src/ocean_tokenizer/latent_experiment.py")}
    analysis, anchor_analysis = data.analysis(), data.analysis()
    cfg = LatentOceanConfig(n_obs_features=62, n_query_features=58, n_local_features=66,
            n_sat_features=56, variant=args.variant, width=args.width, n_latents=args.latents,
            n_heads=args.heads, n_blocks=args.blocks, n_query_blocks=args.query_blocks,
            n_experts=args.experts, slots_per_expert=args.slots_per_expert,
            use_latent=not args.latent_off, use_local=not args.local_off)
    checkpoint = None
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        if any(checkpoint["data"][name] != data.metadata[name]
               for name in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")):
            raise ValueError("checkpoint data mismatch")
        cfg = LatentOceanConfig(**checkpoint["config"])
        if args.resume_training:
            if "continuation" not in checkpoint:
                raise ValueError("checkpoint has no complete continuation state; use --warm-start instead")
            if checkpoint.get("source_sha256") != source_hashes:
                raise ValueError("source changed since checkpoint; use --warm-start for a new experiment")
            for name in ("seed", "steps", "queries", "context_profiles", "eval_context_profiles",
                         "lr", "analysis_lr", "weight_decay", "nll_weight", "val_every", "val_cells",
                         "eval_chunk", "freeze_analysis", "analysis_only", "amp", "latent_off", "local_off"):
                setattr(args, name, checkpoint["training"][name])
    model = LatentOceanModel(cfg).to(data.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        analysis.load_state_dict(checkpoint["analysis"], strict=True)
        if "anchor_analysis" in checkpoint:
            anchor_analysis.load_state_dict(checkpoint["anchor_analysis"], strict=True)
        if args.evaluate_only:
            args.context_profiles = checkpoint["training"]["context_profiles"]
            args.freeze_analysis = checkpoint["training"]["freeze_analysis"]
            args.analysis_only = checkpoint["training"].get("analysis_only", False)
            args.eval_context_profiles = checkpoint["training"]["eval_context_profiles"]
            args.amp = checkpoint["training"]["amp"]
            args.val_cells = checkpoint["training"]["val_cells"]
            args.latent_off = not cfg.use_latent
            args.local_off = not cfg.use_local
    if args.freeze_analysis:
        analysis.requires_grad_(False)
    anchor_analysis.requires_grad_(False)
    val = data.eval_months("validation", args.val_cells)
    identity = {"validation": data.identity_check("validation", val)}
    geometries, fixed_baselines = {}, {}

    def key(ev):
        return ev.month, len(ev.level)

    def geometry(ev):
        k = key(ev)
        if k not in geometries:
            geometries[k] = data.geometry(ev.source, ev.target[ev.profile], analysis.k)
        return geometries[k]

    def geom_chunk(g, start, stop):
        return {name: value[start:stop] for name, value in g.items()}

    @torch.no_grad()
    def evaluate(evs, with_model=True, save_arrays=None):
        model.eval()
        scores = Scores(data.levels, data.norm.std)
        base_scores = Scores(data.levels, data.norm.std)
        pred_all, std_all, target_all = [], [], []
        diagnostics = []
        for ev in evs:
            g = geometry(ev)
            base_chunks, pred_chunks, std_chunks = [], [], []
            state = None
            context_seed = [EVAL_SEED, ev.month]
            for start in range(0, len(ev.level), args.eval_chunk):
                stop = min(start+args.eval_chunk, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                scene = data.scene(ev.source, rows, lev, analysis if with_model else anchor_analysis,
                    args.eval_context_profiles or args.context_profiles, context_seed, geom_chunk(g, start, stop))
                if state is None:
                    state = model.encode(scene) if with_model and not args.analysis_only else None
                if with_model and not args.analysis_only:
                    with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                        enabled=args.amp and data.device.type == "cuda"):
                        prediction = model.decode(scene, state)
                    p, sigma = prediction["mean"].float(), prediction["std"].float()
                    if not torch.isfinite(p).all() or not torch.isfinite(sigma).all():
                        raise FloatingPointError("nonfinite evaluation output")
                    if start == 0:
                        diagnostics.append({name: value.float().cpu().tolist()
                            for name, value in prediction["diagnostics"].items()})
                else:
                    p, sigma = scene["baseline"], torch.full_like(scene["baseline"], .6)
                pred_chunks.append(p.cpu().numpy())
                std_chunks.append(sigma.cpu().numpy())
                if key(ev) not in fixed_baselines:
                    fixed = data.scene(ev.source, rows, lev, anchor_analysis,
                        1, context_seed, geom_chunk(g, start, stop))["baseline"]
                    base_chunks.append(fixed.cpu().numpy())
            p, sigma = np.concatenate(pred_chunks), np.concatenate(std_chunks)
            if key(ev) not in fixed_baselines:
                fixed_baselines[key(ev)] = np.concatenate(base_chunks)
            tgt, _ = data.targets(ev.target[ev.profile], ev.level)
            y = tgt.cpu().numpy()
            for j, ch in enumerate(CH):
                scores.add(ch, p[:, j], y[:, j], ev.level)
                base_scores.add(ch, fixed_baselines[key(ev)][:, j], y[:, j], ev.level)
            pred_all.append(p); std_all.append(sigma); target_all.append(y)
        p, sigma, y = np.concatenate(pred_all), np.concatenate(std_all), np.concatenate(target_all)
        if save_arrays is not None:
            np.savez_compressed(save_arrays, mean=p, std=sigma, target=y,
                month=np.concatenate([np.full(len(ev.level), ev.month, dtype=np.int16) for ev in evs]),
                profile=np.concatenate([ev.target[ev.profile] for ev in evs]),
                level=np.concatenate([ev.level for ev in evs]),
                baseline=np.concatenate([fixed_baselines[key(ev)] for ev in evs]))
        model.train()
        return {"scores": scores.result(), "baseline_scores": base_scores.result(),
                "uncertainty": uncertainty_metrics(p, sigma, y, np.isfinite(y), None),
                "diagnostics": diagnostics}

    print(json.dumps({"tag": args.tag, "config": cfg.to_dict(), "seed": args.seed,
                      "neural_parameters": 0 if args.analysis_only else sum(p.numel() for p in model.parameters()),
                      "train_analysis": not args.freeze_analysis, "data_preparation_s": time.time()-started}, ensure_ascii=False), flush=True)
    history = []
    best = {"score": float("inf"), "step": 0, "state": None}
    initial = None
    if not args.evaluate_only:
        def model_state(step, validation):
            return {"format_version": 2, "model": cpu_state(model), "analysis": cpu_state(analysis),
                    "anchor_analysis": cpu_state(anchor_analysis),
                    "config": cfg.to_dict(), "seed": args.seed, "step": step,
                    "data": data.metadata,
                    "training": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
                    "anchor_first_guess": data.fg_checkpoint,
                    "normalization": {"mean": {ch: torch.as_tensor(data.norm.mean[ch]) for ch in CH},
                                      "std": {ch: torch.as_tensor(data.norm.std[ch]) for ch in CH}},
                    "source_sha256": source_hashes, "validation": validation}
        if not args.resume_training:
            initial = evaluate(val)
            print(f"initial validation T {initial['scores']['TEMP']['J']:.5f} S {initial['scores']['SALT']['J']:.5f}", flush=True)
            initial_state = model_state(0, initial)
            best = {"score": initial["scores"]["macro_z"], "step": 0, "state": initial_state}
            atomic_save(initial_state, out/f"best_seed{args.seed}.pt")
        groups = [] if args.analysis_only else [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
        if not args.freeze_analysis:
            groups.append({"params": analysis.parameters(), "lr": args.analysis_lr, "weight_decay": 0.})
        optimizer = torch.optim.AdamW(groups)
        warmup = min(300, max(1, args.steps//10))
        def schedule(step):
            if step < warmup:
                return (step+1)/warmup
            ratio = (step-warmup)/max(1, args.steps-warmup)
            return .1+.9*.5*(1+math.cos(math.pi*ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        rng = np.random.default_rng(args.seed)
        training_loss = []
        start_step, prior_runtime = 1, 0.
        if args.resume_training:
            continuation = checkpoint["continuation"]
            optimizer.load_state_dict(continuation["optimizer"])
            scheduler.load_state_dict(continuation["scheduler"])
            restore_random_state(rng, continuation["random"])
            best = continuation["best"]
            history = continuation["history"]
            initial = continuation["initial"]
            training_loss = continuation["training_loss"]
            start_step = checkpoint["step"]+1
            prior_runtime = continuation["runtime_s"]
            print(f"resumed complete training state at step {start_step-1}; best step {best['step']}", flush=True)
        def save_last(step):
            state = model_state(step, history[-1] if history else None)
            state["continuation"] = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "random": random_state(rng), "best": best, "history": history, "initial": initial,
                "training_loss": training_loss[-args.val_every:], "runtime_s": prior_runtime+time.time()-started}
            atomic_save(state, out/f"last_seed{args.seed}.pt")
        if not args.resume_training:
            save_last(0)
        optimizer_start = time.time()
        for step in range(start_step, args.steps+1):
            m, src, rows, levels = data.training_draw(rng, args.queries)
            scene = data.scene(src, rows, levels, analysis, args.context_profiles, rng)
            target, valid = data.targets(rows, levels)
            with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                enabled=args.amp and data.device.type == "cuda"):
                prediction = (model(scene) if not args.analysis_only else
                              {"mean": scene["baseline"], "std": torch.full_like(scene["baseline"], .6),
                               "aux_loss": scene["baseline"].new_zeros(())})
                loss, components = masked_losses(prediction, target, valid, args.nll_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not args.freeze_analysis:
                torch.nn.utils.clip_grad_norm_(analysis.parameters(), 1.)
            optimizer.step(); scheduler.step()
            with torch.no_grad():
                for name, p in analysis.named_parameters():
                    if name == "log_gamma":
                        p.clamp_(-8., 5.)
                    elif name == "log_ell_s":
                        p.clamp_(-4., 5.)
                    elif name == "log_ell_t":
                        p.clamp_(math.log(.5), math.log(365.))
                    else:
                        p.clamp_(math.log(10.), math.log(2000.))
            training_loss.append(float(components["mse"].detach()))
            if step <= 3 or step % 100 == 0:
                print(f"step {step} train_mse {training_loss[-1]:.5f} elapsed {time.time()-optimizer_start:.1f}s", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                ev = evaluate(val)
                sc = ev["scores"]
                record = {"step": step, "training_mse": float(np.mean(training_loss[-args.val_every:])),
                          "validation_macro_z": sc["macro_z"], "validation_TEMP_J": sc["TEMP"]["J"],
                          "validation_SALT_J": sc["SALT"]["J"], "elapsed_s": prior_runtime+time.time()-started}
                history.append(record)
                improved = sc["macro_z"] < best["score"]
                if improved:
                    state = model_state(step, ev)
                    best = {"score": sc["macro_z"], "step": step, "state": state}
                    atomic_save(state, out/f"best_seed{args.seed}.pt")
                print("validation " + json.dumps(record) + (" *" if improved else ""), flush=True)
            if step % args.checkpoint_every == 0 or step % args.val_every == 0 or step == args.steps:
                save_last(step)
        model.load_state_dict(best["state"]["model"], strict=True)
        analysis.load_state_dict(best["state"]["analysis"], strict=True)
        checkpoint = best["state"]
    elif checkpoint is None:
        raise ValueError("--evaluate-only requires --load-checkpoint")
    final_validation = evaluate(val, save_arrays=out/f"validation_seed{args.seed}.npz")
    summary = {"tag": args.tag, "seed": args.seed, "config": cfg.to_dict(), "data": data.metadata,
               "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "params": 0 if args.analysis_only else sum(p.numel() for p in model.parameters()),
               "architecture": "analysis_only" if args.analysis_only else cfg.variant,
               "analysis_params": sum(p.numel() for p in analysis.parameters()),
               "best_step": checkpoint["step"], "initial": initial, "validation": final_validation,
               "history": history, "runtime_s": (prior_runtime if not args.evaluate_only else 0.)+time.time()-started,
               "source_sha256": source_hashes}
    if args.development:
        dev = data.eval_months("development", 0)
        identity["development"] = data.identity_check("development", dev)
        summary["development"] = evaluate(dev, save_arrays=out/f"development_seed{args.seed}.npz")
        summary["runtime_s"] = (prior_runtime if not args.evaluate_only else 0.)+time.time()-started
    if previous_summary is not None:
        if (previous_summary["config"] != cfg.to_dict()
                or previous_summary["best_step"] != checkpoint["step"]
                or any(previous_summary["data"][name] != data.metadata[name]
                       for name in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint"))):
            raise ValueError("evaluation checkpoint differs from completed training summary")
        summary["evaluation"] = {"runtime_s": time.time()-started, "source_sha256": source_hashes,
                                 "load_checkpoint": str(args.load_checkpoint), "development": args.development}
        for name in ("initial", "history", "training", "source_sha256"):
            summary[name] = previous_summary[name]
        summary["runtime_s"] = previous_summary["runtime_s"]+time.time()-started
        if not args.development and "development" in previous_summary:
            summary["development"] = previous_summary["development"]
            if "development" in previous_summary.get("identity_check", {}):
                identity["development"] = previous_summary["identity_check"]["development"]
    summary["identity_check"] = identity
    temporary = done.with_name(done.name+".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, done)
    print("finished " + json.dumps({"summary": str(done), "best_step": summary["best_step"],
          "validation": {ch: summary["validation"]["scores"][ch]["J"] for ch in CH},
          "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
