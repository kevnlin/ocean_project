"""The official 4DVarNet on the paper's two tasks, trained and chosen as KA-Ocean was (2026-10-21).

Zihao asked for a published method run from its authors' code. 4DVarNet (Fablet et al.; the
CIA-Oceanix ``4dvarnet-starter`` at a pinned commit) is the learned variational interpolation of
ocean data assimilation; ``fourdvar_grid.FourDVarGridModel`` runs its solver, gradient model and
prior unchanged on one month's state of T and S at 20 levels and the SST/SSS/SSH grids
(``docs/superpowers/specs/2026-10-21-official-4dvarnet-baseline.md``):

* data, splits, inputs and cells are those of KA-Ocean's runs (runner ``101``, method A): CESM2
  ``cesm2_long_uniform`` with 40 training years (1964-2003) and 2004-2005 for selection, or the real
  2016-2020 Argo cohort with 2021 for selection; the same scenes, the same training draws at lead 0
  (30 % of the month's floats held out as targets, 1,024 cells a step), the same 8,000-cell
  selection draw;
* the official hyperparameters (10 solver steps of rate 1e3, a 32-channel prior at half
  resolution, a 48-channel ConvLSTM gradient model with dropout 0.1), objective (50 x the
  reconstruction MSE + 1 x the prior cost; the Sobel term needs dense truth and is dropped) and
  optimiser (Adam at 1e-3 with the prior at half the rate, cosine annealing, gradient norm clipped
  at 0.5), for KA-Ocean's number of steps (70,000 on CESM2, 30,000 on real data);
* 4DVarNet reconstructs a month: at a forecast lead the input month's reconstruction is persisted
  to the target month, like the persisted OI;
* the checkpoint, and whether the official final autoencoder projection is applied, are chosen by
  the lead-0 mean standardised RMSE on the selection draw; the chosen checkpoint is then scored on
  every cell of the selection years at every lead (``summary["forecast"]["best"]``).

  .venv/bin/python experiments/synthetic/160_official_4dvarnet.py --dataset cesm2 --seed 1234 --tag fv_cesm2_s1234
  .venv/bin/python experiments/synthetic/160_official_4dvarnet.py --dataset real_global --seed 1234 --tag fv_real_s1234
  ... --load-checkpoint OUT/last_seed1234.pt --resume-training     continues an interrupted run
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
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer.forecast_scenes import LEADS
from ocean_tokenizer.fourdvar_grid import OUTPUTS, FourDVarGridModel
from ocean_tokenizer.fourdvar_synthetic import FourDVarConfig
from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
from ocean_tokenizer.latent_experiment import masked_losses
from ocean_tokenizer.point_baselines import CH, Scores
from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
from ocean_tokenizer.synthetic_latent_experiment import EVAL_SEED

OUTPUT = ROOT / "outputs/fourdvarnet_20261021"
#: KA-Ocean's settings (runner 101, method A) on each dataset
DATASETS = {"cesm2": {"steps": 70000, "val_every": 2000, "val_years": [2004, 2005], "context_profiles": 6080},
            "real_global": {"steps": 30000, "val_every": 1000, "val_years": [2021], "context_profiles": 0}}
ADAPTATION = {
    "unchanged": ["GradSolver", "ConvLstmGradModel", "BilinAEPriorCost", "final autoencoder projection (optional)",
                  "default hyperparameters", "training objective weights", "optimiser and its prior rate"],
    "state": "global 1-degree grid; T at every level, S at every level, then SST, SSS, SSH (the time-window "
             "channels of the original task are depth/variable channels here)",
    "observation_cost": "bilinear read of the state at every input profile's exact position vs its values, plus the "
                        "surface channels vs the surface grids where they exist",
    "initialisation": "profile cell means and the surface grids; zero where nothing is observed (official)",
    "supervision": "1,024 held-out cells a step, channel-balanced MSE as KA-Ocean; no dense truth, so no Sobel term",
    "forecast": "the input month's reconstruction read at the target month (persisted reconstruction)",
    "uncertainty": "none: the solver is deterministic"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--steps", type=int, default=None, help="default: KA-Ocean's (70,000 CESM2, 30,000 real)")
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--val-years", nargs="+", type=int, default=None)
    p.add_argument("--context-profiles", type=int, default=None)
    p.add_argument("--queries", type=int, default=1024)
    p.add_argument("--val-cells", type=int, default=8000)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--eval-chunk", type=int, default=2048)
    p.add_argument("--cohort", default="cesm2_long_uniform")
    p.add_argument("--train-years", type=int, default=40, help="CESM2: the last N years before 2004")
    p.add_argument("--satellite-cache", type=Path, default=ROOT / "outputs/long_record_20261017/satellites.npz")
    p.add_argument("--lagged-surface", type=Path, default=ROOT / "outputs/forecast_20261019/lagged_surface.npz")
    p.add_argument("--surface-grids", type=Path, default=ROOT / "outputs/grid_ka_20261019/surface_grids.npz")
    p.add_argument("--oi-summary", type=Path,
                   default=ROOT / "outputs/audit/synthetic/fixed_baselines_cesm2_long_uniform/summary.json")
    p.add_argument("--real-caches", type=Path, default=ROOT / "outputs/real_global_20261019")
    p.add_argument("--real-oi-summary", type=Path,
                   default=ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json")
    p.add_argument("--official-repository", type=Path, default=ROOT / "external/4dvarnet-starter")
    p.add_argument("--solver-steps", type=int, default=10)
    p.add_argument("--solver-lr", type=float, default=1e3)
    p.add_argument("--prior-hidden", type=int, default=32)
    p.add_argument("--gradient-hidden", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--load-checkpoint", type=Path)
    p.add_argument("--resume-training", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    for name, value in DATASETS[args.dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def atomic_save(state, path):
    temporary = path.with_name(path.name + ".tmp")
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


def scenes(args):
    if args.dataset == "real_global":
        return RealGridForecastScenes(ROOT, args.device, caches=args.real_caches, oi_summary=args.real_oi_summary,
                                      leads=LEADS)
    return GridForecastScenes(ROOT, args.device, surface_grids=args.surface_grids, leads=LEADS,
                              lagged_surface=args.lagged_surface, train_years=args.train_years, surface=True,
                              satellite_cache=args.satellite_cache, oi_summary=args.oi_summary, cohort=args.cohort)


def main():
    args = arguments()
    if args.val_cells != 8000:
        raise ValueError("the selection draw is KA-Ocean's: 8,000 cells a target month")
    if args.resume_training != (args.load_checkpoint is not None):
        raise ValueError("--load-checkpoint resumes a run: pass it with --resume-training")
    out = args.output or OUTPUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"summary_seed{args.seed}.json"
    if done.exists():
        raise SystemExit(f"refusing to overwrite finished run: {done}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run outside the device-restricted sandbox or choose --device cpu")
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = scenes(args)
    split_of_year = getattr(data, "split_of_year", lambda year: {2004: "validation", 2005: "development"}.get(year))
    if not (np.allclose(data.grid_lat, -89.5 + np.arange(180)) and np.allclose(data.grid_lon, .5 + np.arange(360))):
        raise ValueError("the surface grids are not the 1-degree grid of the 4DVarNet state")
    cfg = FourDVarConfig(levels=data.n_levels, height=180, width=360, n_step=args.solver_steps,
                         lr_grad=args.solver_lr, prior_hidden=args.prior_hidden,
                         gradient_hidden=args.gradient_hidden, surface=True)
    model = FourDVarGridModel(args.official_repository, cfg).to(data.device)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (
        Path(__file__), *(ROOT / "src/ocean_tokenizer" / name for name in (
            "fourdvar_grid.py", "fourdvar_synthetic.py", "forecast_scenes.py", "grid_forecast_scenes.py",
            "real_forecast_scenes.py", "long_record_scenes.py", "latent_experiment.py")))}
    training = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("load_checkpoint", "resume_training")}
    # what the scorers read from a gridded checkpoint's training record
    training.update(amp=False, eval_context_profiles=None, leads=list(LEADS), surface=True, scale=None)
    normalization = {kind: {ch: torch.as_tensor(getattr(data.norm, kind)[ch]) for ch in CH} for kind in ("mean", "std")}
    checkpoint = None
    if args.resume_training:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        if "continuation" not in checkpoint or checkpoint["source_sha256"] != source_hashes:
            raise ValueError("not a continuation checkpoint of this source")
        if checkpoint["config"] != cfg.to_dict() or checkpoint["training"] != training:
            raise ValueError("the resumed run had other settings")
        if any(checkpoint["data"][k] != data.metadata[k] for k in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")):
            raise ValueError("the resumed run read other data")
        model.load_state_dict(checkpoint["model"], strict=True)
    analysis = data.analysis()
    val = [ev for ev in data.forecast_months(args.val_years, args.val_cells) if ev.lead == 0]
    identity = {"validation": data.identity_check(
        "validation", [ev for ev in val if split_of_year(2000 + ev.month // 12) == "validation"])}
    geometries = {}

    @torch.no_grad()
    def evaluate(evs, outputs=OUTPUTS, keep=False, save_arrays=None):
        """Scores of every output path by lead and target year, and the persisted OI's, on the same cells."""
        model.eval()
        leads, years = sorted({ev.lead for ev in evs}), sorted({2000 + ev.month // 12 for ev in evs})
        new = lambda: Scores(data.levels, data.norm.std)
        model_scores = {(o, lead, y): new() for o in outputs for lead in leads for y in (None, *years)}
        oi_scores = {(lead, y): new() for lead in leads for y in (None, *years)}
        arrays = {name: [] for name in ("mean", "target", "baseline", "month", "lead", "profile", "level")}
        for ev in evs:
            key = (ev.month, ev.lead, len(ev.level))
            g = geometries.get(key) or data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
            if keep:
                geometries[key] = g
            state, parts, base = None, {o: [] for o in outputs}, []
            for start in range(0, len(ev.level), args.eval_chunk):
                stop = min(start + args.eval_chunk, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                scene = data.scene(ev.source, rows, lev, analysis, args.context_profiles, [EVAL_SEED, ev.month],
                                   {name: value[start:stop] for name, value in g.items()})
                if state is None:
                    state = model.encode(scene)
                for o in outputs:
                    p = data.to_z(model.decode(scene, state, o)["mean"].float(), rows, lev)
                    if not torch.isfinite(p).all():
                        raise FloatingPointError(f"nonfinite 4DVarNet output, month {ev.month}, lead {ev.lead}")
                    parts[o].append(p.cpu().numpy())
                base.append(data.to_z(scene["baseline"], rows, lev).cpu().numpy())
            y = data.targets(ev.target[ev.profile], ev.level)[0].cpu().numpy()
            b, year = np.concatenate(base), 2000 + ev.month // 12
            for j, ch in enumerate(CH):
                for when in (None, year):
                    oi_scores[ev.lead, when].add(ch, b[:, j], y[:, j], ev.level)
                    for o in outputs:
                        model_scores[o, ev.lead, when].add(ch, np.concatenate(parts[o])[:, j], y[:, j], ev.level)
            if save_arrays is not None:
                for name, value in (("mean", np.concatenate(parts[outputs[0]])), ("target", y), ("baseline", b),
                                    ("month", np.full(len(ev.level), ev.month, dtype=np.int16)),
                                    ("lead", np.full(len(ev.level), ev.lead, dtype=np.int8)),
                                    ("profile", ev.target[ev.profile]), ("level", ev.level)):
                    arrays[name].append(value)
        result = {o: {"by_lead": {str(lead): {
            "scores": model_scores[o, lead, None].result(), "baseline_scores": oi_scores[lead, None].result(),
            "by_year": {str(y): {"scores": model_scores[o, lead, y].result(),
                                 "baseline_scores": oi_scores[lead, y].result()} for y in years}}
            for lead in leads}} for o in outputs}
        for o in outputs:
            result[o]["selection_score"] = result[o]["by_lead"]["0"]["scores"]["macro_z"] if 0 in leads else None
        if save_arrays is not None:
            np.savez_compressed(save_arrays, **{name: np.concatenate(v) for name, v in arrays.items()},
                                normalization_scale=np.stack([data.norm.std[ch] for ch in CH], -1),
                                normalization_offset=np.stack([data.norm.mean[ch] for ch in CH], -1))
        model.train()
        return result

    def chosen(result):
        """The output path with the lower lead-0 selection score."""
        output = min(result, key=lambda o: result[o]["selection_score"])
        return output, result[output]["selection_score"]

    def model_state(step, output, validation):
        return {"format_version": 1, "model": cpu_state(model), "config": cfg.to_dict(), "output": output,
                "seed": args.seed, "step": step, "data": data.metadata, "training": training,
                "normalization": normalization, "official_source": model.official_source, "adaptation": ADAPTATION,
                "source_sha256": source_hashes, "validation": validation}

    print(json.dumps({"tag": args.tag, "seed": args.seed, "config": cfg.to_dict(),
                      "params": sum(p.numel() for p in model.parameters()), "official_source": model.official_source,
                      "data_preparation_s": time.time() - started}), flush=True)
    solver = model.net.solver
    # the official optimiser (``src/utils.py::cosanneal_lr_adam``): the prior learns at half the rate
    optimizer = torch.optim.Adam([{"params": solver.grad_mod.parameters(), "lr": args.lr},
                                  {"params": solver.obs_cost.parameters(), "lr": args.lr},
                                  {"params": solver.prior_cost.parameters(), "lr": args.lr / 2}], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    if checkpoint is not None:
        continuation = checkpoint["continuation"]
        optimizer.load_state_dict(continuation["optimizer"])
        scheduler.load_state_dict(continuation["scheduler"])
        restore_random_state(rng, continuation["random"])
        best, history, initial = continuation["best"], continuation["history"], continuation["initial"]
        losses, prior_runtime, start_step = continuation["losses"], continuation["runtime_s"], checkpoint["step"] + 1
        print(f"resumed at step {start_step - 1}; best step {best['step']} ({best['output']})", flush=True)
    else:
        initial = evaluate(val, keep=True)
        output, score = chosen(initial)
        best = {"score": score, "step": 0, "output": output, "state": model_state(0, output, initial)}
        atomic_save(best["state"], out / f"best_seed{args.seed}.pt")
        history, losses, prior_runtime, start_step = [], [], 0., 1
        print("initial selection score " + json.dumps({o: initial[o]["selection_score"] for o in OUTPUTS}), flush=True)

    def save_last(step):
        state = model_state(step, best["output"], history[-1] if history else None)
        state["continuation"] = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "random": random_state(rng), "best": best, "history": history, "initial": initial,
                                 "losses": losses[-args.val_every:], "runtime_s": prior_runtime + time.time() - started}
        atomic_save(state, out / f"last_seed{args.seed}.pt")

    if checkpoint is None:
        save_last(0)
    clock = time.time()
    for step in range(start_step, args.steps + 1):
        # KA-Ocean's training draw at lead 0: 4DVarNet reconstructs the month of its inputs
        leads, data.leads = data.leads, (0,)
        try:
            month, src, rows, levels = data.training_draw(rng, args.queries)
        finally:
            data.leads = leads
        scene = data.scene(src, rows, levels, analysis, args.context_profiles, rng)
        target, valid = data.network_targets(rows, levels)
        state = model.encode(scene)
        mean = model.decode(scene, state)["mean"]
        _, parts = masked_losses({"mean": mean, "std": torch.ones_like(mean)}, target, valid)
        # the official objective without its Sobel term: 50 x reconstruction + 1 x prior cost
        loss = 50 * parts["mse"] + state["prior_cost"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        scheduler.step()
        losses.append([float(parts["mse"].detach()), float(state["prior_cost"].detach())])
        if step <= 3 or step % 100 == 0:
            print(f"step {step} train_mse {losses[-1][0]:.5f} prior_cost {losses[-1][1]:.5f} "
                  f"elapsed {time.time() - clock:.1f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            result = evaluate(val, keep=True)
            output, score = chosen(result)
            record = {"step": step, "training_mse": float(np.mean([l[0] for l in losses[-args.val_every:]])),
                      "prior_cost": float(np.mean([l[1] for l in losses[-args.val_every:]])),
                      "selection_score": {o: result[o]["selection_score"] for o in OUTPUTS},
                      "lead0_rmse": {o: {ch: result[o]["by_lead"]["0"]["scores"][ch]["rmse_physical"] for ch in CH}
                                     for o in OUTPUTS},
                      "elapsed_s": prior_runtime + time.time() - started}
            history.append(record)
            improved = score < best["score"]
            if improved:
                best = {"score": score, "step": step, "output": output, "state": model_state(step, output, result)}
                atomic_save(best["state"], out / f"best_seed{args.seed}.pt")
            print("validation " + json.dumps(record) + (f" * {output}" if improved else ""), flush=True)
        if step % args.checkpoint_every == 0 or step % args.val_every == 0 or step == args.steps:
            save_last(step)
    model.load_state_dict(best["state"]["model"], strict=True)
    output = best["output"]
    model.output = output
    final = evaluate(val, (output,))[output]
    full = data.forecast_months(args.val_years, 0)
    forecast = evaluate(full, (output,), save_arrays=out / f"forecast_seed{args.seed}.npz")[output]
    summary = {"tag": args.tag, "seed": args.seed, "dataset": args.dataset, "completed_steps": args.steps,
               "config": cfg.to_dict(), "output": output, "data": data.metadata, "training": training,
               "params": sum(p.numel() for p in model.parameters()), "architecture": "official_4dvarnet",
               "official_source": model.official_source, "adaptation": ADAPTATION,
               "selection_metric": "lead-0 mean temperature/salinity standardized RMSE on the selection draw; "
                                   "chooses the checkpoint and the output path",
               "normalization": {kind: {ch: v.tolist() for ch, v in normalization[kind].items()} for kind in normalization},
               "best_step": best["step"], "initial": initial, "validation": final, "leads": list(LEADS),
               "val_years": args.val_years, "forecast": {"best": forecast}, "history": history,
               "identity_check": identity,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if data.device.type == "cuda" else None,
               "runtime_s": prior_runtime + time.time() - started, "source_sha256": source_hashes}
    temporary = done.with_name(done.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, done)
    print("finished " + json.dumps({"summary": str(done), "best_step": best["step"], "output": output,
          "lead0_rmse": {ch: forecast["by_lead"]["0"]["scores"][ch]["rmse_physical"] for ch in CH},
          "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
