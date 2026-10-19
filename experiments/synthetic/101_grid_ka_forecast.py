"""Runner 94 with the gridded state model: one state per month, moved forward, read with kriging.

A copy of ``94_forecast_reconstruction.py`` (the data, the leads, the loss, the optimiser, the
schedule, the checkpoint criterion and every score are 94's) with the model replaced by
``grid_kriging_ocean.GridKrigingModel`` on ``grid_forecast_scenes.GridForecastScenes``:

  --surface-grids     dense SST/SSS/SSH grids of every month (``100_surface_grids.py``)
  --background B      joint (one state from everything) or prior (the innovations and the
                      reconstruction background come from a surface-only prior state; the
                      profiles build the analysis state the forecast evolves)
  --kriging K         read-out of the 32 nearest profiles: ka_flow (default), ka_stationary,
                      ka_oi, or none (the gridded background alone); the kriging starts from the
                      cohort's OI length scales and error ratios, expanded from bands to levels
  --state-channels, --encoder-channels, --evolution-channels, --hidden   model size

  --dataset real_global   the real 2016-2023 Argo cohort with OISST, OISSS and DUACS SLA
                      (``real_forecast_scenes``; caches of ``105``, OI of ``106``); pass
                      --val-years 2021. The training split is 2016-2020 (--train-years is unused).
  --init-from PATH    start from the model weights of another run's checkpoint (another
                      dataset is allowed: weights only, a new optimiser and schedule)

The model never reads the frozen OI (the persisted OI is still scored as the reference) and
works in per-depth z units, so --scale is refused. Log length scales, log error ratios and
the log decay times get no weight decay.

94's options, for reference:

  --leads 0 1 3 6   one model for every lead: each training step draws a lead and a target
                    month whose input month is in the window too; the lead reaches the model
                    as the time from its neighbours to the query (``forecast_scenes``)
  --val-years       target years that choose the checkpoint, 2004 and 2005 by default: the
                    mean over leads of the per-variable standardized RMSE on 8,000 cells a month
  --lagged-surface  ``93_lagged_surface.py``: a query's surface features are its input month's

The baseline at every lead is persisted OI: the frozen OI from month t read at month t + lead.
At the end the checkpoint chosen on the validation years and the model of the last step are both
scored on every cell of those years, by lead and by year, temperature and salinity separately.
The model, its recipes, the loss, the optimiser and the schedule are 60's.
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

from ocean_tokenizer import local_scale
from ocean_tokenizer.latent_experiment import masked_losses
from ocean_tokenizer.synthetic_latent_experiment import EVAL_SEED
from ocean_tokenizer.forecast_scenes import LEADS
from ocean_tokenizer.grid_forecast_scenes import GridForecastScenes
from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
from ocean_tokenizer.grid_kriging_ocean import BACKGROUNDS, KRIGING, GridKrigingConfig, GridKrigingModel, GridSpec
from ocean_tokenizer.reconstruction_metrics import evaluate_reconstruction, inverse_standardize
from ocean_tokenizer.innovation_ocean import RECIPES
from ocean_tokenizer.point_baselines import CH, Scores, band_of_levels

OUTPUT = ROOT / "outputs/grid_ka_20261019"
FORECAST = ROOT / "outputs/forecast_20261019"
#: the observation-operator recipes need operator caches these scenes do not build
OPERATOR_RECIPES = ("operator_direct", "operator_update", "all_modules")


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["dense", "soft_moe", "local_transformer"], default="dense")
    p.add_argument("--recipe", choices=[r for r in RECIPES if r not in OPERATOR_RECIPES], default="local_control")
    p.add_argument("--train-years", type=int, default=4, help="training window: the last N years before 2004")
    p.add_argument("--cohort", default="cesm2_long_uniform", help="synthetic cohort, by file name without .nc")
    p.add_argument("--scale", choices=["anom_cell", "anom_month", "res_cell", "res_month"], default=None,
                   help="table of 80_local_scale_tables.py the local branch's units are taken from")
    p.add_argument("--scale-tables", type=Path, default=ROOT / "outputs/long_record_20261017/scale_tables.npz")
    p.add_argument("--leads", nargs="+", type=int, default=list(LEADS), help="forecast leads in months")
    p.add_argument("--val-years", nargs="+", type=int, default=[2004, 2005],
                   help="target years that choose the checkpoint and are scored at the end")
    p.add_argument("--lagged-surface", type=Path, default=FORECAST / "lagged_surface.npz")
    p.add_argument("--surface-grids", type=Path, default=OUTPUT / "surface_grids.npz")
    p.add_argument("--dataset", choices=["cesm2", "real_global"], default="cesm2")
    p.add_argument("--real-caches", type=Path, default=ROOT / "outputs/real_global_20261019")
    p.add_argument("--real-oi-summary", type=Path,
                   default=ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json")
    p.add_argument("--init-from", type=Path, default=None,
                   help="model weights of another run's checkpoint (any dataset); a new optimiser and schedule")
    p.add_argument("--kriging", choices=KRIGING, default="ka_flow")
    p.add_argument("--background", choices=BACKGROUNDS, default="joint",
                   help="prior: innovations against a surface-only prior state; joint: one state reads everything")
    p.add_argument("--state-channels", type=int, default=128)
    p.add_argument("--encoder-channels", nargs="+", type=int, default=[64, 128, 192, 256])
    p.add_argument("--evolution-channels", nargs="+", type=int, default=[64, 128, 192])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--max-shift-m", type=float, default=100.)
    p.add_argument("--covariance-rank", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--latents", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--query-blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--slots-per-expert", type=int, default=2)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--queries", type=int, default=1024, help="individual depth queries per optimizer step")
    p.add_argument("--context-profiles", type=int, default=6080,
                   help="6080 retains every source profile, matching the old model input pool")
    p.add_argument("--eval-context-profiles", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--analysis-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--nll-weight", type=float, default=.02)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-cells", type=int, default=8000)
    p.add_argument("--eval-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--freeze-analysis", action="store_true", default=True, help="exact validation-tuned OI is always fixed")
    p.add_argument("--surface", action="store_true", help="explicit synthetic satellite arm, requiring matching cache")
    p.add_argument("--satellite-cache", type=Path)
    p.add_argument("--oi-summary", type=Path)
    p.add_argument("--load-checkpoint", type=Path, default=None)
    p.add_argument("--warm-start", action="store_true", help="load model weights and restart the optimizer/schedule")
    p.add_argument("--resume-training", action="store_true", help="restore optimizer, scheduler, sampling and torch RNG state")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--analysis-only", action="store_true", help="evaluate frozen exact OI without neural correction")
    p.add_argument("--latent-off", action="store_true", help="local and query-MLP control without shared observation latent")
    p.add_argument("--local-off", action="store_true", help="retain original OI, disable added learned local evidence path")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--development", action="store_true", help="score all matched previously used 2005 test points after freezing")
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


def selection_score(scores):
    """Old 62 checkpoint criterion: mean per-variable standardized RMSE."""
    return float(scores["macro_z"])


def diagnostic_summary(value):
    value = value.detach().float().cpu()
    if value.numel() <= 64:
        return value.tolist()
    return {"shape": list(value.shape), "mean": float(value.mean()),
            "sd": float(value.std(unbiased=False)), "min": float(value.min()), "max": float(value.max())}


def main():
    args = arguments()
    if args.resume_training and (args.load_checkpoint is None or args.warm_start or args.evaluate_only):
        raise ValueError("--resume-training requires --load-checkpoint and excludes warm-start/evaluate-only")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.val_cells != 8000:
        raise ValueError("matched synthetic protocol requires the exact old 8000-cell validation cap")
    if args.development:
        raise ValueError("--development is replaced by the forecast evaluation of --val-years")
    if args.load_checkpoint and (args.resume_training or args.evaluate_only):
        saved = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        args.seed = saved["seed"]
        args.surface = saved["training"]["surface"]
        if args.surface and args.satellite_cache is None:
            args.satellite_cache = Path(saved["training"]["satellite_cache"])
        # the data a checkpoint was trained on
        args.cohort, args.train_years = saved["training"]["cohort"], saved["training"]["train_years"]
        args.scale = saved["training"]["scale"]
        args.leads, args.val_years = saved["training"]["leads"], saved["training"]["val_years"]
        args.lagged_surface = Path(saved["training"]["lagged_surface"])
        args.surface_grids = Path(saved["training"]["surface_grids"])
        if args.scale is not None:
            args.scale_tables = Path(saved["training"]["scale_tables"])
        del saved
    if args.scale is not None:
        raise ValueError("the gridded model works in per-depth z units; --scale is not available")
    if not args.surface:
        raise ValueError("the gridded model reads the surface grids; pass --surface and its caches")
    started = time.time()
    out = args.output or OUTPUT / args.tag
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
    if args.oi_summary is None:
        args.oi_summary = ROOT/"outputs/audit/synthetic"/f"fixed_baselines_{args.cohort}"/"summary.json"
    if args.dataset == "real_global":
        data = RealGridForecastScenes(ROOT, args.device, caches=args.real_caches, oi_summary=args.real_oi_summary,
                                      leads=args.leads)
    else:
        data = GridForecastScenes(ROOT, args.device, surface_grids=args.surface_grids, leads=args.leads,
                                  lagged_surface=args.lagged_surface if args.surface else None,
                                  train_years=args.train_years, surface=args.surface, satellite_cache=args.satellite_cache,
                                  oi_summary=args.oi_summary, cohort=args.cohort)
    split_of_year = getattr(data, "split_of_year", lambda year: {2004: "validation", 2005: "development"}.get(year))
    grid = GridSpec(lat0=float(data.grid_lat[0]), dlat=float(data.grid_lat[1] - data.grid_lat[0]),
                    lon0=float(data.grid_lon[0]), dlon=float(data.grid_lon[1] - data.grid_lon[0]),
                    n_lat=len(data.grid_lat), n_lon=len(data.grid_lon))
    if not (np.allclose(np.diff(data.grid_lat), grid.dlat) and np.allclose(np.diff(data.grid_lon), grid.dlon)
            and abs(grid.dlon * grid.n_lon - 360.) < 1e-6):
        raise ValueError("the surface grids are not a regular global latitude-longitude grid")
    if args.scale is not None:
        with np.load(args.scale_tables, allow_pickle=False) as tables:
            table_metadata = json.loads(str(tables["metadata"].item()))
            if table_metadata["cohort_fingerprint"] != data.metadata["cohort_fingerprint"]:
                raise ValueError("scale tables were fitted on another cohort")
            scale = local_scale.lookup(tables[args.scale], data.c.lat, data.c.lon, data.c.month_index % 12)
        data.metadata["local_scale"] = {"table": args.scale, "scale_by": "query", "loss_units": "network",
            "tables_sha256": hashlib.sha256(Path(args.scale_tables).read_bytes()).hexdigest(),
            **{name: table_metadata[name] for name in ("fit", "definition", "cell_deg", "month_window", "pseudo", "floor")}}
        data.configure(None, scale, "query")
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT/"src/ocean_tokenizer/latent_ocean.py", ROOT/"src/ocean_tokenizer/latent_experiment.py", ROOT/"src/ocean_tokenizer/synthetic_latent_experiment.py", ROOT/"src/ocean_tokenizer/reconstruction_metrics.py", ROOT/"src/ocean_tokenizer/innovation_ocean.py", ROOT/"src/ocean_tokenizer/innovation_scenes.py", ROOT/"src/ocean_tokenizer/profile_alignment.py", ROOT/"src/ocean_tokenizer/correlated_ocean_update.py", ROOT/"src/ocean_tokenizer/ocean_observation_operator.py", ROOT/"src/ocean_tokenizer/radius_scale_scenes.py", ROOT/"src/ocean_tokenizer/local_scale.py", ROOT/"src/ocean_tokenizer/long_record_scenes.py", ROOT/"src/ocean_tokenizer/query_scaled_innovation.py", ROOT/"src/ocean_tokenizer/forecast_scenes.py", ROOT/"src/ocean_tokenizer/grid_forecast_scenes.py", ROOT/"src/ocean_tokenizer/grid_kriging_ocean.py", ROOT/"src/ocean_tokenizer/kriging_attention.py", ROOT/"src/ocean_tokenizer/real_forecast_scenes.py", ROOT/"src/ocean_tokenizer/real_global.py")}
    analysis, anchor_analysis = data.analysis(), data.analysis()
    bands = band_of_levels(data.levels)
    cfg = GridKrigingConfig(grid=grid.as_tuple(), n_levels=len(data.levels), n_query_features=58,
            state_channels=args.state_channels, encoder_channels=tuple(args.encoder_channels),
            evolution_channels=tuple(args.evolution_channels), hidden=args.hidden, max_lead=max(args.leads),
            kriging=args.kriging, background=args.background,
            kriging_ell_km=[[data.oi_selection[ch][b]["L_km"] for b in bands] for ch in CH],
            kriging_gamma=[[data.oi_selection[ch][b]["gamma"] for b in bands] for ch in CH])
    checkpoint = None
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        if any(checkpoint["data"][name] != data.metadata[name]
               for name in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")):
            raise ValueError("checkpoint data mismatch")
        cfg = GridKrigingConfig(**checkpoint["config"])
        if args.resume_training:
            if "continuation" not in checkpoint:
                raise ValueError("checkpoint has no complete continuation state; use --warm-start instead")
            if checkpoint.get("source_sha256") != source_hashes:
                raise ValueError("source changed since checkpoint; use --warm-start for a new experiment")
            for name in ("seed", "steps", "queries", "context_profiles", "eval_context_profiles",
                         "lr", "analysis_lr", "weight_decay", "nll_weight", "val_every", "val_cells",
                         "eval_chunk", "freeze_analysis", "analysis_only", "amp", "latent_off", "local_off", "surface", "recipe", "max_shift_m", "covariance_rank",
                         "kriging", "background", "state_channels", "encoder_channels", "evolution_channels", "hidden"):
                setattr(args, name, checkpoint["training"][name])
    model = GridKrigingModel(cfg).to(data.device)
    if args.init_from is not None:
        if checkpoint is not None:
            raise ValueError("--init-from starts a new run; it excludes --load-checkpoint")
        source = torch.load(args.init_from, map_location="cpu", weights_only=True)
        # the kriging length scales and error ratios of the configuration only initialise
        # parameters (or the ka_oi buffers), which the loaded state replaces; the rest must match
        mine, theirs = cfg.to_dict(), GridKrigingConfig(**source["config"]).to_dict()
        differs = {k for k in mine if mine[k] != theirs[k]} - {"kriging_ell_km", "kriging_gamma"}
        if differs:
            raise ValueError(f"--init-from: the model configuration differs in {sorted(differs)}")
        model.load_state_dict(source["model"], strict=True)
        data.metadata["init_from"] = {"path": str(args.init_from), "step": source["step"],
                                      "dataset": source["data"].get("cohort", source["data"].get("cohort_fingerprint")),
                                      "sha256": hashlib.sha256(Path(args.init_from).read_bytes()).hexdigest()}
        del source
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
    val = data.forecast_months(args.val_years, args.val_cells)
    identity = {"validation": data.identity_check(
        "validation", [ev for ev in val if ev.lead == 0 and split_of_year(2000 + ev.month // 12) == "validation"])}
    geometries, fixed_baselines = {}, {}

    def key(ev):
        return ev.month, ev.lead, len(ev.level)

    def geometry(ev):
        k = key(ev)
        if k not in geometries:
            geometries[k] = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
        return geometries[k]

    def geom_chunk(g, start, stop):
        return {name: value[start:stop] for name, value in g.items()}

    @torch.no_grad()
    def evaluate(evs, with_model=True, save_arrays=None, full_cells=False):
        """Scores by lead and by target year, the model's and persisted OI's on the same cells."""
        model.eval()
        leads = sorted({ev.lead for ev in evs})
        years = sorted({2000 + ev.month // 12 for ev in evs})
        new = lambda: Scores(data.levels, data.norm.std)
        scores, base_scores = {lead: new() for lead in leads}, {lead: new() for lead in leads}
        year_scores = {(lead, y): new() for lead in leads for y in years}
        year_base = {(lead, y): new() for lead in leads for y in years}
        arrays = {name: [] for name in ("mean", "std", "target", "baseline", "month", "lead", "profile", "level")}
        diagnostics = []
        for ev in evs:
            # every cell of 96 month-leads would hold ~8 GB of neighbour searches: those are not kept
            g = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level) if full_cells else geometry(ev)
            base_chunks, pred_chunks, std_chunks = [], [], []
            state = None
            context_seed = [EVAL_SEED, ev.month]
            for start in range(0, len(ev.level), args.eval_chunk):
                stop = min(start+args.eval_chunk, len(ev.level))
                rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                scene = data.scene(ev.source, rows, lev, analysis if with_model else anchor_analysis,
                    args.eval_context_profiles or args.context_profiles, context_seed, geom_chunk(g, start, stop))
                if with_model and not args.analysis_only:
                    with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                        enabled=args.amp and data.device.type == "cuda"):
                        if state is None:
                            state = model.encode(scene)
                        prediction = model.decode(scene, state)
                    p = data.to_z(prediction["mean"].float(), rows, lev)
                    sigma = data.to_z(prediction["std"].float(), rows, lev)
                    if not torch.isfinite(p).all() or not torch.isfinite(sigma).all():
                        raise FloatingPointError("nonfinite evaluation output")
                    if start == 0 and ev.lead == leads[0]:
                        diagnostics.append({name: diagnostic_summary(value)
                            for name, value in prediction["diagnostics"].items()})
                else:
                    p = data.to_z(scene["baseline"], rows, lev)
                    sigma = torch.full_like(p, .6)
                pred_chunks.append(p.cpu().numpy())
                std_chunks.append(sigma.cpu().numpy())
                if key(ev) not in fixed_baselines:
                    # the frozen OI of the input month: persisted OI at a lead
                    base_chunks.append(data.to_z(scene["baseline"], rows, lev).cpu().numpy())
            p, sigma = np.concatenate(pred_chunks), np.concatenate(std_chunks)
            if key(ev) not in fixed_baselines:
                fixed_baselines[key(ev)] = np.concatenate(base_chunks)
            baseline = fixed_baselines[key(ev)]
            tgt, _ = data.targets(ev.target[ev.profile], ev.level)
            y = tgt.cpu().numpy()
            year = 2000 + ev.month // 12
            for j, ch in enumerate(CH):
                for model_scores, oi_scores in ((scores[ev.lead], base_scores[ev.lead]),
                                                (year_scores[ev.lead, year], year_base[ev.lead, year])):
                    model_scores.add(ch, p[:, j], y[:, j], ev.level)
                    oi_scores.add(ch, baseline[:, j], y[:, j], ev.level)
            if save_arrays is not None:
                for name, value in (("mean", p), ("std", sigma), ("target", y), ("baseline", baseline),
                                    ("month", np.full(len(ev.level), ev.month, dtype=np.int16)),
                                    ("lead", np.full(len(ev.level), ev.lead, dtype=np.int8)),
                                    ("profile", ev.target[ev.profile]), ("level", ev.level)):
                    arrays[name].append(value)
        by_lead = {str(lead): {"scores": scores[lead].result(), "baseline_scores": base_scores[lead].result(),
                               "by_year": {str(y): {"scores": year_scores[lead, y].result(),
                                                    "baseline_scores": year_base[lead, y].result()} for y in years}}
                   for lead in leads}
        oi_identity = {}
        if 0 in leads:
            # lead 0 is the usual evaluation: its OI must replay 44 on the cells 44 scored
            for y in years:
                split = split_of_year(y)
                if split == ("development" if full_cells else "validation"):
                    oi_identity[split] = data.baseline_identity_check(split, by_lead["0"]["by_year"][str(y)]["baseline_scores"])
        mean = lambda pick: float(np.mean([pick(by_lead[str(lead)]["scores"]) for lead in leads]))
        pooled = {"macro_z": mean(lambda s: s["macro_z"]),
                  **{ch: {"rmse_physical": mean(lambda s: s[ch]["rmse_physical"])} for ch in CH}}
        if save_arrays is not None:
            np.savez_compressed(save_arrays, **{name: np.concatenate(v) for name, v in arrays.items()},
                normalization_scale=np.stack([data.norm.std[ch] for ch in CH], -1),
                normalization_offset=np.stack([data.norm.mean[ch] for ch in CH], -1))
        model.train()
        if full_cells and data.device.type == "cuda":
            torch.cuda.empty_cache()
        return {"scores": pooled, "by_lead": by_lead, "oi_identity_check": oi_identity,
                "cells": "every cell" if full_cells else f"{args.val_cells} a target month",
                "selection_score": "mean over leads of the per-variable standardized RMSE",
                "diagnostics": diagnostics}

    print(json.dumps({"tag": args.tag, "config": cfg.to_dict(), "seed": args.seed,
                      "neural_parameters": 0 if args.analysis_only else sum(p.numel() for p in model.parameters()),
                      "train_analysis": not args.freeze_analysis, "data_preparation_s": time.time()-started}, ensure_ascii=False), flush=True)
    history = []
    forecast_last = None
    best = {"score": float("inf"), "step": 0, "state": None}
    initial = None
    prior_runtime = 0.
    if args.analysis_only and not args.evaluate_only:
        initial = evaluate(val)
        checkpoint = {"step": 0}
    elif not args.evaluate_only:
        def model_state(step, validation):
            return {"format_version": 2, "model": cpu_state(model), "analysis": cpu_state(analysis),
                    "anchor_analysis": cpu_state(anchor_analysis),
                    "config": cfg.to_dict(), "seed": args.seed, "step": step,
                    "data": data.metadata,
                    "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "anchor_first_guess": data.fg_checkpoint,
                    "normalization": {"mean": {ch: torch.as_tensor(data.norm.mean[ch]) for ch in CH},
                                      "std": {ch: torch.as_tensor(data.norm.std[ch]) for ch in CH}},
                    "source_sha256": source_hashes, "validation": validation}
        if not args.resume_training:
            initial = evaluate(val)
            print("initial validation physical RMSE " + json.dumps({ch: initial["scores"][ch]["rmse_physical"] for ch in CH}), flush=True)
            initial_state = model_state(0, initial)
            best = {"score": selection_score(initial["scores"]), "step": 0, "state": initial_state}
            atomic_save(initial_state, out/f"best_seed{args.seed}.pt")
        no_decay = lambda n: n.startswith("kriging.log_") or n == "log_decay_months"
        scales = [p for n, p in model.named_parameters() if no_decay(n)]
        others = [p for n, p in model.named_parameters() if not no_decay(n)]
        groups = [] if args.analysis_only else [{"params": others, "lr": args.lr, "weight_decay": args.weight_decay},
                                                {"params": scales, "lr": args.lr, "weight_decay": 0.}]
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
            target, valid = data.network_targets(rows, levels)
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
                          "validation_mean_standardized_rmse": selection_score(sc),
                          "validation_temperature_rmse_C": sc["TEMP"]["rmse_physical"], "validation_salinity_rmse_PSU": sc["SALT"]["rmse_physical"],
                          "by_lead": {lead: {"macro_z": v["scores"]["macro_z"], "TEMP": v["scores"]["TEMP"]["rmse_physical"],
                                             "SALT": v["scores"]["SALT"]["rmse_physical"]} for lead, v in ev["by_lead"].items()},
                          "elapsed_s": prior_runtime+time.time()-started}
                history.append(record)
                improved = selection_score(sc) < best["score"]
                if improved:
                    state = model_state(step, ev)
                    best = {"score": selection_score(sc), "step": step, "state": state}
                    atomic_save(state, out/f"best_seed{args.seed}.pt")
                print("validation " + json.dumps(record) + (" *" if improved else ""), flush=True)
            if step % args.checkpoint_every == 0 or step % args.val_every == 0 or step == args.steps:
                save_last(step)
        full = data.forecast_months(args.val_years, 0)
        # the model of the last step, before the chosen checkpoint replaces it
        forecast_last = evaluate(full, full_cells=True)
        model.load_state_dict(best["state"]["model"], strict=True)
        analysis.load_state_dict(best["state"]["analysis"], strict=True)
        checkpoint = best["state"]
    elif checkpoint is None:
        raise ValueError("--evaluate-only requires --load-checkpoint")
    final_validation = evaluate(val)
    full = data.forecast_months(args.val_years, 0)
    development = [ev for ev in full if ev.lead == 0 and split_of_year(2000 + ev.month // 12) == "development"]
    if development:
        identity["development"] = data.identity_check("development", development)
    forecast_best = evaluate(full, save_arrays=out/f"forecast_seed{args.seed}.npz", full_cells=True)
    summary = {"completed_steps": checkpoint["training"]["steps"] if args.evaluate_only else args.steps,
               "tag": args.tag, "seed": args.seed, "config": cfg.to_dict(), "data": data.metadata,
               "train_years": args.train_years, "training_months": len(data.months("train")),
               "loss_units": "per-depth z" if args.scale is None else "local scale " + args.scale,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated()/2**20 if data.device.type == "cuda" else None,
               "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "params": 0 if args.analysis_only else sum(p.numel() for p in model.parameters()),
               "architecture": "analysis_only" if args.analysis_only else "grid_kriging",
               "kriging": cfg.kriging,
               "analysis_params": sum(p.numel() for p in analysis.parameters()),
               "selection_metric": "mean temperature/salinity standardized RMSE, using per-depth training standard deviations; validation only, identical to old 62 checkpoint criterion",
               "normalization": {"mean": {ch: data.norm.mean[ch].tolist() for ch in CH},
                                 "std": {ch: data.norm.std[ch].tolist() for ch in CH}},
               "best_step": checkpoint["step"], "initial": initial, "validation": final_validation,
               "leads": sorted(set(args.leads)), "val_years": args.val_years,
               "forecast": {"best": forecast_best, "last": forecast_last},
               "history": history, "runtime_s": (prior_runtime if not args.evaluate_only else 0.)+time.time()-started,
               "source_sha256": source_hashes}
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
        if previous_summary.get("forecast", {}).get("last") is not None:
            summary["forecast"]["last"] = previous_summary["forecast"]["last"]
    summary["identity_check"] = identity
    temporary = done.with_name(done.name+".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, done)
    print("finished " + json.dumps({"summary": str(done), "best_step": summary["best_step"],
          "validation": {ch: summary["validation"]["scores"][ch]["rmse_physical"] for ch in CH},
          "runtime_s": summary["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
