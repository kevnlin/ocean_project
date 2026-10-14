"""How much does a trained model use its distant neighbours? No training.

Scores trained Local Transformer checkpoints on the 2004 validation cells with
the local branch's neighbours cut off beyond a radius, for a range of radii.
Weights, the frozen OI anchor and the shared latent are as trained; only the
neighbours farther than the radius are masked, at evaluation.

Without a radius the script must reproduce the validation score stored in each
checkpoint, and exits if it does not.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from ocean_tokenizer.latent_ocean import LatentOceanConfig, LatentOceanModel
from ocean_tokenizer.point_baselines import CH, Scores
from ocean_tokenizer.radius_scale_scenes import RadiusScaleScenes
from ocean_tokenizer.synthetic_latent_experiment import EVAL_SEED


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="best_seed*.pt of runs with the same inputs")
    p.add_argument("--radii", type=float, nargs="+",
                   default=[100., 150., 200., 250., 300., 400., 500., 750., 1000.])
    p.add_argument("--satellite-cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-chunk", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.checkpoints]
    surface = {bool(s["training"]["surface"]) for s in saved}
    if len(surface) != 1:
        raise ValueError("checkpoints of different inputs cannot share one scene set")
    surface = surface.pop()
    if any(s["training"].get("scale") or s["training"].get("local_radius_km") for s in saved):
        raise ValueError("this test is for checkpoints trained without a scale or a radius")
    if any(s["config"].get("recipe", "local_control") != "local_control" for s in saved):
        raise ValueError("only the plain Local Transformer is supported")
    cache = args.satellite_cache or (Path(saved[0]["training"]["satellite_cache"]) if surface else None)
    data = RadiusScaleScenes(ROOT, args.device, surface=surface, satellite_cache=cache)
    analysis = data.analysis().requires_grad_(False)
    val = data.eval_months("validation", 8000)
    data.identity_check("validation", val)
    amp = lambda enabled: torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                         enabled=enabled and data.device.type == "cuda")
    geometries = {ev.month: data.geometry(ev.source, ev.target[ev.profile], 32, ev.level) for ev in val}
    targets = {ev.month: data.targets(ev.target[ev.profile], ev.level)[0].cpu().numpy() for ev in val}
    radii = [None] + sorted(args.radii)
    results = []
    for path, state in zip(args.checkpoints, saved):
        if any(state["data"][name] != data.metadata[name]
               for name in ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")):
            raise ValueError(f"checkpoint data mismatch: {path}")
        cfg = LatentOceanConfig(**{name: value for name, value in state["config"].items()
                                   if name in LatentOceanConfig.__dataclass_fields__})
        model = LatentOceanModel(cfg).to(data.device).eval()
        model.load_state_dict(state["model"], strict=True)
        training = state["training"]
        context = training["eval_context_profiles"] or training["context_profiles"]
        scores = {radius: Scores(data.levels, data.norm.std) for radius in radii}
        kept = {radius: [0., 0., 0] for radius in radii}      # neighbours, queries with none, queries
        with torch.no_grad():
            for ev in val:
                encoded = None
                predictions = {radius: [] for radius in radii}
                for start in range(0, len(ev.level), args.eval_chunk):
                    stop = min(start + args.eval_chunk, len(ev.level))
                    rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
                    chunk = {name: value[start:stop] for name, value in geometries[ev.month].items()}
                    for radius in radii:
                        data.configure(radius, None)
                        scene = data.scene(ev.source, rows, lev, analysis, context,
                                           [EVAL_SEED, ev.month], chunk)
                        with amp(training["amp"]):
                            if encoded is None:
                                encoded = model.encode(scene)
                            mean = model.decode(scene, encoded)["mean"].float()
                        if not torch.isfinite(mean).all():
                            raise FloatingPointError("nonfinite evaluation output")
                        predictions[radius].append(mean.cpu().numpy())
                        count = scene["local_valid"].any(-1).sum(-1)
                        kept[radius][0] += float(count.sum())
                        kept[radius][1] += float((count == 0).sum())
                        kept[radius][2] += len(count)
                for radius in radii:
                    prediction = np.concatenate(predictions[radius])
                    for j, ch in enumerate(CH):
                        scores[radius].add(ch, prediction[:, j], targets[ev.month][:, j], ev.level)
        stored = state["validation"]["scores"]
        rows = []
        for radius in radii:
            sc = scores[radius].result()
            n, none, total = kept[radius]
            rows.append({"radius_km": radius, "TEMP": sc["TEMP"]["rmse_physical"],
                         "SALT": sc["SALT"]["rmse_physical"], "macro_z": sc["macro_z"],
                         "neighbours_kept": n / total, "share_of_queries_with_none": none / total})
        for ch in CH:
            if abs(rows[0][ch] / stored[ch]["rmse_physical"] - 1.) > 2e-3:
                raise SystemExit(f"{path}: without a radius {ch} is {rows[0][ch]}, the checkpoint "
                                 f"stores {stored[ch]['rmse_physical']}")
        results.append({"checkpoint": str(path), "seed": state["seed"], "step": state["step"],
                        "stored_validation": {ch: stored[ch]["rmse_physical"] for ch in CH},
                        "by_radius": rows})
        print(json.dumps(results[-1]), flush=True)
    data.configure(None, None)
    summary = {"split": "validation 2004, 8,000 cells a month", "surface": surface,
               "note": "radius applied at evaluation only; weights as trained without one",
               "radii_km": radii, "runs": results, "runtime_s": time.time() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, args.output)
    print("wrote " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
