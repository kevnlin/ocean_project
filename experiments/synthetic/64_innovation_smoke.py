"""Engineering-only actual-data forward/backward check, never a result table."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from ocean_tokenizer.innovation_ocean import InnovationOceanConfig, InnovationOceanModel, RECIPES
from ocean_tokenizer.innovation_scenes import InnovationOceanScenes
from ocean_tokenizer.latent_experiment import masked_losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--queries", type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(4)
    data = InnovationOceanScenes(ROOT, a.device, surface=True,
        satellite_cache=ROOT / "outputs/synthetic_matched_20261007/satellites.npz",
        operator_cache=ROOT / "outputs/innovation_20261007/operators.npz")
    rng = np.random.default_rng(20261007)
    _, source, rows, levels = data.training_draw(rng, a.queries)
    analysis = data.analysis().requires_grad_(False)
    scene = data.scene(source, rows, levels, analysis, 6080, 20261007)
    target, valid = data.targets(rows, levels)
    results = []
    for recipe in RECIPES:
        start = time.time()
        torch.manual_seed(20261007)
        config = InnovationOceanConfig(n_obs_features=62, n_query_features=58, n_local_features=66,
            n_sat_features=56, variant="local_transformer", width=192, n_latents=96, n_blocks=6,
            n_query_blocks=2, n_heads=8, recipe=recipe)
        model = InnovationOceanModel(config).to(data.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for step in range(3):
            with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                                enabled=data.device.type == "cuda"):
                result = model(scene)
                loss, components = masked_losses(result, target, valid, .02)
            if not torch.isfinite(result["mean"]).all() or not torch.isfinite(result["std"]).all():
                raise FloatingPointError(f"Nonfinite predictions: {recipe}")
            if step == 0 and not torch.equal(result["mean"], scene["baseline"]):
                raise ValueError(f"Initialization differs from OI: {recipe}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
                raise FloatingPointError(f"Nonfinite gradients: {recipe}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        row = {"recipe": recipe, "steps": 3, "finite_forward_backward": True,
               "initial_OI_exact": True, "neural_parameters": sum(v.numel() for v in model.parameters()),
               "seconds": time.time()-start}
        results.append(row); print(json.dumps(row), flush=True)
        del model, optimizer, result, loss
        if data.device.type == "cuda":
            torch.cuda.empty_cache()
    out = ROOT / "outputs/innovation_20261007/engineering_smoke.json"
    out.write_text(json.dumps({"status": "engineering_only; not an efficacy experiment",
        "queries": a.queries, "full_source_pool": True, "surface": True, "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
