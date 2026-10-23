"""Robustness to sparser observations: frozen models on their selection years with fewer input profiles.

For every month of the selection years (CESM2 2004-2005, real 2021) a fraction of the input
profiles is kept (1, 1/2, 1/4, 1/8), drawn once per month with a fixed seed, and every model and the
OI baseline see the same subset; queries are unchanged. Models are frozen (no retraining, no test
year is read). Writes ``outputs/sparsity_20261020/<dataset>.json`` with the RMSE by lead, fraction,
model and seed.

  .venv/bin/python experiments/synthetic/124_sparsity_robustness.py cesm2|real [--seeds 1234 1235 1236]
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import sealed_scoring as SS

spec = importlib.util.spec_from_file_location("align117", ROOT / "experiments/synthetic/117_kernel_alignment.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
spec = importlib.util.spec_from_file_location("tables118", ROOT / "experiments/synthetic/118_validation_tables.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

OUTPUT = ROOT / "outputs/sparsity_20261020"
FRACTIONS = (1.0, 0.5, 0.25, 0.125)
LEADS = (0, 6)
MODELS = {"cesm2": {"KA-Ocean": (T.PILOT, "gj_flow"), "softmax read-out": (T.DEV / "grid_ka_70k_20261019", "gj_softmax"),
                    "ARMOR3D-style OI": (T.FORMAL, "c_gp_oi"), "Aardvark-style": (T.PILOT, "gj_none")},
          "real": {"KA-Ocean": (T.REAL, "scratch"), "ARMOR3D-style OI": (T.FORMAL, "r_gp_oi"),
                   "Aardvark-style": (T.FORMAL, "r_none")}}
SEALED_YEARS = {2006, 2007, 2022, 2023, 2024, 2025}


def thinned(events, fraction, seed=20261020):
    """The events with ``fraction`` of each input month's profiles, the same draw for every model."""
    import numpy as np
    out = []
    for ev in events:
        source = np.asarray(ev.source)
        keep = max(1, int(round(fraction * len(source))))
        if keep < len(source):
            rng = np.random.default_rng(seed + int(ev.month - ev.lead))
            source = np.sort(rng.choice(source, keep, replace=False))
        out.append(dataclasses.replace(ev, source=source))
    return out


def evaluate(path, fractions, device):
    import torch
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training = checkpoint["training"]
    if set(training["val_years"]) & SEALED_YEARS:
        raise SystemExit(f"{path}: selection years include a sealed year")
    data = align.scenes(training, device)
    model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(data.device).eval()
    events = [ev for ev in data.forecast_months(list(training["val_years"]), 8000) if ev.lead in LEADS]
    context = training.get("eval_context_profiles") or training["context_profiles"]
    out = {}
    for fraction in fractions:
        collected = SS.collect(model, data, data.analysis(), thinned(events, fraction), context, training["amp"])
        out[str(fraction)] = {}
        for lead, a in collected.items():
            m = SS.metrics(a["mean"], None, a["base"], a["target"], a["level"], data.norm, data.levels)
            out[str(fraction)][str(lead)] = {ch: {"model": m[ch]["model"]["rmse"], "oi": m[ch]["persisted_oi"]["rmse"]}
                                             for ch in ("TEMP", "SALT")}
        print(json.dumps({"run": Path(path).parent.name, "fraction": fraction,
                          "lead0_T": round(out[str(fraction)]["0"]["TEMP"]["model"], 4),
                          "oi_T": round(out[str(fraction)]["0"]["TEMP"]["oi"], 4)}), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{args.dataset}.json"
    results = json.loads(target.read_text()) if target.exists() else {}
    for name, (folder, tag) in MODELS[args.dataset].items():
        for seed in args.seeds:
            key = f"{name}|{seed}"
            if key in results:
                continue
            results[key] = evaluate(folder / f"{tag}_s{seed}" / f"best_seed{seed}.pt", FRACTIONS, args.device)
            target.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
