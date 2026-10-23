"""Cost of the read-out: wall time of one month's reconstruction with KA, softmax attention or none.

Frozen CESM2 checkpoints (seed 1234) on one selection-year month at lead 0: the encoder runs once and
every query cell of the month is decoded (bf16, as in evaluation). Reports the median over repeats of
the encode and decode times and the parameter counts. Writes ``outputs/efficiency_20261020.json``.

  .venv/bin/python experiments/synthetic/126_efficiency.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("align117", ROOT / "experiments/synthetic/117_kernel_alignment.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)
spec = importlib.util.spec_from_file_location("tables118", ROOT / "experiments/synthetic/118_validation_tables.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

RUNS = {"KA (flow kernel)": (T.PILOT, "gj_flow"), "softmax attention": (T.DEV / "grid_ka_70k_20261019", "gj_softmax"),
        "no read-out": (T.PILOT, "gj_none")}


def main():
    from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
    device = torch.device("cuda")
    out, data = {}, None
    for name, (folder, tag) in RUNS.items():
        checkpoint = torch.load(folder / f"{tag}_s1234" / "best_seed1234.pt", map_location="cpu", weights_only=True)
        training = checkpoint["training"]
        data = data or align.scenes(training, device)
        model = GridKrigingModel(GridKrigingConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device).eval()
        ev = [e for e in data.forecast_months(list(training["val_years"]), 0) if e.lead == 0][0]
        geometry = data.geometry(ev.source, ev.target[ev.profile], 32, ev.level)
        analysis, context = data.analysis(), training.get("eval_context_profiles") or training["context_profiles"]
        scenes = []
        for start in range(0, len(ev.level), 4096):
            stop = min(start + 4096, len(ev.level))
            rows, lev = ev.target[ev.profile[start:stop]], ev.level[start:stop]
            scenes.append(data.scene(ev.source, rows, lev, analysis, context, [20260918, ev.month],
                                     {k: v[start:stop] for k, v in geometry.items()}))
        encode, decode = [], []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for repeat in range(6):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                state = model.encode(scenes[0])
                torch.cuda.synchronize(); t1 = time.perf_counter()
                for scene in scenes:
                    model.decode(scene, state)
                torch.cuda.synchronize(); t2 = time.perf_counter()
                if repeat:                                                   # the first repeat warms up
                    encode.append(t1 - t0)
                    decode.append(t2 - t1)
        out[name] = {"cells": int(len(ev.level)), "encode_ms": 1e3 * float(np.median(encode)),
                     "decode_ms": 1e3 * float(np.median(decode)),
                     "params": int(sum(p.numel() for p in model.parameters()))}
        print(name, json.dumps(out[name]), flush=True)
    target = ROOT / "outputs/efficiency_20261020.json"
    target.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "runs": out}, indent=1) + "\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
