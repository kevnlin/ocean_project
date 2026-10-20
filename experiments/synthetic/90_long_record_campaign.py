"""Queue of the training-window x model-size grid on the 1964-2005 CESM2 cohort.

Every run is ``89_long_record_reconstruction.py`` with Zihao's shared T/S
alignment model (recipe ``profile_shared``, Local Transformer, the settings of
``61_innovation_campaign.py``), Argo plus surface inputs (SST, SSS, steric
SSH) and the local OI-error scale by depth x cell (``--scale res_cell``),
15,000 steps, checkpoint chosen on 2004 and 2005 scored at the end of the run.
Two things vary:

  training window   4, 10, 20 or 40 years ending in 2003 (``--train-years``)
  model size        s: 64 latent slots, width 192 (the multi-modal setup)
                    m: 128 slots, width 256
                    l: 256 slots, width 384
                    Heads (8), blocks (6) and query blocks (2) stay 61's.

A run is named ``y<years>_<size>_s<seed>``; with ``--steps`` other than 15,000
(the follow-up of 2026-10-17 trains 30,000) the name starts with ``k<thousands>_``
and the learning-rate schedule spans the longer run. Seeds are outermost and, within a
seed, the 4- and 40-year windows come first, so the two ends of the grid
finish before its middle. Interrupted runs continue from their last state; a
failed run is retried twice. A second invocation exits at once while one is
running. ``outputs/long_record_20261017/chain.sh`` prepares the data and starts
this; ``watchdog.sh`` there restarts the chain if it dies.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/89_long_record_reconstruction.py"
STEPS = 15000
YEARS = (4, 40, 10, 20)
SIZES = {"s": {"latents": 64, "width": 192}, "m": {"latents": 128, "width": 256},
         "l": {"latents": 256, "width": 384}}
SEEDS = (1234, 1235, 1236)
#: 61_innovation_campaign.py's settings for every recipe; latents and width vary here
REFERENCE = ["--recipe", "profile_shared", "--variant", "local_transformer", "--blocks", "6",
             "--query-blocks", "2", "--heads", "8", "--experts", "4", "--steps", str(STEPS),
             "--context-profiles", "6080", "--queries", "1024", "--val-every", "1000",
             "--checkpoint-every", "500", "--amp"]

_spec = importlib.util.spec_from_file_location("radius_scale_campaign", ROOT / "experiments/synthetic/82_radius_scale_campaign.py")
_queue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_queue)
write_json, slots, queue = _queue.write_json, _queue.slots, _queue.queue


def job(years, size, seed, output=OUTPUT, steps=STEPS):
    tag = f"y{years}_{size}_s{seed}" if steps == STEPS else f"k{steps // 1000}_y{years}_{size}_s{seed}"
    folder = output / tag
    reference = list(REFERENCE)
    reference[reference.index("--steps") + 1] = str(steps)
    command = [sys.executable, "-u", str(RUNNER), *reference,
               "--latents", str(SIZES[size]["latents"]), "--width", str(SIZES[size]["width"]),
               "--train-years", str(years), "--cohort", "cesm2_long_uniform",
               "--scale", "res_cell", "--scale-tables", str(output / "scale_tables.npz"),
               "--surface", "--satellite-cache", str(output / "satellites.npz"),
               "--development", "--seed", str(seed), "--tag", tag, "--output", str(folder)]
    return {"tag": tag, "years": years, "size": size, "seed": seed, "steps": steps,
            "output": str(folder), "command": command}


def jobs(years=YEARS, sizes=tuple(SIZES), seeds=SEEDS, output=OUTPUT, steps=STEPS):
    return [job(y, s, seed, output, steps) for seed in seeds for y in years for s in sizes]


def summary_of(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def completed(item):
    summary = summary_of(item)
    if summary is None:
        return False
    if (summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"]
            or summary.get("train_years") != item["years"] or "development" not in summary):
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--sizes", nargs="+", choices=tuple(SIZES), default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--stagger", type=float, default=180., help="seconds between two launches")
    parser.add_argument("--steps", type=int, default=STEPS, help="training steps of every run")
    parser.add_argument("--name", default="queue", help="names this invocation's status file")
    args = parser.parse_args()
    lock = (OUTPUT / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    for needed in ("scale_tables.npz", "satellites.npz"):
        if not (OUTPUT / needed).exists():
            raise SystemExit(f"prepare {OUTPUT / needed} first")
    requested = jobs(args.years, args.sizes, args.seeds, OUTPUT, args.steps)
    places = slots([int(g) for g in args.gpus.split(",")], args.per_gpu)
    pending = [j for j in requested if not completed(j)]
    # 82's queue judges a finished run by its own completion rule
    _queue.completed = completed
    started = time.time()

    def status(extra):
        write_json(OUTPUT / f"status_{args.name}.json", {"requested": len(requested),
            "completed": sum(completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    manifest = "campaign.json" if args.steps == STEPS else f"campaign_k{args.steps // 1000}.json"
    write_json(OUTPUT / manifest, {"steps": args.steps, "years": args.years, "sizes": SIZES,
        "seeds": args.seeds, "reference": REFERENCE, "runner": str(RUNNER.relative_to(ROOT)),
        "validation_year": 2004, "test_year": 2005, "jobs": [j["tag"] for j in requested]})
    print(json.dumps({"requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        queue(pending, places, "train", args.stagger, status)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
