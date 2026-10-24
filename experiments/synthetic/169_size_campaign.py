"""Queue of the size study (spec ``docs/superpowers/specs/2026-10-23-ka-size-study-design.md``):
KA-Ocean (``101``, method A) on ``cesm2_vlong_uniform`` with a 100-year training window (1904-2003),
200,000 steps, leads 0/1/3/6, the checkpoint chosen on 2004 and 2005 held out, at three sizes, one
width factor on every channel count of A:

  y100_x1      9.4M parameters (A's size)
  y100_x1375  17.7M
  y100_x175   28.6M

Seeds are outermost, so every size has seed 1234 before a second seed starts. Every run keeps a model
snapshot every 5,000 steps; its resume point is every 500. The queue (a GPU memory check before each
launch, retries from the last resume point, finished runs skipped) is ``150``'s.

  .venv/bin/python experiments/synthetic/169_size_campaign.py --smoke --seeds 1234 --gpus 0,2,3
  .venv/bin/python experiments/synthetic/169_size_campaign.py --gpus 0,2,3,4,6 --per-gpu 1
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/size_20261023"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS, SMOKE_STEPS = 200000, 600
SEEDS = (1234, 1235, 1236)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_queue = _script("150_wang_campaign")
VLONG, VLONG_NEEDS = _queue.cohort_options("cesm2_vlong_uniform", 100)
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "--test-years", "2005", "--kriging", "ka_flow"]
#: arm -> width factor on A's channel counts
FACTORS = {"y100_x1": 1., "y100_x1375": 1.375, "y100_x175": 1.75}


def size_options(factor):
    width = lambda channels: [str(round(c * factor)) for c in channels]
    return ["--state-channels", *width([128]), "--encoder-channels", *width([64, 128, 192, 256]),
            "--evolution-channels", *width([64, 128, 192]), "--hidden", *width([256])]


def job(arm, seed, smoke=False):
    steps, every = (SMOKE_STEPS, 300) if smoke else (STEPS, 2000)
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    command = [sys.executable, "-u", str(RUNNER), *COMMON, *VLONG, *size_options(FACTORS[arm]),
               "--steps", str(steps), "--val-every", str(every), "--checkpoint-every", str(min(every, 500)),
               "--keep-every", str(300 if smoke else 5000),
               "--seed", str(seed), "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in VLONG_NEEDS], "command": command}


def jobs(arms, seeds, smoke=False):
    return [job(arm, seed, smoke) for seed in seeds for arm in arms]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(FACTORS), default=list(FACTORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,2,3,4,6")
    parser.add_argument("--per-gpu", type=int, default=1, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=40000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600.)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--stagger", type=float, default=120., help="seconds between two launches")
    args = parser.parse_args()
    name = "smoke" if args.smoke else "queue"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / f"{name}.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    requested = jobs(args.arms, args.seeds, args.smoke)
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not _queue.completed(j)]
    started = time.time()

    def status(extra):
        _queue.write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(_queue.completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        _queue.write_json(OUTPUT / "campaign.json", {"seeds": args.seeds, "steps": STEPS, "factors": FACTORS,
            "options": {arm: COMMON + VLONG + size_options(f) for arm, f in FACTORS.items()},
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested]})
    print(json.dumps({"requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        _queue.queue(pending, places, args.stagger, status, min_free_mb=args.min_free_mb,
                     retry_after=args.retry_after, attempts_allowed=args.attempts)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
