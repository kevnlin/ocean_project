"""Queue of the FuXi comparison (spec ``docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md``):
a FuXi-Ocean-style model (``101 --backbone fuxi --kriging none``) trained as the size study trained KA-Ocean
x1.375 (``169``): ``cesm2_vlong_uniform`` with a 100-year window (1904-2003), 200,000 steps, leads 0/1/3/6,
the checkpoint chosen on 2004 and 2005 held out, seeds 1234-1236. The state (176 channels) and the read-out
(hidden 352) are x1.375's; the backbone's sizes are the config's defaults (17.5M parameters in all). No
snapshots are kept. The queue (a GPU memory check before each launch, retries from the last resume point,
finished runs skipped) is ``150``'s; with ``--min-free-mb 80000`` a run starts only on an open GPU.

  .venv/bin/python experiments/synthetic/172_fuxi_campaign.py --smoke --seeds 1234 --min-free-mb 80000
  nohup setsid .venv/bin/python experiments/synthetic/172_fuxi_campaign.py --min-free-mb 80000 \
      > outputs/fuxi_20261026/queue.log 2>&1 < /dev/null &
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
OUTPUT = ROOT / "outputs/fuxi_20261026"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS, SMOKE_STEPS = 200000, 600
SEEDS = (1234, 1235, 1236)
ARM = "y100_fuxi"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_queue = _script("150_wang_campaign")
VLONG, VLONG_NEEDS = _queue.cohort_options("cesm2_vlong_uniform", 100)
#: 169's options with the FuXi-style backbone in place of ka_flow, at x1.375's state and read-out
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "--test-years", "2005", "--backbone", "fuxi", "--kriging", "none",
          "--state-channels", "176", "--hidden", "352"]


def job(seed, smoke=False):
    steps, every = (SMOKE_STEPS, 300) if smoke else (STEPS, 2000)
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{ARM}_s{seed}"
    command = [sys.executable, "-u", str(RUNNER), *COMMON, *VLONG,
               "--steps", str(steps), "--val-every", str(every), "--checkpoint-every", str(min(every, 500)),
               "--seed", str(seed), "--tag", f"{ARM}_s{seed}", "--output", str(folder)]
    return {"tag": f"{ARM}_s{seed}", "arm": ARM, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in VLONG_NEEDS], "command": command}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="candidates; a launch waits for free memory")
    parser.add_argument("--per-gpu", type=int, default=1, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=80000, help="free GPU memory a launch needs")
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
    requested = [job(seed, args.smoke) for seed in args.seeds]
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not _queue.completed(j)]
    started = time.time()

    def status(extra):
        _queue.write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(_queue.completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        _queue.write_json(OUTPUT / "campaign.json", {"seeds": args.seeds, "steps": STEPS, "options": COMMON + VLONG,
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
