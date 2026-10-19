"""Queue of the forecast runs: the medium model, 4- and 40-year windows, 3 seeds, 70,000 steps.

Every run is ``94_forecast_reconstruction.py`` with Zihao's shared T/S alignment model (recipe
``profile_shared``, the settings of ``61_innovation_campaign.py``) at the medium size of the
long-record grid (128 latent slots, width 256), Argo plus SST, SSS and steric SSH, the local
OI-error scale by depth x cell, one model for leads 0, 1, 3 and 6 months, the checkpoint chosen
on targets in 2004-2005. A run is named ``f_y<years>_s<seed>``; seeds are outermost. Interrupted
runs continue from their last state, and a second invocation exits at once while one is
running. ``outputs/forecast_20261019/chain3.sh`` starts this.

GPUs 2 and 3 are shared with another user's job of about 60 GB each, so a run starts only on a
GPU with ``--min-free-mb`` free, and a failed run (an out-of-memory crash when that job grows) is
put back after ``--retry-after`` seconds, up to ``--attempts`` times, without stopping the others.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/forecast_20261019"
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/94_forecast_reconstruction.py"
STEPS = 70000
YEARS = (4, 40)
SEEDS = (1234, 1235, 1236)
LEADS = (0, 1, 3, 6)
REFERENCE = ["--recipe", "profile_shared", "--variant", "local_transformer", "--blocks", "6",
             "--query-blocks", "2", "--heads", "8", "--experts", "4", "--latents", "128", "--width", "256",
             "--context-profiles", "6080", "--queries", "1024", "--val-every", "2000",
             "--checkpoint-every", "500", "--amp"]

_spec = importlib.util.spec_from_file_location("radius_scale_campaign", ROOT / "experiments/synthetic/82_radius_scale_campaign.py")
_queue = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_queue)
write_json, slots, training_command = _queue.write_json, _queue.slots, _queue.training_command


def job(years, seed, output=OUTPUT, steps=STEPS):
    tag = f"f_y{years}_s{seed}"
    folder = output / tag
    command = [sys.executable, "-u", str(RUNNER), *REFERENCE, "--steps", str(steps),
               "--train-years", str(years), "--cohort", "cesm2_long_uniform",
               "--scale", "res_cell", "--scale-tables", str(LONG / "scale_tables.npz"),
               "--surface", "--satellite-cache", str(LONG / "satellites.npz"),
               "--lagged-surface", str(output / "lagged_surface.npz"),
               "--leads", *map(str, LEADS), "--val-years", "2004", "2005",
               "--seed", str(seed), "--tag", tag, "--output", str(folder)]
    return {"tag": tag, "years": years, "seed": seed, "steps": steps, "output": str(folder), "command": command}


def jobs(years=YEARS, seeds=SEEDS, output=OUTPUT, steps=STEPS):
    return [job(y, seed, output, steps) for seed in seeds for y in years]


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if (summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"]
            or summary.get("train_years") != item["years"] or "forecast" not in summary):
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


def free_mb(gpu, wait=40.):
    """Free memory of one GPU in MiB, or None if nvidia-smi does not answer in ``wait`` seconds.

    nvidia-smi can hang uninterruptibly on this server, so a call that does not finish is
    abandoned rather than waited for."""
    try:
        process = subprocess.Popen(["timeout", "-s", "KILL", "30", "nvidia-smi", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits", "-i", str(gpu)],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return None
    deadline = time.time() + wait
    while process.poll() is None:
        if time.time() > deadline:
            return None
        time.sleep(.5)
    try:
        return int(process.stdout.read().strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def queue(pending, places, stagger, status, *, min_free_mb, retry_after, attempts_allowed):
    """82's queue, with a free-memory check before a launch and patient retries."""
    pending, running, attempts, last_launch, not_before = list(pending), {}, {}, 0., {}
    while pending or running:
        for place in places:
            if place in running or not pending or time.time() - last_launch < stagger:
                continue
            ready = [j for j in pending if not_before.get(j["tag"], 0) <= time.time()]
            if not ready:
                break
            free = free_mb(place[0])
            if free is None or free < min_free_mb:
                continue
            choice = ready[0]
            pending.remove(choice)
            folder = Path(choice["output"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "train.log").open("a")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(place[0]), "OCEAN_ROOT": str(ROOT),
                   "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(training_command(choice), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[place] = process, log, choice
            attempts[choice["tag"]] = attempts.get(choice["tag"], 0) + 1
            last_launch = time.time()
            print(json.dumps({"phase": "train", "started": choice["tag"], "gpu": place[0], "free_mb": free,
                              "pid": process.pid}), flush=True)
        for place, (process, log, item) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[place]
            if code or not completed(item):
                if attempts[item["tag"]] >= attempts_allowed:
                    raise RuntimeError(f"{item['tag']} failed {attempts_allowed} times; see {item['output']}/train.log")
                not_before[item["tag"]] = time.time() + retry_after
                pending.append(item)
                print(json.dumps({"retry": item["tag"], "exit_code": code, "after_s": retry_after}), flush=True)
            else:
                print(json.dumps({"phase": "train", "completed": item["tag"]}), flush=True)
        status({"running": {f"{gpu}.{worker}": item[2]["tag"] for (gpu, worker), item in running.items()},
                "pending": [item["tag"] for item in pending], "attempts": attempts})
        if pending or running:
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=16000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600., help="seconds before a failed run is retried")
    parser.add_argument("--attempts", type=int, default=12, help="launches of one run before the queue gives up")
    parser.add_argument("--stagger", type=float, default=180., help="seconds between two launches")
    parser.add_argument("--name", default="queue", help="names this invocation's status file")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    for needed in (LONG / "scale_tables.npz", LONG / "satellites.npz", OUTPUT / "lagged_surface.npz"):
        if not needed.exists():
            raise SystemExit(f"prepare {needed} first")
    requested = jobs(args.years, args.seeds, OUTPUT, args.steps)
    places = slots([int(g) for g in args.gpus.split(",")], args.per_gpu)
    pending = [j for j in requested if not completed(j)]
    started = time.time()

    def status(extra):
        write_json(OUTPUT / f"status_{args.name}.json", {"requested": len(requested),
            "completed": sum(completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    write_json(OUTPUT / "campaign.json", {"steps": args.steps, "years": args.years, "seeds": args.seeds,
        "leads_months": list(LEADS), "size": {"latents": 128, "width": 256}, "reference": REFERENCE,
        "runner": str(RUNNER.relative_to(ROOT)), "selection_years": [2004, 2005],
        "jobs": [j["tag"] for j in requested]})
    print(json.dumps({"requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        queue(pending, places, args.stagger, status, min_free_mb=args.min_free_mb,
              retry_after=args.retry_after, attempts_allowed=args.attempts)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
