"""Queue of the GLORYS12 simulator runs (spec ``docs/superpowers/specs/2026-10-21-glorys-osse.md``):
every run 100,000 steps, leads 0/1/3/6, a 20-year training window (GLORYS 1997-2016), checkpoint
chosen on the cohort's 2004-2005 (GLORYS 2017-2018), three seeds, on ``glorys12_uniform`` with
its surface caches.

  g_ka        KA-Ocean: ``--kriging ka_flow`` (joint background)
  g_armor     ARMOR3D-style OI: ``--background prior --kriging ka_oi``
  g_none      Aardvark-style: ``--kriging none``
  g_softmax   softmax read-out: ``--kriging softmax``

As in ``150``: a run waits until the cohort, its caches and its OI selection exist, starts only
on a GPU with ``--min-free-mb`` free, a failed run continues from its last checkpoint, finished
runs are skipped. Outputs under ``outputs/glorys_20261021/runs/``.

  .venv/bin/python experiments/synthetic/158_glorys_queue.py --smoke --arms g_ka --seeds 1234 --gpus 7 --min-free-mb 10000
  .venv/bin/python experiments/synthetic/158_glorys_queue.py --gpus 0,1,5,6 --per-gpu 2 --min-free-mb 24000
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/glorys_20261021/runs"
CACHES = ROOT / "outputs/glorys_20261021/glorys12_uniform"
COHORT = "glorys12_uniform"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS = 100000
TRAIN_YEARS = 20
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "2005", "--val-every", "2000", "--checkpoint-every", "500",
          "--steps", str(STEPS), "--cohort", COHORT, "--train-years", str(TRAIN_YEARS), "--context-profiles", "6080",
          "--satellite-cache", str(CACHES / "satellites.npz"), "--lagged-surface", str(CACHES / "lagged_surface.npz"),
          "--surface-grids", str(CACHES / "surface_grids.npz")]
NEEDS = [ROOT / f"data/synthetic_argo/{COHORT}.nc", CACHES / "satellites.npz", CACHES / "lagged_surface.npz",
         CACHES / "surface_grids.npz", ROOT / f"outputs/audit/synthetic/fixed_baselines_{COHORT}/summary.json"]
#: arm -> options
ARMS = {
    "g_ka": ["--kriging", "ka_flow"],
    "g_armor": ["--background", "prior", "--kriging", "ka_oi"],
    "g_none": ["--kriging", "none"],
    "g_softmax": ["--kriging", "softmax"],
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def job(arm, seed, smoke=False):
    options = COMMON + ARMS[arm]
    steps = STEPS
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    if smoke:
        options, steps = options + ["--steps", "40", "--val-every", "20", "--checkpoint-every", "20"], 40
    command = [sys.executable, "-u", str(RUNNER), *options, "--seed", str(seed),
               "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in NEEDS], "command": command}


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"] or "forecast" not in summary:
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


def ready(item):
    return all(Path(p).exists() for p in item["needs"])


def command_for(item):
    last = Path(item["output"]) / f"last_seed{item['seed']}.pt"
    if last.exists():
        return item["command"] + ["--resume-training", "--load-checkpoint", str(last)]
    return list(item["command"])


def free_mb(gpu, wait=40.):
    """Free memory of one GPU in MiB, or None if nvidia-smi does not answer (it can hang here)."""
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
    pending, running, attempts, last_launch, not_before = list(pending), {}, {}, 0., {}
    while pending or running:
        for place in places:
            if place in running or not pending or time.time() - last_launch < stagger:
                continue
            candidates = [j for j in pending if not_before.get(j["tag"], 0) <= time.time() and ready(j)]
            if not candidates:
                break
            free = free_mb(place[0])
            if free is None or free < min_free_mb:
                continue
            choice = candidates[0]
            pending.remove(choice)
            folder = Path(choice["output"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "train.log").open("a")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(place[0]), "PYTHONPATH": str(ROOT / "src"),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(command_for(choice), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
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
                "pending": [item["tag"] for item in pending],
                "waiting_for_data": [item["tag"] for item in pending if not ready(item)], "attempts": attempts})
        if pending or running:
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,5,6")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=24000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600.)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--stagger", type=float, default=120., help="seconds between two launches")
    parser.add_argument("--name", default="queue", help="names this invocation's lock and status files")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / ("smoke.lock" if args.smoke else f"{args.name}.lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    requested = [job(arm, seed, args.smoke) for arm in args.arms for seed in args.seeds]
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not completed(j)]
    started, name = time.time(), "smoke" if args.smoke else args.name

    def status(extra):
        write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        write_json(OUTPUT / f"campaign_{args.name}.json", {"seeds": args.seeds, "steps": STEPS, "cohort": COHORT,
            "train_years": TRAIN_YEARS, "common": COMMON, "arms": ARMS, "needs": [str(p) for p in NEEDS],
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested],
            "spec": "docs/superpowers/specs/2026-10-21-glorys-osse.md"})
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
