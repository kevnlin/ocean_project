"""Queue of KA-Ocean on real Argo + satellites with a 10-year training window (2011-2020).

The paper's real-data run (``107`` scratch: ``101_grid_ka_forecast.py --dataset real_global``,
background "joint", kriging attention with the flow kernel, leads 0/1/3/6, every input profile of a
month in the state, checkpoint chosen on 2021) with the training split 2011-2020 instead of
2016-2020 and everything else unchanged: the normalisation, the targets, the 2021 validation cells
and the frozen OI selection are the 2016-2020-fitted ones (``--real-train-split``, the caches of
``162`` on the 2011-2024 cohort, the OI cohort fingerprint of ``106``). 100,000 steps per the
user's decision (2026-10-21), three seeds, so the result can be scored later as a declared extra
pass on the opened 2024 test.

  r_y10   KA-Ocean, --kriging ka_flow, --background joint, training years 2011-2020

As in ``150``: a run starts only on a GPU with ``--min-free-mb`` free, a failed run continues from
its last checkpoint, finished runs are skipped.

  .venv/bin/python experiments/real_data/163_real_y10_campaign.py --gpus 2,3,4 --per-gpu 1 --min-free-mb 40000
  .venv/bin/python experiments/real_data/163_real_y10_campaign.py --smoke --seeds 1234 --gpus 7 --min-free-mb 10000
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
OUTPUT = ROOT / "outputs/real_y10_20261021"
RUNS = OUTPUT / "runs"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS = 100000
#: the 2016-2023 cohort on which the frozen OI selection (106) was made (as 133)
OI_COHORT = "6366666ef8ad4ec33388fd8cc7d7f1666236d8f29158b28bee4818f16584e3a1"
OI_SUMMARY = ROOT / "outputs/audit/real/fixed_baselines_real_global/summary.json"
REFERENCE = ["--dataset", "real_global", "--surface", "--background", "joint", "--kriging", "ka_flow",
             "--context-profiles", "0", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
             "--val-years", "2021", "--val-every", "1000", "--checkpoint-every", "500", "--steps", str(STEPS)]
Y10 = ["--real-caches", str(OUTPUT), "--real-oi-summary", str(OI_SUMMARY), "--real-years", "2011", "2024",
       "--real-train-split", "2011", "2020", "--real-oi-cohort", OI_COHORT]
NEEDS = [OUTPUT / "satellites.npz", OUTPUT / "lagged_surface.npz", OUTPUT / "surface_grids.npz", OI_SUMMARY]
#: arm -> (options, files that must exist before it starts)
ARMS = {"r_y10": (REFERENCE + Y10, NEEDS)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def job(arm, seed, smoke=False):
    options, needs = ARMS[arm]
    steps = STEPS
    folder = (OUTPUT / "smoke" if smoke else RUNS) / f"{arm}_s{seed}"
    if smoke:
        options, steps = options + ["--steps", "40", "--val-every", "20", "--checkpoint-every", "20"], 40
    command = [sys.executable, "-u", str(RUNNER), *options, "--seed", str(seed),
               "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": [str(p) for p in needs], "command": command}


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
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--per-gpu", type=int, default=1, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=40000, help="free GPU memory a launch needs")
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
        write_json(OUTPUT / f"campaign_{args.name}.json", {"seeds": args.seeds, "steps": STEPS,
            "arms": {arm: {"options": options, "needs": [str(p) for p in needs]} for arm, (options, needs) in ARMS.items()},
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested],
            "gpus": args.gpus, "per_gpu": args.per_gpu, "min_free_mb": args.min_free_mb})
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
