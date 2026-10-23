"""Queue of the experiments after the 2026-10-21 group meeting (spec
``docs/superpowers/specs/2026-10-21-wang-feedback-experiments.md``): every run 100,000 steps,
leads 0/1/3/6, checkpoint chosen on 2004-2005, three seeds.

CESM2 arms (``cesm2_long_uniform`` unless noted; the frozen method's options otherwise):

  c_flow100    the method at 100k steps: the comparison basis of every arm below
  c_deep       KA with a Gaussian kernel in a learned feature space (``ka_deep``)
  c_product    the flow kernel times the feature Gaussian (``ka_product``)
  c_multiscale two flow kernels, the second three times longer (``ka_multiscale``)
  c_mv         the flow kernel with temperature and salinity solved jointly (``--multivariate``)
  c_density    density augmentation: each step keeps 1,520-6,080 of the month's input profiles
  c_draws      cohort ``cesm2_long_aug3`` (three draws a month before 2000), 6,080 inputs a step
  c_both       the same cohort with 1,520-6,080 inputs a step
  c_y154       cohort ``cesm2_vlong_uniform``, training window 1850-2003 (154 years)

A cohort arm waits until its caches exist (``44``, ``88``, ``93``, ``100`` on that cohort). As in
``111``: a run starts only on a GPU with ``--min-free-mb`` free, a failed run continues from its last
checkpoint, finished runs are skipped.

  .venv/bin/python experiments/synthetic/150_wang_campaign.py --gpus 0,1,5,6,7 --per-gpu 2
  .venv/bin/python experiments/synthetic/150_wang_campaign.py --smoke --arms c_deep --seeds 1234 --gpus 0
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
OUTPUT = ROOT / "outputs/wang_20261021"
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS = 100000
COMMON = ["--surface", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
          "--val-years", "2004", "2005", "--val-every", "2000", "--checkpoint-every", "500",
          "--steps", str(STEPS), "--kriging", "ka_flow"]
CESM2 = ["--satellite-cache", str(LONG / "satellites.npz"), "--cohort", "cesm2_long_uniform",
         "--train-years", "40", "--context-profiles", "6080"]


def cohort_options(name, train_years):
    """A cohort built by the data stream: its caches live under ``outputs/cohort_<name>``."""
    folder = ROOT / f"outputs/cohort_{name}"
    options = ["--satellite-cache", str(folder / "satellites.npz"), "--lagged-surface", str(folder / "lagged_surface.npz"),
               "--surface-grids", str(folder / "surface_grids.npz"), "--cohort", name,
               "--train-years", str(train_years), "--context-profiles", "6080"]
    needs = [ROOT / f"data/synthetic_argo/{name}.nc", folder / "satellites.npz", folder / "lagged_surface.npz",
             folder / "surface_grids.npz", ROOT / f"outputs/audit/synthetic/fixed_baselines_{name}/summary.json"]
    return options, needs


AUG3, AUG3_NEEDS = cohort_options("cesm2_long_aug3", 40)
VLONG, VLONG_NEEDS = cohort_options("cesm2_vlong_uniform", 154)
#: arm -> (options, files that must exist before it starts)
ARMS = {
    "c_flow100": (CESM2, []),
    "c_deep": (CESM2 + ["--kriging", "ka_deep"], []),
    "c_product": (CESM2 + ["--kriging", "ka_product"], []),
    "c_multiscale": (CESM2 + ["--kriging", "ka_multiscale"], []),
    "c_mv": (CESM2 + ["--multivariate"], []),
    "c_density": (CESM2 + ["--thin-inputs", "1520", "6080"], []),
    "c_draws": (AUG3 + ["--thin-inputs", "6080", "6080"], AUG3_NEEDS),
    "c_both": (AUG3 + ["--thin-inputs", "1520", "6080"], AUG3_NEEDS),
    "c_y154": (VLONG, VLONG_NEEDS),
    # 2026-10-23 (Zihao: run the 154-year baselines too): are the extra years KA's gain or everyone's?
    "c_armor154": (VLONG + ["--background", "prior", "--kriging", "ka_oi"], VLONG_NEEDS),
    "c_none154": (VLONG + ["--kriging", "none"], VLONG_NEEDS),
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def job(arm, seed, smoke=False):
    options, needs = ARMS[arm]
    options = COMMON + options
    steps = STEPS
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
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
    parser.add_argument("--gpus", default="0,1,5,6,7")
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
        write_json(OUTPUT / f"campaign_{args.name}.json", {"seeds": args.seeds, "steps": STEPS,
            "arms": {arm: {"options": options, "needs": [str(p) for p in needs]} for arm, (options, needs) in ARMS.items()},
            "runner": str(RUNNER.relative_to(ROOT)), "jobs": [j["tag"] for j in requested]})
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
