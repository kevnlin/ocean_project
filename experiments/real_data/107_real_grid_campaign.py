"""Queue of the gridded model on real Argo + satellites: from scratch, zero shot, fine-tuned.

Every run is ``101_grid_ka_forecast.py --dataset real_global`` (``real_forecast_scenes``: the
2016-2023 cohort, OISST/OISSS/DUACS inputs, OI of ``106``), the method of ``103`` (background
"joint", kriging attention with the flow kernel), leads 0, 1, 3 and 6, checkpoint chosen on
2021, every input profile of a month in the state (``--context-profiles 0``):

  scratch   trained on 2016-2020 only, 30,000 steps (60 training months overfit the CESM2
            4-year arm after about 15-20k steps, so the checkpoint choice matters more than length)
  zeroshot  the CESM2 70,000-step model of the same seed (``103`` gj_flow), scored with no update
  finetune  that model trained further on 2016-2020: 10,000 steps at learning rate 1e-4

A zeroshot or finetune run waits until its CESM2 model has finished. Seeds 1234-1236 pair each
real run with the CESM2 run of the same seed. As in 103: a run starts only on a GPU with
``--min-free-mb`` free and a failed run continues from its last checkpoint.
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
OUTPUT = ROOT / "outputs/real_global_20261019"
#: the CESM2 70k runs of 103, in the frozen ka-pilot worktree
PRETRAINED = Path("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019")
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
STEPS = 70000
#: arm -> (steps, extra options); the first arms start first
ARMS = {"scratch": (30000, ["--val-every", "1000", "--checkpoint-every", "500"]),
        "zeroshot": (0, []),
        "finetune": (10000, ["--val-every", "500", "--checkpoint-every", "500", "--lr", "1e-4"])}
REFERENCE = ["--dataset", "real_global", "--surface", "--background", "joint", "--kriging", "ka_flow",
             "--context-profiles", "0", "--queries", "1024", "--amp", "--leads", "0", "1", "3", "6",
             "--val-years", "2021"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def pretrained(seed):
    return PRETRAINED / f"gj_flow_s{seed}"


def job(arm, seed, smoke=False):
    steps, extra = ARMS[arm]
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    if smoke and steps:
        steps, extra = 40, ["--val-every", "20", "--checkpoint-every", "20"] + extra[4:]
    init = [] if arm == "scratch" else ["--init-from", str(pretrained(seed) / f"best_seed{seed}.pt")]
    command = [sys.executable, "-u", str(RUNNER), *REFERENCE, *extra, *init, "--steps", str(steps),
               "--seed", str(seed), "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": None if arm == "scratch" else str(pretrained(seed) / f"summary_seed{seed}.json"),
            "command": command}


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if (summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"]
            or summary["data"].get("cohort") != "real_global 2016-2023" or "forecast" not in summary):
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


def ready(item):
    return item["needs"] is None or Path(item["needs"]).exists()


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
            ready_now = [j for j in pending if not_before.get(j["tag"], 0) <= time.time() and ready(j)]
            ready = ready_now
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
                "pending": [item["tag"] for item in pending], "attempts": attempts})
        if pending or running:
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 1235, 1236])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=14000, help="free GPU memory a launch needs")
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
        write_json(OUTPUT / f"campaign_{args.name}.json", {"seeds": args.seeds, "arms": ARMS,
            "pretrained": str(PRETRAINED),
            "reference": REFERENCE, "runner": str(RUNNER.relative_to(ROOT)), "selection_years": [2021],
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
