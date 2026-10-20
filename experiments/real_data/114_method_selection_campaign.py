"""Queue of the method-selection runs (2026-10-20): the candidates' real-data runs, and one ablation.

The final method is chosen among three variants by the rule of ``115_method_selection.py``, fixed
before these runs finished (``docs/superpowers/specs/2026-10-20-method-selection.md``):

  A  U-Net + kriging attention (flow kernel), joint background: 103's gj_flow and 107's scratch arm
  B  the same with the prior background (the month's profiles enter only through the kriging
     innovations): 111's c_gp_flow on CESM2, ``rb_prior`` here on real data
  C  Poseidon-T (public weights) + kriging attention, joint background: 112's p_ka on CESM2,
     ``rc_poseidon`` here on real data

Real-data arms use 107's scratch setup (real 2016-2020, 30,000 steps, checkpoint chosen on 2021);
``pr_random`` is C with randomly initialised Poseidon-T weights at 112's CESM2 setup (70,000 steps,
checkpoint chosen on 2004-2005): does the public pretraining or the architecture help. Seeds
1234-1236; no run reads a sealed year. As in 112: a run starts only on a GPU with ``--min-free-mb``
free, and a failed run continues from its last checkpoint.
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
OUTPUT = ROOT / "outputs/selection_20261020"
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
REAL = ["--dataset", "real_global", "--surface", "--context-profiles", "0", "--queries", "1024", "--amp",
        "--leads", "0", "1", "3", "6", "--val-years", "2021"]
CESM2 = ["--context-profiles", "6080", "--queries", "1024", "--amp", "--train-years", "40",
         "--cohort", "cesm2_long_uniform", "--surface", "--satellite-cache", str(LONG / "satellites.npz"),
         "--leads", "0", "1", "3", "6", "--val-years", "2004", "2005"]
#: dataset -> (reference options, steps, schedule)
SETUPS = {"real": (REAL, 30000, ["--val-every", "1000", "--checkpoint-every", "500"]),
          "cesm2": (CESM2, 70000, ["--val-every", "2000", "--checkpoint-every", "500"])}
#: arm -> (dataset, background, backbone, Poseidon-T initialisation); the first arms start first
ARMS = {"rb_prior": ("real", "prior", "unet", "pretrained"),
        "rc_poseidon": ("real", "joint", "poseidon", "pretrained"),
        "pr_random": ("cesm2", "joint", "poseidon", "random")}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def job(arm, seed, smoke=False):
    dataset, background, backbone, init = ARMS[arm]
    reference, steps, schedule = SETUPS[dataset]
    if smoke:
        steps, schedule = 40, ["--val-every", "20", "--checkpoint-every", "20"]
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    command = [sys.executable, "-u", str(RUNNER), *reference, *schedule, "--steps", str(steps),
               "--background", background, "--kriging", "ka_flow", "--backbone", backbone,
               "--poseidon-init", init, "--seed", str(seed), "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "dataset": dataset,
            "background": background, "backbone": backbone, "poseidon_init": init, "output": str(folder),
            "command": command}


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    config = summary["config"]
    if (summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"]
            or summary.get("kriging") != "ka_flow" or config.get("background") != item["background"]
            or config.get("backbone") != item["backbone"] or config.get("poseidon_init") != item["poseidon_init"]
            or (summary["data"].get("cohort") == "real_global 2016-2023") != (item["dataset"] == "real")
            or "forecast" not in summary):
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


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
    parser.add_argument("--gpus", default="1,3")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=30000, help="free GPU memory a launch needs")
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
        write_json(OUTPUT / f"campaign_{args.name}.json", {"seeds": args.seeds, "arms": ARMS, "setups": SETUPS,
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
