"""Queue of the formal experiments of the gridded model (frozen method: background "joint",
kriging attention with the flow kernel, lead evolution, lead decay), three seeds each.

CESM2 (``cesm2_long_uniform``, 70,000 steps, leads 0/1/3/6, checkpoint chosen on 2004-2005; the
sealed 2006-2007 test is scored once at the end with ``109``):

  c_gp_oi     ARMOR3D-style baseline: satellite-only prior background + kriging with the frozen OI kernel
  c_gp_flow   background ablation: the satellite-only prior with the flow kernel
  c_noevo     no lead evolution of the state
  c_nodecay   no damping of the read-out with the lead
  c_y4        a 4-year training window (2000-2003): what 40 years of simulation buy
  c_noise     pretraining with input noise (profiles up to 0.5, surface up to 0.3 z), for transfer

Real Argo + satellites (2016-2020 training, checkpoint chosen on 2021; 2022-2023 scored once):

  r_gp_oi, r_none            ARMOR3D-style and Aardvark-style baselines, from scratch, 30,000 steps
  r_scratch_y1, r_scratch_y2     from scratch on the last 1 or 2 training years, 30,000 steps
  r_finetune_y1, r_finetune_y2   the CESM2 model (103 gj_flow, same seed) fine-tuned on those years,
                             10,000 steps at learning rate 1e-4
  r_zeroshot_noise, r_finetune_noise, r_finetune_noise_y1   the same from c_noise (waits for it)

The 5-year scratch, zero-shot and fine-tune runs are 107's. A run that starts from another run's
model waits until that model has finished. As in 103/107: a run starts only on a GPU with
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
OUTPUT = ROOT / "outputs/formal_20261019"
LONG = ROOT / "outputs/long_record_20261017"
#: the CESM2 70k runs of 103, in the frozen ka-pilot worktree
PRETRAINED = Path("/DATA2/linke/ocean_project_ka/outputs/grid_ka_70k_20261019")
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/101_grid_ka_forecast.py"
CESM2 = ["--surface", "--satellite-cache", str(LONG / "satellites.npz"), "--cohort", "cesm2_long_uniform",
         "--train-years", "40", "--context-profiles", "6080", "--queries", "1024", "--amp",
         "--leads", "0", "1", "3", "6", "--val-years", "2004", "2005", "--val-every", "2000",
         "--checkpoint-every", "500", "--steps", "70000"]
REAL = ["--dataset", "real_global", "--surface", "--context-profiles", "0", "--queries", "1024", "--amp",
        "--leads", "0", "1", "3", "6", "--val-years", "2021"]
SCRATCH = ["--steps", "30000", "--val-every", "1000", "--checkpoint-every", "500"]
FINETUNE = ["--steps", "10000", "--val-every", "500", "--checkpoint-every", "500", "--lr", "1e-4"]
#: arm -> (common options, arm options, the run it starts from: None, "flow" (103 gj_flow) or "noise")
ARMS = {
    "c_gp_oi": (CESM2, ["--background", "prior", "--kriging", "ka_oi"], None),
    "c_gp_flow": (CESM2, ["--background", "prior", "--kriging", "ka_flow"], None),
    "c_noevo": (CESM2, ["--kriging", "ka_flow", "--no-evolution"], None),
    "c_nodecay": (CESM2, ["--kriging", "ka_flow", "--no-lead-decay"], None),
    "c_y4": (CESM2, ["--kriging", "ka_flow", "--train-years", "4"], None),
    "c_noise": (CESM2, ["--kriging", "ka_flow", "--profile-noise", "0.5", "--surface-noise", "0.3"], None),
    "r_gp_oi": (REAL, SCRATCH + ["--background", "prior", "--kriging", "ka_oi"], None),
    "r_none": (REAL, SCRATCH + ["--kriging", "none"], None),
    "r_scratch_y1": (REAL, SCRATCH + ["--kriging", "ka_flow", "--real-train-years", "1"], None),
    "r_scratch_y2": (REAL, SCRATCH + ["--kriging", "ka_flow", "--real-train-years", "2"], None),
    "r_finetune_y1": (REAL, FINETUNE + ["--kriging", "ka_flow", "--real-train-years", "1"], "flow"),
    "r_finetune_y2": (REAL, FINETUNE + ["--kriging", "ka_flow", "--real-train-years", "2"], "flow"),
    "r_zeroshot_noise": (REAL, ["--steps", "0", "--kriging", "ka_flow"], "noise"),
    "r_finetune_noise": (REAL, FINETUNE + ["--kriging", "ka_flow"], "noise"),
    "r_finetune_noise_y1": (REAL, FINETUNE + ["--kriging", "ka_flow", "--real-train-years", "1"], "noise"),
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source(kind, seed):
    """The run a fine-tune starts from: its chosen checkpoint and its summary."""
    folder = PRETRAINED / f"gj_flow_s{seed}" if kind == "flow" else OUTPUT / f"c_noise_s{seed}"
    return folder / f"best_seed{seed}.pt", folder / f"summary_seed{seed}.json"


def job(arm, seed, smoke=False):
    common, extra, start = ARMS[arm]
    options = common + extra
    steps = int(options[len(options) - 1 - options[::-1].index("--steps") + 1])
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    if smoke and steps:
        options, steps = options + ["--steps", "40", "--val-every", "20", "--checkpoint-every", "20"], 40
    init, needs = [], None
    if start is not None:
        checkpoint, needs = source(start, seed)
        init = ["--init-from", str(checkpoint)]
    command = [sys.executable, "-u", str(RUNNER), *options, *init, "--seed", str(seed),
               "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "output": str(folder),
            "needs": None if needs is None else str(needs), "command": command}


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"] or "forecast" not in summary:
        raise ValueError(f"incomplete or mismatched training summary: {item['tag']}")
    return True


def has_pretrained(item):
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
            ready = [j for j in pending if not_before.get(j["tag"], 0) <= time.time() and has_pretrained(j)]
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
            "runner": str(RUNNER.relative_to(ROOT)),
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
