"""Queue of the kriging-attention pilot: six arms of runner 97, reconstruction, one seed.

Every run is ``97_kriging_pilot.py`` at the long-record reference (89's medium model: recipe
``profile_shared``, 128 latent slots, width 256; Argo plus SST, SSS and steric SSH; the local
OI-error scale by depth x cell; the 4-year window 2000-2003; 15,000 steps; checkpoint chosen
on 2004; 2005 scored at the end). The arms change only how the local path weighs the 32
nearest profiles and whether the model sees the frozen OI:

  sm_oi        softmax, OI anchor (89's model, the reference)
  kaflow_oi    kriging attention with a flow-dependent kernel, OI anchor
  sm_nooi      softmax, no OI anchor
  kaflow_nooi  kriging attention with a flow-dependent kernel, no OI anchor
  kaoi_nooi    kriging attention with the OI's fixed kernel, no OI anchor
  kast_nooi    kriging attention with a learned stationary kernel, no OI anchor

They answer: does kriging beat softmax with the anchor (kaflow_oi vs sm_oi); can the model
drop the frozen OI (kaflow_nooi vs sm_oi); is kriging what makes that work (vs sm_nooi);
which part of the kernel matters (kaoi_nooi, kast_nooi, kaflow_nooi).

The GPUs are shared with another user's job of about 65 GB each, so a run starts only on a
GPU with ``--min-free-mb`` free; a failed run is put back and continues from its last state.
``--smoke`` runs the chosen arms for 40 steps into ``smoke/`` as an engineering check.
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
OUTPUT = ROOT / "outputs/ka_pilot_20261019"
LONG = ROOT / "outputs/long_record_20261017"
RUNNER = ROOT / "experiments/synthetic/97_kriging_pilot.py"
STEPS = 15000
#: arm -> (local weighting, OI anchor); the first arms start first
ARMS = {"sm_oi": ("softmax", True), "kaflow_oi": ("ka_flow", True), "sm_nooi": ("softmax", False),
        "kaflow_nooi": ("ka_flow", False), "kaoi_nooi": ("ka_oi", False), "kast_nooi": ("ka_stationary", False)}
REFERENCE = ["--recipe", "profile_shared", "--variant", "local_transformer", "--blocks", "6",
             "--query-blocks", "2", "--heads", "8", "--experts", "4", "--latents", "128", "--width", "256",
             "--context-profiles", "6080", "--queries", "1024", "--amp", "--train-years", "4",
             "--cohort", "cesm2_long_uniform", "--scale", "res_cell",
             "--scale-tables", str(LONG / "scale_tables.npz"),
             "--surface", "--satellite-cache", str(LONG / "satellites.npz")]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def job(arm, seed, smoke=False):
    weighting, anchor = ARMS[arm]
    steps = 40 if smoke else STEPS
    folder = (OUTPUT / "smoke" if smoke else OUTPUT) / f"{arm}_s{seed}"
    schedule = (["--val-every", "20", "--checkpoint-every", "20"] if smoke else
                ["--val-every", "1000", "--checkpoint-every", "500", "--development"])
    command = [sys.executable, "-u", str(RUNNER), *REFERENCE, *schedule, "--steps", str(steps),
               "--local-weighting", weighting, *([] if anchor else ["--no-oi-anchor"]),
               "--seed", str(seed), "--tag", f"{arm}_s{seed}", "--output", str(folder)]
    return {"tag": f"{arm}_s{seed}", "arm": arm, "seed": seed, "steps": steps, "weighting": weighting,
            "oi_anchor": anchor, "output": str(folder), "command": command}


def completed(item):
    path = Path(item["output"]) / f"summary_seed{item['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if (summary.get("completed_steps") != item["steps"] or summary.get("seed") != item["seed"]
            or summary.get("local_weighting") != item["weighting"] or summary.get("oi_anchor") != item["oi_anchor"]):
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--per-gpu", type=int, default=2, help="workers sharing one GPU")
    parser.add_argument("--min-free-mb", type=int, default=14000, help="free GPU memory a launch needs")
    parser.add_argument("--retry-after", type=float, default=600.)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--stagger", type=float, default=120., help="seconds between two launches")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / ("smoke.lock" if args.smoke else "queue.lock")).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another queue is running")
    requested = [job(arm, seed, args.smoke) for seed in args.seeds for arm in args.arms]
    places = [(int(gpu), worker) for gpu in args.gpus.split(",") for worker in range(args.per_gpu)]
    pending = [j for j in requested if not completed(j)]
    started, name = time.time(), "smoke" if args.smoke else "queue"

    def status(extra):
        write_json(OUTPUT / f"status_{name}.json", {"requested": len(requested),
            "completed": sum(completed(j) for j in requested), "started_unix": started,
            "updated_unix": time.time(), **extra})

    if not args.smoke:
        write_json(OUTPUT / "campaign.json", {"steps": STEPS, "seeds": args.seeds, "arms": ARMS,
            "reference": REFERENCE, "runner": str(RUNNER.relative_to(ROOT)), "selection_year": 2004,
            "development_year": 2005, "jobs": [j["tag"] for j in requested]})
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
