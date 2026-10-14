"""Queue of the radius / local-scale runs on the OI-anchored model.

Every run is the Local Transformer 192 of the matched campaign
(``51_matched_campaign.py``) with one or two options of
``81_radius_scale_reconstruction.py``. An arm is named by its options, joined
with ``+``:

  control                  no option (the matched campaign's model)
  r250, r500, ...          --local-radius-km
  l64, l32, ...            --latents (slots of the shared latent; the reference has 96)
  qanom_cell, qanom_month, --scale (a table of 80_local_scale_tables.py), a query's
  qres_cell, qres_month    whole neighbourhood in units of the query's own entry
  q<table>_lossz           the same with --loss-units z
  <table>, <table>_lossz   the first design, --scale-by profile: each profile
                           divided by its own entry (distorts the field; kept so
                           the two finished runs can be replayed)

``--phase train`` runs the requested arms, inputs and seeds, several workers a
GPU, continuing interrupted runs from their last state. ``--phase development``
scores 2005 from the selected checkpoints; run it only once the arms to report
are fixed on the 2004 scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/radius_scale_20261008"
RUNNER = ROOT / "experiments/synthetic/81_radius_scale_reconstruction.py"
STEPS = 15000
#: the Local Transformer 192 settings of 51_matched_campaign.py
REFERENCE = ["--variant", "local_transformer", "--width", "192", "--latents", "96",
             "--blocks", "6", "--query-blocks", "2", "--heads", "8", "--experts", "4",
             "--steps", str(STEPS), "--context-profiles", "6080", "--queries", "1024",
             "--val-every", "1000", "--checkpoint-every", "500", "--amp"]
TABLES = ("anom_cell", "anom_month", "res_cell", "res_month")
INPUTS = ("argo", "surface")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def flags(arm):
    """Runner options of an arm name; at most one radius and one scale."""
    atoms = arm.split("+")
    if atoms == ["control"]:
        return []
    radius, slots, scale = [], [], []
    for atom in atoms:
        name = atom.removesuffix("_lossz")
        by = "query" if name[1:] in TABLES and name[:1] == "q" else "profile"
        table = name[1:] if by == "query" else name
        if atom[:1] == "r" and atom[1:].isdigit() and int(atom[1:]) > 0:
            radius.append(["--local-radius-km", atom[1:]])
        elif atom[:1] == "l" and atom[1:].isdigit() and int(atom[1:]) > 0:
            slots.append(["--latents", atom[1:]])
        elif table in TABLES:
            scale.append(["--scale", table, "--scale-by", by]
                         + (["--loss-units", "z"] if atom != name else []))
        else:
            raise ValueError(f"arm {arm!r}: {atom!r} is not control, r<km>, l<slots>, [q]<table> or [q]<table>_lossz")
    if len(radius) > 1 or len(slots) > 1 or len(scale) > 1:
        raise ValueError(f"arm {arm!r} names more than one radius, slot count or scale")
    return [x for group in radius + slots + scale for x in group]


def job(arm, inputs, seed, output=OUTPUT, cohort=None):
    """One run; ``cohort`` names another synthetic cohort, kept in its own subfolder."""
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {INPUTS}")
    output = Path(output) / cohort if cohort is not None else Path(output)
    tag = f"{arm.replace('+', '_')}_{inputs}_s{seed}"
    folder = output / tag
    options = flags(arm)
    command = [sys.executable, "-u", str(RUNNER), "--tag", tag, "--output", str(folder),
               "--seed", str(seed), *REFERENCE, *options]
    if "--scale" in options:
        command += ["--scale-tables", str(output / "scale_tables.npz")]
    if inputs == "surface":
        command += ["--surface", "--satellite-cache", str(output / "satellites.npz")]
    if cohort is not None:
        command += ["--cohort", cohort, "--oi-summary",
                    str(ROOT / "outputs/audit/synthetic" / f"fixed_baselines_{cohort}" / "summary.json")]
    return {"tag": tag, "arm": arm, "inputs": inputs, "seed": seed, "output": str(folder),
            "command": command}


def jobs(arms, inputs, seeds, output=OUTPUT, cohort=None):
    """Seeds outermost, so every arm has one seed before any has two."""
    return [job(arm, kind, seed, output, cohort) for seed in seeds for kind in inputs for arm in arms]


def summary_of(job):
    path = Path(job["output"]) / f"summary_seed{job['seed']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def completed(job):
    summary = summary_of(job)
    if summary is None:
        return False
    if summary.get("completed_steps") != STEPS or summary.get("seed") != job["seed"]:
        raise ValueError(f"incomplete or mismatched training summary: {job['tag']}")
    return True


def scored_2005(job):
    summary = summary_of(job)
    return summary is not None and "development" in summary


def training_command(job):
    last = Path(job["output"]) / f"last_seed{job['seed']}.pt"
    if last.exists():
        return job["command"] + ["--resume-training", "--load-checkpoint", str(last)]
    return list(job["command"])


def development_command(job):
    return job["command"] + ["--evaluate-only", "--development", "--load-checkpoint",
                             str(Path(job["output"]) / f"best_seed{job['seed']}.pt")]


def slots(gpus, per_gpu):
    if len(set(gpus)) != len(gpus) or not gpus or per_gpu < 1:
        raise ValueError("GPU indices must be distinct and each needs at least one worker")
    return [(gpu, worker) for gpu in gpus for worker in range(per_gpu)]


def queue(pending, places, phase, stagger, status):
    """Run ``pending`` on ``places``; a failed job is retried twice from its last state."""
    pending, running, attempts, last_launch = list(pending), {}, {}, 0.
    while pending or running:
        for place in places:
            if place in running or not pending or time.time() - last_launch < stagger:
                continue
            choice = pending.pop(0)
            folder = Path(choice["output"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / f"{phase}.log").open("a")
            command = training_command(choice) if phase == "train" else development_command(choice)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(place[0]), "OCEAN_ROOT": str(ROOT),
                   "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[place] = process, log, choice
            attempts[choice["tag"]] = attempts.get(choice["tag"], 0) + 1
            last_launch = time.time()
            print(json.dumps({"phase": phase, "started": choice["tag"], "gpu": place[0],
                              "pid": process.pid}), flush=True)
        for place, (process, log, item) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[place]
            finished = completed(item) if phase == "train" else scored_2005(item)
            if code or not finished:
                if attempts[item["tag"]] >= 3:
                    for other, handle, _ in running.values():
                        other.terminate()
                        handle.close()
                    raise RuntimeError(f"{phase} failed: {item['tag']} (exit {code}); "
                                       f"see {item['output']}/{phase}.log")
                pending.append(item)
                print(json.dumps({"retry": item["tag"], "exit_code": code}), flush=True)
            else:
                print(json.dumps({"phase": phase, "completed": item["tag"]}), flush=True)
        status({"running": {f"{gpu}.{worker}": item[2]["tag"] for (gpu, worker), item in running.items()},
                "pending": [item["tag"] for item in pending], "attempts": attempts})
        if pending or running:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--inputs", nargs="+", choices=INPUTS, default=["argo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--gpus", default="6")
    parser.add_argument("--per-gpu", type=int, default=3, help="workers sharing one GPU")
    parser.add_argument("--stagger", type=float, default=60., help="seconds between two launches")
    parser.add_argument("--phase", choices=("train", "development"), default="train")
    parser.add_argument("--name", default="queue", help="names this invocation's status file")
    parser.add_argument("--cohort", default=None,
                        help="another synthetic cohort (file name without .nc)")
    args = parser.parse_args()
    requested = jobs(args.arms, args.inputs, args.seeds, OUTPUT, args.cohort)
    places = slots([int(g) for g in args.gpus.split(",")], args.per_gpu)
    folder = OUTPUT / args.cohort if args.cohort else OUTPUT
    if any("--scale" in j["command"] for j in requested) and not (folder / "scale_tables.npz").exists():
        raise SystemExit("run 80_local_scale_tables.py first")
    if "surface" in args.inputs and not (folder / "satellites.npz").exists():
        raise SystemExit(f"the surface input needs the matched cache at {folder / 'satellites.npz'}")
    if args.phase == "train":
        pending = [j for j in requested if not completed(j)]
    else:
        if not all(completed(j) for j in requested):
            raise SystemExit("every requested run must finish training before 2005 is scored")
        pending = [j for j in requested if not scored_2005(j)]
    started = time.time()

    def status(extra):
        write_json(folder / f"status_{args.name}.json", {"phase": args.phase, "requested": len(requested),
            "started_unix": started, "updated_unix": time.time(), **extra})

    print(json.dumps({"phase": args.phase, "requested": len(requested), "to_run": len(pending),
                      "places": [f"{gpu}.{worker}" for gpu, worker in places]}), flush=True)
    try:
        queue(pending, places, args.phase, args.stagger, status)
        status({"running": {}, "pending": [], "finished": True, "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
