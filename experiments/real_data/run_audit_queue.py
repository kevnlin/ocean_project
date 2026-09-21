"""Run the audit's training jobs across an EXPLICIT list of GPUs.

The host is shared with other tenants and which cards they hold changes, so the
GPU list is never inferred and never grown automatically: pass ``--gpus`` from a
fresh ``nvidia-smi`` reading.  The runner only refuses to start if a named card
has less free memory than ``--min-free-mb`` at launch.

  .venv/bin/python experiments/real_data/run_audit_queue.py --queue overfit --gpus 2,3
  .venv/bin/python experiments/real_data/run_audit_queue.py --queue ablation --gpus 0,1,2,3 --jobs 12
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
DRIVER = os.path.join(ROOT, "experiments", "real_data", "62_sanity_train.py")
LOGS = os.path.join(ROOT, "logs", "audit")

#: (tag, extra args). Ablations are ONE switch each, against the same baseline:
#: the global Perceiver-IO (D4RT) model with DFS mass, every profile a month
#: delivers, the satellite-era split. Arms that only make sense on a regional
#: box (coordinate rescaling to the box, GAOT anchors on a box) are not run.
ABLATIONS = [
    ("baseline", []),
    ("mass_uniform", ["--ablation", "mass_uniform"]),
    ("mass_count", ["--ablation", "mass_count"]),
    ("qc", ["--ablation", "qc"]),
    ("anomaly_exact", ["--ablation", "anomaly_exact"]),
    # per-level tokens raise a global month from ~30 k to ~100 k tokens and the
    # DFS neighbour search is quadratic, so this switch is measured at cap 1000
    # against `cap1000` (same cap, one switch apart), not against `baseline`
    ("level_tokens_cap1000", ["--ablation", "level_tokens,cap1000"]),
    ("refiner_local", ["--ablation", "refiner_local"]),
    ("refiner_gate1", ["--ablation", "refiner_gate1"]),
    ("no_latent", ["--ablation", "no_latent"]),
    ("no_target_dropout", ["--ablation", "no_target_dropout"]),
    # 8 months per optimiser step at a quarter of the steps: twice the baseline's
    # samples, so a gain is about gradient noise rather than about seeing more data
    ("batch8", ["--ablation", "batch8", "--steps", "3000", "--val-every", "250"]),
    # what a per-month cap costs when the whole ocean shares it
    ("cap1000", ["--ablation", "cap1000"]),
    # every other data/locality fix together, at every profile (level tokens
    # are left out for the cost reason above)
    ("fixed_stack", ["--ablation", "qc,anomaly_exact,refiner_local,refiner_gate1"]),
    # the backbone replacement for the Perceiver-IO fuse stage: Latent Neural
    # Operator physics-cross-attention, size-matched (405 543 vs 405 063),
    # same D4RT decoder, same DFS evidence
    ("backbone_lno", ["--backbone", "lno"]),
    ("backbone_lno_uniform", ["--backbone", "lno", "--ablation", "mass_uniform"]),
]
FIXED = "refiner_local,refiner_gate1"
OVERFIT = [("mem_d4rt", ["--mode", "memorise", "--steps", "4000"]),
           ("copy_d4rt", ["--mode", "copy", "--steps", "4000"]),
           ("small8_d4rt", ["--mode", "small", "--overfit-months", "8",
                            "--steps", "8000"]),
           ("mem_fixed", ["--mode", "memorise", "--steps", "4000", "--ablation", FIXED]),
           ("copy_fixed", ["--mode", "copy", "--steps", "4000", "--ablation", FIXED])]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="ablation", choices=["ablation", "overfit"])
ap.add_argument("--gpus", required=True, help="explicit list, e.g. 0,2")
ap.add_argument("--jobs", type=int, default=6, help="concurrent jobs in total")
ap.add_argument("--regions", default="global")
ap.add_argument("--seeds", default="1234,1235")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--min-free-mb", type=int, default=24000,
                help="a job starts only on a listed GPU with at least this much "
                     "free memory; below it the runner waits. A global Perceiver "
                     "job peaks near 21 GB")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

gpus = [int(g) for g in args.gpus.split(",") if g != ""]
free = {}
for line in subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=True).stdout.strip().splitlines():
    i, f, u = (int(x) for x in line.split(","))
    free[i] = (f, u)
print("using gpus " + ", ".join(f"{g} ({free[g][0]} MiB free, {free[g][1]}% util)"
                                for g in gpus), flush=True)

jobs = []
for region in args.regions.split(","):
    for seed in args.seeds.split(","):
        for tag, extra in (OVERFIT if args.queue == "overfit" else ABLATIONS):
            cmd = [PY, DRIVER, "--region", region, "--seed", seed, "--tag", tag,
                   "--wandb", "--leads", "0"]
            if "--steps" not in extra:
                cmd += ["--steps", str(args.steps)]
            jobs.append((f"{region}/{tag}/s{seed}", cmd + extra))

os.makedirs(LOGS, exist_ok=True)
qlog = open(os.path.join(LOGS, f"queue_{args.queue}.log"), "a")


def say(msg):
    line = f"{time.strftime('%H:%M')} {msg}"
    print(line, flush=True); qlog.write(line + "\n"); qlog.flush()


say(f"queue {args.queue}: {len(jobs)} jobs on gpus {gpus}, {args.jobs} at a time")
if args.dry_run:
    for name, cmd in jobs:
        print(" ", name, " ".join(cmd[2:]))
    raise SystemExit(0)

def free_mb():
    out = {}
    for line in subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines():
        i, f = (int(x) for x in line.split(","))
        out[i] = f
    return out


def summary_path(name):
    region, tag, seed = name.split("/")
    return os.path.join(ROOT, "outputs", "audit", region, tag,
                        f"summary_seed{seed[1:]}.json")


def already_running(name):
    """A job started by an earlier runner (still alive after a runner restart)."""
    _, tag, seed = name.split("/")
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("62_sanity_train.py" in l and f"--tag {tag} " in l + " "
               and f"--seed {seed[1:]} " in l + " " for l in ps.splitlines())


# resumable: a job with a summary is done; one still running from an earlier
# runner is left alone (it keeps its GPU memory, which the free-memory gate sees)
pending = []
for name, cmd in jobs:
    if os.path.exists(summary_path(name)):
        say(f"skip {name} (summary exists)")
    elif already_running(name):
        say(f"skip {name} (already running)")
    else:
        pending.append((name, cmd))

running, done = [], 0
while pending or running:
    fm = free_mb()
    cand = sorted((g for g in gpus if fm.get(g, 0) >= args.min_free_mb),
                  key=lambda g: -fm[g])
    if pending and len(running) < args.jobs and cand:
        name, cmd = pending.pop(0)
        g = cand[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", WANDB_SILENT="true",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        path = os.path.join(LOGS, name.replace("/", "_") + ".log")
        fh = open(path, "w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        running.append((name, p, fh, g, time.time()))
        say(f"start {name} gpu{g} ({fm[g]} MiB free)")
        time.sleep(90)      # let it allocate before the next free-memory reading
        continue
    time.sleep(20)
    for r in list(running):
        name, p, fh, g, t0 = r
        if p.poll() is not None:
            fh.close(); running.remove(r); done += 1
            say(f"{'ok  ' if p.returncode == 0 else 'FAIL'} {name} gpu{g} "
                f"{time.time()-t0:.0f}s ({done} finished this runner, "
                f"{len(pending)} pending)")
say("ALL DONE")
