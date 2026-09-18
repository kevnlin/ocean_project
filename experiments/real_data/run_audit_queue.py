"""Run the audit's training jobs across an EXPLICIT list of GPUs.

The host is shared with other tenants and which cards they hold changes, so the
GPU list is never inferred and never grown automatically: pass ``--gpus`` from a
fresh ``nvidia-smi`` reading.  The runner only refuses to start if a named card
has less free memory than ``--min-free-mb`` at launch.

  .venv/bin/python experiments/real_data/run_audit_queue.py --queue overfit --gpus 0,2
  .venv/bin/python experiments/real_data/run_audit_queue.py --queue ablation --gpus 0,2 --jobs 6
"""
from __future__ import annotations

import argparse, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.path.join(ROOT, ".venv", "bin", "python")
DRIVER = os.path.join(ROOT, "experiments", "real_data", "62_sanity_train.py")
LOGS = os.path.join(ROOT, "logs", "audit")

#: (tag, extra args).  Ablations are ONE switch each, against the same baseline.
ABLATIONS = [
    ("baseline", []),
    ("mass_uniform", ["--ablation", "mass_uniform"]),
    ("mass_count", ["--ablation", "mass_count"]),
    ("qc", ["--ablation", "qc"]),
    ("anomaly_exact", ["--ablation", "anomaly_exact"]),
    ("level_tokens", ["--ablation", "level_tokens"]),
    ("refiner_local", ["--ablation", "refiner_local"]),
    ("refiner_gate1", ["--ablation", "refiner_gate1"]),
    ("coords_region", ["--ablation", "coords_region"]),
    ("all_profiles", ["--ablation", "all_profiles"]),
    ("no_latent", ["--ablation", "no_latent"]),
    ("no_target_dropout", ["--ablation", "no_target_dropout"]),
    # 8 months per optimiser step at a quarter of the steps: twice the baseline's
    # samples, so a gain is about gradient noise rather than about seeing more data
    ("batch8", ["--ablation", "batch8", "--steps", "3000", "--val-every", "250"]),
    ("fixed_stack", ["--ablation", "qc,anomaly_exact,level_tokens,refiner_local,"
                     "refiner_gate1,coords_region,all_profiles"]),
    ("backbone_setconv", ["--backbone", "setconv"]),
    ("backbone_setconv_all", ["--backbone", "setconv", "--ablation", "all_profiles"]),
    ("backbone_gaot", ["--backbone", "gaot"]),
]
#: the gyre repeats only the arms whose question is regional
GYRE = {"baseline", "qc", "level_tokens", "refiner_local", "coords_region",
        "all_profiles", "fixed_stack", "backbone_setconv",
        "backbone_setconv_all", "mass_uniform"}

OVERFIT = [("mem_d4rt", ["--mode", "memorise", "--steps", "4000"]),
           ("copy_d4rt", ["--mode", "copy", "--steps", "4000"]),
           ("small8_d4rt", ["--mode", "small", "--overfit-months", "8",
                            "--steps", "8000"]),
           ("mem_setconv", ["--mode", "memorise", "--steps", "4000",
                            "--backbone", "setconv"]),
           ("copy_setconv", ["--mode", "copy", "--steps", "4000",
                             "--backbone", "setconv"]),
           ("small8_setconv", ["--mode", "small", "--overfit-months", "8",
                               "--steps", "8000", "--backbone", "setconv"]),
           ("mem_fixed", ["--mode", "memorise", "--steps", "4000", "--ablation",
                          "level_tokens,refiner_local,refiner_gate1,coords_region"]),
           ("copy_fixed", ["--mode", "copy", "--steps", "4000", "--ablation",
                           "level_tokens,refiner_local,refiner_gate1,coords_region"])]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="ablation", choices=["ablation", "overfit"])
ap.add_argument("--gpus", required=True, help="explicit list, e.g. 0,2")
ap.add_argument("--jobs", type=int, default=6, help="concurrent jobs in total")
ap.add_argument("--regions", default="gulfstream,npac_gyre")
ap.add_argument("--seeds", default="1234,1235")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--min-free-mb", type=int, default=15000)
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
for g in gpus:
    if free[g][0] < args.min_free_mb:
        raise SystemExit(f"gpu {g}: only {free[g][0]} MiB free — pick another card")
print("using gpus " + ", ".join(f"{g} ({free[g][0]} MiB free, {free[g][1]}% util)"
                                for g in gpus), flush=True)

jobs = []
for region in args.regions.split(","):
    for seed in args.seeds.split(","):
        for tag, extra in (OVERFIT if args.queue == "overfit" else ABLATIONS):
            if args.queue == "ablation" and region != "gulfstream" and tag not in GYRE:
                continue
            cmd = [PY, DRIVER, "--region", region, "--seed", seed, "--tag", tag,
                   "--wandb", "--leads", "0"]
            if "--steps" not in extra:
                cmd += ["--steps", str(args.steps)]
            if args.queue == "ablation" and tag.startswith("backbone_setconv"):
                cmd += ["--setconv-width", "28"]
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

running, done, i = [], 0, 0
while i < len(jobs) or running:
    while i < len(jobs) and len(running) < args.jobs:
        name, cmd = jobs[i]
        g = gpus[i % len(gpus)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", WANDB_SILENT="true")
        path = os.path.join(LOGS, name.replace("/", "_") + ".log")
        fh = open(path, "w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                             cwd=ROOT)
        running.append((name, p, fh, g, time.time()))
        say(f"start {name} gpu{g}")
        i += 1
    time.sleep(5)
    for r in list(running):
        name, p, fh, g, t0 = r
        if p.poll() is not None:
            fh.close(); running.remove(r); done += 1
            say(f"{'ok  ' if p.returncode == 0 else 'FAIL'} {name} gpu{g} "
                f"{time.time()-t0:.0f}s ({done}/{len(jobs)})")
say("ALL DONE")
