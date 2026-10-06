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

#: the 20 k contribution study: one switch per thing this model claims to add,
#: all against the same baseline, tags prefixed so the 12 k audit runs stand.
#: 20 000 steps, checkpoint every 5 000; everything else as in the audit.
CONTRIB = [("k20_baseline", []),
           # evidence
           ("k20_mass_uniform", ["--ablation", "mass_uniform"]),
           ("k20_mass_count", ["--ablation", "mass_count"]),
           ("k20_refiner_no_mass", ["--ablation", "refiner_no_mass"]),
           # architecture
           ("k20_backbone_lno", ["--backbone", "lno"]),
           ("k20_no_latent", ["--ablation", "no_latent"]),
           ("k20_no_refiner", ["--ablation", "no_refiner"]),
           ("k20_no_refslots", ["--ablation", "no_refslots"]),
           ("k20_no_experts", ["--ablation", "no_experts"]),
           # objective, and the audit's refiner fix
           ("k20_loss_balanced", ["--ablation", "loss_balanced"]),
           ("k20_refiner_gate1", ["--ablation", "refiner_gate1"])]
OVERFIT = [("mem_d4rt", ["--mode", "memorise", "--steps", "4000"]),
           ("copy_d4rt", ["--mode", "copy", "--steps", "4000"]),
           ("small8_d4rt", ["--mode", "small", "--overfit-months", "8",
                            "--steps", "8000"]),
           ("mem_fixed", ["--mode", "memorise", "--steps", "4000", "--ablation", FIXED]),
           ("copy_fixed", ["--mode", "copy", "--steps", "4000", "--ablation", FIXED])]

#: the 2026-09-26 plan on the SYNTHETIC CESM2 cohort (known truth, Argo-only,
#: uniform random positions): these queues always run with --region synthetic.
#: syn_overfit: memorise one fixed month / copy the inputs, 20 k steps, and the
#: bottleneck split — latent + local refiner (base), latent only (no_refiner),
#: raw local tokens only (no_latent) — plus the refiner fix and per-level tokens.
LOCAL = ["--refiner-km", "150", "--refiner-gate", "1.0"]
SYN_OVERFIT = []
for _mode, _pre in (("memorise", "syn_mem"), ("copy", "syn_copy")):
    _m = ["--mode", _mode, "--steps", "20000", "--val-every", "1000"]
    SYN_OVERFIT += [(f"{_pre}_base", _m),
                    (f"{_pre}_latent_only", _m + ["--ablation", "no_refiner"]),
                    (f"{_pre}_tokens_only", _m + ["--ablation", "no_latent"]),
                    (f"{_pre}_local", _m + LOCAL)]
# (per-level tokens were dropped from this queue: 4x the tokens and a quadratic
# DFS search projected ~50 h per 20 k-step run on two shared GPUs)
SYN_OVERFIT.insert(1, ("syn_mem_wd0", ["--mode", "memorise", "--steps", "20000",
                                       "--val-every", "1000", "--weight-decay", "0"]))
#: syn_refiner: held-out training, 12 k steps. The refiner's initial reach x
#: gate on the at-position target, plus the registered cell-centre target at the
#: registered init (plan items 4 and 5). Registered init = (0.35, 0.35, 0.30,
#: 3.0) on (lon/180, lat/90, depth/1000 m, month): ~3 500 km, 300 m, gate 0.05.
EXACT = ["--ablation", "anomaly_exact"]
SYN_REFINER = [("syn_reg_cell", []), ("syn_reg", EXACT),
               ("syn_reg_g1", EXACT + ["--refiner-gate", "1.0"])]
for _km in (150, 500, 1500):
    for _g, _gv in (("g005", "0.05"), ("g1", "1.0")):
        SYN_REFINER.append((f"syn_r{_km}_{_g}",
                            EXACT + ["--refiner-km", str(_km), "--refiner-gate", _gv]))
#: syn_final: plan items 7-8 on the validated configuration — the at-position
#: target and the refiner init selected on validation in syn_refiner
#: (500 km / 100 m, gate 1.0; lowest validation macro J over 3 seeds). That
#: configuration at 32 slots, Perceiver-IO, DFS is `syn_r500_g1` itself, which
#: serves as the reference and is not re-run. Mass rules go through --mass-mode
#: so they do not overwrite the anomaly_exact ablation.
FIX = EXACT + ["--refiner-km", "500", "--refiner-gate", "1.0"]
SYN_FINAL = [("syn_fix_l64", FIX + ["--n-latent", "64"]),
             ("syn_fix_l128", FIX + ["--n-latent", "128"]),
             ("syn_fix_uniform", FIX + ["--mass-mode", "uniform"]),
             ("syn_fix_count", FIX + ["--mass-mode", "count"]),
             ("syn_fix_lno", FIX + ["--backbone", "lno"]),
             ("syn_fix_lno_uniform", FIX + ["--backbone", "lno", "--mass-mode", "uniform"]),
             ("syn_fix_lno_count", FIX + ["--backbone", "lno", "--mass-mode", "count"]),
             # item 7 on the PhCA-style fuse too: only the slot count changes
             # (position-MLP width stays 96), against `syn_fix_lno` at 32 slots
             ("syn_fix_lno_s64", FIX + ["--backbone", "lno", "--n-slots", "64"]),
             ("syn_fix_lno_s128", FIX + ["--backbone", "lno", "--n-slots", "128"])]
#: syn_refiner_k15: the syn_refiner queue re-run at 15 k steps (2026-10-04).
#: Every seed of the arm selected at 12 k had its best validation score at the
#: last step, so those runs were still improving when they stopped. Same nine
#: arms, tags prefixed so the 12 k runs stand.
SYN_REFINER_K15 = [(f"k15_{_tag}", _extra + ["--steps", "15000"])
                   for _tag, _extra in SYN_REFINER]
#: syn_refiner_k15_l64: the same 15 k sweep with 64 latent slots instead of 32
#: (--n-latent, the switch of `syn_fix_l64`; each slot stays 64 channels wide).
SYN_REFINER_K15_L64 = [(f"k15_l64_{_tag}", _extra + ["--n-latent", "64", "--steps", "15000"])
                       for _tag, _extra in SYN_REFINER]
#: syn_mass_k15_l64: the uniform and count mass rules of syn_final on the
#: current model (64 slots, 15 k steps, validated refiner init). Their DFS
#: reference is `k15_l64_syn_r500_g1` of the queue above, which is not re-run.
SYN_MASS_K15_L64 = [(f"k15_l64_{_tag}", _extra + ["--n-latent", "64", "--steps", "15000"])
                    for _tag, _extra in SYN_FINAL
                    if _tag in ("syn_fix_uniform", "syn_fix_count")]
SYN_QUEUES = {"syn_overfit": SYN_OVERFIT, "syn_refiner": SYN_REFINER,
              "syn_final": SYN_FINAL, "syn_refiner_k15": SYN_REFINER_K15,
              "syn_refiner_k15_l64": SYN_REFINER_K15_L64,
              "syn_mass_k15_l64": SYN_MASS_K15_L64}

#: pre-shutdown validation on REAL Argo (2026-09-29 plan, experiments A-C).
#: `baseline` / `anomaly_exact` seeds 1234-1235 already exist and are reused;
#: seed 1236 is added. B uses the refiner the synthetic audit selected.
R500 = ["--refiner-km", "500", "--refiner-dz", "100", "--refiner-gate", "1.0"]
_mem = ["--mode", "memorise", "--steps", "20000", "--val-every", "1000"]
REAL_FINAL = [
    ("real_exact_r500_g1", ["--ablation", "anomaly_exact"] + R500),   # B (fix)
    ("baseline", []),                                                  # A (old), seed 1236
    ("anomaly_exact", ["--ablation", "anomaly_exact"]),                # A (new) / B (old)
    ("real_cell_r500_g1", R500),                                       # A at the local refiner
    ("real_mem20k", _mem),                                             # C
    ("real_mem20k_latent_only", _mem + ["--ablation", "no_refiner"]),
    ("real_mem20k_tokens_only", _mem + ["--ablation", "no_latent"])]
REAL_QUEUES = {"real_final": REAL_FINAL[:4], "real_overfit": REAL_FINAL[4:]}

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="ablation",
                choices=["ablation", "overfit", "contrib"] + sorted(SYN_QUEUES)
                + sorted(REAL_QUEUES))
ap.add_argument("--gpus", required=True, help="explicit list, e.g. 0,2")
ap.add_argument("--jobs", type=int, default=6, help="concurrent jobs in total")
ap.add_argument("--regions", default="global")
ap.add_argument("--seeds", default="1234,1235")
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--ckpt-every", type=int, default=0,
                help="pass through to 62: save a checkpoint every n steps")
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
if args.queue in SYN_QUEUES:
    args.regions = "synthetic"
for region in args.regions.split(","):
    for seed in args.seeds.split(","):
        queue = {"overfit": OVERFIT, "contrib": CONTRIB,
                 **SYN_QUEUES, **REAL_QUEUES}.get(args.queue, ABLATIONS)
        for tag, extra in queue:
            cmd = [PY, DRIVER, "--region", region, "--seed", seed, "--tag", tag,
                   "--wandb", "--leads", "0"]
            if "--steps" not in extra:
                cmd += ["--steps", str(args.steps)]
            if args.ckpt_every:
                cmd += ["--ckpt-every", str(args.ckpt_every)]
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
