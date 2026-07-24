"""Week-4 queue runner: all 9 full-scale runs (3 variants x 3 seeds).

Dispatches experiments/18_full_train.py jobs with bounded concurrency:
up to --per-gpu concurrent runs on GPU 7 (always ours) and on GPU 6 only
while it is free (memory below --free-mib, checked twice 20 s apart before
every dispatch — it may be occupied by someone else's job).

Seed-1234 jobs run first so the decisive variant comparison lands earliest.
Jobs whose outputs/cache/<tag>.json already reports status "done" are
skipped, so the queue is safe to restart.  Extra args after ``--`` are
passed through to every training run (single source of truth for the
final config), e.g.:

    nohup python experiments/run_full_queue.py -- --warmup 1000 \
        --obs-query-frac 0.25 > outputs/queue.log 2>&1 &
"""
import os, sys, json, time, subprocess

ROOT = "/home/nvidia/ocean_project"
CACHE = os.path.join(ROOT, "outputs", "cache")
PY = "/home/nvidia/.venv/bin/python"

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--per-gpu", type=int, default=3)
ap.add_argument("--free-mib", type=int, default=2000)
ap.add_argument("--stagger", type=int, default=90, help="seconds between launches")
ap.add_argument("--gpus", default="7,6", help="preference order; non-7 GPUs "
                "are used only when free")
ap.add_argument("--prefix", default="full", help="run tag prefix "
                "(<prefix>_<variant>_s<seed>)")
ap.add_argument("extra", nargs="*", help="args after -- go to 18_full_train.py")
args = ap.parse_args()
EXTRA = args.extra
PREFIX = args.prefix

JOBS = [(v, s) for s in (1234, 1235, 1236)
        for v in ("mbca", "perceiver", "resampler")]
GPUS = [g.strip() for g in args.gpus.split(",")]


def gpu_mem_used(gpu: str) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if idx == gpu:
            return int(mem)
    return 1 << 30


def gpu_free_for_us(gpu: str) -> bool:
    """GPU 7 is ours unconditionally; others must look idle twice, 20s apart."""
    if gpu == "7":
        return True
    if gpu_mem_used(gpu) >= args.free_mib:
        return False
    time.sleep(20)
    return gpu_mem_used(gpu) < args.free_mib


def job_done(tag: str) -> bool:
    p = os.path.join(CACHE, f"{tag}.json")
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") == "done"
    except Exception:
        return False


running = []          # (proc, tag, gpu)
pending = list(JOBS)
last_launch = 0.0
print(f"queue: {len(pending)} jobs, per_gpu={args.per_gpu}, extra={EXTRA}",
      flush=True)

while pending or running:
    # reap finished
    still = []
    for proc, tag, gpu in running:
        if proc.poll() is None:
            still.append((proc, tag, gpu))
        else:
            ok = job_done(tag)
            print(f"[{time.strftime('%H:%M:%S')}] {tag} on gpu{gpu} exited "
                  f"rc={proc.returncode} done={ok}", flush=True)
    running = still

    # dispatch
    while pending and time.time() - last_launch >= args.stagger:
        variant, seed = pending[0]
        tag = f"{PREFIX}_{variant}_s{seed}"
        if job_done(tag):
            print(f"skip {tag} (already done)", flush=True)
            pending.pop(0)
            continue
        slot_gpu = None
        for g in GPUS:
            if sum(1 for _, _, gg in running if gg == g) < args.per_gpu \
                    and gpu_free_for_us(g):
                slot_gpu = g
                break
        if slot_gpu is None:
            break
        pending.pop(0)
        log = open(os.path.join(ROOT, "outputs", f"{tag}.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot_gpu)
        cmd = [PY, os.path.join(ROOT, "experiments", "18_full_train.py"),
               "--variant", variant, "--seed", str(seed), "--tag", tag] + EXTRA
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        running.append((proc, tag, slot_gpu))
        last_launch = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] launched {tag} on gpu{slot_gpu} "
              f"(pid {proc.pid}); {len(pending)} pending", flush=True)
    time.sleep(30)

print("ALL RUNS COMPLETE", flush=True)
for variant, seed in JOBS:
    tag = f"{PREFIX}_{variant}_s{seed}"
    p = os.path.join(CACHE, f"{tag}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        t = d.get("test", {})
        print(f"  {tag}: TEMP={t.get('TEMP', float('nan')):.4f} "
              f"SALT={t.get('SALT', float('nan')):.4f} "
              f"(best step {d.get('best', {}).get('step')})", flush=True)
