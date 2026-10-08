"""Run the preregistered validation-only ocean architecture search on free GPUs.

This launcher never selects using development scores. All job commands and
completion codes are saved before/after training. Child stdout goes to its own
log, so failures are inspectable and finished experiments are never overwritten.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--phase", choices=["search", "replicate", "ablate"], default="search")
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/latent_ocean/campaign_20261007.json")
    p.add_argument("--seeds", default="1235,1236")
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    if args.phase == "search":
        jobs = []
        for variant in ("dense", "soft_moe", "local_transformer"):
            for capacity in ("base", "large"):
                jobs.append({"tag": f"{variant}_{capacity}_s1234", "variant": variant,
                    "width": 128 if capacity == "base" else 192,
                    "latents": 64 if capacity == "base" else 96,
                    "blocks": 4 if capacity == "base" else 6,
                    "context_profiles": 512 if capacity == "base" else 768,
                    "experts": 4 if capacity == "base" else 8,
                    "seed": 1234, "steps": args.steps})
        jobs.append({"tag": "analysis_only_s1234", "variant": "dense", "width": 64,
                     "latents": 32, "blocks": 2, "context_profiles": 1,
                     "experts": 4, "seed": 1234, "steps": args.steps, "analysis_only": True})
        manifest = {"version": 1, "phase": "search", "selection": "2021 validation macro_z only",
                    "development": "never opened during search; previously used development years",
                    "anchor": "fixed saved crossfit first guess; kriging head trained jointly",
                    "jobs": jobs, "start_unix": time.time(), "gpus": gpus}
        if args.manifest.exists():
            prior = json.loads(args.manifest.read_text())
            if prior.get("jobs") != jobs:
                raise SystemExit("existing campaign manifest differs; use a new --manifest path")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2)+"\n")
    elif args.phase == "replicate":
        frozen = json.loads(args.manifest.read_text())
        if "selected" not in frozen:
            raise SystemExit("replication requires a frozen selected-configuration manifest")
        jobs = []
        for cfg in frozen["selected"]:
            for seed in map(int, args.seeds.split(",")):
                jobs.append({**cfg, "tag": cfg["tag"].rsplit("_s", 1)[0]+f"_s{seed}",
                             "seed": seed, "steps": args.steps})
        manifest = {"phase": "replicate", "frozen_selection": str(args.manifest), "jobs": jobs}
    else:
        manifest = json.loads(args.manifest.read_text())
        jobs = manifest["jobs"]
    pending = []
    for job in jobs:
        folder = ROOT / "outputs/latent_ocean" / job["tag"]
        if (folder/f"summary_seed{job['seed']}.json").exists():
            print(f"already completed {job['tag']}", flush=True)
        else:
            pending.append(job)
    running, completed, failures = {}, [], []
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            folder = ROOT / "outputs/latent_ocean" / job["tag"]
            folder.mkdir(parents=True, exist_ok=True)
            logpath = folder / "training.log"
            command = [sys.executable, str(ROOT/"experiments/real_data/69_latent_ocean.py"),
                       "--tag", job["tag"], "--variant", job["variant"],
                       "--width", str(job["width"]), "--latents", str(job["latents"]),
                       "--blocks", str(job["blocks"]), "--context-profiles", str(job["context_profiles"]),
                       "--experts", str(job["experts"]), "--seed", str(job["seed"]),
                       "--steps", str(job["steps"]), "--queries", "1024", "--val-every", "500", "--amp"]
            if job.get("analysis_only"):
                command.append("--analysis-only")
            for field in ("freeze_analysis", "latent_off", "local_off"):
                if job.get(field):
                    command.append("--"+field.replace("_", "-"))
            last = folder / f"last_seed{job['seed']}.pt"
            if last.exists():
                command.extend(["--load-checkpoint", str(last), "--resume-training"])
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OCEAN_ROOT": str(ROOT),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            log = logpath.open("a")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (process, log, job, time.time())
            print(json.dumps({"launched": job["tag"], "gpu": gpu, "pid": process.pid, "log": str(logpath)}), flush=True)
        for gpu, item in list(running.items()):
            process, log, job, started = item
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = {"tag": job["tag"], "returncode": code, "runtime_s": time.time()-started}
            (completed if code == 0 else failures).append(result)
            del running[gpu]
            print(json.dumps({"finished": result, "pending": len(pending)}), flush=True)
        status = {**manifest, "completed": completed, "failures": failures,
                  "running": {str(gpu): item[2]["tag"] for gpu, item in running.items()},
                  "pending": [job["tag"] for job in pending], "updated_unix": time.time()}
        status_path = args.manifest.with_name(args.manifest.stem+f"_{args.phase}_status.json")
        status_path.write_text(json.dumps(status, indent=2)+"\n")
        if pending or running:
            time.sleep(5)
    if failures:
        raise SystemExit(f"{len(failures)} campaign jobs failed; see their logs")
    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
