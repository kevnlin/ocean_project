"""Complete the validation-selected ocean latent campaign, including controls.

Every training job keeps the registered 6000 steps. The earlier development
years are opened only after architecture selection and all training finish.
Restarting this driver reuses completed jobs and complete continuation states.
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
OUTPUT = ROOT / "outputs/latent_ocean"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def ablation_jobs(selected, summaries, seeds=(1234, 1235, 1236)):
    neural = [job for job in selected if not job.get("analysis_only")]
    winner = min(neural, key=lambda job: summaries[job["tag"]]["validation"]["scores"]["macro_z"])
    prefix = winner["tag"].rsplit("_s", 1)[0]
    jobs = []
    for name, flags in (("full", {}), ("latent_off", {"latent_off": True}),
                        ("local_off", {"local_off": True})):
        for seed in seeds:
            jobs.append({**winner, **flags, "freeze_analysis": True, "seed": seed,
                         "tag": f"{prefix}_fixed_oi_{name}_s{seed}"})
    return {"version": 1, "selection": "overall neural winner on 2021 validation only",
            "selected_neural": winner["tag"], "development": "never used to select controls",
            "jobs": jobs}


def evaluate_jobs(jobs, gpus, status_callback):
    pending = []
    for job in jobs:
        folder = OUTPUT / job["tag"]
        summary = folder / f"summary_seed{job['seed']}.json"
        if not summary.exists():
            raise ValueError(f"training is incomplete before frozen evaluation: {job['tag']}")
        saved = json.loads(summary.read_text())
        if (saved.get("tag") != job["tag"] or saved.get("seed") != job["seed"]
                or any(saved.get("training", {}).get(name) != job[name]
                       for name in ("steps", "context_profiles"))
                or any(saved.get("config", {}).get(name) != job[field]
                       for name, field in (("variant", "variant"), ("width", "width"),
                                           ("n_latents", "latents"), ("n_blocks", "blocks"),
                                           ("n_experts", "experts")))):
            raise ValueError(f"training summary differs from registered job: {job['tag']}")
        for flag in ("analysis_only", "freeze_analysis", "latent_off", "local_off"):
            if bool(saved.get("training", {}).get(flag)) != bool(job.get(flag)):
                raise ValueError(f"training summary {flag} differs from registered job: {job['tag']}")
        if "development" in saved and (folder/f"development_seed{job['seed']}.npz").exists():
            continue
        pending.append(job)
    running = {}
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            folder = OUTPUT / job["tag"]
            log = (folder/"development_evaluation.log").open("a")
            command = [sys.executable, "-u", str(ROOT/"experiments/real_data/69_latent_ocean.py"),
                       "--tag", job["tag"], "--seed", str(job["seed"]),
                       "--load-checkpoint", str(folder/f"best_seed{job['seed']}.pt"),
                       "--evaluate-only", "--development"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OCEAN_ROOT": str(ROOT),
                   "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = process, log, job
            print(json.dumps({"evaluating": job["tag"], "gpu": gpu, "pid": process.pid}), flush=True)
        for gpu, (process, log, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            if code:
                for other, handle, _ in running.values():
                    other.terminate()
                    handle.close()
                raise RuntimeError(f"development evaluation failed: {job['tag']} ({code})")
            print(json.dumps({"evaluated": job["tag"]}), flush=True)
        status_callback({"running": {str(g): item[2]["tag"] for g, item in running.items()},
                         "pending": [job["tag"] for job in pending]})
        if pending or running:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("provide distinct GPU indices")
    manifest = OUTPUT/"campaign_20261007.json"
    selected_path = OUTPUT/"selected_20261007.json"
    ablation_path = OUTPUT/"ablations_20261007.json"
    state_path = OUTPUT/"full_campaign_status_20261007.json"
    started = time.time()
    phase = "prepare"

    def status(extra=None):
        write_json(state_path, {"phase": phase, "gpus": gpus, "started_unix": started,
                               "updated_unix": time.time(), **(extra or {})})

    def run(number, *arguments):
        script = next((ROOT/"experiments/real_data").glob(f"{number}_*.py"))
        print(json.dumps({"phase": phase, "script": str(script), "arguments": list(map(str, arguments))}), flush=True)
        status()
        subprocess.run([sys.executable, "-u", str(script), *map(str, arguments)], cwd=ROOT, check=True)

    try:
        if not args.skip_prepare:
            run("73")
        phase = "search"
        run("70", "--gpus", args.gpus, "--phase", "search", "--manifest", manifest)
        phase = "freeze_selection"
        run("71", "--manifest", manifest, "--freeze-selection", selected_path)
        selected = json.loads(selected_path.read_text())["selected"]
        phase = "replicate"
        run("70", "--gpus", args.gpus, "--phase", "replicate", "--manifest", selected_path)
        summaries = {job["tag"]: json.loads((OUTPUT/job["tag"]/f"summary_seed{job['seed']}.json").read_text())
                     for job in selected}
        ablations = ablation_jobs(selected, summaries)
        if ablation_path.exists() and json.loads(ablation_path.read_text()) != ablations:
            raise ValueError("refusing to change previously frozen component controls")
        write_json(ablation_path, ablations)
        phase = "ablate"
        run("70", "--gpus", args.gpus, "--phase", "ablate", "--manifest", ablation_path)
        phase = "final_validation_selection"
        rule_path = OUTPUT/"final_selection_rule_20261007.json"
        if not rule_path.exists():
            run("76", "--register-only")
        run("76")
        if not (OUTPUT/"final_selection_20261007.json").exists():
            raise ValueError("final validation-only ensemble recommendation is not frozen")
        evaluation = [{**job, "seed": seed,
                       "tag": job["tag"].rsplit("_s", 1)[0]+f"_s{seed}"}
                      for job in selected for seed in (1234, 1235, 1236)] + ablations["jobs"]
        phase = "frozen_development_evaluation"
        evaluate_jobs(evaluation, gpus, status)
        phase = "reports"
        run("71", "--manifest", manifest, "--ablation-manifest", ablation_path)
        run("72", "--manifest", manifest)
        phase = "complete"
        status({"training_runs": 24, "development_evaluations": len(evaluation),
                "runtime_s": time.time()-started})
        print("full campaign complete", flush=True)
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
