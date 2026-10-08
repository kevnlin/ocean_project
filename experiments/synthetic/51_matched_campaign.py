"""Registered 3-seed, 15,000-step matched CESM2 reconstruction campaign.

Training and architecture selection use 2000-2004 only. All 2005 prediction
exports are deferred until the validation-only selection artifact is frozen.
Run this driver in a memory-limited user systemd service on available GPUs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/synthetic_matched_20261007"
SEEDS = (1234, 1235, 1236)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def registered_jobs():
    jobs = []
    recipes = [
        ("previous_token64", "previous", False, []),
        ("dense64", "latent", False, ["--variant", "dense"]),
        ("soft_moe192", "latent", False, ["--variant", "soft_moe"]),
        ("dense192", "latent", False, ["--variant", "dense"]),
        ("local_transformer192", "latent", False, ["--variant", "local_transformer"]),
        ("soft_moe192_latent_off", "latent", False, ["--variant", "soft_moe", "--latent-off"]),
        ("soft_moe192_local_off", "latent", False, ["--variant", "soft_moe", "--local-off"]),
        ("official4dvarnet", "fourdvar", False, []),
        ("dense192", "latent", True, ["--variant", "dense"]),
        ("soft_moe192", "latent", True, ["--variant", "soft_moe"]),
        ("soft_moe192_latent_off", "latent", True, ["--variant", "soft_moe", "--latent-off"]),
        ("official4dvarnet", "fourdvar", True, []),
        ("previous_token64", "previous_surface", True, []),
    ]
    for family, kind, surface, flags in recipes:
        for seed in SEEDS:
            tag = f"{family}_{'surface' if surface else 'argo'}_s{seed}"
            folder = OUTPUT / tag
            command = [sys.executable, "-u"]
            if kind in ("previous", "previous_surface"):
                script = "52_previous_token_surface.py" if surface else "49_previous_token_matched.py"
                command += [str(ROOT / "experiments/synthetic" / script),
                    "--region", "synthetic", "--mode", "train", "--backbone", "d4rt",
                    "--mass-mode", "dfs", "--ablation", "anomaly_exact", "--seed", str(seed),
                    "--n-latent", "64", "--refiner-km", "500", "--refiner-gate", "1.0",
                    "--steps", "15000", "--tag", tag, "--out-root", str(folder),
                    "--checkpoint-every", "500", "--no-final-development"]
                if surface:
                    command += ["--surface", "--satellite-cache", str(OUTPUT / "satellites.npz")]
            elif kind == "latent":
                command += [str(ROOT / "experiments/synthetic/46_latent_reconstruction.py"),
                    "--tag", tag, "--output", str(folder), "--seed", str(seed),
                    "--steps", "15000", "--context-profiles", "6080", "--queries", "1024",
                    "--val-every", "1000", "--checkpoint-every", "500", "--amp", *flags]
                if family != "dense64":
                    command += ["--width", "192", "--latents", "96", "--blocks", "6",
                                "--query-blocks", "2", "--heads", "8", "--experts", "4"]
                if surface:
                    command += ["--surface", "--satellite-cache", str(OUTPUT / "satellites.npz")]
            else:
                command += [str(ROOT / "experiments/synthetic/48_official_4dvarnet.py"),
                    "--tag", tag, "--output", str(folder), "--seed", str(seed),
                    "--steps", "15000", "--val-every", "1000", "--checkpoint-every", "500",
                    "--no-final-development"]
                if surface:
                    command += ["--surface", "--satellite-cache", str(OUTPUT / "satellites.npz")]
            jobs.append({"tag": tag, "family": family, "kind": kind, "seed": seed,
                         "surface": surface, "steps": 15000, "output": str(folder),
                         "command": command})
    return jobs


def completed(job):
    path = Path(job["output"]) / f"summary_seed{job['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    steps = summary.get("completed_steps")
    if steps is None:
        steps = max((record["step"] for record in summary.get("history", [])), default=0)
    if steps != job["steps"]:
        raise ValueError(f"Incomplete training summary: {job['tag']} ({steps})")
    if summary.get("seed") != job["seed"]:
        raise ValueError(f"Summary seed differs: {job['tag']}")
    return True


def ready(job):
    if not Path(job["command"][2]).exists():
        return False
    if job["kind"] in ("fourdvar", "previous_surface"):
        if not (OUTPUT / f"ready_{job['kind']}.json").exists():
            return False
    return not job["surface"] or (OUTPUT / "satellites.npz").exists()


def training_command(job):
    command = list(job["command"])
    folder, seed = Path(job["output"]), job["seed"]
    if job["kind"] in ("previous", "previous_surface"):
        if (folder / f"last_training_seed{seed}.pt").exists():
            command += ["--resume-training"]
    elif (folder / f"last_seed{seed}.pt").exists():
        command += ["--resume-training", "--load-checkpoint", str(folder / f"last_seed{seed}.pt")]
    return command


def evaluation_command(job):
    command = list(job["command"])
    if job["kind"] in ("previous", "previous_surface"):
        command.remove("--no-final-development")
        command += ["--evaluation-only"]
    else:
        if "--no-final-development" in command:
            command.remove("--no-final-development")
        command += ["--evaluate-only", "--development", "--load-checkpoint",
                    str(Path(job["output"]) / f"best_seed{job['seed']}.pt")]
    return command


def evaluate_fixed_mlp(selection, gpu, *, wait=False):
    """Export the original 30-epoch auxiliary baseline after selection only."""
    while not selection.exists():
        if not wait:
            raise ValueError("Freeze validation selection before fixed MLP evaluation")
        time.sleep(5)
    folder = OUTPUT / "fixed_pointwise_mlp"
    if (folder / "development_seed1234.npz").exists():
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OCEAN_ROOT": str(ROOT),
           "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    command = [sys.executable, "-u", str(ROOT / "experiments/synthetic/54_fixed_pointwise_mlp.py"),
               "--seed", "1234", "--out", str(folder), "--evaluation-only", "--development",
               "--selection", str(selection)]
    with (folder / "evaluation.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def queue(jobs, gpus, phase, status):
    pending = [job for job in jobs if not completed(job)] if phase == "train" else list(jobs)
    running, attempts = {}, {}
    while pending or running:
        for gpu in gpus:
            if gpu in running:
                continue
            choice = next((j for j in pending if ready(j)), None)
            if choice is None:
                continue
            pending.remove(choice)
            folder = Path(choice["output"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / f"{phase}.log").open("a")
            command = training_command(choice) if phase == "train" else evaluation_command(choice)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OCEAN_ROOT": str(ROOT),
                   "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = process, log, choice
            attempts[choice["tag"]] = attempts.get(choice["tag"], 0) + 1
            print(json.dumps({"phase": phase, "started": choice["tag"], "gpu": gpu, "pid": process.pid}), flush=True)
        for gpu, (process, log, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[gpu]
            if code:
                if attempts[job["tag"]] < 3:
                    pending.append(job)
                    print(json.dumps({"retry": job["tag"], "exit_code": code,
                                      "continuation": "complete optimizer and RNG state"}), flush=True)
                else:
                    for other, handle, _ in running.values():
                        other.terminate()
                        handle.close()
                    raise RuntimeError(f"{phase} failed: {job['tag']} ({code}); see {job['output']}/{phase}.log")
            else:
                if phase == "train" and not completed(job):
                    raise ValueError(f"Run exited without complete summary: {job['tag']}")
                print(json.dumps({"phase": phase, "completed": job["tag"]}), flush=True)
        status({"running": {str(g): item[2]["tag"] for g, item in running.items()},
                "pending": [job["tag"] for job in pending], "attempts": attempts})
        if pending or running:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,6,7")
    parser.add_argument("--phase", choices=("register", "train", "evaluate", "all", "auxiliary-evaluate"), default="all")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus) or not gpus:
        raise ValueError("GPU indices must be distinct")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "auxiliary-evaluate":
        evaluate_fixed_mlp(OUTPUT / "selection.json", gpus[0], wait=True)
        return
    manifest_path = OUTPUT / "campaign.json"
    manifest = {"version": 1, "experiment": "CESM2 matched ocean reconstruction",
        "train_years": [2000, 2001, 2002, 2003], "validation_year": 2004,
        "previously_used_test_year": 2005, "inputs_per_month": 6080,
        "training_steps": 15000, "seeds": list(SEEDS),
        "primary_metrics": ["temperature_RMSE_C", "salinity_RMSE_PSU"],
        "selection": "minimum validation ensemble mean per-variable standardized RMSE, training depth-wise scales; include deterministic OI",
        "baseline_dir": str(OUTPUT / "previous_baselines"), "jobs": registered_jobs()}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to alter the registered campaign")
    write_json(manifest_path, manifest)
    if args.phase == "register":
        print(f"registered {len(manifest['jobs'])} full training runs: {manifest_path}")
        return
    started, phase = time.time(), args.phase

    def status(extra=None):
        write_json(OUTPUT / "status.json", {"phase": phase, "gpus": gpus,
            "started_unix": started, "updated_unix": time.time(), **(extra or {})})

    report = ROOT / "experiments/synthetic/53_matched_reconstruction_report.py"
    selection = OUTPUT / "selection.json"
    try:
        if args.phase in ("train", "all"):
            phase = "train"
            queue(manifest["jobs"], gpus, phase, status)
        if args.phase in ("evaluate", "all"):
            if not all(completed(job) for job in manifest["jobs"]):
                raise ValueError("All registered training must finish before evaluation")
            phase = "validation_selection"
            status()
            subprocess.run([sys.executable, str(report), "--campaign", str(manifest_path),
                            "--freeze-selection", str(selection)], cwd=ROOT, check=True)
            if not selection.exists():
                raise ValueError("Validation selection was not frozen")
            phase = "frozen_development_evaluation"
            queue(manifest["jobs"], gpus, "evaluate", status)
            evaluate_fixed_mlp(selection, gpus[0])
            phase = "report"
            status()
            subprocess.run([sys.executable, str(report), "--campaign", str(manifest_path),
                            "--selection", str(selection), "--report",
                            str(ROOT / "reports/synthetic/matched_reconstruction_20261007.md"),
                            "--output", str(OUTPUT / "metrics.json")], cwd=ROOT, check=True)
            if not (OUTPUT / "metrics.json").exists():
                raise ValueError("Final metrics are missing; a pending report is not completion")
        phase = "complete" if args.phase != "train" else "training_complete"
        status({"training_runs": len(manifest["jobs"]), "runtime_s": time.time() - started})
    except Exception as error:
        status({"failure": repr(error)})
        raise


if __name__ == "__main__":
    main()
