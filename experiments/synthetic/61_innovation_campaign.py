"""Registered full-budget mechanism contrasts; independent of the original 39.

The original Local Transformer Argo control is reused at identical settings.
Fresh controls and variants train from scratch. All registered three-seed jobs
finish before validation freezes selection and 2005 development is exported.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/innovation_20261007"
OLD = ROOT / "outputs/synthetic_matched_20261007"
SEEDS = (1234, 1235, 1236)
RECIPES = ("local_control", "local_latent_off", "local_off", "profile_none", "profile_shared",
           "profile_independent", "cov_diagonal", "cov_correlated", "aligned_correlated",
           "operator_direct", "operator_update", "all_modules")
CONTRASTS = (
    ("local_latent_off", "local_control", "Shared-latent contribution in the strongest local backbone"),
    ("local_off", "local_control", "Numerical local evidence in the strongest local backbone"),
    ("local_control", "profile_none", "Complete-profile context"),
    ("profile_none", "profile_shared", "Shared T/S depth alignment beyond complete-profile input"),
    ("profile_independent", "profile_shared", "Joint T/S displacement constraint"),
    ("cov_diagonal", "cov_correlated", "Profile/source correlation beyond the same low-rank state covariance"),
    ("profile_shared", "aligned_correlated", "Correlated update added to aligned profiles"),
    ("operator_direct", "operator_update", "Explicit observation residual/operator update"),
    ("aligned_correlated", "all_modules", "Operator increment in the combined system"),
)


def write_json(path, value):
    path = Path(path); temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def registered_jobs():
    previous = json.loads((OLD / "campaign.json").read_text())
    jobs = []
    for surface in (False, True):
        for recipe in RECIPES:
            if not surface and recipe in ("operator_direct", "operator_update", "all_modules"):
                continue
            for seed in SEEDS:
                if recipe == "local_control" and not surface:
                    original = next(j for j in previous["jobs"] if j["family"] == "local_transformer192"
                                    and not j["surface"] and j["seed"] == seed)
                    jobs.append({**original, "family": recipe, "reuse": True,
                                 "original_family": "local_transformer192"})
                    continue
                tag = f"{recipe}_{'surface' if surface else 'argo'}_s{seed}"
                folder = OUTPUT / tag
                command = [sys.executable, "-u", str(ROOT / "experiments/synthetic/60_innovation_reconstruction.py"),
                    "--recipe", recipe, "--variant", "local_transformer", "--width", "192", "--latents", "96",
                    "--blocks", "6", "--query-blocks", "2", "--heads", "8", "--context-profiles", "6080",
                    "--queries", "1024", "--steps", "15000", "--val-every", "1000", "--checkpoint-every", "500",
                    "--seed", str(seed), "--tag", tag, "--output", str(folder), "--amp"]
                if surface:
                    command += ["--surface", "--satellite-cache", str(OLD / "satellites.npz"),
                                "--operator-cache", str(OUTPUT / "operators.npz")]
                jobs.append({"tag": tag, "family": recipe, "kind": "latent", "surface": surface,
                    "seed": seed, "steps": 15000, "output": str(folder), "command": command, "reuse": False})
    return jobs


def completed(job):
    path = Path(job["output"]) / f"summary_seed{job['seed']}.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    steps = summary.get("completed_steps", max((h["step"] for h in summary.get("history", [])), default=0))
    if steps != 15000 or summary["seed"] != job["seed"]:
        raise ValueError(f"Incomplete formal training: {job['tag']}")
    return True


def training_command(job):
    command = list(job["command"])
    last = Path(job["output"]) / f"last_seed{job['seed']}.pt"
    if last.exists():
        command += ["--resume-training", "--load-checkpoint", str(last)]
    return command


def report():
    script = ROOT / "experiments/synthetic/62_innovation_report.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def available_gpus(gpus, occupied, min_free_mb):
    # Do not add jobs to GPUs currently occupied by the original ocean queue.
    state_path = OLD / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    busy = set(map(int, state.get("running", {})))
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    stats = {int(parts[0]): (int(parts[1]), int(parts[2]))
             for parts in (line.split(",") for line in output.strip().splitlines())}
    return [g for g in gpus if g not in occupied and g not in busy
            and stats[g][0] >= min_free_mb and stats[g][1] < 40]


def queue(jobs, gpus, status, *, evaluation=False, stress=False, min_free_mb=32000):
    pending = [j for j in jobs if (not j.get("reuse") and not completed(j))] if not evaluation else list(jobs)
    if evaluation:
        pending = [j for j in pending if not (Path(j["output"]) / f"development_seed{j['seed']}.npz").exists()]
    if stress:
        pending = [j for j in jobs if not (Path(j["output"]) / f"stress_seed{j['seed']}.json").exists()]
    running, attempts = {}, {}
    while pending or running:
        for gpu in available_gpus(gpus, running, min_free_mb):
            if not pending:
                break
            job = pending.pop(0)
            folder = Path(job["output"]); folder.mkdir(parents=True, exist_ok=True)
            if stress:
                command = [sys.executable, "-u", str(ROOT / "experiments/synthetic/65_innovation_stress.py"),
                    "--checkpoint", str(folder / f"best_seed{job['seed']}.pt"),
                    "--output", str(folder / f"stress_seed{job['seed']}.json"),
                    "--seed", "20261007", "--queries-per-month", "256"]
            elif evaluation:
                command = list(job["command"])
                command += ["--evaluate-only", "--development", "--load-checkpoint", str(folder / f"best_seed{job['seed']}.pt")]
            else:
                command = training_command(job)
            phase = "stress" if stress else "evaluate" if evaluation else "train"
            handle = (folder / f"{phase}.log").open("a")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OCEAN_ROOT": str(ROOT),
                   "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[gpu] = (process, handle, job)
            attempts[job["tag"]] = attempts.get(job["tag"], 0) + 1
            print(json.dumps({"phase": phase, "started": job["tag"], "gpu": gpu, "pid": process.pid}), flush=True)
        for gpu, (process, handle, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[gpu]
            if code:
                if attempts[job["tag"]] < 3:
                    pending.append(job)
                else:
                    raise RuntimeError(f"Three failures for {job['tag']}; see its log")
            else:
                if not evaluation and not stress and not completed(job):
                    raise RuntimeError(f"Training exited without full-budget summary: {job['tag']}")
                report()
        status({"running": {str(g): job["tag"] for g, (_, _, job) in running.items()},
                "pending": [j["tag"] for j in pending], "attempts": attempts})
        if pending or running:
            time.sleep(20)


def freeze_selection(manifest_path, manifest):
    if not all(completed(j) for j in manifest["jobs"]):
        raise ValueError("Finish every registered full-budget seed before development scoring")
    spec = importlib.util.spec_from_file_location("matched_report", ROOT / "experiments/synthetic/53_matched_reconstruction_report.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ref = module.load_reference(manifest["baseline_dir"], "validation")
    groups, contracts = {}, []
    for job in manifest["jobs"]:
        prediction, contract = module.validation_prediction(job, ref)
        groups.setdefault((job["surface"], job["family"]), []).append(prediction)
        contract.pop("summary_sha256_at_freeze", None)
        contract.pop("array_sha256_at_freeze", None)
        contracts.append(contract)
    candidates = [{"surface": s, "family": f,
                   "score": module.standardized_rmse(module.ensemble(p)["mean"], ref)}
                  for (s, f), p in groups.items()]
    frozen = {"rule": "minimum ensemble per-variable standardized RMSE on 2004; separate input arms",
              "campaign_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "validation_contracts": contracts, "candidates": candidates,
              "selected": {str(s): min((c for c in candidates if c["surface"] == s), key=lambda c: c["score"])
                           for s in (False, True)}}
    path = OUTPUT / "selection.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Frozen selection changed")
    write_json(path, frozen)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("register", "all", "train", "evaluate"), default="all")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--min-free-mb", type=int, default=32000)
    a = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = registered_jobs()
    manifest = {"version": 1, "experiment": "CESM2 complete-profile, correlation and observation-operator contrasts",
        "train_years": [2000, 2001, 2002, 2003], "validation_year": 2004,
        "development_year_previously_used": 2005, "training_steps": 15000, "seeds": list(SEEDS),
        "inputs_per_month": 6080, "source_profiles_training": 4256, "queries_per_step": 1024,
        "normalization": "same training-year-only normalization as original campaign",
        "baseline_dir": str(OLD / "previous_baselines"), "jobs": jobs,
        "paired_contrasts": [{"control": c, "candidate": v, "question": q} for c, v, q in CONTRASTS],
        "covariance_scope": "PSD within each query neighborhood, not one global joint state posterior",
        "uncertainty_scope": "coherent covariance inside update modules; final signed-gate uncertainty remains a heuristic requiring calibration",
        "hypothesis_status": "candidate mechanisms; no novelty or efficacy claim before matched results",
        "stress_tests": ["duplicate source record ID", "independent repeated measurement", "shared profile bias",
                         "masked depths", "observation noise", "sparse profile coverage", "query chunk invariance"]}
    path = OUTPUT / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to alter registered innovation campaign")
    write_json(path, manifest)
    if a.phase == "register":
        print(json.dumps({"jobs": len(jobs), "reused": sum(j["reuse"] for j in jobs), "manifest": str(path)}))
        return
    gpus = list(map(int, a.gpus.split(",")))
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU indices must be distinct")
    phase = "train"
    def status(extra=None):
        write_json(OUTPUT / "status.json", {"phase": phase, "updated_unix": time.time(),
            "training_complete": sum(completed(j) for j in jobs), "registered": len(jobs), **(extra or {})})
    try:
        if a.phase in ("all", "train"):
            queue(jobs, gpus, status, min_free_mb=a.min_free_mb)
        if a.phase in ("all", "evaluate"):
            if not all(completed(j) for j in jobs):
                raise ValueError("Finish every registered training before stress and development")
            phase = "source_robustness"; status()
            queue(jobs, gpus, status, stress=True, min_free_mb=a.min_free_mb)
            phase = "validation_selection"; status()
            freeze_selection(path, manifest)
            phase = "development"; status()
            queue(jobs, gpus, status, evaluation=True, min_free_mb=a.min_free_mb)
        phase = "complete" if a.phase != "train" else "training_complete"
        status(); report()
    except Exception as error:
        status({"failure": repr(error)}); report()
        raise


if __name__ == "__main__":
    main()
