"""Replay the fixed CESM2 scattered-profile baselines at their original cells.

The OI setting is read from the existing 2004-validation selection; nothing is
tuned on 2005.  Arrays use the latent evaluator's month/profile/level/target/
mean/baseline names.  Physical arrays are anomalies, in degC and PSU.  Additional
absolute arrays reconstruct the field with the saved position climatology so
that R2/correlation can explicitly distinguish anomaly from absolute state.

No uncertainty is assigned to deterministic predictions.  Missing historical
MLP/token weights are reported, never replaced with newly trained predictions.
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ocean_tokenizer import synth_argo_eval as E
from ocean_tokenizer.baselines import MLP
from ocean_tokenizer.audit_tools import cohort_path
from ocean_tokenizer.point_baselines import (
    CH, Scores, band_of_levels, mlp_point_features, nearest_profile, oi_points,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scoring_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in ("month", "profile", "level", "target"):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def compare_scores(actual: dict, expected: dict, name: str) -> None:
    for ch in CH:
        if actual[ch]["n"] != expected[ch]["n"]:
            raise ValueError(f"{name}/{ch}: scored-cell count changed")
        for key in ("rmse_z", "rmse_physical", "climatology_z"):
            if not np.isclose(actual[ch][key], expected[ch][key], rtol=1e-5, atol=1e-8):
                raise ValueError(f"{name}/{ch}/{key}: prediction replay differs from history")


def replay_month(c, obs, ev, selection):
    src, tgt = ev["src"], ev["tgt"]
    bands = band_of_levels(c.levels)
    results = {"climatology": np.zeros((tgt.size, c.levels.size, 2)),
               "nearest_profile": np.empty((tgt.size, c.levels.size, 2)),
               "oi": np.empty((tgt.size, c.levels.size, 2))}
    for ci, ch in enumerate(CH):
        for li in range(c.levels.size):
            args = (c.lat[src], c.lon[src], obs[ch][src, li], c.lat[tgt], c.lon[tgt])
            results["nearest_profile"][:, li, ci] = nearest_profile(*args)
            setting = selection[ch][str(bands[li])]
            results["oi"][:, li, ci] = oi_points(
                *args, L_km=setting["L_km"], gamma=setting["gamma"], k=setting["k"])
    return {name: value[ev["prof"], ev["lev"]] for name, value in results.items()}


def load_optional_mlp(root: Path, seed: int):
    directory = root / "outputs/audit/synthetic/pointwise_mlp"
    checkpoint = directory / f"model_seed{seed}.pt"
    if not checkpoint.exists():
        return None
    summary = json.loads((directory / f"summary_seed{seed}.json").read_text())
    model = MLP(len(summary["features"]), summary["hidden"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model, summary, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["validation", "development"],
                        choices=["validation", "development"])
    parser.add_argument("--mlp-seeds", nargs="+", type=int, default=[1234])
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "outputs/synthetic_matched_20261007/previous_baselines"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ref_path = root / "outputs/audit/synthetic/fixed_baselines/summary.json"
    reference = json.loads(ref_path.read_text())
    cpath = Path(cohort_path(str(root), "synthetic"))
    actual_sha = sha256(cpath)
    if actual_sha != reference["cohort"]["sha256"]:
        raise ValueError("Synthetic cohort differs from the validation-tuned OI cohort")
    c, norm, obs = E.load(str(root))
    # ArgoCohort.load and the climatology both use the stable month ordering.
    with xr.open_dataset(cpath) as ds:
        order = np.argsort(np.asarray(ds.month_index.values, int), kind="stable")
        clim = np.stack([np.asarray(ds[f"CLIM_POS_{ch}"].values, float)[order]
                         for ch in CH], axis=-1)
    norm_mean = np.stack([norm.mean[ch] for ch in CH], axis=-1)
    norm_std = np.stack([norm.std[ch] for ch in CH], axis=-1)
    optional = {seed: load_optional_mlp(root, seed) for seed in args.mlp_seeds}
    inventory = {
        "cohort_sha256": actual_sha, "cohort_path": str(cpath),
        "splits": E.SPLITS, "channels": list(CH), "physical_units": ["degC", "PSU"],
        "input_profiles_per_month": E.N_INPUT, "query_profiles_per_month": E.N_QUERY,
        "target": "climatology interpolated to each profile position; train-only normalization",
        "oi_selection_source": str(ref_path), "oi_selection_sha256": sha256(ref_path),
        "runner_sha256": sha256(Path(__file__)), "uncertainty": "not available for deterministic rows",
        "missing_checkpoints": [str(root / "outputs/audit/synthetic/pointwise_mlp" /
                                     f"model_seed{seed}.pt") for seed, item in optional.items()
                                if item is None],
        "exports": {},
    }
    for split in args.splits:
        evs = E.eval_set(c, obs, split, E.EVAL_CELLS if split == "validation" else 0)
        methods = ["climatology", "nearest_profile", "oi"] + [
            f"pointwise_mlp_s{seed}" for seed, item in optional.items() if item is not None]
        collected = {name: [] for name in methods}
        common = {name: [] for name in ("month", "profile", "level", "target", "climatology_physical")}
        scores = {name: Scores(c.levels, norm.std) for name in methods}
        for ev in evs:
            predictions = replay_month(c, obs, ev, reference["oi_selection"])
            for seed, item in optional.items():
                if item is None:
                    continue
                model, _, _ = item
                features = mlp_point_features(
                    c.lat[ev["src"]], c.lon[ev["src"]],
                    {ch: obs[ch][ev["src"]] for ch in CH}, c.lat[ev["tgt"]],
                    c.lon[ev["tgt"]], c.levels, month=ev["month"] % 12 + 1)
                with torch.no_grad():
                    p = model(torch.from_numpy(features)).numpy().astype(np.float64)
                predictions[f"pointwise_mlp_s{seed}"] = p[
                    ev["prof"] * c.levels.size + ev["lev"]]
            target = np.stack([ev["target"][ch] for ch in CH], axis=-1)
            rows = ev["tgt"][ev["prof"]]
            common["month"].append(np.full(ev["lev"].size, ev["month"], dtype=np.int64))
            common["profile"].append(rows.astype(np.int64))
            common["level"].append(ev["lev"].astype(np.int64))
            common["target"].append(target)
            common["climatology_physical"].append(clim[rows, ev["lev"]])
            for name in methods:
                collected[name].append(predictions[name])
                for ci, ch in enumerate(CH):
                    scores[name].add(ch, predictions[name][:, ci], target[:, ci], ev["lev"])
            print(f"{split}: month {ev['month']} replayed ({time.time() - start:.1f}s)", flush=True)
        arrays = {name: np.concatenate(parts) for name, parts in common.items()}
        sd, mu = norm_std[arrays["level"]], norm_mean[arrays["level"]]
        arrays.update(levels=np.asarray(c.levels), normalization_mean=norm_mean,
                      normalization_std=norm_std,
                      target_physical=arrays["target"] * sd + mu)
        arrays["target_absolute_physical"] = arrays["target_physical"] + arrays["climatology_physical"]
        fingerprint = scoring_fingerprint(arrays)
        baseline = np.concatenate(collected["oi"])
        inventory["exports"][split] = {}
        for name in methods:
            scored = scores[name].result()
            if name in reference["baselines"]:
                compare_scores(scored, reference["baselines"][name]["scores"][split], name)
            else:
                seed = int(name.rsplit("s", 1)[-1])
                compare_scores(scored, optional[seed][1]["scores"][split], name)
            prediction = np.concatenate(collected[name])
            mean_physical = prediction * sd + mu
            metadata = {"method": name, "split": split, "cohort_sha256": actual_sha,
                        "scoring_fingerprint": fingerprint, "physical_arrays": "anomalies",
                        "uncertainty_available": False, "oi_tuned_on": "2004 validation only"}
            artifact = out / f"{name}_{split}.npz"
            temporary = out / f".{artifact.name}.tmp"
            with temporary.open("wb") as f:
                np.savez_compressed(
                    f, **arrays, mean=prediction, baseline=baseline,
                    mean_physical=mean_physical,
                    mean_absolute_physical=mean_physical + arrays["climatology_physical"],
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
            temporary.replace(artifact)
            # Compatibility scores are kept solely to validate the original RMSE.
            # The formal multi-metric report is generated from these full arrays.
            inventory["exports"][split][name] = {
                "path": str(artifact), "sha256": sha256(artifact),
                "scoring_fingerprint": fingerprint, "historical_rmse_replay_verified": True,
                "n": {ch: scored[ch]["n"] for ch in CH},
                "rmse_physical": {ch: scored[ch]["rmse_physical"] for ch in CH}}
    inventory["runtime_s"] = time.time() - start
    (out / "replay_manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory["exports"], indent=2), flush=True)


if __name__ == "__main__":
    main()
