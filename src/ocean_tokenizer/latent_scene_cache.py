"""Exact, file-backed preparation cache for the matched ocean experiments.

Preparation is an explicit, single-process operation. Training only reads a
published manifest and refuses changed scientific inputs or damaged artifacts.
The cache changes storage and startup memory, never sampling or normalization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .argo_obs import ArgoCohort, ArgoNorm

FORMAT = 1
TENSOR_FIELDS = ("x", "y", "fg", "valid", "innovation", "error", "error_valid",
                 "lat", "lon", "day", "depth", "state", "state_ok")
COHORT_FIELDS = ("levels", "month_index", "grid_y", "grid_x", "lat", "lon", "wmo",
                 "year", "year_split", "float_split", "TEMP", "SALT", "TEMP_ERR",
                 "SALT_ERR", "ECCO_TEMP", "ECCO_SALT")


def cache_directory(root, anchor_tag="anc_satday_dfs"):
    return Path(root) / "outputs/latent_ocean/prepared_scenes" / anchor_tag


def file_digest(path):
    """Streaming SHA256: no full input or artifact is loaded into process RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return {"sha256": h.hexdigest(), "bytes": Path(path).stat().st_size}


def provenance(root, anchor_tag="anc_satday_dfs"):
    root = Path(root)
    anchor = f"outputs/audit/global/{anchor_tag}"
    inputs = ("data/argo_cohort/global_global.nc", "data/argo_cohort/global_anomcell.npz",
              "data/argo_cohort/global_satellite.npz", f"{anchor}/first_guess_seed1234.pt",
              f"{anchor}/first_guess_predictions_seed1234.pt",
              "outputs/audit/global/anc_satday_kriging/model_seed1234.pt",
              "outputs/audit/global/setconv_surface/summary_seed1234.json")
    source = Path(__file__).parent
    recipes = ("latent_experiment.py", "latent_scene_cache.py", "argo_obs.py",
               "audit_tools.py", "anchored.py", "point_baselines.py")
    return {"inputs": {name: file_digest(root / name) for name in inputs},
            "recipes": {name: file_digest(source / name) for name in recipes}}


def provenance_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _artifact(path, base, array=None):
    result = {"path": str(path.relative_to(base)), **file_digest(path)}
    if array is not None:
        result.update(shape=list(array.shape), dtype=array.dtype.str)
    return result


def _checked_path(base, record):
    path = (base / record["path"]).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("prepared-scene cache artifact leaves its directory")
    if not path.is_file() or file_digest(path) != {k: record[k] for k in ("sha256", "bytes")}:
        raise ValueError(f"prepared-scene cache artifact integrity mismatch: {record['path']}")
    return path


def prepare_scene_cache(root, anchor_tag="anc_satday_dfs", directory=None, *, replace=False):
    """Publish a verified cache atomically after one ordinary CPU preparation.

    The manifest is the only publication point. An interrupted preparation has
    no readable cache, and concurrent training cannot observe partial arrays.
    """
    from .latent_experiment import OceanScenes

    root = Path(root)
    base = Path(directory) if directory is not None else cache_directory(root, anchor_tag)
    original = provenance(root, anchor_tag)
    key = provenance_key(original)
    manifest_path = base / "manifest.json"
    if manifest_path.exists() and not replace:
        current = json.loads(manifest_path.read_text())
        if current.get("format") != FORMAT or current.get("cache_key") != key:
            raise ValueError("existing prepared-scene cache provenance changed; prepare with --replace")
        # Verify all artifacts before reporting an existing cache as prepared.
        for record in [*current["arrays"].values(), current["state"]]:
            _checked_path(base, record)
        return current

    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=base))
    try:
        scenes = OceanScenes(root, device="cpu", anchor_tag=anchor_tag, use_prepared_cache=False)
        arrays = {}
        for field in TENSOR_FIELDS:
            value = getattr(scenes, field).detach().cpu().numpy()
            path = staging / f"tensor_{field}.npy"
            np.save(path, value, allow_pickle=False)
            arrays[f"tensor.{field}"] = _artifact(path, staging, value)
        for field in COHORT_FIELDS:
            value = getattr(scenes.c, field, None)
            if value is None:
                continue
            value = np.asarray(value)
            path = staging / f"cohort_{field}.npy"
            np.save(path, value, allow_pickle=False)
            arrays[f"cohort.{field}"] = _artifact(path, staging, value)
        state = {"metadata": scenes.metadata, "fg_checkpoint": scenes.fg_checkpoint,
                 "reference": scenes.reference, "months_available": sorted(scenes.months_available),
                 "norm": {name: {ch: torch.as_tensor(value).clone() for ch, value in getattr(scenes.norm, name).items()}
                          for name in ("mean", "std")},
                 "cohort_region": scenes.c.region, "cohort_grid": list(scenes.c.grid)}
        state_path = staging / "state.pt"
        torch.save(state, state_path)
        state_record = _artifact(state_path, staging)
        if provenance(root, anchor_tag) != original:
            raise ValueError("scientific inputs changed during prepared-scene cache creation")
        # A content-addressed generation can be reused if another prepare won.
        datasets = base / "datasets"
        datasets.mkdir(exist_ok=True)
        generation = datasets / key
        if generation.exists():
            # Refuse overwriting a damaged previously published generation.
            for record in [*arrays.values(), state_record]:
                _checked_path(generation, record)
            shutil.rmtree(staging)
        else:
            staging.rename(generation)
        for record in [*arrays.values(), state_record]:
            record["path"] = str(Path("datasets") / key / record["path"])
        manifest = {"format": FORMAT, "cache_key": key, "anchor_tag": anchor_tag,
                    "provenance": original, "arrays": arrays, "state": state_record,
                    "storage": "copy-on-write mmap; exact tensor and cohort dtypes",
                    "n_profiles": scenes.metadata["n_profiles"], "n_levels": scenes.n_levels}
        temporary_manifest = base / f".manifest-{os.getpid()}.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_scene_cache(scenes, anchor_tag="anc_satday_dfs", directory=None):
    """Return False only when no published cache exists; stale caches raise."""
    base = Path(directory) if directory is not None else cache_directory(scenes.root, anchor_tag)
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("anchor_tag") != anchor_tag:
        raise ValueError("prepared-scene cache format or anchor mismatch")
    current = provenance(scenes.root, anchor_tag)
    if manifest.get("provenance") != current or manifest.get("cache_key") != provenance_key(current):
        raise ValueError("prepared-scene cache provenance mismatch; rerun 73_prepare_latent_scenes.py --replace")
    mapped = {}
    for name, record in manifest["arrays"].items():
        # Mode c is writable without changing the cache file. On CPU, tensors
        # share these file-backed pages; CUDA copies only prepared arrays.
        value = np.load(_checked_path(base, record), mmap_mode="c", allow_pickle=False)
        if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
            raise ValueError(f"prepared-scene array contract mismatch: {name}")
        mapped[name] = value
    state = torch.load(_checked_path(base, manifest["state"]), map_location="cpu", weights_only=True)
    kwargs = {name: mapped[f"cohort.{name}"] for name in COHORT_FIELDS if f"cohort.{name}" in mapped}
    scenes.c = ArgoCohort(region=state["cohort_region"], grid=tuple(state["cohort_grid"]), **kwargs)
    uniq, start = np.unique(scenes.c.month_index, return_index=True)
    stop = np.r_[start[1:], scenes.c.month_index.size]
    scenes.c._by_month = {int(m): (int(a), int(b)) for m, a, b in zip(uniq, start, stop)}
    for field in TENSOR_FIELDS:
        setattr(scenes, field, torch.from_numpy(mapped[f"tensor.{field}"]).to(scenes.device))
    scenes.norm = ArgoNorm(**{name: {ch: value.numpy() for ch, value in state["norm"][name].items()}
                            for name in ("mean", "std")})
    scenes.levels = scenes.c.levels
    scenes.n_levels = len(scenes.levels)
    scenes.metadata, scenes.fg_checkpoint, scenes.reference = (state[name] for name in ("metadata", "fg_checkpoint", "reference"))
    scenes.months_available = set(state["months_available"])
    scenes._months_cache, scenes._training_pools = {}, {}
    # Keep arrays alive explicitly, including CPU tensor backing stores.
    scenes._prepared_mmaps = mapped
    scenes.prepared_cache = {"manifest": str(manifest_path), "cache_key": manifest["cache_key"]}
    return True
