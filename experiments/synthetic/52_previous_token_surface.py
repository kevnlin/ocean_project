"""Original 64-slot token architecture with the new documented CESM2 surfaces.

The old architecture, profile compression, query decoder, refiner, loss and
15,000-step training recipe are preserved.  Shared 50 surface_z fields replace
the real-satellite ingest in 62.  SST/SSS use the original pair patch encoder;
steric sea level uses its original SSH patch encoder.  This is a newly trained
multimodal arm, not an exact replay of the unlocated pasted historical numbers.

The legacy token architecture receives dense 3x3-degree patches, while 46 uses
the same products sampled at profile/query positions. This presentation
difference belongs to the architecture and must be stated in comparisons.
The original model is deterministic: no predictive standard deviation is
invented.  Normalization scales and complete mean/target arrays are exported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
LEGACY = ROOT / "experiments/real_data/62_sanity_train.py"
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_surface_cache(path, c, norm, device, *, require_global_grid=True):
    import torch
    from ocean_tokenizer.latent_experiment import array_fingerprint
    from ocean_tokenizer.synth_argo_eval import SPLITS

    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        features = cache["features"].copy()
        months = cache["grid_month"].copy()
        z = cache["surface_z"].copy()
        lat, lon = cache["grid_lat"].copy(), cache["grid_lon"].copy()
        for name, expected in (("month_index", c.month_index), ("lat", c.lat), ("lon", c.lon)):
            if not np.array_equal(cache[name], expected):
                raise ValueError(f"surface cache {name} is not the exact cohort")
        mean = np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], axis=-1)
        std = np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], axis=-1)
        if not np.array_equal(cache["normalization_mean"], mean) or not np.array_equal(cache["normalization_std"], std):
            raise ValueError("surface cache profile normalization differs")
    standardized = np.stack([norm.z(ch, getattr(c, ch)) for ch in ("TEMP", "SALT")], axis=-1)
    cohort_fingerprint = array_fingerprint((c.month_index, c.lat, c.lon, c.wmo,
                                           c.levels, standardized))
    if metadata.get("cohort_fingerprint") != cohort_fingerprint:
        raise ValueError("surface cache anomaly/cohort fingerprint differs")
    if metadata.get("normalization_scope") != "training years only" or metadata.get("splits") != {k: list(v) for k, v in SPLITS.items()}:
        raise ValueError("surface cache is not training-only normalized under the registered splits")
    if metadata.get("surface_variables") != ["SST", "SSS", "steric_SSH"]:
        raise ValueError("surface channel order must be SST, SSS, steric_SSH")
    if metadata.get("surface_is_independent_of_truth") is not False or metadata.get("satellite_noise") != "none":
        raise ValueError("expected documented noise-free truth-derived synthetic fields")
    if metadata.get("p_ref_dbar") != 990.0:
        raise ValueError("synthetic steric height reference pressure differs")
    if features.shape != (len(c.lat), 56) or not np.isfinite(features).all():
        raise ValueError("surface features must be finite [n_profiles,56]")
    if metadata.get("feature_fingerprint") != array_fingerprint((features,)):
        raise ValueError("surface profile features were changed")
    if not np.array_equal(months, np.unique(c.month_index)):
        raise ValueError("surface month coverage/order differs from the exact cohort")
    if z.shape != (len(months), 3, len(lat), len(lon)):
        raise ValueError("surface grid tensor shape differs")
    if metadata.get("surface_fingerprint") != array_fingerprint((months, z)):
        raise ValueError("surface fields were changed")
    if require_global_grid and (not np.array_equal(lat, np.arange(180) - 89.5)
                                or not np.array_equal(lon, np.arange(360) + .5)):
        raise ValueError("legacy surface encoder requires the registered global 1-degree grid")
    # The original dense encoder receives precisely the same normalized fields
    # used to create 46's sampled profile features; no new normalization is fit.
    surface_z, surface = {}, {}
    names = ("SST", "SSS", "SLA")
    for i, month in enumerate(months):
        tensor = torch.as_tensor(z[i], dtype=torch.float32, device=device)
        surface_z[int(month)] = {name: tensor[k] for k, name in enumerate(names)}
        # SetConv's original auxiliary order is SST/SLA/SSS + shared validity.
        ordered = tensor[[0, 2, 1]]
        valid = torch.isfinite(ordered).all(0, keepdim=True).to(torch.float32)
        surface[int(month)] = torch.cat([ordered.nan_to_num(), valid])
    return {
        "SURF": surface, "SURF_Z": surface_z,
        "GRID_LAT": torch.as_tensor(lat, dtype=torch.float32, device=device),
        "GRID_LON": torch.as_tensor(lon, dtype=torch.float32, device=device),
        "_synthetic_surface_metadata": metadata,
        "_synthetic_surface_cache_sha256": sha256(path),
        "_synthetic_surface_cache_path": str(path),
    }


class SurfaceASTAdapter:
    """Change only parser validation and synthetic surface ingest in old setup."""

    def __getattr__(self, name):
        return getattr(ast, name)

    def parse(self, source, filename="<unknown>", mode="exec", **kwargs):
        tree = ast.parse(source, filename=filename, mode=mode, **kwargs)
        if filename != str(LEGACY):
            return tree
        body = []
        removed_rejection = replaced_ingest = checked_arguments = False
        native_synthetic_ingest = False
        # The newer runner removed its Argo-only rejection and explicitly
        # selects a CESM2 store. Accept that reviewed revision too, rather than
        # silently accepting any future removal of the original safeguard.
        native_ingest = ast.parse(
            '_o = _xr.open_zarr(os.path.join(ROOT, "data", '
            '*(("synthetic_argo", "cesm2_surface_1deg.zarr") '
            'if args.region == "synthetic" else ("real_obs_1deg.zarr",))))'
        ).body[0]
        for node in tree.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.region == 'synthetic' and args.surface":
                removed_rejection = True
                continue
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.surface":
                native_synthetic_ingest = any(
                    ast.dump(child, include_attributes=False)
                    == ast.dump(native_ingest, include_attributes=False)
                    for child in node.body
                )
                body += ast.parse(
                    "globals().update(_matched_surface_loader(c, norm, dev))\n"
                    "print('loaded shared CESM2 SST/SSS/steric-SSH grids for the original dense patch encoders', flush=True)\n").body
                replaced_ingest = True
                continue
            body.append(node)
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets)
                    and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "ap.parse_args"):
                body += ast.parse("_matched_validate_args(args)\n").body
                checked_arguments = True
        if not ((removed_rejection or native_synthetic_ingest)
                and replaced_ingest and checked_arguments):
            raise ValueError("Original runner setup changed; surface adaptation must be reviewed")
        tree.body = body
        return ast.fix_missing_locations(tree)


def validate_legacy_args(args, diagnostic=False):
    if not args.surface or args.region != "synthetic" or args.backbone != "d4rt" or args.mode != "train":
        raise ValueError("surface arm requires the original synthetic d4rt training architecture")
    if args.surface_vars != "SST,SLA,SSS" or args.surface_patch != 3:
        raise ValueError("preserve the original SST/SSS pair plus SLA encoders, with 3x3 patches")
    if args.mass_mode != "dfs" or args.ablation != "anomaly_exact" or args.leads != "0":
        raise ValueError("preserve DFS weighting, exact position anomaly, and reconstruction setting")
    if diagnostic:
        return
    required = {"steps": 15000, "n_latent": 64, "d_model": 64, "n_self_blocks": 2,
                "n_dec_blocks": 2, "n_heads": 4, "target_dropout": .2, "queries": 1024,
                "batch": 1, "n_profiles": 0, "eval_cells": 8000, "lr": .001,
                "warmup": 300, "weight_decay": .01, "refiner_km": 500.,
                "refiner_gate": 1., "refiner_dz": 100., "val_every": 1000}
    for key, value in required.items():
        if getattr(args, key) != value:
            raise ValueError(f"matched legacy surface arm requires {key}={value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--satellite-cache", type=Path,
                        default=ROOT / "outputs/synthetic_matched_20261007/satellites.npz")
    parser.add_argument("--diagnostic", action="store_true")
    extra, forwarded = parser.parse_known_args()
    cache_path = extra.satellite_cache.resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"prepare shared fields with 50 first: {cache_path}")
    spec = importlib.util.spec_from_file_location("previous_token_resume_engine", ROOT / "experiments/synthetic/49_previous_token_matched.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.ast = SurfaceASTAdapter()
    original_execute, original_contract, original_export = engine.execute_nodes, engine.source_contract, engine.export_predictions
    namespace = {}

    def execute(nodes, ns):
        ns.update(_matched_surface_loader=lambda c, norm, dev: load_surface_cache(cache_path, c, norm, dev),
                  _matched_validate_args=lambda args: validate_legacy_args(args, extra.diagnostic))
        original_execute(nodes, ns)
        namespace.update(ns)

    def contract():
        return {**original_contract(), str(RUNNER.relative_to(ROOT)): sha256(RUNNER),
                "shared_satellite_cache": sha256(cache_path)}

    def export(ns, baseline_dir, output):
        original_export(ns, baseline_dir, output)
        for split in ("validation", "development"):
            if split not in ns["eval_sets"]:
                continue
            path = output / f"{split}_predictions_seed{ns['args'].seed}.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata.update(method="previous_64_slot_token_surface", satellite_cache_sha256=sha256(cache_path),
                            surface_metadata=ns["_synthetic_surface_metadata"],
                            surface_presentation="dense 3x3-degree SST/SSS pair and steric-SSH patch encoders",
                            historical_pasted_multimodal_artifact_replayed=False,
                            surface_runner_sha256=sha256(RUNNER))
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            temporary = Path(str(path) + ".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, **arrays)
            temporary.replace(path)

    engine.execute_nodes, engine.source_contract, engine.export_predictions = execute, contract, export
    # 49's strict Argo-only guard is bypassed because 52 supplies the equally
    # strict surface configuration check before model setup. Its training AST,
    # optimizer, RNG checkpoints and resume validation remain untouched.
    defaults = ["--region", "synthetic", "--mode", "train", "--backbone", "d4rt",
                "--mass-mode", "dfs", "--ablation", "anomaly_exact", "--n-latent", "64",
                "--refiner-km", "500", "--refiner-gate", "1.0", "--steps", "15000",
                "--surface", "--diagnostic", "--no-final-development"]
    argv = sys.argv
    sys.argv = [str(RUNNER), *defaults, *forwarded]
    try:
        engine.main()
    finally:
        sys.argv = argv
    if not namespace:
        return
    args = namespace["args"]
    path = Path(namespace["OUT"]) / f"summary_seed{args.seed}.json"
    if not path.exists():
        return  # a partial diagnostic/continuation is not a completed result
    summary = json.loads(path.read_text())
    summary.update(rerun_of="original_64slot_architecture_with_documented_CESM2_surfaces",
                   historical_setting=False, matched_registered_architecture=not extra.diagnostic,
                   surface_cache={"path": str(cache_path), "sha256": sha256(cache_path),
                                  "metadata": namespace["_synthetic_surface_metadata"]},
                   surface_presentation="dense 3x3-degree SST/SSS pair and steric-SSH patch encoders",
                   historical_pasted_multimodal_artifact_replayed=False)
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
