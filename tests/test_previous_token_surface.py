from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer.latent_experiment import array_fingerprint
from ocean_tokenizer.synth_argo_eval import SPLITS

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/synthetic/52_previous_token_surface.py"
spec = importlib.util.spec_from_file_location("previous_surface_fixture", PATH)
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)


class Norm:
    mean = {"TEMP": np.array([.1, .2]), "SALT": np.array([.01, .02])}
    std = {"TEMP": np.array([1., 2.]), "SALT": np.array([.1, .2])}

    def z(self, channel, values):
        return (values - self.mean[channel]) / self.std[channel]


@pytest.fixture
def cache_fixture(tmp_path):
    c = SimpleNamespace(month_index=np.array([0, 0, 1, 1]),
                        lat=np.array([-1., 1., -1., 1.]), lon=np.arange(4.) + 10.,
                        wmo=np.array(["A", "B", "C", "D"]), levels=np.array([5., 15.]),
                        TEMP=np.arange(8.).reshape(4, 2), SALT=np.arange(8.).reshape(4, 2) / 100)
    norm = Norm()
    standardized = np.stack([norm.z(ch, getattr(c, ch)) for ch in ("TEMP", "SALT")], -1)
    features = np.zeros((4, 56), dtype=np.float32)
    z = np.stack([np.full((2, 4), float(k+1), dtype=np.float32) for k in range(3)])[None].repeat(2, 0)
    months = np.array([0, 1])
    metadata = {
        "cohort_fingerprint": array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, standardized)),
        "normalization_scope": "training years only", "splits": {k: list(v) for k, v in SPLITS.items()},
        "surface_variables": ["SST", "SSS", "steric_SSH"], "p_ref_dbar": 990.,
        "surface_is_independent_of_truth": False, "satellite_noise": "none",
        "feature_fingerprint": array_fingerprint((features,)),
        "surface_fingerprint": array_fingerprint((months, z)),
    }
    data = {"features": features, "month_index": c.month_index, "lat": c.lat, "lon": c.lon,
            "grid_month": months, "surface_z": z, "grid_lat": np.array([-45., 45.]),
            "grid_lon": np.array([45., 135., 225., 315.]),
            "normalization_mean": np.stack([norm.mean[ch] for ch in ("TEMP", "SALT")], -1),
            "normalization_std": np.stack([norm.std[ch] for ch in ("TEMP", "SALT")], -1)}
    path = tmp_path / "satellites.npz"

    def save():
        np.savez(path, **data, metadata=json.dumps(metadata))

    save()
    return path, c, norm, data, metadata, save


def test_shared_grid_channel_mapping_preserves_original_pair_and_ssh_paths(cache_fixture):
    path, c, norm, *_ = cache_fixture
    state = surface.load_surface_cache(path, c, norm, "cpu", require_global_grid=False)
    assert list(state["SURF_Z"]) == [0, 1]
    assert torch.equal(state["SURF_Z"][0]["SST"], torch.ones(2, 4))
    assert torch.equal(state["SURF_Z"][0]["SSS"], torch.full((2, 4), 2.))
    assert torch.equal(state["SURF_Z"][0]["SLA"], torch.full((2, 4), 3.))
    assert torch.equal(state["SURF"][0][:, 0, 0], torch.tensor([1., 3., 2., 1.]))


@pytest.mark.parametrize("change, message", [
    ("position", "exact cohort"), ("target", "anomaly/cohort"),
    ("profile_std", "normalization differs"), ("scope", "training-only"),
    ("channels", "channel order"), ("truth_independence", "truth-derived"),
    ("pressure", "reference pressure"), ("feature", "features were changed"),
    ("field", "fields were changed"), ("months", "coverage/order"),
])
def test_changed_evidence_or_preprocessing_is_rejected(cache_fixture, change, message):
    path, c, norm, data, metadata, save = cache_fixture
    if change == "position":
        data["lat"] = data["lat"] + .1
    elif change == "target":
        c.TEMP = c.TEMP + .1
    elif change == "profile_std":
        data["normalization_std"] = data["normalization_std"] + .1
    elif change == "scope":
        metadata["normalization_scope"] = "all years"
    elif change == "channels":
        metadata["surface_variables"] = ["SST", "steric_SSH", "SSS"]
    elif change == "truth_independence":
        metadata["surface_is_independent_of_truth"] = True
    elif change == "pressure":
        metadata["p_ref_dbar"] = 1000.
    elif change == "feature":
        data["features"][0, 44] += 1
    elif change == "field":
        data["surface_z"][0, 0, 0, 0] += 1
    elif change == "months":
        data["grid_month"] = data["grid_month"][::-1]
    save()
    with pytest.raises(ValueError, match=message):
        surface.load_surface_cache(path, c, norm, "cpu", require_global_grid=False)


def test_production_grid_is_enforced(cache_fixture):
    path, c, norm, *_ = cache_fixture
    with pytest.raises(ValueError, match="global 1-degree grid"):
        surface.load_surface_cache(path, c, norm, "cpu")


def surface_runner_source(*, legacy_rejection=False, store=None):
    tree = ast.parse(surface.LEGACY.read_text())
    if legacy_rejection:
        parser_index = next(i for i, node in enumerate(tree.body)
                            if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "args"
                                    for t in node.targets))
        tree.body[parser_index + 1:parser_index + 1] = ast.parse(
            "if args.region == 'synthetic' and args.surface:\n"
            "    raise SystemExit('synthetic task is Argo-only')\n"
        ).body
        store = "real_obs_1deg.zarr"
    if store is not None:
        ingest = next(node for node in tree.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "args.surface")
        index = next(i for i, node in enumerate(ingest.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_o"
                             for t in node.targets))
        ingest.body[index] = ast.parse(
            f"_o = _xr.open_zarr(os.path.join(ROOT, 'data', {store!r}))"
        ).body[0]
    return ast.unparse(ast.fix_missing_locations(tree))


@pytest.mark.parametrize("legacy_rejection", [False, True])
def test_surface_adapter_preserves_entire_original_training_loop_and_model(legacy_rejection):
    source = surface_runner_source(legacy_rejection=legacy_rejection)
    original = ast.parse(source, filename=str(surface.LEGACY))
    adapted = surface.SurfaceASTAdapter().parse(source, filename=str(surface.LEGACY))
    definitions = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                               for n in tree.body
                               if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    loop = lambda tree: next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "step")
    assert definitions(original) == definitions(adapted)
    assert ast.dump(loop(original), include_attributes=False) == ast.dump(loop(adapted), include_attributes=False)
    assert "real_obs_1deg.zarr" not in ast.unparse(adapted)
    assert "_matched_surface_loader(c, norm, dev)" in ast.unparse(adapted)
    assert "_matched_validate_args(args)" in ast.unparse(adapted)


@pytest.mark.parametrize("store", ["real_obs_1deg.zarr", "unreviewed_surface.zarr"])
def test_removed_rejection_requires_reviewed_native_synthetic_ingest(store):
    source = surface_runner_source(store=store)
    with pytest.raises(ValueError, match="setup changed"):
        surface.SurfaceASTAdapter().parse(source, filename=str(surface.LEGACY))


def test_registered_training_recipe_rejects_changed_budget_or_dropout():
    values = dict(surface=True, region="synthetic", backbone="d4rt", mode="train",
                  surface_vars="SST,SLA,SSS", surface_patch=3, mass_mode="dfs",
                  ablation="anomaly_exact", leads="0", steps=15000, n_latent=64,
                  d_model=64, n_self_blocks=2, n_dec_blocks=2, n_heads=4,
                  target_dropout=.2, queries=1024, batch=1, n_profiles=0,
                  eval_cells=8000, lr=.001, warmup=300, weight_decay=.01,
                  refiner_km=500., refiner_gate=1., refiner_dz=100., val_every=1000)
    surface.validate_legacy_args(SimpleNamespace(**values))
    for name, value in (("steps", 6000), ("target_dropout", 0.), ("n_latent", 32)):
        with pytest.raises(ValueError, match=name):
            surface.validate_legacy_args(SimpleNamespace(**{**values, name: value}))


def test_original_multimodal_model_reaches_both_surface_encoders_on_cpu():
    from ocean_tokenizer.fusion import build_fusion_model

    levels = np.array([5., 15., 25., 35., 45., 55., 65., 85., 105., 125.,
                       145., 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7])
    model = build_fusion_model("d4rt", SimpleNamespace(depth=levels), d_model=64,
                               n_latent=64, n_heads=4, n_self_blocks=2, n_dec_blocks=2,
                               max_lead=1, seed=1234, query_chunk=2048,
                               with_ssh=True, patch_surf=(3, 3))
    torch.manual_seed(17)
    grid = {"lat": torch.linspace(-80., 80., 9), "lon": torch.arange(12.) * 30. + 15.,
            "month": torch.tensor([1])}
    observations = {
        "profiles": {"prof": torch.randn(1, 3, 2, 20),
                     "lat": torch.tensor([[-10., 0., 10.]]),
                     "lon": torch.tensor([[10., 120., 250.]]), "month": torch.tensor([1])},
        "surf": {"field": torch.randn(1, 2, 9, 12), **grid},
        "ssh": {"field": torch.randn(1, 1, 9, 12), **grid},
    }
    query = torch.tensor([[[1., 30., 5., 1.], [2., 150., 125., 1.]]])
    tokens = model.encode(observations, batch=1, device="cpu")
    mean = model.decode(model.fuse(tokens), query, None, torch.zeros(1, 2, dtype=torch.long))
    assert mean.shape == (1, 2, 2)
    assert torch.isfinite(mean).all()
    mean.square().sum().backward()
    for encoder in ("surf", "ssh"):
        gradients = [p.grad for p in model.encoders[encoder].parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
