"""Original recipe preservation, exact epoch recovery and common exports."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "experiments/synthetic/54_fixed_pointwise_mlp.py"
SPEC = importlib.util.spec_from_file_location("fixed_pointwise_recovery", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_original_draws_features_and_training_loop_execute_unchanged_nodes():
    source = R.LEGACY.read_text()
    setup, draw, loop = R.original_parts(source)
    original = ast.parse(source)
    start = next(i for i, node in enumerate(original.body) if R.assigned(node, "rng"))
    end = next(i for i, node in enumerate(original.body) if isinstance(node, ast.For)
               and isinstance(node.target, ast.Name) and node.target.id == "ep")
    assert ast.dump(ast.Module(body=draw, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=original.body[start:end], type_ignores=[]), include_attributes=False)
    assert ast.dump(loop, include_attributes=False) == ast.dump(original.body[end], include_attributes=False)
    for name in ("features", "score"):
        expected = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        actual = next(node for node in setup if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def test_default_setup_never_builds_development_scoring_points():
    setup, _, _ = R.original_parts(R.LEGACY.read_text())
    test = next(node for node in setup if R.assigned(node, "test"))
    assert isinstance(test.value, ast.List) and not test.value.elts
    calls = [node for statement in setup for node in ast.walk(statement)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "eval_set"]
    assert len(calls) == 1
    assert calls[0].args[2].value == "validation"


def test_epoch_recovery_preserves_original_adam_and_batch_shuffle(tmp_path):
    _, _, original_loop = R.original_parts(R.LEGACY.read_text())
    generator = np.random.default_rng(22)
    inputs = torch.from_numpy(generator.normal(size=(21, 9)).astype("float32"))
    targets = torch.from_numpy(generator.normal(size=(21, 2)).astype("float32"))

    def initialize():
        torch.manual_seed(1234); np.random.seed(31)
        model = torch.nn.Sequential(torch.nn.Linear(9, 16), torch.nn.SiLU(), torch.nn.Linear(16, 2))
        return {"torch": torch, "np": np, "time": SimpleNamespace(time=lambda: 0.), "t0": 0.,
                "dev": "cpu", "model": model, "opt": torch.optim.Adam(model.parameters(), lr=.001),
                "lossf": torch.nn.MSELoss(), "Xt": inputs, "Yt": targets,
                "nb": 3, "C": SimpleNamespace(MLP_BATCH=8), "epochs": 4,
                "rng": np.random.default_rng(1234)}

    def run(ns, start, end, hook=lambda *args: None):
        loop = copy.deepcopy(original_loop)
        loop.iter = ast.parse("range(_start, _end)", mode="eval").body
        loop.body += ast.parse("_hook(ep+1, train_mse)").body
        ns.update(_start=start, _end=end, _hook=hook)
        R.execute([loop], ns)

    full = initialize()
    run(full, 0, 4)
    expected = R.cpu_copy(full["model"].state_dict())
    original = initialize()
    run(original, 0, 2)
    checkpoint = {"model": R.cpu_copy(original["model"].state_dict()),
                  "optimizer": R.cpu_copy(original["opt"].state_dict()), "random": R.random_state(original["rng"])}
    path = tmp_path / "last.pt"
    R.atomic_save(checkpoint, path)
    recovered = torch.load(path, weights_only=True)
    resumed = initialize()
    resumed["model"].load_state_dict(recovered["model"])
    resumed["opt"].load_state_dict(recovered["optimizer"])
    R.restore_random_state(resumed["rng"], recovered["random"])
    run(resumed, 2, 4)
    for key, value in expected.items():
        torch.testing.assert_close(value, resumed["model"].state_dict()[key], rtol=0, atol=0)
    assert resumed["train_mse"] == full["train_mse"]
    for key, value in full["opt"].state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, resumed["opt"].state_dict()["state"][key][name], rtol=0, atol=0)
    assert resumed["rng"].bit_generator.state == full["rng"].bit_generator.state


def export_fixture(tmp_path):
    base = tmp_path / "canonical"
    base.mkdir()
    levels = np.array([5., 100.])
    target = np.array([[1., 2.], [2., 3.], [3., np.nan], [4., 5.]])
    mean, std = np.array([[.1, .2], [.3, .4]]), np.array([[1., 2.], [3., 4.]])
    arrays = {"month": np.full(4, 48, dtype=np.int64), "profile": np.array([2, 2, 3, 3]),
              "level": np.array([0, 1, 0, 1]), "target": target,
              "mean": np.zeros((4, 2)), "climatology_physical": np.arange(8.).reshape(4, 2),
              "normalization_mean": mean, "normalization_std": std}
    np.savez_compressed(base / "oi_validation.npz", **arrays)
    ev = {"src": np.array([0, 1]), "tgt": np.array([2, 3]), "month": 48,
          "prof": np.array([0, 0, 1, 1]), "lev": np.array([0, 1, 0, 1]),
          "target": {ch: target[:, j] for j, ch in enumerate(("TEMP", "SALT"))}}
    model = torch.nn.Linear(9, 2)
    with torch.no_grad():
        model.weight.zero_(); model.bias.copy_(torch.tensor([1., 2.]))
    ns = {"c": SimpleNamespace(levels=levels), "norm": SimpleNamespace(
                mean={ch: mean[:, j] for j, ch in enumerate(("TEMP", "SALT"))},
                std={ch: std[:, j] for j, ch in enumerate(("TEMP", "SALT"))}),
          "model": model, "features": lambda *args: np.zeros((4, 9), dtype="float32"),
          "CH": ("TEMP", "SALT"), "synth_argo_eval": SimpleNamespace(eval_set=lambda *args: [ev]),
          "OBS": {}, "EVAL_CELLS": 8000, "identity_check": lambda *args: {"passed": True},
          "score": lambda *args: {}, "dev": "cpu", "args": SimpleNamespace(seed=1234),
          "C": SimpleNamespace(MLP_EPOCHS=30)}
    return ns, base, arrays


def test_export_uses_exact_canonical_identity_and_both_physical_targets(tmp_path):
    ns, base, expected = export_fixture(tmp_path)
    result = R.export_predictions(ns, "validation", base, tmp_path)
    assert result["physical_metrics"]["TEMP"]["n"] == 4
    assert result["physical_metrics"]["SALT"]["n"] == 3
    for ch in ("TEMP", "SALT"):
        assert result["absolute_physical_metrics"][ch]["rmse"] == pytest.approx(result["physical_metrics"][ch]["rmse"])
    with np.load(tmp_path / "validation_seed1234.npz") as saved:
        assert "std" not in saved
        for key in ("month", "profile", "level", "target"):
            np.testing.assert_array_equal(saved[key], expected[key])
        np.testing.assert_array_equal(saved["baseline"], expected["mean"])
        np.testing.assert_array_equal(saved["mean_absolute_physical"], saved["mean_physical"] + expected["climatology_physical"])


def test_export_rejects_a_changed_scoring_target(tmp_path):
    ns, base, expected = export_fixture(tmp_path)
    expected["target"] = expected["target"].copy()
    expected["target"][0, 0] += .1
    np.savez_compressed(base / "oi_validation.npz", **expected)
    with pytest.raises(ValueError, match="target"):
        R.export_predictions(ns, "validation", base, tmp_path)


def test_training_data_fingerprint_includes_shape_dtype_and_values():
    data = np.arange(6, dtype="float32").reshape(2, 3)
    assert R.array_sha256(data) == R.array_sha256(data.copy())
    assert R.array_sha256(data) != R.array_sha256(data.astype("float64"))
    assert R.array_sha256(data) != R.array_sha256(data.reshape(3, 2))
    changed = data.copy(); changed[0, 0] += 1
    assert R.array_sha256(data) != R.array_sha256(changed)
