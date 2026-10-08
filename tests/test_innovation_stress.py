"""Stress protocol invariants: only sources change, labels/support are fixed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/65_innovation_stress.py"
spec = importlib.util.spec_from_file_location("innovation_stress_test", SCRIPT)
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


def _data():
    generator = torch.Generator().manual_seed(420)
    values = torch.randn(8, 4, 2, generator=generator)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[0, 1, 1] = False
    values[0, 1, 1] = float("nan")
    scale = torch.tensor([[2.0, 0.1]]).expand(4, -1).clone()
    return SimpleNamespace(device=torch.device("cpu"), n_levels=4,
        y=values.clone(), innovation=torch.nan_to_num(values), valid=valid.clone(),
        valid_np=valid.numpy().copy(), profile_std=scale,
        profile_physical=values * scale, profile_absolute=values * scale + 35.0,
        indices=lambda rows: torch.as_tensor(rows, dtype=torch.long))


@pytest.mark.parametrize("condition", stress.CONDITIONS)
def test_perturbations_preserve_query_labels_and_restore_all_source_fields(condition):
    data = _data()
    source = np.array([0, 2, 4, 6])
    queries = np.array([1, 3, 5, 7])
    tensors = {name: getattr(data, name).clone() for name in
               ("y", "innovation", "valid", "profile_physical", "profile_absolute")}
    valid_np = data.valid_np.copy()
    with stress.perturbed_source(data, source, queries, condition, 48, 20261007) as (retained, count):
        assert np.isin(retained, source).all()
        assert count["n_source_profiles_used"] == len(retained)
        torch.testing.assert_close(data.y, tensors["y"], equal_nan=True)
        for name in tensors:
            torch.testing.assert_close(getattr(data, name)[queries], tensors[name][queries], equal_nan=True)
        if "noise" in condition or "bias" in condition:
            delta_physical = data.profile_physical[source] - tensors["profile_physical"][source]
            delta_innovation = (data.innovation[source] - tensors["innovation"][source]) * data.profile_std
            torch.testing.assert_close(delta_physical[data.valid[source]], delta_innovation[data.valid[source]], rtol=2e-5, atol=1e-6)
    for name in tensors:
        torch.testing.assert_close(getattr(data, name), tensors[name], equal_nan=True)
    np.testing.assert_array_equal(data.valid_np, valid_np)


def test_failed_inference_restores_source_evidence():
    data = _data()
    original = data.innovation.clone()
    with pytest.raises(RuntimeError, match="inference failed"):
        with stress.perturbed_source(data, [0, 2], [1, 3], "profile_common_bias", 48, 20261007):
            raise RuntimeError("inference failed")
    torch.testing.assert_close(data.innovation, original)


def test_source_target_overlap_is_rejected_before_mutation():
    data = _data()
    with pytest.raises(ValueError, match="overlap"):
        with stress.perturbed_source(data, [0, 2], [2, 3], "standard", 48, 20261007):
            pass


def test_noise_marginals_are_matched_but_depth_correlation_differs():
    source = np.arange(2000)
    _, _, common = stress.perturbation_plan(source, 20, "profile_common_bias", 48, 20261007)
    _, _, independent = stress.perturbation_plan(source, 20, "independent_depth_noise", 48, 20261007)
    np.testing.assert_array_equal(common[:, 0], common[:, -1])
    assert not np.array_equal(independent[:, 0], independent[:, -1])
    for noise in (common, independent):
        np.testing.assert_allclose(noise.std(axis=(0, 1)), stress.NOISE_STD, rtol=0.04)
    _, mask, _ = stress.perturbation_plan(source, 20, "mask_depths_30pct", 48, 20261007)
    np.testing.assert_array_equal(mask[..., 0], mask[..., 1])
    assert abs(mask.mean() - 0.70) < 0.01


@pytest.mark.parametrize("condition", stress.CONDITIONS)
def test_plans_are_deterministic_given_registered_seed_month_and_condition(condition):
    first = stress.perturbation_plan(np.arange(40), 20, condition, 48, 20261007)
    second = stress.perturbation_plan(np.arange(40), 20, condition, 48, 20261007)
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
