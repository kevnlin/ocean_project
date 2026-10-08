"""Observed-profile alignment checks that exercise active, differentiable shifts."""
import copy

import pytest
import torch

from ocean_tokenizer.profile_alignment import FullProfileAlignment


WIDTH = 12
QUERY_KEYS = (
    "local_profile_values", "local_profile_valid", "local_profile_absolute",
    "local_profile_climatology", "query_level", "query_coord", "local_features",
    "local_innovation", "local_valid",
)
NEIGHBOR_KEYS = (
    "local_profile_values", "local_profile_valid", "local_profile_absolute",
    "local_profile_climatology", "local_features", "local_innovation", "local_valid",
)


def _scene(*, partial=False, absolute=False):
    generator = torch.Generator().manual_seed(124)
    depths = torch.tensor([0.0, 50.0, 100.0, 200.0, 400.0], dtype=torch.float64)
    q, k = 4, 3
    values = torch.stack((depths * 0.012, -depths * 0.0007), dim=-1)
    values = values[None, None].expand(q, k, -1, -1).clone()
    values += torch.arange(k, dtype=values.dtype)[None, :, None, None] * 0.04
    valid = torch.ones_like(values, dtype=torch.bool)
    if partial:
        valid[0, 0, 1, 1] = False
        valid[1, 1] = False
        valid[2, 2, 2, 0] = False
    mean = torch.stack((-0.2 + depths * 0.001, 0.03 - depths * 0.00001), dim=-1)
    std = torch.stack((0.5 + depths * 0.003, 0.05 + depths * 0.0001), dim=-1)
    query_level = torch.tensor([1, 2, 3, 0])
    original = (values[torch.arange(q), :, query_level] - mean[query_level, None]) / std[query_level, None]
    original_valid = valid[torch.arange(q), :, query_level]
    original = torch.where(original_valid, original, 0.0)
    coord = torch.randn(q, 4, dtype=values.dtype, generator=generator)
    coord[:, 2] = depths[query_level]
    scene = dict(
        local_profile_values=values,
        local_profile_valid=valid,
        profile_depths=depths,
        profile_normalization_mean=mean,
        profile_normalization_std=std,
        query_level=query_level,
        query_coord=coord,
        local_features=torch.randn(q, k, 66, dtype=values.dtype, generator=generator),
        local_innovation=original,
        local_valid=original_valid,
    )
    scene["local_features"][..., 56:58] = 0.0
    if absolute:
        climate = torch.stack((20.0 - depths * 0.02, 35.0 + depths * 0.001), dim=-1)
        climate = climate[None, None].expand_as(values).clone()
        # Source-specific climates must stay attached to each observed profile.
        climate += torch.arange(k, dtype=values.dtype)[None, :, None, None] * 0.1
        scene["local_profile_climatology"] = climate
        scene["local_profile_absolute"] = climate + values
    h = torch.randn(q, WIDTH, dtype=values.dtype, generator=generator)
    return scene, h


def _model(mode, shift=None):
    torch.manual_seed(82)
    model = FullProfileAlignment(WIDTH, mode=mode).double().eval()
    if shift is not None:
        with torch.no_grad():
            model.shift_head[-1].bias.copy_(torch.atanh(torch.tensor(shift, dtype=torch.float64) / 100.0))
    return model


def _select(scene, index):
    return {key: value[index] if key in QUERY_KEYS else value for key, value in scene.items()}


@pytest.mark.parametrize("mode", ("none", "shared", "independent"))
@pytest.mark.parametrize("absolute", (False, True))
def test_zero_shift_preserves_original_evidence_and_partial_masks_exactly(mode, absolute):
    scene, h = _scene(partial=True, absolute=absolute)
    updated, diagnostics = _model(mode)(scene, h)
    assert torch.equal(updated["local_innovation"], scene["local_innovation"])
    assert torch.equal(updated["local_valid"], scene["local_valid"])
    assert updated["local_features"] is scene["local_features"]
    assert updated["profile_embedding"].shape == (4, 3, WIDTH)
    assert not updated["profile_embedding"][1, 1].any()
    assert not diagnostics["profile_shift_m"].any()
    assert diagnostics["profile_absolute_used"].item() == float(absolute)
    expected_profiles = torch.where(scene["local_profile_valid"], scene["local_profile_values"], 0.0)
    assert torch.equal(updated["aligned_profile_values"], expected_profiles)
    assert torch.equal(updated["aligned_profile_valid"], scene["local_profile_valid"])


def test_control_reads_complete_profile_context_without_changing_numeric_evidence():
    scene, h = _scene()
    model = _model("none")
    first, _ = model(scene, h)
    changed = copy.deepcopy(scene)
    changed["local_profile_values"][0, 0, 4, 0] += 10.0
    second, _ = model(changed, h)
    assert first["local_innovation"] is scene["local_innovation"]
    assert first["local_valid"] is scene["local_valid"]
    assert not torch.equal(first["profile_embedding"][0, 0], second["profile_embedding"][0, 0])
    assert torch.equal(first["profile_embedding"][1:], second["profile_embedding"][1:])


def test_shared_shift_is_exactly_shared_and_normalizes_at_destination_query_level():
    scene, h = _scene()
    updated, diagnostics = _model("shared", [25.0])(scene, h)
    shifts = diagnostics["profile_shift_m"]
    assert torch.equal(shifts[..., 0], shifts[..., 1])
    torch.testing.assert_close(shifts, torch.full_like(shifts, 25.0))
    physical = (scene["local_profile_values"][0, 0, 1] + scene["local_profile_values"][0, 0, 2]) / 2
    expected = (physical - scene["profile_normalization_mean"][1]) / scene["profile_normalization_std"][1]
    torch.testing.assert_close(updated["local_innovation"][0, 0], expected)
    # Standardizing at source depth is observably a different operation.
    source_normalized = (
        (scene["local_profile_values"][0, 0, 1] - scene["profile_normalization_mean"][1])
        / scene["profile_normalization_std"][1]
        + (scene["local_profile_values"][0, 0, 2] - scene["profile_normalization_mean"][2])
        / scene["profile_normalization_std"][2]
    ) / 2
    assert not torch.allclose(expected, source_normalized)


def test_independent_shift_can_sample_different_temperature_and_salinity_depths():
    scene, h = _scene()
    updated, diagnostics = _model("independent", [25.0, -25.0])(scene, h)
    torch.testing.assert_close(diagnostics["profile_shift_m"][0, 0], torch.tensor([25.0, -25.0], dtype=h.dtype))
    expected_physical = torch.tensor([75.0 * 0.012, -25.0 * 0.0007], dtype=h.dtype)
    expected = (expected_physical - scene["profile_normalization_mean"][1]) / scene["profile_normalization_std"][1]
    torch.testing.assert_close(updated["local_innovation"][0, 0], expected)


def test_missing_endpoints_are_not_bridged_and_shared_channels_keep_separate_masks():
    scene, h = _scene()
    scene["local_profile_valid"][0, 0, 2, 1] = False
    updated, diagnostics = _model("shared", [25.0])(scene, h)
    assert updated["local_valid"][0, 0, 0]
    assert not updated["local_valid"][0, 0, 1]
    assert updated["local_innovation"][0, 0, 1] == 0
    # A more distant valid level must not silently bridge the missing endpoint.
    assert scene["local_profile_valid"][0, 0, 3, 1]
    assert torch.equal(diagnostics["profile_shift_m"][..., 0], diagnostics["profile_shift_m"][..., 1])


def test_out_of_support_depths_are_invalidated_without_clamping_or_extrapolating():
    scene, h = _scene()
    updated, diagnostics = _model("shared", [-25.0])(scene, h)
    assert (diagnostics["profile_sample_depth_m"][3] < scene["profile_depths"][0]).all()
    assert not updated["local_valid"][3].any()
    assert not updated["local_innovation"][3].any()
    assert not updated["aligned_profile_valid"][:, :, 0].any()
    high, _ = _model("shared", [25.0])(scene, h)
    assert not high["aligned_profile_valid"][:, :, -1].any()


def test_exact_observed_endpoint_remains_valid_with_missing_adjacent_level():
    scene, h = _scene()
    scene["local_profile_valid"][0, 0, 2] = False
    updated, _ = _model("shared")(scene, h)
    assert updated["local_valid"][0, 0].all()
    assert torch.equal(updated["local_innovation"][0, 0], scene["local_innovation"][0, 0])


@pytest.mark.parametrize("absolute", (False, True))
def test_zero_initialized_offsets_receive_gradient_through_observed_interpolation(absolute):
    scene, h = _scene(absolute=absolute)
    model = _model("shared")
    updated, _ = model(scene, h)
    updated["local_innovation"][0, 0, 0].backward()
    assert torch.isfinite(model.shift_head[-1].bias.grad).all()
    assert model.shift_head[-1].bias.grad.abs().sum() > 0
    assert model.shift_head[-1].weight.grad.abs().sum() > 0


def test_interpolation_backpropagates_the_physical_endpoint_weights():
    scene, h = _scene()
    scene["local_profile_values"].requires_grad_()
    model = _model("independent", [25.0, -25.0])
    updated, _ = model(scene, h)
    updated["local_innovation"][0, 0, 0].backward()
    gradients = scene["local_profile_values"].grad
    expected = 0.5 / scene["profile_normalization_std"][1, 0]
    torch.testing.assert_close(gradients[0, 0, 1, 0], expected)
    torch.testing.assert_close(gradients[0, 0, 2, 0], expected)
    assert torch.count_nonzero(gradients) == 2


def test_absolute_alignment_subtracts_unshifted_source_climatology_and_shifts_all_levels():
    scene, h = _scene(absolute=True)
    updated, _ = _model("shared", [25.0])(scene, h)
    raw = (scene["local_profile_absolute"][0, 0, 1] + scene["local_profile_absolute"][0, 0, 2]) / 2
    anomaly = raw - scene["local_profile_climatology"][0, 0, 1]
    expected = (anomaly - scene["profile_normalization_mean"][1]) / scene["profile_normalization_std"][1]
    torch.testing.assert_close(updated["local_innovation"][0, 0], expected)
    torch.testing.assert_close(updated["aligned_profile_values"][0, 0, 1], anomaly)
    # Native depth 100 samples raw depth 125, but keeps the native 100 climate.
    raw125 = 0.75 * scene["local_profile_absolute"][0, 0, 2] + 0.25 * scene["local_profile_absolute"][0, 0, 3]
    expected125 = raw125 - scene["local_profile_climatology"][0, 0, 2]
    torch.testing.assert_close(updated["aligned_profile_values"][0, 0, 2], expected125)
    assert not torch.allclose(updated["aligned_profile_values"][0, 0, 1],
                              0.5 * (scene["local_profile_values"][0, 0, 1] + scene["local_profile_values"][0, 0, 2]))


@pytest.mark.parametrize("absolute", (False, True))
def test_nonzero_background_innovations_remain_continuous_at_zero_shift(absolute):
    scene, h = _scene(absolute=absolute)
    background = torch.tensor([0.8, -0.35], dtype=h.dtype)
    scene["local_features"][..., 56:58] = background
    scene["local_innovation"] = scene["local_innovation"] - background
    initial, _ = _model("shared")(scene, h)
    assert torch.equal(initial["local_innovation"], scene["local_innovation"])
    nearby, _ = _model("shared", [1e-5])(scene, h)
    torch.testing.assert_close(nearby["local_innovation"][0, 0],
                               scene["local_innovation"][0, 0], rtol=0, atol=1e-5)
    shifted, _ = _model("shared", [25.0])(scene, h)
    without_background = copy.deepcopy(scene)
    without_background["local_features"][..., 56:58] = 0.0
    no_background, _ = _model("shared", [25.0])(without_background, h)
    torch.testing.assert_close(shifted["local_innovation"][0, 0],
                               no_background["local_innovation"][0, 0] - background)


@pytest.mark.parametrize("absolute", (False, True))
def test_poisoned_masked_channels_do_not_influence_embeddings_or_active_shifts(absolute):
    scene, h = _scene(partial=True, absolute=absolute)
    model = _model("independent")
    with torch.no_grad():
        model.shift_head[-1].weight.normal_(std=0.003)
    poisoned = copy.deepcopy(scene)
    poisoned["local_profile_values"][~scene["local_profile_valid"]] = float("nan")
    if absolute:
        poisoned["local_profile_absolute"][~scene["local_profile_valid"]] = float("inf")
    expected, expected_diag = model(scene, h)
    actual, actual_diag = model(poisoned, h)
    for key in ("profile_embedding", "local_innovation", "aligned_profile_values"):
        torch.testing.assert_close(actual[key], expected[key])
        assert torch.isfinite(actual[key]).all()
    assert torch.equal(actual["local_valid"], expected["local_valid"])
    torch.testing.assert_close(actual_diag["profile_shift_m"], expected_diag["profile_shift_m"])


def test_active_alignment_preserves_query_chunking_and_neighbor_permutation():
    scene, h = _scene(absolute=True)
    model = _model("independent")
    with torch.no_grad():
        model.shift_head[-1].weight.normal_(std=0.005)
    full, diagnostics = model(scene, h)
    order = torch.tensor([2, 0, 3, 1])
    permuted, permuted_diag = model(_select(scene, order), h[order])
    neighbor_order = torch.tensor([2, 0, 1])
    reordered = {
        key: value[:, neighbor_order] if key in NEIGHBOR_KEYS else value
        for key, value in scene.items()
    }
    neighbors, _ = model(reordered, h)
    parts = [model(_select(scene, slice(a, b)), h[a:b])[0] for a, b in ((0, 1), (1, 3), (3, 4))]
    for key in ("local_innovation", "profile_embedding", "aligned_profile_values"):
        torch.testing.assert_close(permuted[key], full[key][order], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(neighbors[key], full[key][:, neighbor_order], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.cat([part[key] for part in parts]), full[key], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(permuted_diag["profile_shift_m"], diagnostics["profile_shift_m"][order])


@pytest.mark.parametrize("mode", ("none", "shared", "independent"))
def test_empty_queries_and_neighbors_have_finite_shapes(mode):
    scene, h = _scene()
    for empty in ("query", "neighbor"):
        if empty == "query":
            case, query_h = _select(scene, slice(0, 0)), h[:0]
        else:
            case = {key: value[:, :0] if key in NEIGHBOR_KEYS else value for key, value in scene.items()}
            query_h = h
        updated, diagnostics = _model(mode)(case, query_h)
        assert updated["profile_embedding"].shape[:2] == updated["local_innovation"].shape[:2]
        assert torch.isfinite(updated["profile_embedding"]).all()
        assert torch.isfinite(diagnostics["profile_aligned_coverage"])


def test_single_level_profiles_support_only_that_observed_depth():
    scene, h = _scene()
    for key in ("local_profile_values", "local_profile_valid"):
        scene[key] = scene[key][:, :, :1]
    for key in ("profile_depths", "profile_normalization_mean", "profile_normalization_std"):
        scene[key] = scene[key][:1]
    scene["query_level"].zero_()
    scene["query_coord"][:, 2] = 0
    updated, _ = _model("shared", [25.0])(scene, h)
    assert not updated["local_valid"].any()
    assert not updated["aligned_profile_valid"].any()


def test_requires_complete_and_monotonic_physical_normalization_contract():
    scene, h = _scene()
    broken = copy.deepcopy(scene)
    broken["profile_depths"][2] = broken["profile_depths"][1]
    with pytest.raises(ValueError, match="strictly increasing"):
        _model("shared")(broken, h)
    broken = copy.deepcopy(scene)
    broken["profile_normalization_std"][0, 0] = 0
    with pytest.raises(ValueError, match="positive std"):
        _model("shared")(broken, h)
    broken = copy.deepcopy(scene)
    broken["local_profile_absolute"] = scene["local_profile_values"]
    with pytest.raises(ValueError, match="both absolute"):
        _model("shared")(broken, h)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("active", (False, True))
def test_bfloat16_context_preserves_fp32_salinity_and_depth_interpolation(device, active):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 not supported")
    scene, h = _scene(absolute=True)
    scene = {
        key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
        for key, value in scene.items()
    }
    h = h.to(device=device, dtype=torch.bfloat16)
    model = _model("shared", [25.0] if active else None).float().to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        updated, diagnostics = model(scene, h)
    assert updated["local_innovation"].dtype == torch.float32
    assert updated["aligned_profile_values"].dtype == torch.float32
    assert diagnostics["profile_sample_depth_m"].dtype == torch.float32
    if not active:
        assert torch.equal(updated["local_innovation"], scene["local_innovation"])
        return
    # Use the actual (possibly AMP-quantized) predicted shift to isolate the
    # precision of the numeric sampling from the precision of its predictor.
    fraction = diagnostics["profile_shift_m"][0, 0, 1] / 50.0
    raw = scene["local_profile_absolute"][0, 0, 1, 1] + fraction * (
        scene["local_profile_absolute"][0, 0, 2, 1]
        - scene["local_profile_absolute"][0, 0, 1, 1])
    anomaly = raw - scene["local_profile_climatology"][0, 0, 1, 1]
    expected = (anomaly - scene["profile_normalization_mean"][1, 1]) / scene["profile_normalization_std"][1, 1]
    torch.testing.assert_close(updated["local_innovation"][0, 0, 1], expected, rtol=1e-6, atol=1e-6)
    # BF16 subtraction near 35 PSU would collapse this anomaly to zero and
    # materially change the normalized salinity prediction.
    quantized = (-scene["profile_normalization_mean"][1, 1]
                 / scene["profile_normalization_std"][1, 1])
    assert (updated["local_innovation"][0, 0, 1] - quantized).abs() > 0.1
    updated["local_innovation"][0, 0, 1].backward()
    assert torch.isfinite(model.shift_head[-1].bias.grad).all()
    assert model.shift_head[-1].bias.grad.abs().sum() > 0
