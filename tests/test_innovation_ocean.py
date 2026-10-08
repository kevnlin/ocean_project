"""Matched-recipe integration tests on a complete, partially masked scene."""
import copy
import math

import pytest
import torch

from ocean_tokenizer.innovation_ocean import (
    RECIPES, InnovationOceanConfig, InnovationOceanModel,
)
from ocean_tokenizer.ocean_observation_operator import build_surface_observation_operator


def _model(recipe, *, active=False, width=16):
    torch.manual_seed(718)
    config = InnovationOceanConfig(
        n_obs_features=62, n_query_features=58, n_local_features=66,
        n_sat_features=56, variant="local_transformer", recipe=recipe,
        width=width, n_heads=4, n_latents=4, n_blocks=1,
        n_query_blocks=1, covariance_rank=4, n_profile_levels=20,
        use_latent=recipe != "local_latent_off", use_local=recipe != "local_off",
    )
    model = InnovationOceanModel(config).double().eval()
    if active:
        with torch.no_grad():
            for head in model.mean_head:
                head[-1].weight.fill_(.015)
                head[-1].bias.fill_(.04)
            model.local_gate[-1].bias.fill_(math.atanh(.3))
            if model.operator is not None:
                model.column_gate[-1].bias.fill_(math.atanh(.2))
    return model


def _scene():
    generator = torch.Generator().manual_seed(719)
    q, k, l, n, s = 5, 3, 20, 8, 4
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    depths = torch.linspace(5., 1500., l, dtype=torch.float64)
    levels = torch.tensor([0, 5, 8, 11, 16])
    mean = rand(l, 2) * .03
    std = rand(l, 2).abs() * .1 + .8
    values = rand(q, k, 1, 2).expand(q, k, l, 2).clone() * .2
    values += depths[None, None, :, None] * torch.tensor([.001, .0005], dtype=torch.float64)
    values += mean[None, None]
    valid = torch.ones(q, k, l, 2, dtype=torch.bool)
    valid[0, 0, 3, 1] = False
    valid[1, 1, 9:12, 0] = False
    valid[2, 2] = False
    local_features = rand(q, k, 66) * .1
    local_fg = local_features[..., 56:58].clone()
    index = levels[:, None, None, None].expand(q, k, 1, 2)
    native = values.gather(2, index).squeeze(2)
    local_valid = valid.gather(2, index).squeeze(2)
    local_innovation = (native - mean[levels, None]) / std[levels, None] - local_fg
    query_coord = rand(q, 4)
    query_coord[:, 0] *= 30.
    query_coord[:, 1] *= 100.
    query_coord[:, 2] = depths[levels]
    query_coord[:, 3] *= 10.
    obs_coord = rand(n, 4)
    obs_coord[:, 2] = torch.arange(n, dtype=torch.float64) * 100.
    satellite_coord = rand(s, 4)
    satellite_coord[:, 2] = 0.
    offsets = rand(q, k, 4)
    offsets[..., :2] *= 20.
    offsets[..., 3] *= .05
    operator, offset, operator_valid = build_surface_observation_operator(
        mean, std, torch.zeros(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64), n_queries=q)
    operator_valid[0, 1] = False
    operator_valid[2, :2] = False
    climate = torch.tensor([18., 35.], dtype=torch.float64).expand(q, k, l, 2).clone()
    climate[..., 0] -= depths[None, None] * .005
    return {
        "obs_features": rand(n, 62), "obs_coord": obs_coord,
        "obs_innovation": rand(n, 2), "obs_valid": torch.ones(n, 2, dtype=torch.bool),
        "satellite_features": rand(s, 56), "satellite_coord": satellite_coord,
        "satellite_valid": torch.ones(s, dtype=torch.bool),
        "query_features": rand(q, 58), "query_coord": query_coord,
        "baseline": rand(q, 2) * .1, "query_background": rand(q, 2) * .1,
        "query_level": levels, "local_features": local_features,
        "local_offsets": offsets, "local_innovation": local_innovation,
        "local_valid": local_valid, "local_profile_values": values,
        "local_profile_valid": valid, "local_profile_ids": torch.arange(k).expand(q, k),
        "profile_depths": depths, "profile_normalization_mean": mean,
        "profile_normalization_std": std, "local_profile_absolute": values + climate,
        "local_profile_climatology": climate, "surface_observation": rand(q, 3) * .2,
        "surface_operator": operator, "surface_operator_offset": offset,
        "surface_operator_valid": operator_valid,
    }


QUERY_KEYS = {
    "query_features", "query_coord", "baseline", "query_background", "query_level",
    "local_features", "local_offsets", "local_innovation", "local_valid",
    "local_profile_values", "local_profile_valid", "local_profile_ids",
    "local_profile_absolute", "local_profile_climatology", "surface_observation",
    "surface_operator", "surface_operator_offset", "surface_operator_valid",
}


def _queries(scene, selection):
    return {key: value[selection] if key in QUERY_KEYS else value for key, value in scene.items()}


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_preserves_exact_oi_at_zero_initialization(recipe):
    model, scene = _model(recipe), _scene()
    result = model(scene)
    assert torch.equal(result["mean"], scene["baseline"])
    assert torch.isfinite(result["std"]).all()
    assert (result["std"] > 0.).all()


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_has_finite_active_gradients_to_relevant_observations(recipe):
    model, scene = _model(recipe, active=True), _scene()
    for key in ("obs_innovation", "local_innovation", "local_profile_values", "local_profile_absolute", "surface_observation"):
        scene[key].requires_grad_()
    output = model(scene)
    loss = (output["mean"] - .35).square().mean() + .01 * output["std"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0.
    if model.covariance is not None or model.alignment is not None or model.operator is not None:
        evidence_key = "local_profile_absolute" if model.alignment is not None and model.covariance is None else "local_profile_values"
        gradient = scene[evidence_key].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.
    if recipe in ("operator_update", "all_modules"):
        gradient = scene["surface_observation"].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.
    if model.alignment is not None and model.alignment.shift_head is not None:
        gradient = model.alignment.shift_head[-1].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0.


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_query_chunk_and_permutation_invariance(recipe):
    model, scene = _model(recipe, active=True), _scene()
    state = model.encode(scene)
    complete = model.decode(scene, state)
    permutation = torch.tensor([3, 0, 4, 2, 1])
    permuted = model.decode(_queries(scene, permutation), state)
    parts = [model.decode(_queries(scene, slice(start, end)), state)
             for start, end in ((0, 2), (2, 3), (3, 5))]
    for key in ("mean", "std"):
        torch.testing.assert_close(permuted[key], complete[key][permutation], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(torch.cat([part[key] for part in parts]), complete[key], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("recipe", ["cov_diagonal", "cov_correlated", "aligned_correlated", "all_modules"])
def test_covariance_candidate_compares_with_absolute_standardized_oi(recipe):
    model, scene = _model(recipe), _scene()
    with torch.no_grad():
        model.local_gate[-1].bias.fill_(math.atanh(.3))
    candidate = []
    handle = model.covariance.register_forward_hook(lambda module, arguments, output: candidate.append(output[0]))
    output = model(scene)
    handle.remove()
    gate = output["diagnostics"]["covariance_gate"]
    expected = scene["baseline"] + gate * (candidate[0] - scene["baseline"])
    torch.testing.assert_close(output["mean"], expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("recipe", ["aligned_correlated", "all_modules"])
def test_combined_covariance_reads_shifted_complete_profile_values(recipe):
    model, scene = _model(recipe, active=True), _scene()
    observed_values = []
    handle = model.covariance.register_forward_pre_hook(
        lambda module, arguments: observed_values.append(arguments[0]["local_profile_values"].detach().clone()))
    initial = model(scene)["mean"]
    with torch.no_grad():
        model.alignment.shift_head[-1].bias.fill_(.2)
    shifted = model(scene)["mean"]
    handle.remove()
    assert not torch.equal(observed_values[0], observed_values[1])
    assert not torch.allclose(initial, shifted, rtol=1e-8, atol=1e-10)
    assert torch.isfinite(shifted).all()


def test_operator_direct_and_update_share_identical_prior_when_surface_missing():
    direct, update = _model("operator_direct", active=True), _model("operator_update", active=True)
    update.load_state_dict(direct.state_dict(), strict=True)
    scene = _scene()
    scene["surface_operator_valid"].zero_()
    for key in ("mean", "std"):
        torch.testing.assert_close(direct(scene)[key], update(scene)[key], rtol=1e-10, atol=1e-11)
    assert sum(p.numel() for p in direct.parameters()) == sum(p.numel() for p in update.parameters())


def test_operator_update_changes_mean_and_variance_together_for_available_surface():
    model, scene = _model("operator_update", active=True), _scene()
    # Read the native near-surface level so SST/SSS have nonzero direct gain.
    scene["query_level"].zero_()
    scene["query_coord"][:, 2] = scene["profile_depths"][0]
    without = copy.deepcopy(scene)
    without["surface_operator_valid"].zero_()
    absent, present = model(without), model(scene)
    active = scene["surface_operator_valid"][:, :2]
    assert (present["mean"][active] - absent["mean"][active]).abs().sum() > 1e-6
    assert (present["std"][active] < absent["std"][active]).all()


def test_near_zero_profile_shift_retains_native_first_guess_innovation_units():
    model, scene = _model("profile_shared"), _scene()
    h = torch.zeros(5, model.config.width, dtype=torch.float64)
    unshifted, _ = model.alignment(scene, h)
    with torch.no_grad():
        model.alignment.shift_head[-1].bias.fill_(1e-9)
    near_zero, _ = model.alignment(scene, h)
    active = unshifted["local_valid"] & near_zero["local_valid"]
    torch.testing.assert_close(near_zero["local_innovation"][active],
                               unshifted["local_innovation"][active], rtol=1e-6, atol=1e-6)
