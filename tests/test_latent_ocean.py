"""Behavior checks for shared ocean latents with an anchored innovation path.

The nontrivial inference checks activate residual heads first: a zero-initialized
residual would otherwise hide broken masking or query-dependent MoE routing.
"""

import copy
import io
from dataclasses import asdict

import pytest
import torch

from ocean_tokenizer.latent_ocean import (
    LatentOceanConfig,
    LatentOceanModel,
    gaussian_objective,
)


VARIANTS = ("dense", "soft_moe", "local_transformer")
QUERY_KEYS = (
    "query_features", "query_coord", "baseline", "query_background",
    "local_features", "local_offsets", "local_innovation", "local_valid",
)
OBS_KEYS = ("obs_features", "obs_coord", "obs_innovation", "obs_valid")
LOCAL_KEYS = ("local_features", "local_offsets", "local_innovation", "local_valid")


def _model(variant, *, active=False, use_latent=True, use_local=True):
    torch.manual_seed(17)
    config = LatentOceanConfig(
        n_obs_features=5,
        n_query_features=7,
        n_local_features=6,
        n_sat_features=4,
        variant=variant,
        width=24,
        n_latents=6,
        n_heads=4,
        n_blocks=2,
        n_query_blocks=2,
        n_experts=3,
        slots_per_expert=2,
        mlp_ratio=2,
        use_latent=use_latent,
        use_local=use_local,
    )
    model = LatentOceanModel(config).double().eval()
    if active:
        generator = torch.Generator().manual_seed(19)
        with torch.no_grad():
            for head in model.mean_head:
                head[-1].weight.normal_(std=0.08, generator=generator)
                head[-1].bias.fill_(0.025)
            model.local_gate[-1].bias.fill_(0.2)
    return model


def _scene(n_obs=11, n_query=9, n_local=4):
    generator = torch.Generator().manual_seed(23)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    def coord(n, surface=False):
        values = torch.rand(n, 4, generator=generator, dtype=torch.float64)
        values[:, 0] = values[:, 0] * 120 - 60
        values[:, 1] = values[:, 1] * 358 - 179
        values[:, 2] = 0 if surface else values[:, 2] * 1500
        values[:, 3] *= 28
        return values

    scene = dict(
        obs_features=randn(n_obs, 5),
        obs_coord=coord(n_obs),
        obs_innovation=randn(n_obs, 2),
        obs_valid=torch.ones(n_obs, 2, dtype=torch.bool),
        query_features=randn(n_query, 7),
        query_coord=coord(n_query),
        baseline=randn(n_query, 2),
        query_background=randn(n_query, 2),
        local_features=randn(n_query, n_local, 6),
        local_offsets=randn(n_query, n_local, 4) * 10,
        local_innovation=randn(n_query, n_local, 2),
        local_valid=torch.ones(n_query, n_local, 2, dtype=torch.bool),
        satellite_features=randn(7, 4),
        satellite_coord=coord(7, surface=True),
        satellite_valid=torch.ones(7, dtype=torch.bool),
    )
    if n_obs >= 2:
        scene["obs_valid"][0, 1] = False
        scene["obs_valid"][1] = False
    if n_local >= 2 and n_query:
        scene["local_valid"][0, 0, 1] = False
        scene["local_valid"][0, 1] = False
    return scene


def _select_queries(scene, selection):
    return {
        key: value[selection] if key in QUERY_KEYS else value
        for key, value in scene.items()
    }


def _same_predictions(actual, expected):
    for key in ("mean", "std"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("variant", VARIANTS)
def test_neutral_initialization_preserves_the_frozen_baseline_exactly(variant):
    model, scene = _model(variant), _scene()
    with torch.no_grad():
        output = model(scene)
    assert torch.equal(output["mean"], scene["baseline"])
    assert output["std"].shape == scene["baseline"].shape
    assert torch.isfinite(output["std"]).all()
    assert (output["std"] > 0).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_queries_can_be_permuted_and_decoded_alone_or_in_chunks(variant):
    """A query must not borrow information from other targets in its batch."""
    model, scene = _model(variant, active=True), _scene()
    permutation = torch.tensor([8, 1, 5, 0, 7, 3, 6, 2, 4])
    with torch.no_grad():
        state = model.encode(scene)
        full = model.decode(scene, state)
        permuted = model.decode(_select_queries(scene, permutation), state)
        alone = model.decode(_select_queries(scene, slice(3, 4)), state)
        parts = [
            model.decode(_select_queries(scene, slice(start, end)), state)
            for start, end in ((0, 2), (2, 5), (5, 9))
        ]
    assert not torch.equal(full["mean"], scene["baseline"])
    _same_predictions(permuted, {key: full[key][permutation] for key in ("mean", "std")})
    _same_predictions(alone, {key: full[key][3:4] for key in ("mean", "std")})
    _same_predictions(
        {key: torch.cat([part[key] for part in parts]) for key in ("mean", "std")},
        full,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_masked_variables_and_absent_context_cannot_leak_corrupted_values(variant):
    model, scene = _model(variant, active=True), _scene()
    poisoned = copy.deepcopy(scene)
    poisoned["obs_innovation"][~scene["obs_valid"]] = float("nan")
    absent_obs = ~scene["obs_valid"].any(-1)
    poisoned["obs_features"][absent_obs] = float("nan")
    poisoned["obs_coord"][absent_obs] = float("nan")
    poisoned["local_innovation"][~scene["local_valid"]] = float("nan")
    absent_local = ~scene["local_valid"].any(-1)
    poisoned["local_features"][absent_local] = float("nan")
    poisoned["local_offsets"][absent_local] = float("nan")
    # Missing satellite pixels must be inert, including their coordinates.
    scene["satellite_valid"][2] = False
    poisoned["satellite_valid"][2] = False
    poisoned["satellite_features"][2] = float("nan")
    poisoned["satellite_coord"][2] = float("nan")
    with torch.no_grad():
        expected, actual = model(scene), model(poisoned)
    assert torch.isfinite(actual["mean"]).all()
    assert torch.isfinite(actual["std"]).all()
    _same_predictions(actual, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_completely_empty_or_masked_observations_have_finite_fallback(variant):
    model = _model(variant, active=True)
    empty = _scene(n_obs=0, n_local=0)
    empty["satellite_features"] = empty["satellite_features"][:0]
    empty["satellite_coord"] = empty["satellite_coord"][:0]
    empty["satellite_valid"] = empty["satellite_valid"][:0]
    masked = _scene()
    masked["obs_valid"].fill_(False)
    masked["local_valid"].fill_(False)
    masked["satellite_valid"].fill_(False)
    for key in (*OBS_KEYS[:-1], *LOCAL_KEYS[:-1], "satellite_features", "satellite_coord"):
        masked[key].fill_(float("nan"))
    with torch.no_grad():
        for scene in (empty, masked):
            output = model(scene)
            assert torch.isfinite(output["mean"]).all()
            assert torch.isfinite(output["std"]).all()
            assert torch.isfinite(output["aux_loss"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_observation_and_local_neighbor_order_do_not_change_predictions(variant):
    model, scene = _model(variant, active=True), _scene()
    permuted = copy.deepcopy(scene)
    obs_order = torch.tensor([8, 2, 10, 4, 0, 7, 3, 9, 1, 6, 5])
    local_order = torch.tensor([2, 0, 3, 1])
    sat_order = torch.tensor([5, 1, 6, 3, 0, 4, 2])
    for key in OBS_KEYS:
        permuted[key] = scene[key][obs_order]
    for key in LOCAL_KEYS:
        permuted[key] = scene[key][:, local_order]
    for key in ("satellite_features", "satellite_coord", "satellite_valid"):
        permuted[key] = scene[key][sat_order]
    with torch.no_grad():
        _same_predictions(model(permuted), model(scene))


@pytest.mark.parametrize("variant", VARIANTS)
def test_serialized_config_and_strict_checkpoint_replay_the_prediction(variant):
    model, scene = _model(variant, active=True), _scene()
    checkpoint = io.BytesIO()
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = LatentOceanModel(LatentOceanConfig(**saved["config"])).double().eval()
    result = restored.load_state_dict(saved["state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    with torch.no_grad():
        expected, actual = model(scene), restored(scene)
    for key in ("mean", "std"):
        assert torch.equal(actual[key], expected[key])


@pytest.mark.parametrize("variant", VARIANTS)
def test_local_innovation_path_can_use_an_observation_without_global_readout(variant):
    model, scene = _model(variant), _scene()
    scene["baseline"] = scene["query_background"].clone()
    scene["local_innovation"].zero_()
    with torch.no_grad():
        model.local_gate[-1].bias.fill_(0.5)
        before = model(scene)
        changed = copy.deepcopy(scene)
        changed["local_innovation"][3, 0, 0] = 2.0
        after = model(changed)
    assert abs(float(after["mean"][3, 0] - before["mean"][3, 0])) > 1e-6
    torch.testing.assert_close(after["mean"][:3], before["mean"][:3])
    torch.testing.assert_close(after["mean"][4:], before["mean"][4:])


@pytest.mark.parametrize("variant", VARIANTS)
def test_supervised_distribution_loss_reaches_observations_queries_and_local_path(variant):
    """Activated readouts must transmit a real supervised gradient upstream."""
    model, scene = _model(variant, active=True), _scene()
    differentiated = (
        "obs_features", "obs_innovation", "satellite_features",
        "query_features", "local_innovation",
    )
    for key in differentiated:
        scene[key].requires_grad_()
    output = model(scene)
    target = scene["baseline"] + torch.tensor([0.4, -0.3], dtype=torch.float64)
    loss = (
        0.5 * ((target - output["mean"]) / output["std"]).square()
        + output["std"].log()
    ).mean() + 0.01 * output["aux_loss"]
    loss.backward()
    for key in differentiated:
        gradient = scene[key].grad
        assert gradient is not None, key
        assert torch.isfinite(gradient).all(), key
        assert gradient.abs().sum() > 1e-10, key
    # A mask must also prevent training on the missing channel's placeholder.
    assert torch.equal(
        scene["obs_innovation"].grad[~scene["obs_valid"]],
        torch.zeros_like(scene["obs_innovation"].grad[~scene["obs_valid"]]),
    )
    module_names = [
        "mean_head", "local_gate", "std_head", "observation_projection",
        "satellite_projection", "latent_blocks", "query_blocks",
    ]
    if variant == "local_transformer":
        module_names.append("local_transformer")
    for module_name in module_names:
        gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if name.startswith(module_name)
        ]
        assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients), module_name
    if variant == "soft_moe":
        for block in model.latent_blocks:
            assert block.ffn.router.grad is not None
            assert torch.isfinite(block.ffn.router.grad).all()
            assert block.ffn.router.grad.abs().sum() > 0
            for expert in block.ffn.experts:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())


@pytest.mark.parametrize("variant", VARIANTS)
def test_equivalent_longitudes_preserve_geographic_predictions(variant):
    model, scene = _model(variant, active=True), _scene()
    wrapped = copy.deepcopy(scene)
    for key in ("obs_coord", "query_coord", "satellite_coord"):
        wrapped[key][:, 1] += 360.0
    with torch.no_grad():
        expected, actual = model(scene), model(wrapped)
    for key in ("mean", "std"):
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("use_latent,use_local", [(False, True), (True, False), (False, False)])
def test_component_ablations_keep_neutral_mean_equal_to_the_original_baseline(variant, use_latent, use_local):
    model = _model(variant, use_latent=use_latent, use_local=use_local)
    scene = _scene()
    with torch.no_grad():
        output = model(scene)
    assert torch.equal(output["mean"], scene["baseline"])
    assert torch.isfinite(output["std"]).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_latent_disabled_ignores_global_context_and_keeps_local_supervised_gradients(variant):
    model = _model(variant, active=True, use_latent=False)
    scene = _scene()
    poisoned = copy.deepcopy(scene)
    for key in (*OBS_KEYS[:-1], "satellite_features", "satellite_coord"):
        poisoned[key].fill_(float("nan"))
    poisoned["obs_valid"].fill_(False)
    poisoned["satellite_valid"].fill_(False)
    with torch.no_grad():
        _same_predictions(model(scene), model(poisoned))
    scene["local_innovation"].requires_grad_()
    output = model(scene)
    loss = (output["mean"] - scene["baseline"] - .4).square().mean()
    loss.backward()
    assert scene["local_innovation"].grad is not None
    assert scene["local_innovation"].grad.abs().sum() > 1e-10
    for name in ("local_scorer", "local_gate"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(name))
    assert all(p.grad is None for n, p in model.named_parameters()
               if n.startswith(("observation_projection", "latent_blocks", "query_blocks")))


@pytest.mark.parametrize("variant", VARIANTS)
def test_local_disabled_ignores_added_local_inputs_but_preserves_existing_oi_baseline(variant):
    model = _model(variant, active=True, use_local=False)
    scene = _scene()
    poisoned = copy.deepcopy(scene)
    for key in LOCAL_KEYS[:-1]:
        poisoned[key].fill_(float("nan"))
    poisoned["local_valid"].fill_(False)
    with torch.no_grad():
        expected = model(scene)
        _same_predictions(model(poisoned), expected)
    assert not torch.equal(expected["mean"], scene["baseline"])
    assert torch.equal(expected["diagnostics"]["local_gate"], torch.zeros_like(scene["baseline"]))


def test_training_objective_balances_variables_despite_different_valid_counts():
    """Temperature's larger sample count must not silently reweight salinity."""
    mean = torch.tensor([[1.0, 2.0], [3.0, 20.0], [5.0, 30.0]], requires_grad=True)
    std = torch.ones_like(mean, requires_grad=True)
    valid = torch.tensor([[True, True], [True, False], [True, False]])
    target = torch.zeros_like(mean)
    target[~valid] = float("nan")
    objective = gaussian_objective({"mean": mean, "std": std}, target, valid)
    expected = ((1.0 + 9.0 + 25.0) / 3.0 + 4.0) / 2.0
    assert float(objective["mse"].detach()) == pytest.approx(expected)
    objective["loss"].backward()
    assert torch.equal(mean.grad[~valid], torch.zeros_like(mean.grad[~valid]))
    assert torch.isfinite(mean.grad).all()


def test_no_valid_targets_have_zero_finite_loss_and_no_supervised_gradient():
    mean = torch.randn(4, 2, requires_grad=True)
    std = torch.full((4, 2), 0.6, requires_grad=True)
    target = torch.full((4, 2), float("nan"))
    valid = torch.zeros(4, 2, dtype=torch.bool)
    objective = gaussian_objective(
        {"mean": mean, "std": std}, target, valid, mse_weight=1, nll_weight=1,
    )
    assert float(objective["loss"].detach()) == 0.0
    objective["loss"].backward()
    for gradient in (mean.grad, std.grad):
        assert torch.isfinite(gradient).all()
        assert torch.equal(gradient, torch.zeros_like(gradient))
