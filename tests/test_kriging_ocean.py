"""The query-scaled innovation model with kriging attention in its local path."""
import math

import pytest
import torch

from ocean_tokenizer.innovation_ocean import InnovationOceanConfig
from ocean_tokenizer.kriging_ocean import LOCAL_WEIGHTINGS, KrigingOceanConfig, KrigingOceanModel
from ocean_tokenizer.query_scaled_innovation import QueryScaledInnovationModel
from test_innovation_ocean import _scene

ELL = ((250.,) * 20, (300.,) * 20)
GAMMA = ((.01,) * 20, (.03,) * 20)
SHARED = dict(n_obs_features=62, n_query_features=58, n_local_features=66, n_sat_features=56,
              variant="local_transformer", recipe="profile_shared", width=16, n_heads=4, n_latents=4,
              n_blocks=1, n_query_blocks=1, covariance_rank=4, n_profile_levels=20)


def _kriging(weighting, *, active=False):
    torch.manual_seed(718)
    extra = {} if weighting == "softmax" else dict(kriging_ell_km=ELL, kriging_gamma=GAMMA)
    model = KrigingOceanModel(KrigingOceanConfig(**SHARED, local_weighting=weighting, **extra)).double().eval()
    if active:
        _activate(model)
    return model


def _activate(model):
    with torch.no_grad():
        for head in model.mean_head:
            head[-1].weight.fill_(.015)
            head[-1].bias.fill_(.04)
        model.local_gate[-1].bias.fill_(math.atanh(.3))


def test_softmax_weighting_reproduces_the_query_scaled_model_exactly():
    torch.manual_seed(718)
    reference = QueryScaledInnovationModel(InnovationOceanConfig(**SHARED)).double().eval()
    _activate(reference)
    model = _kriging("softmax")
    model.load_state_dict(reference.state_dict(), strict=True)
    scene = _scene()
    expected, result = reference(scene), model(scene)
    assert torch.equal(result["mean"], expected["mean"])
    assert torch.equal(result["std"], expected["std"])


@pytest.mark.parametrize("weighting", LOCAL_WEIGHTINGS)
def test_every_weighting_starts_from_the_anchor(weighting):
    model, scene = _kriging(weighting), _scene()
    result = model(scene)
    assert torch.equal(result["mean"], scene["baseline"])
    assert torch.isfinite(result["std"]).all() and (result["std"] > 0).all()


@pytest.mark.parametrize("weighting", [w for w in LOCAL_WEIGHTINGS if w != "softmax"])
def test_kriging_replaces_the_softmax_candidate(weighting):
    scene = _scene()
    softmax, kriging = _kriging("softmax", active=True), _kriging(weighting, active=True)
    kriging.load_state_dict({**softmax.state_dict(), **{k: v for k, v in kriging.state_dict().items()
                                                       if k.startswith("kriging.")}}, strict=True)
    assert not torch.allclose(kriging(scene)["mean"], softmax(scene)["mean"])


@pytest.mark.parametrize("weighting", LOCAL_WEIGHTINGS)
def test_every_weighting_trains_with_finite_gradients(weighting):
    model, scene = _kriging(weighting, active=True), _scene()
    scene["local_profile_absolute"].requires_grad_()
    output = model(scene)
    loss = (output["mean"] - .35).square().mean() + .01 * output["std"].mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    grad = scene["local_profile_absolute"].grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    if weighting == "ka_flow":
        assert model.kriging.local_head[-1].weight.grad.abs().sum() > 0
    if weighting in ("ka_stationary", "ka_flow"):
        assert model.kriging.log_ell.grad.abs().sum() > 0


@pytest.mark.parametrize("weighting", LOCAL_WEIGHTINGS)
def test_serialized_config_replays_the_prediction(weighting):
    model, scene = _kriging(weighting, active=True), _scene()
    config = KrigingOceanConfig(**model.config.to_dict())
    assert config == model.config
    replay = KrigingOceanModel(config).double().eval()
    replay.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(replay(scene)["mean"], model(scene)["mean"])


def test_kriging_weightings_need_their_initial_settings():
    with pytest.raises(ValueError):
        KrigingOceanConfig(**SHARED, local_weighting="ka_flow")
    with pytest.raises(ValueError):
        KrigingOceanConfig(**SHARED, local_weighting="ka_ridge", kriging_ell_km=ELL, kriging_gamma=GAMMA)
