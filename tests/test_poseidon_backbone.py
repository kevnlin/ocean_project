"""The public Poseidon-T processor as the gridded model's state encoder (a baseline backbone)."""
import pytest
import torch

from ocean_tokenizer.grid_kriging_ocean import GridKrigingConfig, GridKrigingModel
from ocean_tokenizer.poseidon_backbone import RESET_KEYS, PoseidonEncoder, public_weights
from test_grid_kriging_ocean import _config, _scene


def test_the_encoder_maps_a_grid_to_state_channels_on_the_same_grid():
    encoder = PoseidonEncoder(7, 5, initialization="random", seed=1).eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, 7, 13, 26), lead=1)
    assert out.shape == (1, 5, 13, 26) and torch.isfinite(out).all()


def test_pretrained_keeps_the_public_processor_and_both_share_new_adapters():
    pretrained = PoseidonEncoder(7, 5, initialization="pretrained", seed=1).backbone.state_dict()
    random = PoseidonEncoder(7, 5, initialization="random", seed=1).backbone.state_dict()
    public = public_weights()
    processor = next(k for k in public if k not in RESET_KEYS and public[k].ndim == 2)
    assert torch.equal(pretrained[processor], public[processor])
    assert not torch.equal(random[processor], public[processor])
    for key in RESET_KEYS:
        assert torch.equal(pretrained[key], random[key]), key


def test_the_lead_conditions_the_state():
    encoder = PoseidonEncoder(7, 5, initialization="pretrained", seed=1).eval()
    x = torch.randn(1, 7, 13, 26)
    with torch.no_grad():
        assert not torch.allclose(encoder(x, lead=0), encoder(x, lead=3))


def test_a_poseidon_model_trains_and_replays():
    torch.manual_seed(3)
    model = GridKrigingModel(_config(backbone="poseidon", evolution=False, poseidon_init="random"))
    scene = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in _scene(lead=3).items()}
    with torch.no_grad():
        model.mean_head[-1].weight.normal_(0., .1)
    out = model(scene)
    ((out["mean"] - .3).square().mean() + .01 * out["std"].mean()).backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is None or torch.isfinite(g).all() for g in grads.values())
    assert grads["encoder.backbone.patch_recovery.projection.weight"].abs().sum() > 0
    config = GridKrigingConfig(**model.config.to_dict())
    assert config == model.config
    replay = GridKrigingModel(config)
    replay.load_state_dict(model.state_dict(), strict=True)
    model.eval(); replay.eval()
    torch.testing.assert_close(replay(scene)["mean"], model(scene)["mean"])


def test_the_lead_conditioned_backbone_replaces_the_evolution_network():
    with pytest.raises(ValueError):
        _config(backbone="poseidon")                    # evolution defaults to on
    with pytest.raises(ValueError):
        _config(backbone="poseidon", evolution=False, background="prior")
