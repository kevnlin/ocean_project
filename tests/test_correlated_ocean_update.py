"""Numerical mechanism checks, including a direct dense Gaussian reference."""
import copy

import pytest
import torch

from ocean_tokenizer.correlated_ocean_update import CorrelatedOceanUpdate


def _fixture(mode="correlated", q=3, k=3, l=4):
    torch.manual_seed(610)
    model = CorrelatedOceanUpdate(width=12, mode=mode, rank=5).double()
    depths = torch.tensor([10., 100., 350., 700.], dtype=torch.float64)[:l]
    levels = torch.arange(q) % l
    mean = torch.randn(l, 2, dtype=torch.float64) * .2
    std = torch.rand(l, 2, dtype=torch.float64) + .3
    offsets = torch.randn(q, k, 4, dtype=torch.float64)
    offsets[..., :2] *= 30.
    offsets[..., 2] *= 2.
    offsets[..., 3] *= .05
    coord = torch.zeros(q, 4, dtype=torch.float64)
    coord[:, 2] = depths[levels]
    scene = {
        "local_profile_values": torch.randn(q, k, l, 2, dtype=torch.float64) * .4 + mean,
        "local_profile_valid": torch.ones(q, k, l, 2, dtype=torch.bool),
        "profile_depths": depths,
        "profile_normalization_mean": mean,
        "profile_normalization_std": std,
        "query_level": levels,
        "local_offsets": offsets,
        "local_profile_ids": torch.arange(k).expand(q, k),
        "query_coord": coord,
    }
    h = torch.randn(q, 12, dtype=torch.float64)
    return model, scene, h


def _dense_reference(model, scene, h):
    """Explicit dense B/R only in tests, independently checking Woodbury."""
    context = model.context(h)
    p = context[:, model.rank:]
    gates = 2. * context[:, :model.rank].sigmoid()
    state = model._bounded(p[:, 0], .05, 4.)
    nugget = model._bounded(p[:, 1:3], 1e-4, 1.)
    noise = model._bounded(p[:, 3:5], 1e-4, 4.)
    profile = model._bounded(p[:, 5:7], 1e-5, 2.)
    common = model._bounded(p[:, 7], 1e-5, 1.)
    if model.mode == "diagonal":
        profile, common = profile * 0., common * 0.
    q, k, l, _ = scene["local_profile_values"].shape
    query_depth = scene["query_coord"][:, 2]
    fo = model._factors(scene["local_offsets"][:, :, None].expand(q, k, l, 4),
                        scene["profile_depths"][None, None].expand(q, k, l),
                        query_depth, gates, state).reshape(q, k * l * 2, model.rank)
    fq = model._factors(h.new_zeros(q, 1, 1, 4), query_depth[:, None, None],
                        query_depth, gates, state)[:, 0, 0]
    std = scene["profile_normalization_std"][scene["query_level"]]
    y = ((scene["local_profile_values"] - scene["profile_normalization_mean"][None, None])
         / std[:, None, None]).reshape(q, k * l * 2)
    means, variances, eigvalues = [], [], []
    for i in range(q):
        channel = torch.arange(k * l * 2) % 2
        identity = scene["local_profile_ids"][i].repeat_interleave(l * 2)
        matching = (identity[:, None] == identity[None]) & (channel[:, None] == channel[None])
        r = torch.diag(noise[i, channel]) + matching * profile[i, channel][:, None] + common[i]
        covariance = fo[i] @ fo[i].T + r
        cross = fq[i] @ fo[i].T
        means.append(cross @ torch.linalg.solve(covariance, y[i]))
        variances.append(fq[i].square().sum(-1) + nugget[i]
                         - (cross * torch.linalg.solve(covariance, cross.T).T).sum(-1))
        joint = torch.cat((fq[i], fo[i]))
        b = joint @ joint.T
        b[:2, :2] = b[:2, :2] + torch.diag(nugget[i])
        eigvalues.append(torch.linalg.eigvalsh(b))
        assert torch.linalg.eigvalsh(r).min() > 0.
    return torch.stack(means), torch.stack(variances), torch.stack(eigvalues)


@pytest.mark.parametrize("mode", ["diagonal", "correlated"])
def test_woodbury_matches_dense_update_and_joint_state_covariance_is_psd(mode):
    model, scene, h = _fixture(mode)
    candidate, variance, diagnostics = model(scene, h)
    expected, expected_variance, eigenvalues = _dense_reference(model, scene, h)
    torch.testing.assert_close(candidate, expected, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(variance, expected_variance, rtol=1e-10, atol=1e-11)
    assert eigenvalues.min() > -1e-12
    assert (variance > 0.).all()
    assert (variance <= diagnostics["prior_variance"] + 1e-12).all()


@pytest.mark.parametrize("mode", ["diagonal", "correlated"])
def test_empty_or_poisoned_invalid_evidence_returns_zero_prior_mean(mode):
    model, scene, h = _fixture(mode)
    scene["local_profile_valid"].zero_()
    scene["local_profile_values"].fill_(float("nan"))
    scene["local_offsets"].fill_(float("nan"))
    candidate, variance, diagnostics = model(scene, h)
    assert torch.equal(candidate, torch.zeros_like(candidate))
    torch.testing.assert_close(variance, diagnostics["prior_variance"])
    assert torch.isfinite(variance).all()
    assert torch.equal(diagnostics["unique_profile_count"], torch.zeros(3, dtype=torch.long))


def _duplicate_selected_profile(scene, *, independent=False):
    duplicated = copy.deepcopy(scene)
    q = scene["local_profile_values"].shape[0]
    chosen = torch.arange(q) % scene["local_profile_values"].shape[1]
    for key in ("local_profile_values", "local_profile_valid", "local_offsets", "local_profile_ids"):
        extra = scene[key][torch.arange(q), chosen][:, None].clone()
        if key == "local_profile_ids" and independent:
            extra += 10000
        duplicated[key] = torch.cat((scene[key], extra), dim=1)
    return duplicated


def test_selectively_duplicate_record_is_invariant_but_new_independent_record_adds_information():
    model, scene, h = _fixture()
    # Missing individual depths must not break identity normalization.
    scene["local_profile_valid"][0, 0, 1, 1] = False
    scene["local_profile_valid"][2, 2, 0] = False
    base_mean, base_var, base_diagnostics = model(scene, h)
    repeated_mean, repeated_var, repeated_diagnostics = model(_duplicate_selected_profile(scene), h)
    torch.testing.assert_close(repeated_mean, base_mean, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(repeated_var, base_var, rtol=1e-10, atol=1e-11)
    assert torch.equal(repeated_diagnostics["unique_profile_count"], base_diagnostics["unique_profile_count"])
    _, independent_var, independent_diagnostics = model(_duplicate_selected_profile(scene, independent=True), h)
    assert (independent_var < base_var - 1e-9).all()
    assert torch.equal(independent_diagnostics["unique_profile_count"], base_diagnostics["unique_profile_count"] + 1)


def test_diagonal_ablation_treats_duplicate_records_as_independent():
    model, scene, h = _fixture("diagonal")
    _, original_var, _ = model(scene, h)
    _, repeated_var, _ = model(_duplicate_selected_profile(scene), h)
    assert (repeated_var < original_var - 1e-9).all()


def test_larger_independent_measurement_noise_weakens_update_and_increases_posterior_variance():
    model, scene, h = _fixture(q=1, k=1)
    scene["local_profile_values"] = (scene["profile_normalization_mean"][None, None]
                                       + torch.ones_like(scene["local_profile_values"]))
    small_mean, small_var, _ = model(scene, h)
    noisy = copy.deepcopy(scene)
    noisy["local_profile_noise_variance"] = torch.full_like(scene["local_profile_values"], 100.)
    large_mean, large_var, _ = model(noisy, h)
    assert (large_var > small_var).all()
    assert large_mean.square().sum() < small_mean.square().sum()


def test_other_depths_can_update_query_and_observation_level_training_mean_is_retained():
    model, scene, h = _fixture(q=1, k=1)
    scene["local_profile_values"] = scene["profile_normalization_mean"][None, None].clone()
    scene["local_profile_valid"].zero_()
    scene["local_profile_valid"][0, 0, 1, 0] = True
    scene["query_level"].zero_()
    scene["query_coord"][:, 2] = scene["profile_depths"][0]
    neutral, _, _ = model(scene, h)
    assert torch.equal(neutral, torch.zeros_like(neutral))
    scene["local_profile_values"][0, 0, 1, 0] += 1.
    changed, _, _ = model(scene, h)
    assert changed.abs().sum() > 1e-5
    assert not scene["local_profile_valid"][..., 0, :].any()


QUERY_KEYS = ("local_profile_values", "local_profile_valid", "query_level", "local_offsets",
              "local_profile_ids", "query_coord")


def _queries(scene, selection):
    return {key: value[selection] if key in QUERY_KEYS else value for key, value in scene.items()}


def test_query_and_profile_permutation_and_query_chunk_invariance():
    model, scene, h = _fixture()
    baseline = model(scene, h)
    order = torch.tensor([2, 0, 1])
    permuted = model(_queries(scene, order), h[order])
    parts = [model(_queries(scene, slice(i, i + 1)), h[i:i + 1]) for i in range(3)]
    profiles = copy.deepcopy(scene)
    for key in ("local_profile_values", "local_profile_valid", "local_offsets", "local_profile_ids"):
        profiles[key] = profiles[key][:, order]
    profile_permuted = model(profiles, h)
    for index in (0, 1):
        torch.testing.assert_close(permuted[index], baseline[index][order], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(torch.cat([part[index] for part in parts]), baseline[index], rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(profile_permuted[index], baseline[index], rtol=1e-10, atol=1e-11)


def test_covariance_and_raw_evidence_have_finite_nonzero_gradients():
    model, scene, h = _fixture()
    h.requires_grad_()
    scene["local_profile_values"].requires_grad_()
    candidate, variance, _ = model(scene, h)
    loss = (candidate - .3).square().mean() + .2 * variance.mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0.
    assert torch.isfinite(h.grad).all()
    # Initial context weights are zero, but after a learned context is activated,
    # the covariance must transmit gradients to h as well.
    with torch.no_grad():
        model.context.weight.fill_(.01)
    h.grad.zero_()
    model(scene, h)[0].square().mean().backward()
    assert h.grad.abs().sum() > 0.
    assert torch.isfinite(scene["local_profile_values"].grad).all()
    assert scene["local_profile_values"].grad.abs().sum() > 0.


@pytest.mark.parametrize("q,k", [(2, 0), (0, 3)])
def test_zero_query_or_zero_profile_dimensions(q, k):
    model, scene, h = _fixture(q=q, k=k)
    candidate, variance, _ = model(scene, h)
    assert candidate.shape == variance.shape == (q, 2)
    assert torch.isfinite(variance).all()
    assert torch.equal(candidate, torch.zeros_like(candidate))


def test_fp32_solve_survives_autocast_and_reports_finite_uncertainty():
    model, scene, h = _fixture()
    model = model.float()
    scene = {key: value.float() if value.dtype == torch.float64 else value for key, value in scene.items()}
    h = h.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        candidate, variance, diagnostics = model(scene, h)
    assert candidate.dtype == variance.dtype == torch.float32
    assert torch.isfinite(candidate).all() and torch.isfinite(variance).all()
    assert (variance > 0.).all()
    assert (variance <= diagnostics["prior_variance"] + 1e-6).all()


def test_failed_rounded_covariance_solve_receives_explicit_numerical_regularization():
    # I + a large rank-one Gram matrix can lose its unit diagonal in FP32.
    # The fallback must remain finite and identify that changed numerical solve.
    rounded = torch.full((1, 8, 8), 1e8, dtype=torch.float32)
    factor, jitter = CorrelatedOceanUpdate._cholesky(rounded)
    assert (jitter > 0.).all()
    assert torch.isfinite(factor).all()
    torch.testing.assert_close(factor @ factor.transpose(-1, -2),
                               rounded + torch.eye(8)[None] * jitter[:, None, None])
