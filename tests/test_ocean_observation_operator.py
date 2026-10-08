"""Numerical and representation contracts for explicit observation updates."""
import pytest
import torch

from ocean_tokenizer.ocean_observation_operator import (
    ObservationOperatorUpdate, build_surface_observation_operator,
)


def _case(q=4, levels=3):
    generator = torch.Generator().manual_seed(923)
    mean = torch.randn(q, levels, 2, generator=generator, dtype=torch.float64)
    variance = torch.rand(q, levels, 2, generator=generator, dtype=torch.float64) + .3
    observation = torch.randn(q, 3, generator=generator, dtype=torch.float64)
    operator = torch.randn(q, 3, levels * 2, generator=generator, dtype=torch.float64)
    offset = torch.randn(q, 3, generator=generator, dtype=torch.float64)
    valid = torch.ones(q, 3, dtype=torch.bool)
    h = torch.randn(q, 8, generator=generator, dtype=torch.float64)
    return mean, variance, observation, operator, offset, valid, h


def test_matches_analytic_gaussian_posterior_and_same_gain_variance():
    args = _case()
    mean, variance, y, H, offset, valid, h = args
    module = ObservationOperatorUpdate(8, 3, initial_noise_variance=.4).double()
    known_noise = torch.full_like(y, .07)
    out_mean, out_variance, diag = module(*args, observation_noise=known_noise)
    covariance = torch.diag_embed(variance.flatten(1))
    system = H @ covariance @ H.transpose(-1, -2) + torch.eye(3) * .47
    gain = covariance @ H.transpose(-1, -2) @ torch.linalg.inv(system)
    expected_mean = mean.flatten(1) + (gain @ (y - (H @ mean.flatten(1)[..., None]).squeeze(-1) - offset)[..., None]).squeeze(-1)
    expected_covariance = covariance - gain @ H @ covariance
    torch.testing.assert_close(out_mean.flatten(1), expected_mean, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(out_variance.flatten(1), expected_covariance.diagonal(dim1=-2, dim2=-1), rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(diag["gain"], gain, rtol=1e-7, atol=1e-7)
    assert (out_variance >= 0).all()
    assert (out_variance <= variance).all()


def test_no_available_observations_give_exact_identity_and_zero_correction():
    args = list(_case())
    args[5][:] = False
    args[2][:] = float("nan")
    args[3][:] = float("inf")
    args[4][:] = float("nan")
    module = ObservationOperatorUpdate(8, 3).double()
    out_mean, out_variance, diag = module(*args)
    assert torch.equal(out_mean, args[0])
    assert torch.equal(out_variance, args[1])
    assert torch.count_nonzero(diag["mean_correction"]) == 0
    assert torch.count_nonzero(diag["gain"]) == 0
    assert torch.isfinite(out_mean).all()


def test_unavailable_modalities_do_not_affect_available_observation_solve():
    args = list(_case())
    args[5][:, 1:] = False
    module = ObservationOperatorUpdate(8, 3).double()
    full = module(*args)
    reduced_args = [*args[:2], args[2][:, :1], args[3][:, :1], args[4][:, :1], args[5][:, :1], args[6]]
    reduced = module(*reduced_args)
    torch.testing.assert_close(full[0], reduced[0])
    torch.testing.assert_close(full[1], reduced[1])
    args[2][:, 1:] = 1e25
    args[3][:, 1:] = float("nan")
    args[4][:, 1:] = -1e25
    perturbed = module(*args)
    assert torch.equal(full[0], perturbed[0])
    assert torch.equal(full[1], perturbed[1])


def test_nonfinite_observation_is_treated_as_unavailable():
    args = list(_case())
    args[2][0, 0] = float("nan")
    module = ObservationOperatorUpdate(8, 3).double()
    result = module(*args)
    args[5][0, 0] = False
    explicit = module(*args)
    assert torch.equal(result[0], explicit[0])
    assert not result[2]["valid"][0, 0]


def test_each_query_is_independent_of_other_queries_and_chunking():
    args = _case()
    module = ObservationOperatorUpdate(8, 3).double()
    full = module(*args)
    part = module(*(x[1:3] for x in args))
    torch.testing.assert_close(full[0][1:3], part[0])
    torch.testing.assert_close(full[1][1:3], part[1])


def test_low_rank_covariance_couples_temperature_salinity_and_depths():
    mean = torch.zeros(1, 2, 2, dtype=torch.float64)
    variance = torch.ones_like(mean)
    y = torch.tensor([[2.]], dtype=torch.float64)
    H = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
    h = torch.zeros(1, 8, dtype=torch.float64)
    module = ObservationOperatorUpdate(8, 2).double()
    factor = torch.tensor([[[1.], [-.5], [.8], [.2]]], dtype=torch.float64)
    out, out_var, diag = module(mean, variance, y, H, torch.zeros_like(y),
                                torch.ones_like(y, dtype=torch.bool), h,
                                column_covariance_factor=factor)
    covariance = torch.diag_embed(variance.flatten(1)) + factor @ factor.transpose(-1, -2)
    expected_gain = covariance[..., :, :1] / (covariance[:, :1, :1] + .25)
    torch.testing.assert_close(out.flatten(1), (expected_gain * 2).squeeze(-1), rtol=1e-7, atol=1e-7)
    assert out[0, 0, 1] < 0 < out[0, 1, 0]
    assert (out_var <= diag["prior_marginal_variance"]).all()


def test_no_data_variance_includes_supplied_low_rank_prior_covariance():
    args = list(_case())
    args[5][:] = False
    factor = torch.ones(4, 6, 2, dtype=torch.float64) * .2
    out_mean, out_variance, diag = ObservationOperatorUpdate(8, 3).double()(
        *args, column_covariance_factor=factor)
    assert torch.equal(out_mean, args[0])
    torch.testing.assert_close(out_variance, args[1] + .08)
    assert torch.count_nonzero(diag["variance_reduction"]) == 0


def test_finite_gradients_for_mean_covariance_operator_and_noise_parameters():
    args = list(_case())
    for i in (0, 1, 2, 3, 4, 6):
        args[i].requires_grad_(True)
    factor = torch.randn(4, 6, 2, dtype=torch.float64, requires_grad=True)
    module = ObservationOperatorUpdate(8, 3).double()
    out_mean, out_variance, _ = module(*args, column_covariance_factor=factor)
    (out_mean.square().sum() + out_variance.log().sum()).backward()
    for x in [args[i] for i in (0, 1, 2, 3, 4, 6)] + [factor] + list(module.parameters()):
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


def test_amp_keeps_small_system_and_variances_in_fp32():
    args = [x.float() if x.is_floating_point() else x for x in _case()]
    module = ObservationOperatorUpdate(8, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_mean, out_variance, diag = module(*args)
    assert out_variance.dtype == torch.float32
    assert diag["gain"].dtype == torch.float32
    assert torch.isfinite(out_mean).all() and torch.isfinite(out_variance).all()


def test_builder_reproduces_physical_surface_standardization_with_different_climatologies():
    generator = torch.Generator().manual_seed(707)
    q, levels = 4, 3
    physical = torch.randn(q, levels, 2, generator=generator, dtype=torch.float64)
    pmean = torch.randn(levels, 2, generator=generator, dtype=torch.float64)
    pscale = torch.rand(levels, 2, generator=generator, dtype=torch.float64) + .1
    pclim = torch.randn(q, levels, 2, generator=generator, dtype=torch.float64)
    smean = torch.tensor([1., -.5, .2], dtype=torch.float64)
    sscale = torch.tensor([.6, .2, .1], dtype=torch.float64)
    sclim = torch.randn(q, 2, generator=generator, dtype=torch.float64)
    H, offset, valid = build_surface_observation_operator(
        pmean, pscale, smean, sscale,
        profile_climatology=pclim, surface_climatology=sclim)
    standardized = (physical - pclim - pmean) / pscale
    predicted = (H @ standardized.flatten(1)[..., None]).squeeze(-1) + offset
    expected = (physical[:, 0] - sclim - smean[:2]) / sscale[:2]
    torch.testing.assert_close(predicted[:, :2], expected)
    assert valid[:, :2].all() and not valid[:, 2].any()
    assert torch.count_nonzero(H[:, 2]) == 0


def test_builder_linear_depth_weights_and_external_ssh_jacobian():
    pmean = torch.zeros(2, 2, dtype=torch.float64)
    pscale = torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64)
    weights = torch.tensor([.25, .75], dtype=torch.float64)
    jacobian = torch.tensor([[.1, -.2, .3, -.4]], dtype=torch.float64)
    H, offset, valid = build_surface_observation_operator(
        pmean, pscale, torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64),
        surface_depth_weights=weights, ssh_jacobian=jacobian,
        ssh_offset=torch.tensor([.7], dtype=torch.float64))
    torch.testing.assert_close(H[0, 0], torch.tensor([.5, 0., 3., 0.], dtype=torch.float64))
    torch.testing.assert_close(H[0, 1], torch.tensor([0., .75, 0., 3.75], dtype=torch.float64))
    assert torch.equal(H[:, 2], jacobian)
    assert offset[0, 2] == .7
    assert valid.all()


def test_builder_refuses_one_sided_climatology_and_incomplete_ssh_linearization():
    args = (torch.zeros(2, 2), torch.ones(2, 2), torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError, match="both profile and surface climatologies"):
        build_surface_observation_operator(*args, profile_climatology=torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="both a standardized Jacobian and offset"):
        build_surface_observation_operator(*args, ssh_jacobian=torch.zeros(1, 4))


def test_negative_provided_observation_variance_is_rejected():
    args = _case()
    with pytest.raises(ValueError, match="nonnegative"):
        ObservationOperatorUpdate(8, 3).double()(*args, observation_noise=-torch.ones_like(args[2]))


def test_empty_query_chunk_is_supported():
    args = _case(q=0)
    out_mean, out_variance, diag = ObservationOperatorUpdate(8, 3).double()(*args)
    assert out_mean.shape == out_variance.shape == (0, 3, 2)
    assert diag["gain"].shape == (0, 6, 3)


def test_unused_deep_missing_climatology_does_not_hide_surface_operator():
    pmean, pscale = torch.zeros(2, 2), torch.ones(2, 2)
    pclim = torch.tensor([[[12., 35.], [float("nan"), float("nan")]]])
    sclim = torch.tensor([[12., 35.]])
    H, offset, valid = build_surface_observation_operator(
        pmean, pscale, torch.zeros(3), torch.ones(3),
        profile_climatology=pclim, surface_climatology=sclim)
    assert valid[:, :2].all()
    assert torch.isfinite(H).all() and torch.isfinite(offset).all()
