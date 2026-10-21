"""The CLOINet-style read-out: the local kriging solve with the covariance of soft cluster memberships."""
import torch

from ocean_tokenizer.kriging_attention import KrigingAttention, cluster_correlation
from test_kriging_attention import _init, _inputs


def test_cluster_correlation_is_cloinets_membership_similarity():
    m = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.5, .5, 0.]], dtype=torch.float64)
    corr = cluster_correlation(m)
    # B_ab = 1 - sum_c (m_ac - m_bc)^2: one cluster +1, two clusters -1, half-shared 0.5
    torch.testing.assert_close(torch.diagonal(corr), torch.ones(4, dtype=torch.float64))
    assert corr[0, 1] == 1. and corr[0, 2] == -1. and corr[0, 3] == .5 and corr[2, 3] == .5
    torch.testing.assert_close(corr, corr.T)


def test_cluster_mode_has_no_distance_term():
    ell, gamma = _init()
    torch.manual_seed(7)
    cluster = KrigingAttention(16, "ka_cluster", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    near = cluster(h, local, offsets, innovation, valid, level)
    far = cluster(h, local, 10. * offsets, innovation, valid, level)
    torch.testing.assert_close(near["estimate"], far["estimate"])
    torch.testing.assert_close(near["weights"], far["weights"])


def test_cluster_mode_ignores_invalid_neighbours_and_starts_at_the_oi_noise():
    ell, gamma = _init()
    torch.manual_seed(8)
    cluster = KrigingAttention(16, "ka_cluster", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    valid[2, 4, 1] = False
    a = cluster(h, local, offsets, innovation, valid, level)
    innovation[2, 4, 1] = 1e6
    b = cluster(h, local, offsets, innovation, valid, level)
    torch.testing.assert_close(a["estimate"], b["estimate"])
    assert a["weights"][2, 1, 4] == 0
    # the amplitude and error-ratio outputs start at zero: the noise is the OI's gamma
    expected = torch.tensor(gamma, dtype=torch.float64)[:, level].T[..., None].expand_as(a["nugget"])
    torch.testing.assert_close(a["nugget"], expected)


def test_cluster_mode_heads_receive_finite_gradients():
    ell, gamma = _init()
    torch.manual_seed(9)
    cluster = KrigingAttention(16, "ka_cluster", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    out = cluster(h, local, offsets, innovation, valid, level)
    loss = (out["estimate"] - .3).square().mean() + out["variance"].mean()
    loss.backward()
    for head in (cluster.query_head, cluster.local_head):
        grad = head[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
