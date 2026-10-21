"""The DKNN-style read-out: the local kriging solve with a covariance of attention scores."""
import torch

from ocean_tokenizer.kriging_attention import KrigingAttention, score_covariance
from test_kriging_attention import _init, _inputs


def test_score_covariance_is_dknns_scaled_dot_product_without_the_neighbours_own_scores():
    g = torch.Generator().manual_seed(1)
    queries, keys = torch.randn(3, 5, 4, generator=g, dtype=torch.float64), torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    cov = score_covariance(queries, keys)
    full = queries @ keys.transpose(-1, -2) / 2.
    off = ~torch.eye(5, dtype=torch.bool)
    torch.testing.assert_close(cov[:, off], full[:, off])
    # the query keeps its own score (the prior variance); every neighbour's own score is removed
    torch.testing.assert_close(cov[:, 0, 0], full[:, 0, 0])
    assert (torch.diagonal(cov, dim1=-2, dim2=-1)[:, 1:] == 0).all()


def test_scores_mode_has_no_distance_term_and_ignores_invalid_neighbours():
    ell, gamma = _init()
    torch.manual_seed(11)
    scores = KrigingAttention(16, "ka_scores", ell, gamma).double()
    h, local, offsets, innovation, valid, level = _inputs()
    a = scores(h, local, offsets, innovation, valid, level)
    torch.testing.assert_close(a["estimate"], scores(h, local, 5. * offsets, innovation, valid, level)["estimate"])
    valid[1, 2, 0] = False
    b = scores(h, local, offsets, innovation, valid, level)
    innovation[1, 2, 0] = -1e6
    torch.testing.assert_close(b["estimate"], scores(h, local, offsets, innovation, valid, level)["estimate"])
    assert b["weights"][1, 0, 2] == 0


def test_scores_mode_projections_receive_finite_gradients():
    ell, gamma = _init()
    torch.manual_seed(12)
    scores = KrigingAttention(16, "ka_scores", ell, gamma).double()
    out = scores(*_inputs())
    ((out["estimate"] - .3).square().mean() + out["variance"].mean()).backward()
    for layer in (scores.query_head, scores.key_head):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
