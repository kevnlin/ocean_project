"""Query-anchored fusion: the pieces that decide how in-situ innovations reach a query."""
import math

import numpy as np
import pytest
import torch

from ocean_tokenizer import anchored as A
from ocean_tokenizer.point_baselines import oi_points

KM = 111.195


def t(x):
    return torch.tensor(x, dtype=torch.float64)


# ------------------------------------------------------------------ neighbours
def test_neighbours_are_found_across_the_date_line():
    nb = A.neighbours(t([10.0]), t([179.9]), t([10.0, 10.0, 10.0]), t([180.1, 170.0, 0.0]), k=2)
    assert nb.idx[0].tolist() == [0, 1]
    # 0.2 degrees of longitude to the EAST at 10N
    assert float(nb.dx[0, 0]) == pytest.approx(0.2 * KM * math.cos(math.radians(10.0)), abs=0.3)
    assert float(nb.dx[0, 1]) < 0


def test_neighbour_offsets_are_km_east_and_north():
    nb = A.neighbours(t([60.0]), t([20.0]), t([61.0, 60.0]), t([20.0, 21.0]), k=2)
    order = nb.idx[0].tolist()
    east, north = order.index(1), order.index(0)
    assert float(nb.dy[0, north]) == pytest.approx(KM, abs=0.5)
    assert float(nb.dx[0, north]) == pytest.approx(0.0, abs=0.5)
    assert float(nb.dx[0, east]) == pytest.approx(KM * 0.5, abs=0.5)
    assert float(nb.dist[0, east]) == pytest.approx(KM * 0.5, abs=0.5)


def test_neighbours_never_asks_for_more_than_there_are():
    nb = A.neighbours(t([0.0, 1.0]), t([0.0, 1.0]), t([0.0, 0.5, 1.0]), t([0.0, 0.5, 1.0]), k=32)
    assert nb.idx.shape == (2, 3)


# ------------------------------------------------------------------ kriging weights
def test_kriging_weights_reproduce_the_repo_oi():
    rng = np.random.default_rng(0)
    o_lat = 35.0 + rng.uniform(-1.5, 1.5, 30); o_lon = 150.0 + rng.uniform(-1.5, 1.5, 30)
    val = rng.normal(size=30)
    nb = A.neighbours(t([35.0]), t([150.0]), t(o_lat), t(o_lon), k=30)
    w = A.kriging_weights(nb.dx, nb.dy, torch.ones_like(nb.dx, dtype=torch.bool),
                          ell_x=t(150.0), ell_y=t(150.0), gamma=t(0.1))
    mine = float((w * t(val)[nb.idx]).sum())
    ref = float(oi_points(o_lat, o_lon, val, np.array([35.0]), np.array([150.0]),
                          L_km=150.0, gamma=0.1, k=30)[0])
    assert mine == pytest.approx(ref, abs=5e-3)


def test_kriging_returns_a_colocated_value_as_noise_vanishes():
    dx, dy = t([[0.0, 120.0, -90.0]]), t([[0.0, 40.0, 150.0]])
    w = A.kriging_weights(dx, dy, torch.ones(1, 3, dtype=torch.bool),
                          ell_x=t(200.0), ell_y=t(200.0), gamma=t(1e-6))
    assert w[0].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-3)


def test_kriging_gives_a_masked_neighbour_no_weight():
    dx, dy = t([[30.0, 120.0, -90.0]]), t([[10.0, 40.0, 150.0]])
    kw = dict(ell_x=t(200.0), ell_y=t(200.0), gamma=t(0.1))
    w3 = A.kriging_weights(dx, dy, torch.tensor([[True, True, False]]), **kw)
    w2 = A.kriging_weights(dx[:, :2], dy[:, :2], torch.ones(1, 2, dtype=torch.bool), **kw)
    assert float(w3[0, 2]) == 0.0
    assert w3[0, :2].tolist() == pytest.approx(w2[0].tolist(), abs=1e-9)


def test_kriging_counts_a_reingested_observation_once():
    # five copies of one observation act as that observation with a fifth of
    # the error variance, so the analysis moves by a few percent; a plain
    # kernel average of the same seven values moves from 0.63 to 1.47
    dx, dy, v = [60.0, -150.0, 40.0], [20.0, 30.0, -170.0], [2.0, -1.0, 0.5]

    def analysis(dx, dy, v):
        w = A.kriging_weights(t([dx]), t([dy]), torch.ones(1, len(v), dtype=torch.bool),
                              ell_x=t(200.0), ell_y=t(200.0), gamma=t(0.01))
        return float((w * t([v])).sum())

    once = analysis(dx, dy, v)
    five = analysis([dx[0]] * 5 + dx[1:], [dy[0]] * 5 + dy[1:], [v[0]] * 5 + v[1:])
    assert five == pytest.approx(once, abs=0.05 * abs(once))


def test_kriging_uses_the_time_and_state_separations_when_given():
    dx, dy = t([[50.0, 50.0]]), t([[0.0, 0.0]])
    ok = torch.ones(1, 2, dtype=torch.bool)
    kw = dict(ell_x=t(200.0), ell_y=t(200.0), gamma=t(0.1))
    same = A.kriging_weights(dx, dy, ok, **kw)
    late = A.kriging_weights(dx, dy, ok, dt=t([[0.0, 30.0]]), ell_t=t(10.0), **kw)
    other = A.kriging_weights(dx, dy, ok, ds=t([[0.0, 3.0]]), ell_s=t(1.0), **kw)
    assert float(same[0, 0]) == pytest.approx(float(same[0, 1]))
    assert float(late[0, 1]) < 0.05 * float(late[0, 0])
    assert float(other[0, 1]) < 0.05 * float(other[0, 0])


# ------------------------------------------------------------------ softmax weights
def test_softmax_weights_and_the_background_sum_to_one():
    logits = t([[0.3, -1.0, 2.0, 0.0]])
    ok = torch.tensor([[True, True, False, True]])
    w = A.softmax_weights(logits, ok, bg_logit=t(0.5))
    assert float(w[0, 2]) == 0.0
    bg = math.exp(0.5) / (math.exp(0.3) + math.exp(-1.0) + math.exp(0.0) + math.exp(0.5))
    assert float(w.sum()) == pytest.approx(1.0 - bg, abs=1e-9)


def test_a_mass_prior_makes_a_split_observation_count_once():
    v = [2.0, -1.0]

    def analysis(logits, values, mass=None):
        w = A.softmax_weights(t([logits]), torch.ones(1, len(values), dtype=torch.bool),
                              bg_logit=t(0.0),
                              log_mass=None if mass is None else torch.log(t([mass])))
        return float((w * t([values])).sum())

    whole = analysis([0.4, -0.2], v, mass=[1.0, 1.0])
    split = analysis([0.4] * 4 + [-0.2], [v[0]] * 4 + [v[1]], mass=[0.25] * 4 + [1.0])
    counted = analysis([0.4] * 4 + [-0.2], [v[0]] * 4 + [v[1]])
    assert split == pytest.approx(whole, abs=1e-9)
    assert abs(counted - whole) > 0.1


# ------------------------------------------------------------------ evidence of a profile
def test_leverage_shares_one_degree_of_freedom_among_duplicates():
    lat = t([0.0, 30.0, -40.0] + [10.0] * 4)
    lon = t([0.0, 90.0, 200.0] + [300.0] * 4)
    tau = A.profile_leverage(lat, lon, ell_km=200.0, noise=0.01, k=6)
    assert tau[:3].tolist() == pytest.approx([1 / 1.01] * 3, abs=1e-6)
    assert tau[3:].tolist() == pytest.approx([1 / 4.01] * 4, abs=1e-6)


def test_leverage_falls_as_profiles_crowd_together():
    lat = t([0.0, 0.0, 0.0, 50.0]); lon = t([0.0, 0.3, 0.6, 100.0])
    tau = A.profile_leverage(lat, lon, ell_km=200.0, noise=0.01, k=4)
    assert float(tau[1]) < float(tau[0]) < float(tau[3])


# ------------------------------------------------------------------ the analysis module
def _scene(n=40, q=12, f=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    o_lat = 20.0 + 6.0 * torch.rand(n, generator=g, dtype=torch.float64)
    o_lon = 140.0 + 6.0 * torch.rand(n, generator=g, dtype=torch.float64)
    q_lat = 21.0 + 4.0 * torch.rand(q, generator=g, dtype=torch.float64)
    q_lon = 141.0 + 4.0 * torch.rand(q, generator=g, dtype=torch.float64)
    innov = torch.randn(n, f, generator=g, dtype=torch.float64)
    return q_lat, q_lon, o_lat, o_lon, innov


@pytest.mark.parametrize("mode", ["softmax", "dfs", "kriging"])
def test_analysis_adds_nothing_where_no_neighbour_has_a_value(mode):
    q_lat, q_lon, o_lat, o_lon, innov = _scene()
    innov[:, 2] = float("nan")                       # one field missing everywhere
    m = A.InnovationAnalysis(n_fields=6, mode=mode, k=8).double()
    out = m(q_lat, q_lon, o_lat, o_lon, innov,
            log_mass=torch.zeros(40, dtype=torch.float64)).detach()
    assert torch.isfinite(out).all()
    assert float(out[:, 2].abs().max()) == 0.0
    assert float(out[:, 0].abs().max()) > 0.0


@pytest.mark.parametrize("mode", ["softmax", "dfs", "kriging"])
def test_analysis_of_a_query_does_not_depend_on_the_other_queries(mode):
    q_lat, q_lon, o_lat, o_lon, innov = _scene()
    innov[::3, 1] = float("nan")
    m = A.InnovationAnalysis(n_fields=6, mode=mode, k=8).double()
    kw = dict(log_mass=torch.zeros(40, dtype=torch.float64))
    full = m(q_lat, q_lon, o_lat, o_lon, innov, **kw)
    part = m(q_lat[3:5], q_lon[3:5], o_lat, o_lon, innov, **kw)
    assert torch.allclose(full[3:5], part, atol=1e-10)


def test_kriging_analysis_returns_what_a_colocated_profile_measured():
    # with negligible error and a kernel no wider than the spacing (a wide
    # Gaussian over crowded points is singular, and its nugget then matters)
    q_lat, q_lon, o_lat, o_lon, innov = _scene()
    m = A.InnovationAnalysis(n_fields=6, mode="kriging", k=8, init_km=60.0,
                             init_gamma=1e-6).double()
    out = m(o_lat[:5], o_lon[:5], o_lat, o_lon, innov)
    assert torch.allclose(out, innov[:5], atol=1e-3)


@pytest.mark.parametrize("mode", ["softmax", "dfs", "kriging"])
def test_every_parameter_of_the_analysis_receives_a_gradient(mode):
    q_lat, q_lon, o_lat, o_lon, innov = _scene()
    g = torch.Generator().manual_seed(1)
    days = lambda n: 30.0 * torch.rand(n, generator=g, dtype=torch.float64)
    m = A.InnovationAnalysis(n_fields=6, mode=mode, k=8, use_time=True, use_state=True).double()
    out = m(q_lat, q_lon, o_lat, o_lon, innov, q_t=days(12), o_t=days(40),
            q_s=torch.randn(12, generator=g, dtype=torch.float64),
            o_s=torch.randn(40, generator=g, dtype=torch.float64),
            log_mass=-torch.rand(40, generator=g, dtype=torch.float64))
    out.pow(2).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert float(p.grad.abs().sum()) > 0.0, name


def test_first_guess_maps_features_to_every_field():
    net = A.FirstGuess(n_in=7, n_fields=40, width=32, depth=2)
    assert net(torch.zeros(5, 7)).shape == (5, 40)
