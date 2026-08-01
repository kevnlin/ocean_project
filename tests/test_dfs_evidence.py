"""DFS-Attention — the invariants the evidence estimator must satisfy.

These are the *decisive* tests of the method claim (plan Sections 4, 7, 11):
evidence must respond to independent information, not to token count, and it
must be measured at the target reconstruction scale.  Everything here is
architectural — random initialisation, fixed seed — so a passing suite says the
mechanism is right, not that a particular training run went well.

    duplication  exact re-ingestion of one measurement adds NOTHING
    complement   the same column at different depths adds nearly everything
    redundancy   clustered observations are discounted, isolated ones are not
    stratified   dense levels survive across a thermocline, not in a smooth layer
    scale        a finer target retains more dense information (monotone)
    transport    the resampler conserves total evidence exactly
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer import dfs
from ocean_tokenizer.token_api import ProfileEncoder, GridPatchEncoder, MODALITIES
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import split_token, duplicate_tokens, output_change

DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)
TOL_EXACT = 1e-5


class FakeGrid:
    depth = DEPTHS.astype("float64")


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(0)
    return ProfileEncoder(DEPTHS, c_vars=2, d_model=32)


def profiles(lat, lon, seed=0, values=None, **kw):
    lat = np.atleast_1d(np.asarray(lat, "float32"))
    lon = np.atleast_1d(np.asarray(lon, "float32"))
    P = lat.size
    if values is None:
        values = np.random.default_rng(seed).normal(size=(P, 2, D))
    return dict(prof=torch.tensor(np.asarray(values, "float32")[None]),
                lat=torch.tensor(lat[None]), lon=torch.tensor(lon[None]),
                month=torch.tensor([3]), **kw)


def total(tb, target=dfs.PROTOCOL_SCALE):
    return float(dfs.dfs_scores(tb, target).total)


# --------------------------------------------------------------------------
# Duplication: the same measurement, re-ingested, is not new evidence
# --------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_exact_duplication_conserves_total_dfs(enc, factor):
    tb = enc(**profiles(np.linspace(-40, 40, 12), np.linspace(0, 330, 12)))
    base = total(tb)
    idx = tb.mask[0].nonzero().flatten()[:10]
    dup = duplicate_tokens(tb, idx, factor, divide_mass=False)
    assert dup.mask.sum() > tb.mask.sum()             # tokens really did grow
    assert abs(total(dup) - base) <= TOL_EXACT * base


@pytest.mark.parametrize("n", [2, 4, 8])
def test_token_partition_conserves_total_dfs(enc, n):
    tb = enc(**profiles(np.linspace(-40, 40, 12), np.linspace(0, 330, 12)))
    base = total(tb)
    idx = int(tb.mask[0].nonzero()[0])
    assert abs(total(split_token(tb, idx, n)) - base) <= TOL_EXACT * base


def test_realtime_and_delayed_mode_copies_merge(enc):
    """One float cycle delivered twice on two streams = one measurement."""
    vals = np.random.default_rng(1).normal(size=(1, 2, D))
    one = enc(**profiles([10.0], [20.0], values=vals))
    both = enc(**profiles([10.0, 10.0], [20.0, 20.0],
                          values=np.concatenate([vals, vals]),
                          parent=torch.tensor([[0, 0]]),          # same cycle
                          source=torch.tensor([[300, 301]])))     # RT vs DM
    assert both.mask.sum() == 2 * one.mask.sum()
    assert abs(total(both) - total(one)) <= TOL_EXACT * total(one)


def test_two_nearby_profiles_are_redundant_only_at_a_coarse_target(enc):
    """The plan's "two profiles 5 km apart" case: redundant for a coarse
    reconstruction, genuinely two observations for a fine one."""
    solo = enc(**profiles([10.0], [20.0], seed=2))
    pair = enc(**profiles([10.0, 10.045], [20.0, 20.0], seed=2))   # ~5 km
    coarse = total(pair, dfs.COARSE_SCALE) / total(solo, dfs.COARSE_SCALE)
    fine = total(pair, dfs.FINE_SCALE) / total(solo, dfs.FINE_SCALE)
    assert coarse < 1.1                     # coarse target: near-duplicate
    assert fine > 1.4                       # fine target: two observations
    assert fine > coarse


# --------------------------------------------------------------------------
# Vertical complementarity — the plan's central correction
# --------------------------------------------------------------------------
def test_same_site_different_depths_are_complementary(enc):
    """One profile's depth bands describe different water: ~1 DOF each."""
    tb = enc(**profiles([10.0], [20.0], seed=4))
    n_bands = int(tb.mask.sum())
    assert n_bands >= 4
    assert total(tb) > 0.9 * n_bands


def test_clustered_profiles_are_discounted_relative_to_spread(enc):
    rng = np.random.default_rng(5)
    P = 16
    spread = enc(**profiles(rng.uniform(-60, 60, P), rng.uniform(0, 360, P)))
    cluster = enc(**profiles(rng.uniform(9.5, 10.5, P), rng.uniform(20, 21, P)))
    assert total(cluster) < 0.6 * total(spread)
    assert total(spread) > 0.9 * int(spread.mask.sum())


def _column(kind, z, amp=1.0):
    """Anomaly column in z-units, matched in RMS: only the vertical structure
    differs (see experiments/23_dfs_evidence_probes.py)."""
    t = (np.cos(np.pi * z / 2000.0) if kind == "smooth"
         else np.tanh((z - 120.0) / 12.0) * np.exp(-((z - 120.0) / 90.0) ** 2))
    t = t / max(np.sqrt((t ** 2).mean()), 1e-9) * amp
    return np.stack([t, 0.4 * t])[None]


def test_dense_levels_survive_a_thermocline_but_not_a_smooth_column():
    """Same vertical sampling density, same anomaly amplitude, different
    vertical structure.  A smooth column's adjacent tokens are mutually
    redundant; across a sharp gradient they are not, because the vertical
    length scale is stratification-dependent.  Asked at a target fine enough
    for the difference to matter (Δz = 10 m).
    """
    torch.manual_seed(0)
    z = np.arange(2.0, 300.0, 2.0)
    bands = [(zz, zz + 10.0) for zz in np.arange(0.0, 300.0, 10.0)]
    e = ProfileEncoder(DEPTHS, c_vars=2, d_model=32, depth_bands=bands)
    fine = dfs.TargetScale(dz_m=10.0)
    t_smooth = total(e(**profiles([10.0], [20.0], values=_column("smooth", z),
                                  depths=torch.tensor(z.astype("float32")))),
                     fine)
    t_sharp = total(e(**profiles([10.0], [20.0], values=_column("sharp", z),
                                 depths=torch.tensor(z.astype("float32")))),
                    fine)
    assert t_sharp > 1.3 * t_smooth, (t_sharp, t_smooth)


# --------------------------------------------------------------------------
# Target-resolution awareness
# --------------------------------------------------------------------------
def test_finer_target_retains_more_evidence(enc):
    rng = np.random.default_rng(6)
    P = 16
    tb = enc(**profiles(rng.uniform(9.0, 11.0, P), rng.uniform(20, 22, P)))
    coarse = total(tb, dfs.COARSE_SCALE)
    proto = total(tb, dfs.PROTOCOL_SCALE)
    fine = total(tb, dfs.FINE_SCALE)
    assert coarse < proto < fine
    assert coarse < 0.9 * fine        # the sweep must be a real effect


def test_scale_sweep_is_monotone_in_dz(enc):
    rng = np.random.default_rng(7)
    bands = [(z, z + 25.0) for z in np.arange(0.0, 400.0, 25.0)]
    e = ProfileEncoder(DEPTHS, c_vars=2, d_model=32, depth_bands=bands)
    tb = e(**profiles(rng.uniform(9, 11, 4), rng.uniform(20, 22, 4)))
    seq = [total(tb, dfs.TargetScale(dz_m=dz)) for dz in (200., 100., 50., 10.)]
    assert all(a < b for a, b in zip(seq, seq[1:])), seq


# --------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------
def test_low_reliability_is_down_weighted_not_discarded(enc):
    """A QC-flagged observation keeps its place in the redundancy structure
    but contributes less evidence."""
    tb = enc(**profiles(np.linspace(-40, 40, 9), np.linspace(0, 320, 9)))
    good = total(tb)
    tb.reliability = torch.full_like(tb.sigma, 0.2)
    poor = total(tb)
    assert 0.0 < poor < good


def test_permutation_invariance(enc):
    tb = enc(**profiles(np.linspace(-40, 40, 9), np.linspace(0, 320, 9)))
    tau = dfs.dfs_scores(tb).tau
    perm = torch.randperm(tb.emb.shape[1])
    from ocean_tokenizer.invariance import index_tokens
    tau_p = dfs.dfs_scores(index_tokens(tb, perm)).tau
    assert torch.allclose(tau[:, perm], tau_p, atol=1e-6)


def test_masked_padding_contributes_nothing(enc):
    tb = enc(**profiles(np.linspace(-40, 40, 9), np.linspace(0, 320, 9)))
    base = total(tb)
    pad = enc(**profiles(np.linspace(-40, 40, 12), np.linspace(0, 320, 12),
                         values=np.concatenate([
                             np.random.default_rng(8).normal(size=(9, 2, D)),
                             np.full((3, 2, D), np.nan)])))
    tb2 = enc(**profiles(np.linspace(-40, 40, 9), np.linspace(0, 320, 9),
                         values=np.random.default_rng(8).normal(size=(9, 2, D))))
    assert int(pad.mask.sum()) == int(tb2.mask.sum())
    assert abs(total(pad) - total(tb2)) <= TOL_EXACT * max(total(tb2), 1.0)
    assert base > 0


def test_background_modality_is_not_evidence():
    torch.manual_seed(0)
    m = build_fusion_model("dfs", FakeGrid(), d_model=32, n_latent=16,
                           n_heads=4, n_self_blocks=1, patch=(4, 6), seed=3)
    tb = m.encode(_obs_all(), batch=1)
    res = m.evidence(tb)
    assert float(res.by_modality["woa_grid"]) == 0.0
    assert float(res.by_modality["profile"]) > 0.0
    assert float(res.by_modality["surf_grid"]) > 0.0


def test_resampler_transports_evidence_exactly():
    torch.manual_seed(0)
    m = build_fusion_model("dfs", FakeGrid(), d_model=32, n_latent=16,
                           n_heads=4, n_self_blocks=1, patch=(4, 6), seed=3)
    tb = m.encode(_obs_all(), batch=1)
    res = m.evidence(tb)
    obs_mask, _ = m._split(tb)
    _, out_mask, nu = m.resampler(tb.emb, obs_mask, res.tau, tb.modality)
    nu = nu.detach()
    incoming = float((res.tau * obs_mask).sum())
    assert abs(float(nu.sum()) - incoming) <= 1e-4 * incoming
    assert bool((nu[out_mask] > 0).all())


# --------------------------------------------------------------------------
# End-to-end model behaviour
# --------------------------------------------------------------------------
def _obs_all(P=7, seed=0, month=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(1, 2, 8, 12)).astype("float32")
    f[:, :, :2, :3] = np.nan
    vol = rng.normal(size=(1, 2, D, 8, 12)).astype("float32")
    return {
        "profiles": dict(
            prof=torch.tensor(rng.normal(size=(1, P, 2, D)).astype("float32")),
            lat=torch.tensor(rng.uniform(-80, 80, (1, P)).astype("float32")),
            lon=torch.tensor(rng.uniform(0, 360, (1, P)).astype("float32")),
            month=torch.tensor([month])),
        "surf": dict(field=torch.tensor(f), lat=torch.linspace(-80, 80, 8),
                     lon=torch.linspace(0, 345, 12), month=torch.tensor([month])),
        "woa": dict(field=torch.tensor(vol), lat=torch.linspace(-80, 80, 8),
                    lon=torch.linspace(0, 345, 12), month=torch.tensor([month]),
                    depth=torch.tensor(DEPTHS)),
    }


def _queries(n=9, seed=2):
    rng = np.random.default_rng(seed)
    return torch.tensor(np.stack([rng.uniform(-80, 80, n), rng.uniform(0, 360, n),
                                  rng.uniform(0, 985, n), np.full(n, 3.0)],
                                 -1).astype("float32"))[None]


def test_prediction_is_stable_under_exact_duplication():
    """The success criterion's first leg, measured on the model output."""
    torch.manual_seed(0)
    q = _queries()
    changes = {}
    for v in ("perceiver", "dfs"):
        m = build_fusion_model(v, FakeGrid(), d_model=32, n_latent=16,
                               n_heads=4, n_self_blocks=2, patch=(4, 6), seed=3)
        m.eval()
        tb = m.encode(_obs_all(), batch=1)
        idx = ((tb.modality[0] == MODALITIES["profile"]) & tb.mask[0]
               & (tb.parent_id[0] < 3)).nonzero().flatten()
        dup = duplicate_tokens(tb, idx, 4, divide_mass=False)
        changes[v] = output_change(m, tb, dup, q)
    assert changes["dfs"] < 0.5 * changes["perceiver"], changes


def test_gradients_reach_every_branch():
    torch.manual_seed(0)
    m = build_fusion_model("dfs", FakeGrid(), d_model=32, n_latent=16,
                           n_heads=4, n_self_blocks=2, patch=(4, 6), seed=3)
    m(_obs_all(), _queries()).pow(2).mean().backward()
    dead = [n for n, p in m.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    # scale_proj.weight is inactive at the reference scale by construction
    assert dead == ["scale_proj.weight"], dead


def test_learned_scale_residual_is_identity_at_init_and_trainable():
    torch.manual_seed(0)
    plain = dfs.SupportScales(learn_residual=False)
    learned = dfs.SupportScales(learn_residual=True)
    depth = torch.tensor([5.0, 120.0, 500.0, 900.0])
    strat = torch.zeros(4)
    a = plain(depth, strat, dfs.PROTOCOL_SCALE)
    b = learned(depth, strat, dfs.PROTOCOL_SCALE)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)
    b[1].sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in learned.parameters())
