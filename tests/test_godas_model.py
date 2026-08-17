"""The GODAS row model (mentor doc §2.2-§2.6 assembled).

Rows differ ONLY in how observation mass is measured and whether the frozen-OI
residual wraps them, so the tests here pin that the mechanism is genuinely
matched across `dfs` / `uniform` / `count` and that the OI wrapper degrades
cleanly.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.godas_model import GodasRowModel, ROWS, build_row
from ocean_tokenizer.godas_obs import build_sample, ObsConfig


def _fields(nt=6, nz=16, ny=38, nx=26, seed=0):
    rng = np.random.default_rng(seed)
    f = dict(months=np.array([np.datetime64("2000-01") + i for i in range(nt)]),
             TEMP=rng.normal(0, 1, (nt, nz, ny, nx)),
             SALT=rng.normal(0, 1, (nt, nz, ny, nx)),
             SSH=rng.normal(0, 1, (nt, ny, nx)))
    for k in ("TEMP", "SALT"):
        f[k][:, :, :4, :4] = np.nan
    f["SSH"][:, :4, :4] = np.nan
    return f


def _sample(seed=0, **kw):
    return build_sample(_fields(), t_src=3, cfg=ObsConfig(**kw),
                        rng=np.random.default_rng(seed))


NEURAL_ROWS = [r for r in ROWS if r != "objective_interpolation"]


# --------------------------------------------------------------------------
# Every registered row runs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row", NEURAL_ROWS)
def test_every_row_produces_finite_predictions(row):
    torch.manual_seed(0)
    m = build_row(row).eval()
    s = _sample()
    with torch.no_grad():
        out = m(s)
    assert out.shape == (s["query"].shape[0], 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("row", NEURAL_ROWS)
def test_every_row_survives_every_modality_being_dropped(row):
    torch.manual_seed(0)
    m = build_row(row).eval()
    for seed in range(6):
        s = _sample(seed=seed, modality_dropout=0.7, train=True)
        with torch.no_grad():
            out = m(s)
        assert torch.isfinite(out).all(), (row, seed)


# --------------------------------------------------------------------------
# The mass mechanism is matched across rows
# --------------------------------------------------------------------------
def test_uniform_and_dfs_share_the_same_transport_machinery():
    """`uniform` is the matched control: same modules, unit masses."""
    a = build_row("dfs_expertlocal_cbottle")
    b = build_row("uniform_expertlocal_cbottle")
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sa == sb, "matched control must have an identical parameter set"


def test_uniform_uses_unit_masses_and_dfs_does_not():
    torch.manual_seed(0)
    s = _sample()
    dfs_m = build_row("dfs_expertlocal_cbottle").eval()
    uni_m = build_row("uniform_expertlocal_cbottle").eval()
    with torch.no_grad():
        w_dfs = dfs_m.observation_mass(s)
        w_uni = uni_m.observation_mass(s)
    live = s["mask"]
    assert torch.allclose(w_uni[live], torch.ones_like(w_uni[live]))
    assert not torch.allclose(w_dfs[live], torch.ones_like(w_dfs[live]))
    assert float(w_dfs[live].sum()) < float(w_uni[live].sum()), \
        "measured evidence must be scarcer than one-vote-each"


def test_dfs_mass_respects_the_feature_ceiling():
    torch.manual_seed(0)
    m = build_row("dfs_expertlocal_cbottle").eval()
    with torch.no_grad():
        w = m.observation_mass(_sample())
    assert float(w.sum()) <= 32.0 + 1e-6


def test_masked_tokens_carry_no_mass_in_any_row():
    for row in ("dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle",
                "count_expertlocal_cbottle"):
        torch.manual_seed(0)
        m = build_row(row).eval()
        s = _sample()
        with torch.no_grad():
            w = m.observation_mass(s)
        assert float(w[~s["mask"]].abs().sum()) == 0.0, row


# --------------------------------------------------------------------------
# The OI residual wrapper
# --------------------------------------------------------------------------
@pytest.mark.parametrize("row", [r for r in ROWS if r.endswith("oi_expert_cbottle")])
def test_oi_rows_carry_exactly_eight_gates(row):
    m = build_row(row)
    assert m.oi_residual is not None
    assert m.oi_residual.gate_logit.numel() == 8


def test_a_closed_gate_reduces_an_oi_row_to_plain_oi():
    torch.manual_seed(0)
    m = build_row("dfs_oi_expert_cbottle").eval()
    s = _sample()
    with torch.no_grad():
        m.oi_residual.gate_logit.fill_(-30.0)
        blended = m(s)
        oi_only = m.oi_background(s)
    assert torch.allclose(blended, oi_only, atol=1e-5)


def test_non_oi_rows_have_no_gates():
    m = build_row("dfs_expertlocal_cbottle")
    assert m.oi_residual is None


# --------------------------------------------------------------------------
# Query independence survives the whole stack (doc §2.3)
# --------------------------------------------------------------------------
def test_permuting_queries_permutes_predictions():
    torch.manual_seed(0)
    m = build_row("dfs_expertlocal_cbottle").eval()
    s = _sample()
    perm = torch.randperm(s["query"].shape[0])
    with torch.no_grad():
        a = m(s)
        s2 = dict(s); s2["query"] = s["query"][perm]
        b = m(s2)
    assert torch.allclose(b, a[perm], atol=1e-5)
