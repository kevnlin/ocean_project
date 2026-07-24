"""Task 6, Test C — grid retokenization (coarse vs refined tokens).

Two regimes per the brief:
  * EXACT synthetic — a grid token is replaced by equivalent children
    (identical content, mass split): MBCA must be exactly invariant.
  * PHYSICAL — the same field is conservatively refined (2x cells at half
    spacing, children carry the parent value) and re-encoded: content
    embeddings genuinely change, so no method is exactly invariant; what MUST
    hold is the measure-side contract — the modality's total support mass is
    conserved — and MBCA's shift stays below standard attention's.
Reported: prediction difference, latent difference (Task-6 required outputs;
the RMSE difference on trained models lives in the Stage-A toy results, see
reports/invariance_test_summary.md).
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import MODALITIES
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import (split_token, output_change,
                                        latent_change, modality_mass_totals)

D_MODEL, N_LATENT = 32, 16
DEPTHS = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                   186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
D = len(DEPTHS)
TOL_EXACT = 1e-5


class FakeGrid:
    depth = DEPTHS.astype("float64")


def build(variant, seed=3):
    m = build_fusion_model(variant, FakeGrid(), d_model=D_MODEL,
                           n_latent=N_LATENT, n_heads=4, n_self_blocks=2,
                           patch=(4, 6), seed=seed)
    m.eval()
    return m


def base_field(H=8, W=12, seed=1):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(1, 2, H, W)).astype("float32")
    f[:, :, :2, :3] = np.nan                       # a land corner
    return f


def refine(f, k):
    """Conservative refinement: kx cells at 1/k spacing, parent values."""
    return np.repeat(np.repeat(f, k, axis=2), k, axis=3)


def refine_coords(c: torch.Tensor, k: int) -> torch.Tensor:
    """Cell-centre coordinates of the k-fold subdivided cells."""
    step = float(c[1] - c[0])
    off = (torch.arange(k, dtype=c.dtype) + 0.5) / k - 0.5
    return (c[:, None] + off[None, :] * step).reshape(-1)


LAT0 = torch.linspace(-80, 80, 8)
LON0 = torch.linspace(0, 345, 12)


def obs_with_grid(f, k=1, P=5, seed=0, month=3):
    rng = np.random.default_rng(seed)
    lat = LAT0 if k == 1 else refine_coords(LAT0, k)
    lon = LON0 if k == 1 else refine_coords(LON0, k)
    return {
        "profiles": dict(
            prof=torch.tensor(rng.normal(size=(1, P, 2, D)).astype("float32")),
            lat=torch.tensor(rng.uniform(-80, 80, (1, P)).astype("float32")),
            lon=torch.tensor(rng.uniform(0, 360, (1, P)).astype("float32")),
            month=torch.tensor([month])),
        "surf": dict(field=torch.tensor(f), lat=lat, lon=lon,
                     month=torch.tensor([month])),
    }


def queries(Q=9, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)], -1).astype("float32")
    return torch.tensor(q)[None]


# ============================ exact synthetic test ============================
@pytest.mark.parametrize("n", [2, 4, 8])
def test_C_exact_children_mbca_invariant(n):
    """Grid token -> n equivalent children (identical content, mass/n)."""
    model = build("mbca")
    tb = model.encode(obs_with_grid(base_field()), batch=1)
    g = MODALITIES["surf_grid"]
    idx = int(((tb.modality[0] == g) & tb.mask[0]).nonzero()[0])
    rel = output_change(model, tb, split_token(tb, idx, n), queries())
    assert rel < TOL_EXACT, f"exact refinement n={n}: rel {rel:.2e}"


def test_C_exact_children_standard_changes():
    model = build("perceiver")
    tb = model.encode(obs_with_grid(base_field()), batch=1)
    g = MODALITIES["surf_grid"]
    idx = int(((tb.modality[0] == g) & tb.mask[0]).nonzero()[0])
    rel = output_change(model, tb, split_token(tb, idx, 8), queries())
    assert rel > 1e-4


# ============================ physical refinement ============================
def encode_pair(model, k):
    f = base_field()
    tb_c = model.encode(obs_with_grid(f), batch=1)
    tb_r = model.encode(obs_with_grid(refine(f, k), k=k), batch=1)
    return tb_c, tb_r


@pytest.mark.parametrize("k", [2, 4])
def test_C_physical_mass_conserved(k):
    """The measure contract: refining kx multiplies grid tokens by ~k^2 but
    conserves the modality's total support mass (area x spacing shrinks)."""
    model = build("mbca")
    tb_c, tb_r = encode_pair(model, k)
    g = MODALITIES["surf_grid"]
    n_c = int(((tb_c.modality[0] == g) & tb_c.mask[0]).sum())
    n_r = int(((tb_r.modality[0] == g) & tb_r.mask[0]).sum())
    assert n_r >= n_c * (k ** 2) * 0.9             # token count explodes ...
    m_c = modality_mass_totals(tb_c)[g]
    m_r = modality_mass_totals(tb_r)[g]
    assert abs(m_r - m_c) / m_c < 0.05             # ... total mass does not
    # profile modality untouched
    p = MODALITIES["profile"]
    assert abs(modality_mass_totals(tb_c)[p]
               - modality_mass_totals(tb_r)[p]) < 1e-5


def test_C_physical_mbca_more_stable_than_standard():
    """Content genuinely changes (no method is exact) — but MBCA's prediction
    and latent shifts must stay below standard attention's, which lets the
    grid modality's attention mass grow ~4x."""
    q = queries()
    shifts = {}
    for variant in ("perceiver", "mbca"):
        model = build(variant)
        tb_c, tb_r = encode_pair(model, 2)
        shifts[variant] = (output_change(model, tb_c, tb_r, q),
                           latent_change(model, tb_c, tb_r))
    pred_p, lat_p = shifts["perceiver"]
    pred_m, lat_m = shifts["mbca"]
    assert np.isfinite([pred_p, lat_p, pred_m, lat_m]).all()
    assert pred_m < pred_p, f"pred: mbca {pred_m:.4f} vs standard {pred_p:.4f}"
    assert lat_m < lat_p, f"latent: mbca {lat_m:.4f} vs standard {lat_p:.4f}"
