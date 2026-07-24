"""Task 6, Test D — profile vertical resampling.

The same continuous profile represented at sparse / standard / denser depth
sampling must not change the model's answer *because of sample count alone*;
the model must still respond to genuinely new vertical structure.  The
physical-depth-band ProfileEncoder makes token count independent of level
count (bands are fixed), and per-profile mass normalization keeps total mass
fixed, so both the encoder and MBCA contribute to this stability.
"""
import numpy as np
import pytest
import torch

from ocean_tokenizer.token_api import ProfileEncoder
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.invariance import output_change

D_MODEL, N_LATENT = 32, 16
DEPTHS_STD = np.array([5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165,
                       186, 222, 267, 327, 408, 527, 707, 985], dtype="float32")
# sparse = every other level BUT keep the deepest, so vertical *coverage* is
# preserved and only the sampling density changes (dropping 985 m would be a
# genuine loss of information, not a resampling)
DEPTHS_SPARSE = np.append(DEPTHS_STD[::2], DEPTHS_STD[-1])   # 11 levels
_mid = (DEPTHS_STD[:-1] + DEPTHS_STD[1:]) / 2
DEPTHS_DENSE = np.sort(np.concatenate([DEPTHS_STD, _mid]))   # 39 levels


class FakeGrid:
    depth = DEPTHS_STD.astype("float64")


def build(variant, seed=3):
    m = build_fusion_model(variant, FakeGrid(), d_model=D_MODEL,
                           n_latent=N_LATENT, n_heads=4, n_self_blocks=2,
                           patch=(4, 6), seed=seed)
    m.eval()
    return m


def smooth_profiles(P=6, seed=0):
    """Smooth continuous T/S columns: value = f(depth) with random smooth f."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(P, 2, 3))
    lat = rng.uniform(-80, 80, (1, P)).astype("float32")
    lon = rng.uniform(0, 360, (1, P)).astype("float32")

    def sample(depths):
        z = depths / 1000.0
        prof = np.stack([
            a[:, c, 0:1] + a[:, c, 1:2] * z[None] +
            a[:, c, 2:3] * np.sin(2 * np.pi * z[None])
            for c in range(2)], axis=1)                   # (P, 2, D)
        return dict(prof=torch.tensor(prof[None].astype("float32")),
                    lat=torch.tensor(lat), lon=torch.tensor(lon),
                    month=torch.tensor([3]),
                    depths=torch.tensor(depths))
    return sample


def queries(Q=9, seed=2):
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-80, 80, Q), rng.uniform(0, 360, Q),
                  rng.uniform(0, 985, Q), np.full(Q, 3.0)], -1).astype("float32")
    return torch.tensor(q)[None]


# ------------------------------------------------------------ encoder level
def test_D_token_count_independent_of_level_count():
    enc = ProfileEncoder(DEPTHS_STD, d_model=D_MODEL)
    sample = smooth_profiles()
    for depths in (DEPTHS_SPARSE, DEPTHS_STD, DEPTHS_DENSE):
        tb = enc(**sample(depths))
        assert tb.emb.shape[1] == 6 * enc.n_bands        # bands, not levels


def test_D_mass_independent_of_level_count():
    """Per-profile normalized masses are identical across samplings (exact);
    raw represented spans agree to ~the boundary half-intervals."""
    sample = smooth_profiles()
    enc = ProfileEncoder(DEPTHS_STD, d_model=D_MODEL)     # normalized default
    m = [enc(**sample(d)).support_mass for d in
         (DEPTHS_SPARSE, DEPTHS_STD, DEPTHS_DENSE)]
    for a in m[1:]:
        assert torch.allclose(m[0].sum(), a.sum(), atol=1e-5)
    enc_raw = ProfileEncoder(DEPTHS_STD, d_model=D_MODEL,
                             normalize_per_profile=False)
    r = [enc_raw(**sample(d)).support_mass.sum() for d in
         (DEPTHS_SPARSE, DEPTHS_STD, DEPTHS_DENSE)]
    for a in r[1:]:
        assert 0.8 < float(a / r[0]) < 1.25


# ------------------------------------------------------------- model level
@pytest.mark.parametrize("variant", ["perceiver", "mbca"])
def test_D_resampling_shift_small_vs_real_information(variant):
    """Resampling the same smooth curve shifts the output far less than a
    genuinely different set of profiles (the discriminative control) — and
    the model is NOT trivially input-insensitive."""
    model = build(variant)
    q = queries()
    sample = smooth_profiles(seed=0)
    obs_std = {"profiles": sample(DEPTHS_STD)}
    tb_std = model.encode(obs_std, batch=1)

    shifts = {}
    for name, depths in (("sparse", DEPTHS_SPARSE), ("dense", DEPTHS_DENSE)):
        tb = model.encode({"profiles": sample(depths)}, batch=1)
        shifts[name] = output_change(model, tb_std, tb, q)

    other = model.encode({"profiles": smooth_profiles(seed=9)(DEPTHS_STD)},
                         batch=1)
    real_shift = output_change(model, tb_std, other, q)

    assert real_shift > 5e-3, "model ignores its profile input entirely"
    for name, s in shifts.items():
        assert s < 0.5 * real_shift, \
            f"{variant}/{name}: resample shift {s:.4f} vs real {real_shift:.4f}"


def test_D_band_masses_stable_under_resampling():
    """The MBCA weight input moves only marginally under resampling: mean
    band-mass shift < 10%.  The residual comes from the half-interval spill
    of levels adjacent to a band boundary (midpoint quadrature), not from
    level *count* — the count-driven effect would be ~2x."""
    sample = smooth_profiles(seed=0)
    enc = ProfileEncoder(DEPTHS_STD, d_model=D_MODEL)
    m_std = enc(**sample(DEPTHS_STD)).support_mass
    for depths in (DEPTHS_SPARSE, DEPTHS_DENSE):
        m = enc(**sample(depths)).support_mass
        assert float((m - m_std).abs().mean()) < 0.10 * float(m_std.abs().mean())
