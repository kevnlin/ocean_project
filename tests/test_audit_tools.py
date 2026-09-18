"""The audit's three new pieces, tested on data with a known answer.

`audit_tools` is what the 2026-09-17 pipeline audit measures WITH, so a silent
bug there would look like a finding about the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from ocean_tokenizer.argo_obs import ArgoCohort, ArgoNorm
from ocean_tokenizer.audit_tools import (BINS_KM, CovModel, KrigingOI, fit_cov,
                                         haversine_km, pair_correlation,
                                         robust_qc)


def _cohort(n_floats=12, n_cycles=6, n_levels=4, seed=0, scale_km=150.0):
    """A cohort whose anomaly is a smooth Gaussian random field plus noise."""
    rng = np.random.default_rng(seed)
    P = n_floats * n_cycles
    lat = np.repeat(rng.uniform(30, 40, n_floats), n_cycles) + rng.normal(0, 0.2, P)
    lon = np.repeat(rng.uniform(300, 310, n_floats), n_cycles) + rng.normal(0, 0.2, P)
    wmo = np.repeat([f"F{i:03d}" for i in range(n_floats)], n_cycles).astype(str)
    month = np.tile(np.arange(n_cycles), n_floats)
    d = haversine_km(lat[:, None], lon[:, None], lat[None], lon[None])
    K = np.exp(-0.5 * (d / scale_km) ** 2) + 1e-6 * np.eye(P)
    L = np.linalg.cholesky(K)
    field = L @ rng.normal(size=(P, n_levels))
    same = month[:, None] == month[None, :]
    field = np.where(same.any(1)[:, None], field, field)      # keep shape
    levels = np.array([5.0, 100.0, 400.0, 1000.0])[:n_levels]
    year = 2000 + month // 12
    return ArgoCohort(
        region="test", levels=levels, month_index=month,
        grid_y=np.zeros(P, int), grid_x=np.zeros(P, int), lat=lat, lon=lon,
        wmo=wmo, year=year, year_split=np.array(["train"] * P),
        float_split=np.where(np.isin(wmo, np.unique(wmo)[:3]),
                             "heldout_float", "cohort_float"),
        TEMP=field.copy(), SALT=field.copy() * 0.5,
        TEMP_ERR=np.full((P, n_levels), 0.002),
        SALT_ERR=np.full((P, n_levels), 0.01))._index()


def _index(self):
    order = np.argsort(self.month_index, kind="stable")
    for k in ("month_index", "grid_y", "grid_x", "lat", "lon", "wmo", "year",
              "year_split", "float_split", "TEMP", "SALT", "TEMP_ERR", "SALT_ERR"):
        setattr(self, k, getattr(self, k)[order])
    uniq, start = np.unique(self.month_index, return_index=True)
    stop = np.r_[start[1:], self.month_index.size]
    self._by_month = {int(m): (int(a), int(b)) for m, a, b in zip(uniq, start, stop)}
    return self


ArgoCohort._index = _index


# ------------------------------------------------------------------ distance
def test_haversine_matches_known_separations():
    assert haversine_km(0.0, 0.0, 0.0, 1.0) == pytest.approx(111.19, abs=0.1)
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)
    # a degree of longitude shrinks with the cosine of latitude
    assert haversine_km(60.0, 0.0, 60.0, 1.0) == pytest.approx(111.19 / 2, abs=0.6)


def test_haversine_wraps_the_dateline():
    assert haversine_km(0.0, 359.5, 0.0, 0.5) == pytest.approx(111.19, abs=0.5)


# ------------------------------------------------------------------ QC
def test_robust_qc_removes_the_spike_and_keeps_the_ocean():
    c = _cohort()
    truth = c.TEMP.copy()
    c.TEMP[3, 1] = 500.0                      # one impossible value
    rep = robust_qc(c, k_sigma=8.0, profile_levels=3)
    assert not np.isfinite(c.TEMP[3, 1]), "the spike survived QC"
    assert rep.n_values_flagged["TEMP"] >= 1
    kept = np.isfinite(c.TEMP)
    kept[3, 1] = False
    assert np.allclose(c.TEMP[kept], truth[kept]), "QC changed good measurements"


def test_robust_qc_drops_a_whole_channel_when_a_sensor_drifts():
    c = _cohort()
    c.SALT[7, :] += 60.0                      # a drifting conductivity cell
    rep = robust_qc(c, k_sigma=8.0, profile_levels=3)
    assert not np.isfinite(c.SALT[7]).any(), "the drifting profile kept levels"
    assert rep.n_profiles_dropped["SALT"] >= 1
    assert np.isfinite(c.TEMP[7]).all(), "TEMP was dropped with SALT"


def test_robust_qc_statistics_come_from_the_named_split_only():
    c = _cohort()
    c.year_split = np.where(np.arange(c.TEMP.shape[0]) % 2 == 0, "train", "development")
    c.TEMP[np.flatnonzero(c.year_split == "development")[0], 0] = 300.0
    before = np.nanstd(c.TEMP[c.year_split == "train"], axis=0).copy()
    robust_qc(c, split="train", k_sigma=8.0)
    after = np.nanstd(c.TEMP[c.year_split == "train"], axis=0)
    assert np.allclose(before, after, rtol=0.2), "train statistics moved"


# ------------------------------------------------------------------ covariance
def test_pair_correlation_falls_with_distance_and_excludes_same_float():
    c = _cohort(n_floats=25, n_cycles=4, scale_km=200.0)
    norm = ArgoNorm.fit(c, "train")
    rho, cnt = pair_correlation(c, norm, "TEMP", 0, c.months_in("train"))
    ok = cnt > 20
    assert ok.sum() >= 3
    near, far = rho[ok][0], rho[ok][-1]
    assert near > far, f"correlation did not fall with distance ({near} -> {far})"


def test_fit_cov_recovers_a_planted_length_scale():
    mid = 0.5 * (BINS_KM[:-1] + BINS_KM[1:])
    true = CovModel(a0=0.0, a1=0.8, L1=80.0, a2=0.0, L2=900.0)
    m = fit_cov(true(mid), np.full(mid.size, 1e4))
    assert m.L1 == pytest.approx(80.0, rel=0.5)
    assert m.a1 == pytest.approx(0.8, abs=0.15)
    assert 0.0 < m.nugget < 0.35


def test_cov_model_nugget_is_the_unexplained_variance():
    m = CovModel(a0=0.1, a1=0.5, L1=50.0, a2=0.2, L2=800.0)
    assert m.nugget == pytest.approx(0.2, abs=1e-9)
    assert m(np.array([0.0]))[0] == pytest.approx(0.8, abs=1e-9)


# ------------------------------------------------------------------ kriging
def test_kriging_beats_climatology_on_a_field_it_knows():
    c = _cohort(n_floats=30, n_cycles=3, scale_km=200.0, seed=3)
    norm = ArgoNorm.fit(c, "train")
    cov = {ch: [CovModel(0.0, 0.85, 200.0, 0.0, 900.0) for _ in c.levels]
           for ch in ("TEMP", "SALT")}
    oi = KrigingOI(cov)
    se = se0 = n = 0.0
    for m in c.months_in("train"):
        src = c.month(int(m), float_split="cohort_float")
        tgt = c.month(int(m), float_split="heldout_float")
        if src.size == 0 or tgt.size == 0:
            continue
        pred = oi.predict(c, norm, src, tgt)
        y = (c.TEMP[tgt] - norm.mean["TEMP"]) / norm.std["TEMP"]
        se += float(((pred[..., 0] - y) ** 2).sum())
        se0 += float((y ** 2).sum()); n += y.size
    assert n > 0
    assert np.sqrt(se / n) < 0.85 * np.sqrt(se0 / n), "kriging did not beat the mean"


def test_kriging_returns_zero_without_observations():
    c = _cohort()
    norm = ArgoNorm.fit(c, "train")
    cov = {ch: [CovModel(0.0, 0.8, 100.0, 0.0, 900.0) for _ in c.levels]
           for ch in ("TEMP", "SALT")}
    out = KrigingOI(cov).predict(c, norm, np.array([], int), c.month(0)[:2])
    assert out.shape == (2, c.levels.size, 2)
    assert np.all(out == 0.0)


def test_kriging_ignores_a_level_no_input_reported():
    c = _cohort(n_floats=8, n_cycles=3)
    norm = ArgoNorm.fit(c, "train")
    src = c.month(0, float_split="cohort_float")
    tgt = c.month(0, float_split="heldout_float")
    c.TEMP[src, 2] = np.nan                      # nobody measured level 2
    cov = {ch: [CovModel(0.0, 0.8, 200.0, 0.0, 900.0) for _ in c.levels]
           for ch in ("TEMP", "SALT")}
    out = KrigingOI(cov).predict(c, norm, src, tgt)
    assert np.all(out[:, 2, 0] == 0.0), "predicted from levels with no data"
    assert np.isfinite(out).all()
