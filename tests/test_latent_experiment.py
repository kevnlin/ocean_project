"""Scientific data-contract checks for the matched ocean-latent experiments."""

import hashlib
import json

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_tokenizer import anchored as A
from ocean_tokenizer import latent_experiment as E
from ocean_tokenizer.argo_obs import ArgoNorm


class _Cohort:
    def __init__(self):
        self.levels = np.array([0.0, 125.0, 900.0])
        self.lat = np.array([10., 10.4, 9.8, 11., 10.2, 10.7, 9.5, 10.1])
        self.lon = np.array([179.5, 180.2, 178.9, 179.8, 180.4, 179.7, 179.6, 178.8])
        self.wmo = np.array(["101", "101", "102", "103", "900", "900", "901", "104"])
        self.month_index = np.full(8, 192)
        self.year_split = np.full(8, "train")
        self.float_split = np.array(["cohort_float"] * 4 + ["heldout_float"] * 3 + ["cohort_float"])
        generator = np.random.default_rng(7)
        self.TEMP = generator.normal(size=(8, 3))
        self.SALT = generator.normal(size=(8, 3))
        self.TEMP[0, 1] = np.nan
        self.SALT[3, 2] = 11.0
        self.TEMP_ERR = np.full((8, 3), 0.2)
        self.SALT_ERR = np.full((8, 3), 0.4)
        self.TEMP_ERR[0] = [np.nan, np.inf, -np.inf]

    def month(self, month, float_split=None):
        selected = self.month_index == month
        if float_split is not None:
            selected &= self.float_split == float_split
        return np.flatnonzero(selected)

    def months_in(self, split):
        return np.unique(self.month_index)


def _prepared():
    """Prepare a tiny scene directly, without any real-data file access."""
    prepared = E.OceanScenes.__new__(E.OceanScenes)
    prepared.device = torch.device("cpu")
    prepared.c = c = _Cohort()
    prepared.n_levels = 3
    prepared.levels = c.levels
    prepared.months_available = {192}
    prepared._months_cache = {}
    prepared._training_pools = {}
    y = np.stack([c.TEMP, c.SALT], axis=-1)
    prepared.y = torch.tensor(y, dtype=torch.float64)
    prepared.valid = torch.isfinite(prepared.y) & (prepared.y.abs() <= 10)
    profile = torch.arange(8, dtype=torch.float64)[:, None, None]
    level = torch.arange(3, dtype=torch.float64)[None, :, None]
    channel = torch.arange(2, dtype=torch.float64)[None, None, :]
    prepared.fg = profile * 0.01 + level * 0.1 + channel * 0.3
    prepared.innovation = torch.where(prepared.valid, prepared.y - prepared.fg, 0)
    prepared.error = torch.full_like(prepared.y, 0.1)
    prepared.error_valid = torch.ones_like(prepared.y)
    prepared.x = torch.randn(8, 56, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    prepared.x[:, 0] = torch.arange(8)
    prepared.lat = torch.tensor(c.lat)
    prepared.lon = torch.tensor(c.lon)
    prepared.depth = torch.tensor(c.levels)
    prepared.day = torch.tensor([0., 4., 7., 11., 5., 18., 23., 16.], dtype=torch.float64)
    prepared.state = torch.arange(8, dtype=torch.float64) * 0.15
    prepared.state[5] = 0
    prepared.state_ok = torch.ones(8, dtype=torch.bool)
    prepared.state_ok[5] = False
    return prepared


def _analysis(k=3):
    analysis = A.InnovationAnalysis(6, mode="kriging", k=k, use_time=True, use_state=True).double()
    settings = {
        "log_ell_x": [80., 200., 400., 150., 350., 700.],
        "log_ell_y": [100., 250., 500., 200., 300., 550.],
        "log_gamma": [.05, .1, .3, .2, .4, .8],
        "log_ell_t": [5., 8., 12., 7., 11., 18.],
        "log_ell_s": [.5, .8, 1.2, .7, 1.1, 2.],
    }
    with torch.no_grad():
        for name, values in settings.items():
            getattr(analysis, name).copy_(torch.tensor(values).log())
    return analysis


def test_selected_depth_baselines_reproduce_the_existing_full_field_analysis():
    data, analysis = _prepared(), _analysis()
    source = data.c.month(192, "cohort_float")
    rows, levels = np.array([6, 4, 6, 5, 4]), np.array([0, 1, 2, 2, 0])
    flat = data.innovation.transpose(1, 2).reshape(8, 6)
    flat_valid = data.valid.transpose(1, 2).reshape(8, 6)
    flat = torch.where(flat_valid, flat, float("nan"))
    with torch.no_grad():
        correction = analysis(
            data.lat[rows], data.lon[rows], data.lat[source], data.lon[source], flat[source],
            q_t=data.day[rows], o_t=data.day[source], q_s=data.state[rows], o_s=data.state[source],
        )
        selected = torch.stack((torch.tensor(levels), torch.tensor(levels) + 3), dim=-1)
        expected = data.fg[rows, levels] + correction[torch.arange(len(rows))[:, None], selected]
        scene = data.scene(source, rows, levels, analysis)
    torch.testing.assert_close(scene["baseline"], expected, rtol=1e-10, atol=1e-10)
    assert not torch.equal(scene["baseline"][:, 0], scene["baseline"][:, 1])


def test_geometry_keeps_repeated_query_rows_and_date_state_offsets_aligned():
    data = _prepared()
    source = data.c.month(192, "cohort_float")
    rows = np.array([6, 4, 6, 5, 4])
    geometry = data.geometry(source, rows, 3)
    reference = A.neighbours(data.lat[rows], data.lon[rows], data.lat[source], data.lon[source], 3)
    neighbors = torch.tensor(source)[reference.idx]
    assert torch.equal(geometry["neighbors"], neighbors)
    torch.testing.assert_close(geometry["offsets"][..., 0], reference.dx)
    torch.testing.assert_close(geometry["offsets"][..., 1], reference.dy)
    torch.testing.assert_close(geometry["offsets"][..., 2], data.day[neighbors] - data.day[rows, None])
    torch.testing.assert_close(geometry["offsets"][..., 3], data.state[neighbors] - data.state[rows, None])


def test_target_measurements_and_errors_do_not_enter_the_model_scene():
    data, analysis = _prepared(), _analysis()
    source = data.c.month(192, "cohort_float")
    rows, levels = np.array([4, 5, 6]), np.array([2, 0, 1])
    with torch.no_grad():
        expected = data.scene(source, rows, levels, analysis)
        data.y[rows] = float("nan")
        data.innovation[rows] = 1e12
        data.valid[rows] = False
        data.error[rows] = 1e12
        data.error_valid[rows] = 0
        actual = data.scene(source, rows, levels, analysis)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_depth_values_are_preserved_and_latent_budget_does_not_thin_local_evidence():
    data, analysis = _prepared(), _analysis()
    source = data.c.month(192, "cohort_float")
    rows, levels = np.array([4, 6]), np.array([0, 2])
    with torch.no_grad():
        full = data.scene(source, rows, levels, analysis, context_profiles=0)
        budgeted = data.scene(source, rows, levels, analysis, context_profiles=1, context_seed=8)
    assert budgeted["obs_features"].shape == (3, 62)
    assert budgeted["local_features"].shape == (2, analysis.k, 66)
    torch.testing.assert_close(budgeted["obs_coord"][:, 2], data.depth)
    assert torch.unique(budgeted["obs_features"][:, 0]).numel() == 1
    assert torch.unique(budgeted["local_features"][..., 0]).numel() > 1
    for key in ("baseline", "local_features", "local_offsets", "local_innovation", "local_valid"):
        assert torch.equal(full[key], budgeted[key]), key
    context_row = int(budgeted["obs_features"][0, 0])
    torch.testing.assert_close(budgeted["obs_innovation"], data.innovation[context_row])
    assert torch.equal(budgeted["obs_valid"], data.valid[context_row])


def test_training_holds_out_entire_floats_including_repeated_cycles():
    data = _prepared()
    rng = np.random.default_rng(11)
    for _ in range(20):
        month, source, rows, levels = data.training_draw(rng, n_queries=25)
        assert month == 192
        assert len(rows) == len(levels) == 25
        assert len(source) > 0
        assert not np.intersect1d(data.c.wmo[source], data.c.wmo[rows]).size
        assert np.isin(rows, data.c.month(month, "cohort_float")).all()
        assert ((0 <= levels) & (levels < data.n_levels)).all()


def test_evaluation_rejects_a_float_identity_shared_between_context_and_target():
    data = _prepared()
    data.c.wmo[4] = data.c.wmo[0]
    with pytest.raises(ValueError, match="identities overlap"):
        data.eval_months("development", max_cells=0)


def _write_tiny_anchor(tmp_path, monkeypatch, *, defect=None):
    """Use real small checkpoint/NPZ files, with cohort loading stubbed out."""
    c = _Cohort()
    norm = ArgoNorm({ch: np.zeros(3) for ch in E.CH}, {ch: np.ones(3) for ch in E.CH})
    monkeypatch.setattr(E, "load_cohort", lambda *args, **kwargs: (c, None))
    monkeypatch.setattr(E.ArgoNorm, "fit", lambda cohort, split: norm if split == "train" else pytest.fail("nontraining normalization"))
    satellite = {
        key: np.linspace(0.05, 0.3, 8) for key in (
            "SLA_1deg_anom", "SST_1deg_anom", "SSS_1deg_anom", "SST_1deg",
            "SLA_day_c", "SLA_day_e", "SLA_day_w", "SLA_day_n", "SLA_day_s",
            "SLA_clim_1deg", "SLA_1deg",
        )
    }
    satellite["month_index"] = c.month_index.copy()
    satellite["SST_1deg_anom"][1] = np.nan
    satellite["SLA_day_c"][5] = np.nan
    x = E.satellite_features(c, satellite)
    y = np.concatenate([c.TEMP, c.SALT], axis=1)
    cohort_fingerprint = E.array_fingerprint((c.month_index, c.lat, c.lon, c.wmo, c.levels, y))
    checkpoint = {
        "cohort_fingerprint": cohort_fingerprint,
        "feature_fingerprint": hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest(),
        "feature_mean": torch.zeros(56, dtype=torch.float64),
        "feature_std": torch.ones(56, dtype=torch.float64),
    }
    predictions = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 100
    cache = {"cohort_fingerprint": cohort_fingerprint, "training_predictions": "cross-fit by year", "predictions": predictions}
    if defect == "checkpoint_cohort":
        checkpoint["cohort_fingerprint"] = "stale"
    elif defect == "cache_cohort":
        cache["cohort_fingerprint"] = "stale"
    elif defect == "in_sample_training":
        cache["training_predictions"] = "in sample"
    elif defect == "features":
        checkpoint["feature_fingerprint"] = "stale"
    elif defect == "satellite_order":
        satellite["month_index"][0] += 1
    anchor = tmp_path / "outputs/audit/global/anc_satday_dfs"
    anchor.mkdir(parents=True)
    torch.save(checkpoint, anchor / "first_guess_seed1234.pt")
    torch.save(cache, anchor / "first_guess_predictions_seed1234.pt")
    reference = tmp_path / "outputs/audit/global/setconv_surface"
    reference.mkdir(parents=True)
    (reference / "summary_seed1234.json").write_text(json.dumps({"scores": {}}))
    data = tmp_path / "data/argo_cohort"
    data.mkdir(parents=True)
    np.savez(data / "global_satellite.npz", **satellite)
    raw = xr.Dataset({"month_index": ("profile", c.month_index),
                      "time": ("profile", np.arange(8).astype("timedelta64[D]") + np.datetime64("2016-01-01"))})
    raw.to_netcdf(data / "global_global.nc", engine="scipy")
    return c, predictions


def test_small_file_loader_preserves_temperature_salinity_indexing_and_input_qc(tmp_path, monkeypatch):
    c, predictions = _write_tiny_anchor(tmp_path, monkeypatch)
    data = E.OceanScenes(tmp_path)
    torch.testing.assert_close(data.fg[..., 0], predictions[:, :3])
    torch.testing.assert_close(data.fg[..., 1], predictions[:, 3:])
    torch.testing.assert_close(data.y[..., 0], torch.tensor(c.TEMP, dtype=torch.float32), equal_nan=True)
    torch.testing.assert_close(data.y[..., 1], torch.tensor(c.SALT, dtype=torch.float32), equal_nan=True)
    assert not data.valid[0, 1, 0]
    assert not data.valid[3, 2, 1]
    assert torch.equal(data.innovation[~data.valid], torch.zeros_like(data.innovation[~data.valid]))
    assert torch.isfinite(data.x).all()
    assert torch.isfinite(data.error).all()
    assert (data.error >= 0).all()
    assert torch.equal(data.error_valid[0, :, 0], torch.zeros(3))
    assert torch.equal(data.error_valid[..., 1], torch.ones(8, 3))
    assert not data.state_ok[5] and float(data.state[5]) == 0


@pytest.mark.parametrize("defect,message", [
    ("checkpoint_cohort", "cohort/normalization"),
    ("cache_cohort", "cohort/normalization"),
    ("in_sample_training", "cross-fitted"),
    ("features", "satellite features differ"),
    ("satellite_order", "satellite profile ordering"),
])
def test_mismatched_or_in_sample_anchor_caches_are_rejected(tmp_path, monkeypatch, defect, message):
    _write_tiny_anchor(tmp_path, monkeypatch, defect=defect)
    with pytest.raises(ValueError, match=message):
        E.OceanScenes(tmp_path)
