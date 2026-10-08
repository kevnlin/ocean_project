"""Matched synthetic OI/latent contracts; no expensive CESM2 data required."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer import synthetic_latent_experiment as E
from ocean_tokenizer.latent_ocean import LatentOceanConfig, LatentOceanModel
from ocean_tokenizer.oi import _lonlat_to_xyz
from ocean_tokenizer.point_baselines import BANDS, CH, band_of_levels, oi_points


def prepared(device="cpu"):
    data = E.SyntheticOceanScenes.__new__(E.SyntheticOceanScenes)
    data.device = torch.device(device)
    data.surface = False
    data.levels = np.array([5., 125., 900.])
    data.n_levels = 3
    lat = np.array([10., 10.4, 9.8, 11., 10.2, 10.7, 9.5, 10.1, 10.9])
    lon = np.array([179.5, 180.2, 178.9, 179.8, 180.4, 179.7, 179.6, 178.8, 180.1])
    raw = np.random.default_rng(11).normal(size=(9, 3, 2)).astype("float32")
    raw[0, 1, 0] = np.nan
    raw[1, 1, 0] = np.nan
    raw[0, 2, 1] = np.nan
    raw[2, 2, 1] = 15.  # Genuine synthetic extremes must not be QC-clipped.
    data.c = SimpleNamespace(lat=lat, lon=lon, month_index=np.zeros(9, dtype=int),
                             wmo=np.array(["a", "a", "b", "c", "d", "e", "f", "g", "h"]))
    data.oi_selection = {ch: {band: {"L_km": 400.-ci*100+bi*30, "gamma": .03+.02*ci,
                                   "k": [3, 4, 2][min(bi, 2)]}
        for bi, (band, _, _) in enumerate(BANDS)} for ci, ch in enumerate(CH)}
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=data.device)
    data.obs = {ch: raw[..., ci] for ci, ch in enumerate(CH)}
    data.xyz_np = _lonlat_to_xyz(lat, lon)
    data.xyz = tensor(data.xyz_np, torch.float64)
    data.y = tensor(raw)
    data.valid_np = np.isfinite(raw)
    data.valid = torch.isfinite(data.y)
    data.innovation = torch.where(data.valid, data.y, 0.)
    data.fg = torch.zeros_like(data.y)
    data.error = torch.zeros_like(data.y)
    data.error_valid = torch.zeros_like(data.y)
    data.x = tensor(E.position_features(data.c))
    data.lat, data.lon = tensor(lat), tensor(lon)
    data.day, data.depth = tensor(np.zeros(9)), tensor(data.levels)
    return data


def test_exact_spherical_oi_matches_44_with_per_channel_finite_neighbors():
    data, source = prepared(), np.arange(6)
    rows, levels = np.array([6, 7, 8, 7, 6]), np.array([0, 1, 2, 0, 2])
    analysis = data.analysis()
    geometry = data.geometry(source, rows, 32, levels)
    actual = analysis(geometry, data.indices(levels), data.innovation, data.xyz).numpy()
    expected = np.zeros_like(actual)
    bands = band_of_levels(data.levels)
    for channel, ch in enumerate(CH):
        for level in np.unique(levels):
            chosen = levels == level
            setting = data.oi_selection[ch][bands[level]]
            expected[chosen, channel] = oi_points(data.c.lat[source], data.c.lon[source],
                data.obs[ch][source, level], data.c.lat[rows[chosen]], data.c.lon[rows[chosen]],
                **{k: setting[k] for k in ("L_km", "gamma", "k")})
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-6)
    assert data.valid[2, 2, 1] and data.innovation[2, 2, 1] == 15.


def test_geometry_chooses_finite_neighbors_before_localization():
    data = prepared()
    geometry = data.geometry(np.arange(6), np.array([6]), 3, np.array([1]))
    temp = geometry["oi_neighbors"][0, 0][geometry["oi_valid"][0, 0]].numpy()
    assert not np.isin(temp, [0, 1]).any()
    assert len(temp) == 4
    salt = geometry["oi_neighbors"][0, 1][geometry["oi_valid"][0, 1]].numpy()
    assert len(salt) == 4


def test_no_observations_in_a_variable_returns_zero_anomaly():
    data = prepared()
    source = np.arange(6)
    data.valid_np[source, 2, 1] = False
    data.valid[source, 2, 1] = False
    data.innovation[source, 2, 1] = 0.
    geometry = data.geometry(source, np.array([6, 7]), 32, np.array([2, 2]))
    actual = data.analysis()(geometry, data.indices([2, 2]), data.innovation, data.xyz)
    assert torch.equal(actual[:, 1], torch.zeros(2))
    assert torch.isfinite(actual).all()


def test_latent_context_budget_never_thins_full_pool_oi_or_local_evidence():
    data, source = prepared(), np.arange(6)
    rows, levels = np.array([6, 7]), np.array([0, 2])
    full = data.scene(source, rows, levels, data.analysis(), context_profiles=0)
    small = data.scene(source, rows, levels, data.analysis(), context_profiles=1, context_seed=123)
    assert small["obs_features"].shape == (3, 62)
    assert small["local_features"].shape == (2, 6, 66)
    for name in ("baseline", "local_features", "local_innovation", "local_valid", "local_offsets"):
        torch.testing.assert_close(full[name], small[name], rtol=0, atol=0)


def test_argo_only_scene_contains_no_satellite_or_target_measurements():
    data, source = prepared(), np.arange(6)
    rows, levels = np.array([6, 7]), np.array([0, 2])
    before = data.scene(source, rows, levels, data.analysis())
    data.y[rows] = 1e8
    data.innovation[rows] = 1e8
    data.error[rows] = 1e8
    after = data.scene(source, rows, levels, data.analysis())
    assert "satellite_features" not in after and "satellite_coord" not in after
    assert torch.equal(data.x[:, 44:], torch.zeros(9, 12))
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


def test_query_source_overlap_and_missing_depth_selection_fail_closed():
    data = prepared()
    with pytest.raises(ValueError, match="overlap"):
        data.geometry(np.arange(6), np.array([0]), 32, np.array([0]))
    with pytest.raises(ValueError, match="query_levels"):
        data.geometry(np.arange(6), np.array([6]))


@pytest.mark.parametrize("variant", ["dense", "soft_moe", "local_transformer"])
def test_new_backbones_start_exactly_at_frozen_oi_and_decode_independent_queries(variant):
    data = prepared()
    scene = data.scene(np.arange(6), np.array([6, 7, 8]), np.array([0, 1, 2]), data.analysis())
    model = LatentOceanModel(LatentOceanConfig(n_obs_features=62, n_query_features=58,
        n_local_features=66, n_sat_features=0, width=16, n_heads=2, n_latents=4,
        n_blocks=1, n_query_blocks=1, variant=variant))
    with torch.no_grad():
        output = model(scene)
    torch.testing.assert_close(output["mean"], scene["baseline"], rtol=0, atol=0)
    assert torch.isfinite(output["std"]).all() and (output["std"] > 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_finite_neighbour_selection_and_float64_oi_match_cpu():
    cpu, cuda = prepared(), prepared("cuda")
    source, rows, levels = np.arange(6), np.array([6, 7, 8]), np.array([0, 1, 2])
    expected = cpu.scene(source, rows, levels, cpu.analysis())["baseline"]
    actual = cuda.scene(source, rows, levels, cuda.analysis())["baseline"].cpu()
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def test_runner_resume_restores_optimizer_sampler_and_best_checkpoint(tmp_path, monkeypatch):
    """Exercise the actual new driver, including standard physical metrics."""
    import importlib.util
    from pathlib import Path
    from ocean_tokenizer.latent_experiment import EvalMonth

    path = Path(__file__).resolve().parents[1]/"experiments/synthetic/46_latent_reconstruction.py"
    spec = importlib.util.spec_from_file_location("synthetic_runner_under_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class SmallScenes(E.SyntheticOceanScenes):
        def __init__(self, root, device, **kwargs):
            self.__dict__.update(prepared(device).__dict__)
            self.levels = np.array([5., 125., 500., 900.])
            self.n_levels = 4
            self.depth = torch.tensor(self.levels, dtype=torch.float32)
            self.y = torch.cat((self.y[:, :2], self.y[:, :1], self.y[:, 2:]), dim=1)
            self.valid = torch.isfinite(self.y)
            self.valid_np = self.valid.numpy()
            self.innovation = torch.where(self.valid, self.y, 0.)
            self.fg = torch.zeros_like(self.y)
            self.error = torch.zeros_like(self.y)
            self.error_valid = torch.zeros_like(self.y)
            self.norm = SimpleNamespace(mean={ch: np.zeros(4) for ch in CH},
                                        std={ch: np.ones(4) for ch in CH})
            self.c.year_split = np.full(9, "validation")
            self.metadata = {k: "fixture-fixed" for k in
                ("cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")}
            self.fg_checkpoint = {"method": "zero background"}

        def eval_months(self, split, max_cells):
            return [EvalMonth(0, np.arange(6), np.arange(6, 9),
                              np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3))]

        def identity_check(self, split, evs):
            return {"fixture": True}

        def baseline_identity_check(self, split, scores):
            return {"fixture": True}

        def training_draw(self, rng, n_queries):
            return 0, np.arange(6), rng.choice(np.arange(6, 9), n_queries), rng.choice(4, n_queries)

        def scene(self, *args, **kwargs):
            output = super().scene(*args, **kwargs)
            seed = kwargs.get("context_seed", args[5] if len(args) > 5 else None)
            if isinstance(seed, np.random.Generator):
                output["obs_features"] = output["obs_features"]+.01*torch.randn_like(output["obs_features"])
            return output

    monkeypatch.setattr(runner, "SyntheticOceanScenes", SmallScenes)
    def invoke(output, *extra):
        monkeypatch.setattr(runner.sys, "argv", [str(path), "--tag", "fixture", "--output", str(output),
            "--device", "cpu", "--steps", "6", "--queries", "8", "--context-profiles", "3",
            "--width", "16", "--latents", "4", "--blocks", "1", "--query-blocks", "1",
            "--val-every", "3", "--eval-chunk", "8", "--checkpoint-every", "3", *extra])
        runner.main()

    direct, resumed = tmp_path/"direct", tmp_path/"resumed"
    original_threads = torch.get_num_threads()
    try:
        invoke(direct)
        save = runner.atomic_save
        class Interrupted(Exception):
            pass
        def interrupt(state, destination):
            save(state, destination)
            if Path(destination).name == "last_seed1234.pt" and state["step"] == 3:
                raise Interrupted()
        monkeypatch.setattr(runner, "atomic_save", interrupt)
        with pytest.raises(Interrupted):
            invoke(resumed)
        monkeypatch.setattr(runner, "atomic_save", save)
        invoke(resumed, "--resume-training", "--load-checkpoint", str(resumed/"last_seed1234.pt"))
        expected = torch.load(direct/"last_seed1234.pt", weights_only=True)
        actual = torch.load(resumed/"last_seed1234.pt", weights_only=True)
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for name in a:
                    equal(a[name], b[name])
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    equal(x, y)
            else:
                assert a == b
        for name in ("model", "analysis", "anchor_analysis"):
            equal(actual[name], expected[name])
        for name in ("optimizer", "scheduler", "random", "training_loss", "initial"):
            equal(actual["continuation"][name], expected["continuation"][name])
        for name in ("score", "step"):
            equal(actual["continuation"]["best"][name], expected["continuation"]["best"][name])
        for name in ("model", "analysis", "validation"):
            equal(actual["continuation"]["best"]["state"][name], expected["continuation"]["best"]["state"][name])
        with np.load(direct/"validation_seed1234.npz") as a, np.load(resumed/"validation_seed1234.npz") as b:
            assert a.files == b.files
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name])
    finally:
        torch.set_num_threads(original_threads)
