"""Radius limit and local scale on the matched synthetic scenes; no CESM2 data needed."""
import numpy as np
import pytest
import torch

from ocean_tokenizer import radius_scale_scenes as R
from ocean_tokenizer import synthetic_latent_experiment as E
from ocean_tokenizer.latent_ocean import LatentOceanConfig, LatentOceanModel
from test_synthetic_latent_experiment import prepared

SOURCE = np.arange(6)
ROWS, LEVELS = np.array([6, 7, 8, 7, 6]), np.array([0, 1, 2, 0, 2])


def scenes(radius_km=None, scale=None, scale_by="query"):
    data = prepared()
    data.__class__ = R.RadiusScaleScenes
    data.metadata = {}
    data.configure(radius_km=radius_km, scale=scale, scale_by=scale_by)
    return data


def some_scale():
    return np.random.default_rng(3).uniform(.5, 2., size=(9, 3, 2))


def model():
    torch.manual_seed(0)
    net = LatentOceanModel(LatentOceanConfig(n_obs_features=62, n_query_features=58,
        n_local_features=66, n_sat_features=0, variant="local_transformer", width=32,
        n_latents=4, n_heads=4, n_blocks=1, n_query_blocks=1))
    # the local gate starts closed; open it so the local path can change the output
    torch.nn.init.normal_(net.local_gate[-1].weight, std=.5)
    torch.nn.init.normal_(net.local_gate[-1].bias, std=.5)
    return net.eval()


def test_without_radius_or_scale_the_scene_is_the_parents():
    reference = prepared()
    expected = reference.scene(SOURCE, ROWS, LEVELS, reference.analysis())
    data = scenes()
    actual = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_radius_invalidates_far_neighbours_and_nothing_else():
    plain = scenes()
    reference = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    capped = scenes(radius_km=100.)
    actual = capped.scene(SOURCE, ROWS, LEVELS, capped.analysis())
    distance = reference["local_offsets"][..., :2].norm(dim=-1)
    inside = distance <= 100.
    assert inside.any() and not inside.all()
    assert torch.equal(actual["local_valid"], reference["local_valid"] & inside[..., None])
    for name in reference:          # the OI anchor in particular
        if name != "local_valid":
            assert torch.equal(actual[name], reference[name]), name


def test_a_neighbour_beyond_the_radius_cannot_change_the_prediction():
    net = model()
    outputs = {}
    for radius in (None, 100.):
        data = scenes(radius_km=radius)
        scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
        far = scene["local_offsets"][..., :2].norm(dim=-1) > 100.
        moved = dict(scene)
        moved["local_innovation"] = torch.where(far[..., None], scene["local_innovation"] + 3.,
                                                scene["local_innovation"])
        moved["local_features"] = torch.where(far[..., None], scene["local_features"] + 1.,
                                              scene["local_features"])
        with torch.no_grad():
            outputs[radius] = net(scene)["mean"], net(moved)["mean"]
    assert not torch.allclose(*outputs[None], atol=1e-4)    # without a limit it does
    torch.testing.assert_close(*outputs[100.], rtol=0, atol=0)


def test_a_query_with_nobody_inside_the_radius_falls_back_on_the_anchor():
    data = scenes(radius_km=1.)
    scene = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    assert not scene["local_valid"].any()
    with torch.no_grad():
        prediction = model()(scene)
    assert torch.isfinite(prediction["mean"]).all() and torch.isfinite(prediction["std"]).all()
    assert torch.equal(prediction["diagnostics"]["local_gate"], torch.zeros(len(ROWS), 2))


def test_query_scale_divides_a_querys_whole_neighbourhood_by_its_own_scale():
    scale = some_scale()
    plain = scenes()
    reference = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    data = scenes(scale=scale)
    actual = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    s = torch.as_tensor(scale, dtype=torch.float32)[ROWS, LEVELS]
    torch.testing.assert_close(actual["baseline"] * s, reference["baseline"])
    torch.testing.assert_close(actual["local_innovation"] * s[:, None], reference["local_innovation"])
    assert not torch.equal(actual["local_innovation"], reference["local_innovation"])
    for name in reference:          # the shared latent keeps reading per-depth z values
        if name not in ("baseline", "local_innovation"):
            assert torch.equal(actual[name], reference[name]), name


def test_query_scale_keeps_the_neighbours_in_proportion():
    # a weighted mean of the neighbours, converted back, is the unscaled weighted mean:
    # the field the local branch interpolates is not distorted
    scale = some_scale()
    plain, data = scenes(), scenes(scale=scale)
    reference = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    actual = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    weights = torch.rand(reference["local_innovation"].shape[:2] + (1,), generator=torch.Generator().manual_seed(1))
    back = data.to_z((weights * actual["local_innovation"]).sum(1), ROWS, LEVELS)
    torch.testing.assert_close(back, (weights * reference["local_innovation"]).sum(1), rtol=1e-5, atol=1e-6)
    own = scenes(scale=scale, scale_by="profile")
    distorted = own.scene(SOURCE, ROWS, LEVELS, own.analysis())
    back = own.to_z((weights * distorted["local_innovation"]).sum(1), ROWS, LEVELS)
    assert not torch.allclose(back, (weights * reference["local_innovation"]).sum(1), atol=1e-3)


def test_query_scale_works_with_a_thinned_context():
    data = scenes(scale=some_scale())
    data.scene(SOURCE, ROWS, LEVELS, data.analysis(), context_profiles=1, context_seed=5)


def test_profile_scale_divides_each_profile_by_its_own_scale():
    scale = some_scale()
    plain = scenes()
    reference = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    data = scenes(scale=scale, scale_by="profile")
    actual = data.scene(SOURCE, ROWS, LEVELS, data.analysis())
    s = torch.as_tensor(scale, dtype=torch.float32)
    neighbors = plain.geometry(SOURCE, ROWS, 32, LEVELS)["neighbors"]
    lev = torch.as_tensor(LEVELS)
    torch.testing.assert_close(actual["baseline"] * s[ROWS, LEVELS], reference["baseline"])
    torch.testing.assert_close(actual["local_innovation"] * s[neighbors, lev[:, None]],
                               reference["local_innovation"])
    torch.testing.assert_close(actual["obs_innovation"] * s[SOURCE].reshape(-1, 2),
                               reference["obs_innovation"])
    assert not torch.equal(actual["baseline"], reference["baseline"])
    for name in reference:
        if name not in ("baseline", "local_innovation", "obs_innovation"):
            assert torch.equal(actual[name], reference[name]), name


def test_a_scale_of_one_changes_nothing():
    plain = scenes()
    reference = plain.scene(SOURCE, ROWS, LEVELS, plain.analysis())
    for scale_by in ("query", "profile"):
        ones = scenes(scale=np.ones((9, 3, 2)), scale_by=scale_by)
        actual = ones.scene(SOURCE, ROWS, LEVELS, ones.analysis())
        for name in reference:
            assert torch.equal(actual[name], reference[name]), name


def test_network_targets_are_the_targets_in_local_units_and_convert_back():
    scale = some_scale()
    data = scenes(scale=scale)
    target, valid = data.targets(ROWS, LEVELS)
    network, network_valid = data.network_targets(ROWS, LEVELS)
    s = torch.as_tensor(scale, dtype=torch.float32)[ROWS, LEVELS]
    assert torch.equal(valid, network_valid)
    torch.testing.assert_close(network[valid] * s[valid], target[valid])
    torch.testing.assert_close(data.to_z(network, ROWS, LEVELS)[valid], target[valid])
    plain = scenes()
    assert torch.equal(plain.network_targets(ROWS, LEVELS)[0][valid], target[valid])
    assert torch.equal(plain.to_z(target, ROWS, LEVELS)[valid], target[valid])


def test_profile_scale_refuses_a_thinned_context():
    data = scenes(scale=some_scale(), scale_by="profile")
    with pytest.raises(ValueError, match="every source profile"):
        data.scene(SOURCE, ROWS, LEVELS, data.analysis(), context_profiles=1, context_seed=5)
    plain = scenes()
    plain.scene(SOURCE, ROWS, LEVELS, plain.analysis(), context_profiles=1, context_seed=5)


@pytest.mark.parametrize("radius", [0., -5., float("nan")])
def test_a_radius_must_be_positive(radius):
    with pytest.raises(ValueError, match="radius"):
        scenes(radius_km=radius)


def test_a_scale_must_be_positive_finite_and_cover_every_value():
    bad = some_scale()
    bad[0, 0, 0] = 0.
    with pytest.raises(ValueError, match="positive"):
        scenes(scale=bad)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="positive"):
        scenes(scale=bad)
    with pytest.raises(ValueError, match="shape"):
        scenes(scale=some_scale()[:, :2])


def test_metadata_says_what_was_applied():
    assert scenes().metadata == {"local_radius_km": None, "local_scale_sha256": None,
                                 "local_scale_by": None}
    data = scenes(radius_km=250., scale=some_scale())
    assert data.metadata["local_radius_km"] == 250.
    assert data.metadata["local_scale_by"] == "query"
    assert scenes(scale=some_scale(), scale_by="profile").metadata["local_scale_by"] == "profile"
    assert len(data.metadata["local_scale_sha256"]) == 64
    assert data.metadata["local_scale_sha256"] != scenes(scale=some_scale() * 2).metadata["local_scale_sha256"]


def test_folds_partition_a_months_profiles():
    rows = np.arange(100, 123)
    folds = R.fold_rows(rows, 5, seed=1, month=7)
    assert len(folds) == 5 and max(map(len, folds)) - min(map(len, folds)) <= 1
    assert np.array_equal(np.sort(np.concatenate(folds)), rows)
    assert all(np.array_equal(a, b) for a, b in zip(folds, R.fold_rows(rows, 5, seed=1, month=7)))
    assert not np.array_equal(folds[0], R.fold_rows(rows, 5, seed=1, month=8)[0])


def test_crossfit_residual_is_the_frozen_oi_error_from_the_other_folds():
    data = scenes()
    residual = R.crossfit_residual(data, [0], n_folds=3, seed=1, chunk=4)
    assert residual.shape == (9, 3, 2) and residual.dtype == np.float32
    assert np.array_equal(np.isfinite(residual), data.valid_np)
    fold = R.fold_rows(np.arange(9), 3, seed=1, month=0)[1]
    source = np.setdiff1d(np.arange(9), fold)
    rows, levels = np.repeat(fold, 3), np.tile(np.arange(3), len(fold))
    oi = data.analysis()(data.geometry(source, rows, 32, levels), data.indices(levels),
                         data.innovation, data.xyz)
    expected = (data.y[rows, levels] - oi).numpy().reshape(len(fold), 3, 2)
    np.testing.assert_allclose(residual[fold], expected, rtol=1e-5, atol=1e-6)


def test_crossfit_never_predicts_a_profile_from_its_own_value():
    data = scenes()
    before = R.crossfit_residual(data, [0], n_folds=3, seed=1)
    data.y[4, 0, 0] += 2.
    data.innovation[4, 0, 0] += 2.
    after = R.crossfit_residual(data, [0], n_folds=3, seed=1)
    # its own analysis did not move, so its error moved by exactly the change
    assert after[4, 0, 0] - before[4, 0, 0] == pytest.approx(2., abs=1e-5)


def test_crossfit_leaves_other_months_empty():
    data = scenes()
    data.c.month_index = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    residual = R.crossfit_residual(data, [0], n_folds=3, seed=1)
    assert np.isnan(residual[6:]).all() and np.isfinite(residual[:6][data.valid_np[:6]]).all()


def test_an_unknown_way_of_scaling_is_refused():
    with pytest.raises(ValueError, match="scale_by"):
        scenes(scale=some_scale(), scale_by="cell")


def zero_scores(data, ev):
    from ocean_tokenizer.point_baselines import CH, Scores
    scores = Scores(data.levels, data.norm.std)
    for ch in CH:
        target = data.obs[ch][ev.target][ev.profile, ev.level]
        scores.add(ch, np.zeros(len(ev.level)), target, ev.level)
    return scores.result()


def cohort_scenes():
    from types import SimpleNamespace
    from ocean_tokenizer.latent_experiment import EvalMonth
    from ocean_tokenizer.point_baselines import CH
    data = scenes()
    data.norm = SimpleNamespace(std={ch: np.array([1., 2., 3.]) for ch in CH})
    ev = EvalMonth(0, SOURCE, np.array([6, 7, 8]), np.array([0, 1, 2, 1, 0]), LEVELS)
    return data, ev


def test_a_named_cohort_is_checked_against_its_own_climatology_baseline():
    from ocean_tokenizer.point_baselines import CH
    data, ev = cohort_scenes()
    zero = zero_scores(data, ev)
    data.climatology_reference = {"validation": {ch: {"n": zero[ch]["n"], "rmse_z": zero[ch]["rmse_z"]}
                                                 for ch in CH}}
    report = data.identity_check("validation", [ev])
    assert report["TEMP"]["n"] == zero["TEMP"]["n"]
    data.climatology_reference["validation"]["SALT"]["rmse_z"] *= 1.001
    with pytest.raises(ValueError, match="identity"):
        data.identity_check("validation", [ev])
    data.climatology_reference["validation"]["SALT"] = {"n": zero["SALT"]["n"] + 1, "rmse_z": zero["SALT"]["rmse_z"]}
    with pytest.raises(ValueError, match="identity"):
        data.identity_check("validation", [ev])


def test_cohort_reference_is_read_only_from_that_cohorts_baselines(tmp_path):
    import hashlib, json
    data, _ = cohort_scenes()
    data.root = tmp_path
    (tmp_path / "data/synthetic_argo").mkdir(parents=True)
    cohort_file = tmp_path / "data/synthetic_argo/other.nc"
    cohort_file.write_bytes(b"cohort bytes")
    digest = hashlib.sha256(b"cohort bytes").hexdigest()
    scores = {"validation": {"TEMP": {"n": 3, "rmse_z": 1.}, "SALT": {"n": 3, "rmse_z": 1.}}}
    summary = tmp_path / "summary.json"
    def write(path, sha):
        summary.write_text(json.dumps({"cohort": {"path": path, "sha256": sha},
                                       "baselines": {"climatology": {"scores": scores}}}))
    write("data/synthetic_argo/other.nc", digest)
    data.use_cohort_reference("other", summary)
    assert data.climatology_reference == scores and data.metadata["cohort"] == "other"
    write("data/synthetic_argo/cesm2_uniform.nc", digest)
    with pytest.raises(ValueError, match="another cohort"):
        data.use_cohort_reference("other", summary)
    write("data/synthetic_argo/other.nc", "0" * 64)
    with pytest.raises(ValueError, match="rebuilt"):
        data.use_cohort_reference("other", summary)
    with pytest.raises(ValueError, match="baselines"):
        data.use_cohort_reference("other", None)
