"""Continuation checks against the actual runner, without real-data/GPU access."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ocean_tokenizer import anchored as A
from ocean_tokenizer.latent_experiment import EvalMonth


RUNNER_PATH = Path(__file__).resolve().parents[1] / "experiments/real_data/69_latent_ocean.py"
spec = importlib.util.spec_from_file_location("latent_training_runner_under_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal(a, b)
    else:
        assert actual == expected


def test_random_state_replays_torch_and_numpy_sampling(tmp_path):
    rng = np.random.default_rng(819)
    torch.manual_seed(820)
    saved = runner.random_state(rng)
    runner.atomic_save({"continuation": {"random": saved}}, tmp_path / "last.pt")
    expected = (rng.choice(100, 13, replace=False), rng.normal(size=7), torch.randn(11))
    rng.normal(size=100)
    torch.randn(100)
    restored = torch.load(tmp_path / "last.pt", weights_only=True)
    runner.restore_random_state(rng, restored["continuation"]["random"])
    actual = (rng.choice(100, 13, replace=False), rng.normal(size=7), torch.randn(11))
    _assert_equal(actual, expected)


def test_atomic_overwrite_is_loadable_and_failed_write_preserves_last_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    runner.atomic_save({"step": 2, "model": {"weight": torch.ones(3)}}, path)
    runner.atomic_save({"step": 3, "model": {"weight": torch.full((3,), 2.)}}, path)
    assert not path.with_name("last.pt.tmp").exists()
    expected = torch.load(path, weights_only=True)
    assert expected["step"] == 3

    def incomplete_write(state, temporary):
        Path(temporary).write_bytes(b"incomplete checkpoint")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(runner.torch, "save", incomplete_write)
    with pytest.raises(OSError, match="interrupted write"):
        runner.atomic_save({"step": 4}, path)
    _assert_equal(torch.load(path, weights_only=True), expected)


class _ToyScenes:
    """Small deterministic inputs plus stochastic training context features.

    Validation consumes neither global torch randomness nor training sampler
    state. Training intentionally consumes both so an incomplete RNG restore
    changes the subsequent updates. All four score bands are represented.
    """

    def __init__(self, root, device):
        self.device = torch.device(device)
        self.levels = np.array([5., 150., 500., 900.])
        self.norm = SimpleNamespace(mean={ch: np.zeros(4) for ch in ("TEMP", "SALT")},
                                    std={ch: np.ones(4) for ch in ("TEMP", "SALT")})
        self.metadata = {name: "toy-fixed-inputs" for name in (
            "cohort_fingerprint", "feature_fingerprint", "scene_fingerprint")}
        self.fg_checkpoint = {"model": {"fixed": torch.tensor([0.])}}
        self.y = torch.randn(8, 4, 2, generator=torch.Generator().manual_seed(7)) * .2

    def analysis(self):
        return A.InnovationAnalysis(8, mode="kriging", k=3)

    def eval_months(self, split, max_cells):
        return [EvalMonth(192, np.arange(4), np.array([4, 5]),
                          np.repeat(np.arange(2), 4), np.tile(np.arange(4), 2))]

    def identity_check(self, split, evs):
        return {"toy": True}

    def geometry(self, source, rows, k):
        return {"row": torch.as_tensor(rows)[:, None]}

    def training_draw(self, rng, n_queries):
        return 192, np.arange(4), rng.choice([4, 5], n_queries), rng.choice(4, n_queries)

    def targets(self, rows, levels):
        target = self.y[torch.as_tensor(rows), torch.as_tensor(levels)]
        return target, torch.ones_like(target, dtype=torch.bool)

    def scene(self, source, rows, levels, analysis, context_profiles=4,
              context_seed=0, geometry=None):
        q = len(rows)
        training = isinstance(context_seed, np.random.Generator)
        rng = context_seed if training else np.random.default_rng(context_seed)
        obs = torch.as_tensor(rng.normal(size=(8, 62)), dtype=torch.float32)
        if training:
            obs = obs + .1 * torch.randn_like(obs)
        query = torch.zeros(q, 58)
        query[:, 0] = torch.as_tensor(rows, dtype=torch.float32) / 8
        query[:, 1] = torch.as_tensor(levels, dtype=torch.float32) / 4
        lev = torch.as_tensor(levels)
        field = torch.stack((lev, lev + 4), -1)
        baseline = .1 * analysis.log_gamma[field].sigmoid()
        coord = torch.zeros(q, 4)
        coord[:, 0] = torch.as_tensor(rows, dtype=torch.float32)
        coord[:, 2] = torch.as_tensor(self.levels[levels], dtype=torch.float32)
        obscoord = torch.zeros(8, 4)
        obscoord[:, 2] = torch.as_tensor(np.tile(self.levels, 2), dtype=torch.float32)
        return {"obs_features": obs, "obs_coord": obscoord,
                "obs_innovation": torch.linspace(-.3, .3, 16).reshape(8, 2),
                "obs_valid": torch.ones(8, 2, dtype=torch.bool),
                "satellite_features": obs[:2, :56], "satellite_coord": torch.zeros(2, 4),
                "query_features": query, "query_coord": coord,
                "baseline": baseline, "query_background": torch.zeros(q, 2),
                "local_features": torch.zeros(q, 3, 66),
                "local_offsets": torch.zeros(q, 3, 4),
                "local_innovation": torch.linspace(-.2, .4, 6).reshape(1, 3, 2).expand(q, -1, -1),
                "local_valid": torch.ones(q, 3, 2, dtype=torch.bool)}


def _invoke(monkeypatch, output, *extra):
    monkeypatch.setattr(runner.sys, "argv", [str(RUNNER_PATH), "--tag", "toy", "--output", str(output),
        "--device", "cpu", "--steps", "8", "--queries", "6", "--context-profiles", "4",
        "--width", "16", "--latents", "4", "--blocks", "1", "--query-blocks", "1",
        "--val-every", "4", "--eval-chunk", "8", "--checkpoint-every", "2", *extra])
    runner.main()


def test_actual_runner_resumes_identical_updates_optimizer_schedule_and_nested_best(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OceanScenes", _ToyScenes)
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    old_threads = torch.get_num_threads()
    try:
        _invoke(monkeypatch, uninterrupted)
        original_save = runner.atomic_save

        class Interrupted(Exception):
            pass

        def interrupt_after_complete_checkpoint(state, path):
            original_save(state, path)
            if Path(path).name == "last_seed1234.pt" and state["step"] == 4:
                raise Interrupted()

        monkeypatch.setattr(runner, "atomic_save", interrupt_after_complete_checkpoint)
        with pytest.raises(Interrupted):
            _invoke(monkeypatch, resumed)
        partial = torch.load(resumed / "last_seed1234.pt", weights_only=True)
        assert partial["step"] == 4
        assert partial["continuation"]["best"]["state"]["model"]
        monkeypatch.setattr(runner, "atomic_save", original_save)
        _invoke(monkeypatch, resumed, "--load-checkpoint", str(resumed / "last_seed1234.pt"),
                "--resume-training")
        expected = torch.load(uninterrupted / "last_seed1234.pt", weights_only=True)
        actual = torch.load(resumed / "last_seed1234.pt", weights_only=True)
        assert actual["step"] == expected["step"] == 8
        for name in ("model", "analysis", "anchor_analysis"):
            _assert_equal(actual[name], expected[name])
        for name in ("optimizer", "scheduler", "random", "training_loss", "initial"):
            _assert_equal(actual["continuation"][name], expected["continuation"][name])
        assert actual["continuation"]["scheduler"]["last_epoch"] == 8
        assert actual["continuation"]["optimizer"]["state"]
        for name in ("score", "step"):
            _assert_equal(actual["continuation"]["best"][name], expected["continuation"]["best"][name])
        for name in ("model", "analysis", "validation"):
            _assert_equal(actual["continuation"]["best"]["state"][name],
                          expected["continuation"]["best"]["state"][name])
        with np.load(resumed / "validation_seed1234.npz") as a, np.load(uninterrupted / "validation_seed1234.npz") as b:
            for name in a.files:
                assert np.array_equal(a[name], b[name]), name
    finally:
        torch.set_num_threads(old_threads)


def test_frozen_evaluation_keeps_training_history_and_uses_checkpoint_seed(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(runner, "OceanScenes", _ToyScenes)
    old_threads = torch.get_num_threads()
    try:
        _invoke(monkeypatch, tmp_path, "--seed", "1235")
        path = tmp_path / "summary_seed1235.json"
        training = json.loads(path.read_text())
        _invoke(monkeypatch, tmp_path, "--load-checkpoint", str(tmp_path / "best_seed1235.pt"),
                "--evaluate-only", "--development")
        evaluation = json.loads(path.read_text())
        assert not (tmp_path / "summary_seed1234.json").exists()
        for name in ("initial", "history", "training", "source_sha256", "best_step"):
            assert evaluation[name] == training[name], name
        assert evaluation["development"]
        assert evaluation["evaluation"]["development"]
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device for bf16 continuation replay")
def test_cuda_bf16_continuation_replays_actual_cuda_random_sampling(tmp_path, monkeypatch):
    import sys
    class CudaScenes(_ToyScenes):
        def analysis(self):
            return super().analysis().to(self.device)
        def targets(self, *args):
            return tuple(value.to(self.device) for value in super().targets(*args))
        def scene(self, *args, **kwargs):
            output = {name: value.to(self.device) for name, value in super().scene(*args, **kwargs).items()}
            seed = kwargs.get("context_seed", args[5] if len(args) > 5 else None)
            if isinstance(seed, np.random.Generator):
                output["obs_features"] = output["obs_features"]+.1*torch.randn_like(output["obs_features"])
            return output
    original = _invoke
    def invoke_cuda(patch, output, *extra):
        return original(patch, output, *extra, "--device", "cuda", "--amp")
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_ToyScenes", CudaScenes)
    monkeypatch.setattr(module, "_invoke", invoke_cuda)
    test_actual_runner_resumes_identical_updates_optimizer_schedule_and_nested_best(tmp_path, monkeypatch)
