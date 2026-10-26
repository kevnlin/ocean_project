"""Runner 101 on resume: the inputs a checkpoint was trained on, and the provenance of a fine-tuned run."""
import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = script("101_grid_ka_forecast")


def training(**changes):
    record = {"surface": True, "satellite_cache": "/c/satellites.npz", "cohort": "cesm2_long_uniform",
              "train_years": 40, "scale": None, "leads": [0, 1, 3, 6], "val_years": [2004, 2005],
              "lagged_surface": "/c/lagged.npz", "surface_grids": "/c/grids.npz"}
    return {**record, **changes}


def test_a_cesm2_run_reads_the_inputs_it_was_trained_on():
    args = argparse.Namespace(satellite_cache=None)
    runner.restore_training_inputs(args, training())
    assert args.satellite_cache == Path("/c/satellites.npz") and args.surface_grids == Path("/c/grids.npz")
    assert args.cohort == "cesm2_long_uniform" and args.train_years == 40 and args.val_years == [2004, 2005]
    passed = argparse.Namespace(satellite_cache=Path("/elsewhere.npz"))
    runner.restore_training_inputs(passed, training())
    assert passed.satellite_cache == Path("/elsewhere.npz")


def test_a_real_run_without_a_satellite_cache_resumes():
    args = argparse.Namespace(satellite_cache=None)
    runner.restore_training_inputs(args, training(satellite_cache=None, lagged_surface=None, val_years=[2021]))
    assert args.satellite_cache is None and args.lagged_surface is None and args.val_years == [2021]


def checkpoint(source="/p/gj_flow_s1234/best_seed1234.pt"):
    data = {"cohort_fingerprint": "f"}
    if source is not None:
        data["init_from"] = {"path": source, "step": 61000, "sha256": "ab"}
    return {"data": data}


def test_a_resumed_fine_tune_keeps_its_source_whether_or_not_the_queue_repeats_it():
    resumed = lambda init_from: argparse.Namespace(resume_training=True, init_from=init_from)
    recorded = checkpoint()["data"]["init_from"]
    assert runner.resumed_provenance(resumed(None), checkpoint()) == recorded
    assert runner.resumed_provenance(resumed(Path(recorded["path"])), checkpoint()) == recorded
    assert runner.resumed_provenance(resumed(None), checkpoint(None)) is None


def test_init_from_must_name_the_source_of_the_run_it_resumes():
    with pytest.raises(ValueError):
        runner.resumed_provenance(argparse.Namespace(resume_training=True, init_from=Path("/other.pt")), checkpoint())
    with pytest.raises(ValueError):
        runner.resumed_provenance(argparse.Namespace(resume_training=True, init_from=Path("/p.pt")), checkpoint(None))


def test_init_from_still_excludes_evaluating_or_warm_starting_a_checkpoint():
    with pytest.raises(ValueError):
        runner.resumed_provenance(argparse.Namespace(resume_training=False, init_from=Path("/p.pt")), checkpoint())


def test_a_run_from_before_test_years_restores_none():
    args = argparse.Namespace(satellite_cache=None)
    runner.restore_training_inputs(args, training())
    assert args.test_years == []
    runner.restore_training_inputs(args, training(val_years=[2004], test_years=[2005]))
    assert args.val_years == [2004] and args.test_years == [2005]


def test_the_final_scoring_covers_validation_and_test_years_and_refuses_an_overlap():
    assert runner.scored_years([2004], [2005]) == [2004, 2005]
    assert runner.scored_years([2004, 2005], []) == [2004, 2005]
    with pytest.raises(ValueError):
        runner.scored_years([2004, 2005], [2005])


def test_test_years_default_to_none_and_snapshots_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["101", "--tag", "t"])
    args = runner.arguments()
    assert args.test_years == [] and args.val_years == [2004, 2005] and args.keep_every == 0


def test_the_runner_takes_the_fuxi_backbone_and_hashes_its_source(monkeypatch):
    monkeypatch.setattr("sys.argv", ["101", "--tag", "t", "--backbone", "fuxi", "--kriging", "none"])
    args = runner.arguments()
    assert args.backbone == "fuxi" and args.kriging == "none"
    assert 'ROOT/"src/ocean_tokenizer/fuxi_backbone.py"' in Path(runner.__file__).read_text()
