"""Protect full budgets and validation-before-development in the new queue."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("innovation_campaign", ROOT / "experiments/synthetic/61_innovation_campaign.py")
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)


def test_all_hypotheses_have_three_seed_controls_and_full_budgets(tmp_path, monkeypatch):
    # Build the prerequisite registration from code, without local run artifacts.
    spec = importlib.util.spec_from_file_location(
        "matched_campaign_fixture", ROOT / "experiments/synthetic/51_matched_campaign.py")
    matched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matched)
    monkeypatch.setattr(matched, "OUTPUT", tmp_path / "matched")
    monkeypatch.setattr(C, "OLD", tmp_path / "matched")
    C.OLD.mkdir()
    (C.OLD / "campaign.json").write_text(json.dumps({"jobs": matched.registered_jobs()}))
    jobs = C.registered_jobs()
    assert len(jobs) == 63
    assert sum(j["reuse"] for j in jobs) == 3
    groups = {}
    for job in jobs:
        groups.setdefault((job["surface"], job["family"]), set()).add(job["seed"])
        assert job["steps"] == 15000
        assert "--development" not in job["command"]
        assert "--warm-start" not in job["command"]
    assert all(seeds == {1234, 1235, 1236} for seeds in groups.values())
    for control, candidate, _ in C.CONTRASTS:
        arms = [s for s in (False, True) if (s, candidate) in groups]
        assert arms and all((s, control) in groups for s in arms)


def test_partial_checkpoint_cannot_be_published_as_completed(tmp_path):
    job = {"output": str(tmp_path), "seed": 1234, "tag": "trial"}
    (tmp_path / "summary_seed1234.json").write_text(json.dumps({"seed": 1234, "completed_steps": 100}))
    with pytest.raises(ValueError, match="Incomplete formal"):
        C.completed(job)


def test_resumes_full_optimizer_state_from_last_not_selected_best(tmp_path):
    job = {"output": str(tmp_path), "seed": 1234, "command": ["python", "trainer"]}
    (tmp_path / "best_seed1234.pt").touch()
    assert C.training_command(job) == job["command"]
    (tmp_path / "last_seed1234.pt").touch()
    assert C.training_command(job)[-3:] == ["--resume-training", "--load-checkpoint", str(tmp_path / "last_seed1234.pt")]


def test_no_development_selection_before_all_trials_finish(tmp_path):
    manifest = {"jobs": [{"output": str(tmp_path), "seed": 1234, "tag": "trial"}]}
    with pytest.raises(ValueError, match="Finish every"):
        C.freeze_selection(tmp_path / "campaign.json", manifest)
    assert not (tmp_path / "selection.json").exists()
