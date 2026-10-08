"""Protect the registered full-budget and validation-before-test protocol."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/51_matched_campaign.py"
SPEC = importlib.util.spec_from_file_location("matched_campaign", SCRIPT)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def test_incomplete_training_cannot_be_treated_as_completed(tmp_path):
    job = {"output": str(tmp_path), "seed": 1234, "steps": 15000, "tag": "example"}
    summary = tmp_path / "summary_seed1234.json"
    summary.write_text(json.dumps({"seed": 1234, "training": {"steps": 15000},
                                   "history": [{"step": 1000}]}))
    with pytest.raises(ValueError, match="Incomplete training"):
        CAMPAIGN.completed(job)
    summary.write_text(json.dumps({"seed": 1234, "completed_steps": 15000}))
    assert CAMPAIGN.completed(job)


def test_continuation_uses_optimizer_state_and_never_selected_best(tmp_path):
    base = {"output": str(tmp_path), "seed": 1234, "command": ["python", "runner.py"]}
    (tmp_path / "best_seed1234.pt").touch()
    assert CAMPAIGN.training_command({**base, "kind": "latent"}) == base["command"]
    (tmp_path / "last_seed1234.pt").touch()
    assert CAMPAIGN.training_command({**base, "kind": "latent"})[-3:] == [
        "--resume-training", "--load-checkpoint", str(tmp_path / "last_seed1234.pt")]
    (tmp_path / "last_training_seed1234.pt").touch()
    assert CAMPAIGN.training_command({**base, "kind": "previous"})[-1] == "--resume-training"


def test_training_commands_exclude_development_scoring():
    jobs = CAMPAIGN.registered_jobs()
    assert len(jobs) == 39
    assert len({job["tag"] for job in jobs}) == 39
    for job in jobs:
        assert "--development" not in job["command"]
        assert "--evaluate-only" not in job["command"]
        if job["kind"] != "latent":
            assert "--no-final-development" in job["command"]


def test_evaluation_cannot_start_before_every_registered_training_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(CAMPAIGN, "OUTPUT", tmp_path)
    monkeypatch.setattr(CAMPAIGN.sys, "argv", [str(SCRIPT), "--phase", "evaluate"])
    calls = []
    monkeypatch.setattr(CAMPAIGN.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    with pytest.raises(ValueError, match="All registered training"):
        CAMPAIGN.main()
    assert calls == []
    assert not (tmp_path / "selection.json").exists()


def test_registered_recipe_cannot_change_on_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(CAMPAIGN, "OUTPUT", tmp_path)
    monkeypatch.setattr(CAMPAIGN.sys, "argv", [str(SCRIPT), "--phase", "register"])
    CAMPAIGN.main()
    path = tmp_path / "campaign.json"
    manifest = json.loads(path.read_text())
    manifest["training_steps"] = 300
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Refusing to alter"):
        CAMPAIGN.main()
