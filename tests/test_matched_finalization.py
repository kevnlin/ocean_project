"""Independent delivery checks: full budget, every family, and deferred test use."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "experiments/synthetic" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_script("57_finalize_matched_campaign.py")
campaign = load_script("51_matched_campaign.py")


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "OUTPUT", tmp_path)
    jobs = campaign.registered_jobs()
    manifest = {"jobs": jobs}
    frozen = {}
    for job in jobs:
        folder = Path(job["output"]); folder.mkdir()
        summary_path = folder / f"summary_seed{job['seed']}.json"
        summary_path.write_text(json.dumps({"seed": job["seed"], "training": {"steps": 15000},
                                            "history": [{"step": 1000}, {"step": 15000}]}))
        frozen[job["tag"]] = {"summary_path": str(summary_path), "training_contract": {
            "seed": job["seed"], "training": {"steps": 15000}}}
    stats = {channel: {"n": 351895, "rmse": .1} for channel in ("TEMP", "SALT")}
    groups = {(bool(job["surface"]), job["family"]): job["kind"] for job in jobs}
    rows = [{"family": family, "mode": "surface" if surface else "argo", "kind": kind,
             "seeds": [1234, 1235, 1236],
             "seed_metrics": [{"seed": seed, "physical_metrics": copy.deepcopy(stats)}
                              for seed in (1234, 1235, 1236)],
             "ensemble": {"physical_metrics": copy.deepcopy(stats)}}
            for (surface, family), kind in groups.items()]
    rows += [{"family": family, "mode": "argo", "kind": "classical",
              "physical_metrics": copy.deepcopy(stats)}
             for family in ("climatology", "nearest_profile", "oi", "fixed_pointwise_mlp", "prior_learned_oi")]
    metrics = {"test_n_per_variable": {"TEMP": 351895, "SALT": 351895},
               "selection": {"runs": frozen,
                   "selected": {"argo": {"family": "prior_learned_oi"},
                                "surface": {"family": "prior_learned_oi"}}},
               "rows": rows}
    return manifest, metrics


def test_complete_registered_report_is_accepted(artifacts):
    guard.verify(artifacts[1], artifacts[0])


@pytest.mark.parametrize("family", ["prior_learned_oi", "fixed_pointwise_mlp"])
def test_required_strong_and_fixed_baselines_cannot_be_omitted(artifacts, family):
    manifest, metrics = artifacts
    metrics["rows"] = [row for row in metrics["rows"] if row["family"] != family]
    with pytest.raises(ValueError):
        guard.verify(metrics, manifest)


def test_incomplete_scoring_support_is_rejected(artifacts):
    manifest, metrics = artifacts
    metrics["test_n_per_variable"]["SALT"] -= 1
    with pytest.raises(ValueError):
        guard.verify(metrics, manifest)


def test_one_missing_frozen_run_is_rejected(artifacts):
    manifest, metrics = artifacts
    metrics["selection"]["runs"].pop(manifest["jobs"][0]["tag"])
    with pytest.raises(ValueError):
        guard.verify(metrics, manifest)


@pytest.mark.parametrize("damage", ["truncated_manifest", "missing_family", "short_budget"])
def test_39_run_proof_cannot_hide_missing_families_or_budget(artifacts, damage):
    manifest, metrics = artifacts
    if damage == "truncated_manifest":
        manifest["jobs"] = manifest["jobs"][:-3]
        tags = {job["tag"] for job in manifest["jobs"]}
        metrics["selection"]["runs"] = {tag: run for tag, run in metrics["selection"]["runs"].items() if tag in tags}
    elif damage == "missing_family":
        metrics["rows"] = [row for row in metrics["rows"]
                           if not (row["family"] == "official4dvarnet" and row["mode"] == "surface")]
    else:
        first = manifest["jobs"][0]["tag"]
        metrics["selection"]["runs"][first]["training_contract"]["training"]["steps"] = 14000
    with pytest.raises(ValueError):
        guard.verify(metrics, manifest)


def test_evaluation_readiness_requires_every_array_and_fixed_mlp(artifacts, tmp_path, monkeypatch):
    manifest, _ = artifacts
    monkeypatch.setattr(guard, "OUTPUT", tmp_path)
    for job in manifest["jobs"][:-1]:
        folder = Path(job["output"])
        (folder / f"development_seed{job['seed']}.npz").touch()
    assert not guard.evaluated(manifest)
    last = manifest["jobs"][-1]
    folder = Path(last["output"])
    (folder / f"development_predictions_seed{last['seed']}.npz").touch()
    assert not guard.evaluated(manifest)
    fixed = tmp_path / "fixed_pointwise_mlp"; fixed.mkdir()
    (fixed / "development_seed1234.npz").touch()
    assert guard.evaluated(manifest)


def test_nominal_15000_steps_do_not_disguise_short_46_history(artifacts):
    manifest, metrics = artifacts
    job = next(job for job in manifest["jobs"] if job["kind"] == "latent")
    frozen = metrics["selection"]["runs"][job["tag"]]
    assert "completed_steps" not in frozen["training_contract"]
    path = Path(frozen["summary_path"])
    summary = json.loads(path.read_text())
    summary["history"] = [{"step": 1000}, {"step": 14000}]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        guard.verify(metrics, manifest)


def test_finalization_does_not_inspect_evaluation_before_selection(artifacts, tmp_path, monkeypatch):
    manifest, _ = artifacts
    monkeypatch.setattr(guard, "OUTPUT", tmp_path)
    (tmp_path / "campaign.json").write_text(json.dumps(manifest))
    (tmp_path / "status.json").write_text(json.dumps({"phase": "train"}))
    monkeypatch.setattr(guard, "evaluated", lambda _: pytest.fail("development readiness checked before frozen selection"))
    class StopWaiting(Exception):
        pass
    def stop(_):
        raise StopWaiting
    monkeypatch.setattr(guard.time, "sleep", stop)
    with pytest.raises(StopWaiting):
        guard.main()
    assert not (tmp_path / "completion.json").exists()


def test_pending_report_exit_zero_cannot_generate_completion(artifacts, tmp_path, monkeypatch):
    manifest, _ = artifacts
    monkeypatch.setattr(guard, "OUTPUT", tmp_path)
    monkeypatch.setattr(guard, "REPORT", tmp_path / "missing.md")
    (tmp_path / "campaign.json").write_text(json.dumps(manifest))
    (tmp_path / "status.json").write_text(json.dumps({"phase": "complete"}))
    (tmp_path / "selection.json").write_text("{}")
    monkeypatch.setattr(guard, "evaluated", lambda _: True)
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="still pending"):
        guard.main()
    assert not (tmp_path / "completion.json").exists()
    assert json.loads((tmp_path / "finalization_status.json").read_text())["phase"] == "failed"
