"""The complete campaign freezes validation choices before development access."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "experiments/real_data/74_latent_full_campaign.py"
spec = importlib.util.spec_from_file_location("latent_full_campaign", PATH)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def job(variant="dense", seed=1234, analysis_only=False):
    prefix = "analysis_only" if analysis_only else f"{variant}_base"
    return {"tag": f"{prefix}_s{seed}", "variant": variant, "width": 128,
        "latents": 64, "blocks": 4, "context_profiles": 512, "experts": 4,
        "steps": 6000, "seed": seed, **({"analysis_only": True} if analysis_only else {})}


def selected_jobs():
    return [job(), job("soft_moe"), job("local_transformer"), job(analysis_only=True)]


def summaries(selected=None):
    selected = selected or selected_jobs()
    validation = {"dense_base_s1234": .3, "soft_moe_base_s1234": .4,
        "local_transformer_base_s1234": .5, "analysis_only_s1234": .01}
    development = {"dense_base_s1234": 10., "soft_moe_base_s1234": .0001,
        "local_transformer_base_s1234": .001, "analysis_only_s1234": 0.}
    return {j["tag"]: {"tag": j["tag"], "seed": j["seed"],
        "config": {"variant": j["variant"], "width": j["width"], "n_latents": j["latents"],
            "n_blocks": j["blocks"], "n_experts": j["experts"]},
        "training": {"steps": j["steps"], "context_profiles": j["context_profiles"],
            **{flag: j.get(flag, False) for flag in ("analysis_only", "freeze_analysis", "latent_off", "local_off")}},
        "validation": {"scores": {"macro_z": validation.get(j["tag"], .3)}},
        "development": {"scores": {"macro_z": development.get(j["tag"], 10.)}}}
        for j in selected}


def save_training_summary(root, current_job, development=False, array=False):
    folder = root / current_job["tag"]
    folder.mkdir(parents=True, exist_ok=True)
    value = summaries([current_job])[current_job["tag"]]
    if not development:
        value.pop("development")
    (folder / f"summary_seed{current_job['seed']}.json").write_text(json.dumps(value))
    (folder / f"best_seed{current_job['seed']}.pt").touch()
    if array:
        (folder / f"development_seed{current_job['seed']}.npz").touch()
    return folder


def test_component_controls_select_neural_validation_winner_not_development_or_oi():
    selected = selected_jobs()
    stored = summaries(selected)
    result = C.ablation_jobs(selected, stored)
    assert result["selected_neural"] == "dense_base_s1234"
    assert result["selection"] == "overall neural winner on 2021 validation only"
    assert "analysis_only" not in result["selected_neural"]
    # Winner has the worst development score; this must not affect selection.
    stored["soft_moe_base_s1234"]["development"]["scores"]["macro_z"] = -100.
    stored["analysis_only_s1234"]["validation"]["scores"]["macro_z"] = -1000.
    assert C.ablation_jobs(selected, stored) == result


def test_component_selection_works_without_any_development_results():
    selected = selected_jobs()
    stored = summaries(selected)
    for summary in stored.values():
        summary.pop("development")
    assert C.ablation_jobs(selected, stored)["selected_neural"] == "dense_base_s1234"


def test_unselected_summary_cannot_win_component_controls():
    stored = summaries()
    stored["unregistered_s1234"] = {"validation": {"scores": {"macro_z": -100.}}}
    assert C.ablation_jobs(selected_jobs(), stored)["selected_neural"] == "dense_base_s1234"


def test_three_component_controls_have_three_full_budget_seeds_and_correct_flags():
    selected = selected_jobs()
    result = C.ablation_jobs(selected, summaries())
    assert len(result["jobs"]) == 9
    assert len({j["tag"] for j in result["jobs"]}) == 9
    for name, latent_off, local_off in (("full", False, False), ("latent_off", True, False), ("local_off", False, True)):
        control = [j for j in result["jobs"] if f"_fixed_oi_{name}_" in j["tag"]]
        assert {j["seed"] for j in control} == {1234, 1235, 1236}
        for j in control:
            assert j["steps"] == 6000 and j["freeze_analysis"] is True
            assert j.get("latent_off", False) is latent_off
            assert j.get("local_off", False) is local_off
            assert not j.get("analysis_only", False)
            assert (j["variant"], j["width"], j["latents"], j["blocks"], j["context_profiles"]) == ("dense", 128, 64, 4, 512)


def test_control_tags_do_not_overwrite_search_or_replication():
    selected = selected_jobs()
    occupied = {j["tag"].rsplit("_s", 1)[0] + f"_s{seed}" for j in selected for seed in (1234, 1235, 1236)}
    occupied |= {j["tag"].replace("_base_", "_large_") for j in selected}
    controls = C.ablation_jobs(selected, summaries())["jobs"]
    assert not occupied & {j["tag"] for j in controls}


def test_finished_development_and_array_skip_subprocess(tmp_path, monkeypatch):
    current_job = job()
    save_training_summary(tmp_path, current_job, development=True, array=True)
    monkeypatch.setattr(C, "OUTPUT", tmp_path)
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("completed development must not launch"))
    updates = []
    C.evaluate_jobs([current_job], [0], updates.append)
    assert not updates


def test_missing_training_summary_refuses_development_before_any_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUTPUT", tmp_path)
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("incomplete training must not launch"))
    with pytest.raises(ValueError, match="training is incomplete"):
        C.evaluate_jobs([job()], [0], lambda _: None)


def test_one_missing_training_run_prevents_all_queued_evaluations(tmp_path, monkeypatch):
    save_training_summary(tmp_path, job())
    monkeypatch.setattr(C, "OUTPUT", tmp_path)
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("all training must finish before any evaluation"))
    with pytest.raises(ValueError, match="training is incomplete"):
        C.evaluate_jobs([job(), job("soft_moe")], [0], lambda _: None)


@pytest.mark.parametrize("section,field,value", [
    (None, "tag", "old_integration"), (None, "seed", 0),
    ("training", "steps", 5), ("training", "context_profiles", 1),
    ("config", "variant", "soft_moe"), ("config", "width", 64),
    ("config", "n_latents", 32), ("config", "n_blocks", 2), ("config", "n_experts", 8),
    ("training", "analysis_only", True), ("training", "freeze_analysis", True),
    ("training", "latent_off", True), ("training", "local_off", True),
])
def test_existing_summary_must_match_registered_training_even_if_dev_array_exists(tmp_path, monkeypatch, section, field, value):
    current_job = job()
    folder = save_training_summary(tmp_path, current_job, development=True, array=True)
    path = folder / "summary_seed1234.json"
    stored = json.loads(path.read_text())
    target = stored if section is None else stored[section]
    target[field] = value
    path.write_text(json.dumps(stored))
    monkeypatch.setattr(C, "OUTPUT", tmp_path)
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("mismatched training must not launch"))
    with pytest.raises(ValueError, match="differs from registered job"):
        C.evaluate_jobs([current_job], [0], lambda _: None)


@pytest.mark.parametrize("development,array", [(False, False), (True, False), (False, True)])
def test_incomplete_development_evaluates_saved_checkpoint_only(tmp_path, monkeypatch, development, array):
    current_job = job("soft_moe", seed=1235)
    folder = save_training_summary(tmp_path, current_job, development=development, array=array)
    monkeypatch.setattr(C, "OUTPUT", tmp_path)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    calls = []

    class CompleteProcess:
        pid = 42

        def poll(self):
            return 0

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return CompleteProcess()

    monkeypatch.setattr(C.subprocess, "Popen", popen)
    updates = []
    C.evaluate_jobs([current_job], [6], updates.append)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--evaluate-only" in command and "--development" in command
    assert command[command.index("--load-checkpoint") + 1] == str(folder / "best_seed1235.pt")
    assert command[command.index("--seed") + 1] == "1235"
    assert "--warm-start" not in command and "--resume-training" not in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert kwargs["env"]["OCEAN_ROOT"] == str(tmp_path)
    assert kwargs["stdout"].closed
    assert updates == [{"running": {}, "pending": []}]


def test_failed_development_subprocess_is_not_reported_complete(tmp_path, monkeypatch):
    save_training_summary(tmp_path, job())
    monkeypatch.setattr(C, "OUTPUT", tmp_path)

    class FailedProcess:
        pid = 42

        def poll(self):
            return 7

    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: FailedProcess())
    with pytest.raises(RuntimeError, match="development evaluation failed.*7"):
        C.evaluate_jobs([job()], [0], lambda _: None)


def setup_pipeline(tmp_path, monkeypatch):
    scripts = tmp_path / "experiments/real_data"
    scripts.mkdir(parents=True)
    for number in (70, 71, 72, 73, 76):
        (scripts / f"{number}_fake.py").touch()
    output = tmp_path / "outputs/latent_ocean"
    output.mkdir(parents=True)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "OUTPUT", output)
    monkeypatch.setattr(sys, "argv", ["74_latent_full_campaign", "--skip-prepare", "--gpus", "0,1"])
    return output


@pytest.mark.parametrize("selection_ready", [True, False])
def test_pipeline_finishes_all_training_before_21_frozen_evaluations_and_reports(tmp_path, monkeypatch, capsys, selection_ready):
    output = setup_pipeline(tmp_path, monkeypatch)
    selected = selected_jobs()
    events = []

    def run(command, **kwargs):
        number = Path(command[2]).name.split("_", 1)[0]
        phase = command[command.index("--phase") + 1] if "--phase" in command else number
        if "--register-only" in command:
            phase = "register_rule"
        events.append(phase)
        assert kwargs["check"] is True
        if "--freeze-selection" in command:
            Path(command[command.index("--freeze-selection") + 1]).write_text(json.dumps({"selected": selected}))
        if phase == "replicate":
            for current_job in selected:
                for seed in (1234, 1235, 1236):
                    this_job = {**current_job, "tag": current_job["tag"].rsplit("_s", 1)[0] + f"_s{seed}", "seed": seed}
                    save_training_summary(output, this_job)
        if phase == "ablate":
            manifest = json.loads(Path(command[command.index("--manifest") + 1]).read_text())
            assert len(manifest["jobs"]) == 9
            for current_job in manifest["jobs"]:
                assert current_job["steps"] == 6000
                save_training_summary(output, current_job)
        if phase == "register_rule":
            (output/"final_selection_rule_20261007.json").write_text("{}")
        if phase == "76" and selection_ready:
            (output/"final_selection_20261007.json").write_text("{}")

    def evaluate(current_jobs, gpus, callback):
        assert events == ["search", "71", "replicate", "ablate", "register_rule", "76"]
        assert (output/"final_selection_20261007.json").exists()
        assert len(current_jobs) == 21 and len({j["tag"] for j in current_jobs}) == 21
        assert all(j["steps"] == 6000 for j in current_jobs)
        assert gpus == [0, 1]
        for current_job in current_jobs:
            assert (output / current_job["tag"] / f"summary_seed{current_job['seed']}.json").exists()
        events.append("evaluate")
        callback({"running": {}, "pending": []})

    monkeypatch.setattr(C.subprocess, "run", run)
    monkeypatch.setattr(C, "evaluate_jobs", evaluate)
    if not selection_ready:
        with pytest.raises(ValueError, match="recommendation is not frozen"):
            C.main()
        assert "evaluate" not in events
        status = json.loads((output/"full_campaign_status_20261007.json").read_text())
        assert status["phase"] == "final_validation_selection"
        return
    C.main()
    assert events == ["search", "71", "replicate", "ablate", "register_rule", "76", "evaluate", "71", "72"]
    status = json.loads((output / "full_campaign_status_20261007.json").read_text())
    assert status["phase"] == "complete"
    assert status["training_runs"] == 24 and status["development_evaluations"] == 21
    assert not status.get("failure")
    # Manifest and selected paths are pathlib.Path arguments in main(). They
    # must become strings before JSON logging as well as before subprocess.run.
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    subprocess_records = [record for record in records if "arguments" in record]
    assert len(subprocess_records) == 8
    assert all(isinstance(argument, str) for record in subprocess_records for argument in record["arguments"])
    assert str(output / "campaign_20261007.json") in subprocess_records[0]["arguments"]
    assert str(output / "selected_20261007.json") in subprocess_records[1]["arguments"]


def test_training_failure_blocks_development_and_persists_failure_phase(tmp_path, monkeypatch):
    output = setup_pipeline(tmp_path, monkeypatch)

    def fail_training(command, **kwargs):
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(C.subprocess, "run", fail_training)
    monkeypatch.setattr(C, "evaluate_jobs", lambda *a, **k: pytest.fail("failed training must block development"))
    with pytest.raises(subprocess.CalledProcessError):
        C.main()
    status = json.loads((output / "full_campaign_status_20261007.json").read_text())
    assert status["phase"] == "search" and "CalledProcessError" in status["failure"]
