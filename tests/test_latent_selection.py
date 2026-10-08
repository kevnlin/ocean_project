"""Final recommendation must use complete, matched validation ensembles only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/real_data/76_latent_selection.py"
spec = importlib.util.spec_from_file_location("latent_final_selection", SCRIPT)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def arrays(error=0.):
    target = np.column_stack((np.linspace(-1., 1., 24), np.linspace(.2, 1.8, 24))).astype(np.float32)
    return {"mean": target + error, "std": np.ones_like(target), "target": target,
            "baseline": np.zeros_like(target), "month": np.repeat(np.arange(252, 264), 2),
            "profile": np.arange(24), "level": np.tile([0, 1], 12)}


def fixture_campaign(tmp_path, errors=None):
    output = tmp_path / "outputs"
    selected = []
    for variant in ("dense", "soft_moe", "local_transformer", "analysis_only"):
        oi = variant == "analysis_only"
        selected.append({"tag": f"{variant}_base_s1234", "variant": "dense" if oi else variant,
                         "seed": 1234, "steps": 6000, "width": 64 if oi else 128,
                         "latents": 32 if oi else 64, "blocks": 2 if oi else 4,
                         "context_profiles": 1 if oi else 512, "experts": 4, "analysis_only": oi})
    controls = []
    for name in ("full", "latent_off", "local_off"):
        for seed in selection.SEEDS:
            controls.append({**selected[0], "seed": seed, "freeze_analysis": True,
                             "latent_off": name == "latent_off", "local_off": name == "local_off",
                             "tag": f"dense_base_fixed_oi_{name}_s{seed}"})
    selected_path, ablation_path = tmp_path / "selected.json", tmp_path / "ablations.json"
    selected_path.write_text(json.dumps({"selected": selected}))
    ablation_path.write_text(json.dumps({"selected_neural": selected[0]["tag"], "jobs": controls}))
    groups = selection.candidate_jobs({"selected": selected}, {"selected_neural": selected[0]["tag"], "jobs": controls})
    for gi, (prefix, jobs) in enumerate(groups):
        for job in jobs:
            folder = output / job["tag"]
            folder.mkdir(parents=True)
            error = (errors or {}).get(prefix, .1 + .1 * gi)
            if isinstance(error, tuple):
                error = error[selection.SEEDS.index(job["seed"])]
            prediction = arrays(error)
            score = selection.report.z_scores(prediction["mean"], prediction["target"])
            config = {"variant": job["variant"], "width": job["width"], "n_latents": job["latents"],
                      "n_blocks": job["blocks"], "n_experts": job["experts"],
                      "use_latent": not job.get("latent_off", False), "use_local": not job.get("local_off", False)}
            training = {"steps": 6000, "queries": 1024, "context_profiles": job["context_profiles"],
                        "eval_context_profiles": None, "val_cells": 8000, "val_every": 500, "eval_chunk": 2048,
                        "lr": .0003, "analysis_lr": .001, "weight_decay": .01, "nll_weight": .02, "amp": True,
                        **{flag: bool(job.get(flag)) for flag in selection.FLAGS}}
            saved = {"tag": job["tag"], "seed": job["seed"], "config": config, "training": training,
                     "data": {key: key + "-fixed" for key in selection.FINGERPRINTS},
                     "source_sha256": {"training.py": "a" * 64}, "params": 0 if job.get("analysis_only") else 100,
                     "best_step": 500, "history": [{"step": 6000}], "validation": {"scores": score},
                     # Deliberately make validation's loser appear spectacular on the other split.
                     "development": {"scores": {"macro_z": 0. if gi == 1 else 100.}}}
            (folder / f"summary_seed{job['seed']}.json").write_text(json.dumps(saved))
            np.savez(folder / f"validation_seed{job['seed']}.npz", **prediction)
            np.savez(folder / f"development_seed{job['seed']}.npz", **arrays(0. if gi == 1 else 100.))
            (folder / f"best_seed{job['seed']}.pt").write_bytes(b"test-checkpoint-" + str(job["seed"]).encode())
    record_path = tmp_path / "rule.json"
    record = selection.register_rule(record_path)
    return selected_path, ablation_path, output, record, record_path


def change_summary(output, tag, mutate):
    seed = int(tag.rsplit("_s", 1)[1])
    path = output / tag / f"summary_seed{seed}.json"
    stored = json.loads(path.read_text())
    mutate(stored)
    path.write_text(json.dumps(stored))


def test_development_champion_cannot_influence_validation_choice(tmp_path, monkeypatch):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    original = np.load
    paths = []

    def validation_only(path, *args, **kwargs):
        paths.append(Path(path))
        assert Path(path).name.startswith("validation_seed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", validation_only)
    result = selection.build_selection(selected, ablations, output, record)
    assert result["selectedtag"] == "dense_base"
    assert result["scores"]["macro_z"] == pytest.approx(.1, abs=1e-7)
    assert len(result["candidates"]) == 7
    assert paths and all(p.name.startswith("validation_seed") for p in paths)
    assert len(result["inputarray_sha256"]) == 21
    assert len(result["checkpoints"]) == 3
    assert result["seeds"] == [1234, 1235, 1236]
    json.dumps(result, allow_nan=False)


def test_ensemble_mean_is_scored_not_mean_of_seed_errors(tmp_path):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path, {"dense_base": (1., -1., 0.)})
    result = selection.build_selection(selected, ablations, output, record)
    assert result["selectedtag"] == "dense_base"
    assert result["scores"]["macro_z"] < 1e-7


def test_local_only_winner_is_not_forced_to_keep_latent(tmp_path):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path, {"dense_base_fixed_oi_latent_off": .001})
    result = selection.build_selection(selected, ablations, output, record)
    assert result["selectedtag"] == "dense_base_fixed_oi_latent_off"
    assert "no shared latent" in result["selected_architecture"]


def test_missing_seed_refuses_selection_and_does_not_write_choice(tmp_path, monkeypatch, capsys):
    selected, ablations, output, _, record_path = fixture_campaign(tmp_path)
    (output / "dense_base_s1236" / "summary_seed1236.json").unlink()
    destination = tmp_path / "final.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--selected-manifest", str(selected),
                                     "--ablation-manifest", str(ablations), "--output-root", str(output),
                                     "--rule-record", str(record_path), "--output", str(destination)])
    selection.main()
    assert not destination.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "pending" and printed["chosen"] is None


@pytest.mark.parametrize("key", selection.report.IDENTITY_KEYS)
def test_every_scoring_identity_mismatch_refuses_selection(tmp_path, key):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    path = output / "dense_base_s1235" / "validation_seed1235.npz"
    bad = arrays(.1)
    bad[key].flat[0] += 1
    np.savez(path, **bad)
    with pytest.raises(ValueError, match=key):
        selection.build_selection(selected, ablations, output, record)


@pytest.mark.parametrize("key", selection.FINGERPRINTS)
def test_input_fingerprint_mismatch_refuses_selection(tmp_path, key):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    change_summary(output, "dense_base_s1235", lambda d: d["data"].update({key: "wrong"}))
    with pytest.raises(ValueError, match="fingerprints"):
        selection.build_selection(selected, ablations, output, record)


def test_nominal_budget_cannot_disguise_unfinished_training(tmp_path):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    change_summary(output, "dense_base_s1235", lambda d: d.update(history=[{"step": 5000}]))
    with pytest.raises(selection.PendingSelection, match="6000"):
        selection.build_selection(selected, ablations, output, record)


def test_accidental_control_flags_refuse_selection(tmp_path):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    change_summary(output, "dense_base_s1235", lambda d: d["training"].update(latent_off=True))
    with pytest.raises(ValueError, match="latent_off"):
        selection.build_selection(selected, ablations, output, record)


def test_actual_arrays_must_match_saved_validation_score(tmp_path):
    selected, ablations, output, record, _ = fixture_campaign(tmp_path)
    path = output / "dense_base_s1235" / "validation_seed1235.npz"
    np.savez(path, **arrays(.9))
    with pytest.raises(ValueError, match="saved score"):
        selection.build_selection(selected, ablations, output, record)


def test_rule_is_immutable_after_registration(tmp_path):
    _, _, _, _, record_path = fixture_campaign(tmp_path)
    altered = json.loads(record_path.read_text())
    altered["rule"]["criterion"] = "select by something else"
    record_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="changed"):
        selection.register_rule(record_path)
