"""The FuXi comparison: the queue's runs, the transfer scorer's refusals, the report's verdicts and labels."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(command):
    """{flag: its values} of a runner command (``python -u runner ...``)."""
    out, flag = {}, None
    for word in command[3:]:
        if word.startswith("--"):
            flag = word
            out[flag] = []
        else:
            out[flag].append(word)
    return out


campaign = script("172_fuxi_campaign")
size = script("169_size_campaign")
MODEL = {"--backbone", "--kriging", "--encoder-channels", "--evolution-channels", "--keep-every", "--tag", "--output"}


def test_the_fuxi_runs_train_exactly_as_ka_x1375_but_for_the_model():
    fuxi, ka = options(campaign.job(1234)["command"]), options(size.job("y100_x1375", 1234)["command"])
    assert {k: v for k, v in fuxi.items() if k not in MODEL} == {k: v for k, v in ka.items() if k not in MODEL}
    assert fuxi["--backbone"] == ["fuxi"] and fuxi["--kriging"] == ["none"] and ka["--kriging"] == ["ka_flow"]
    assert fuxi["--state-channels"] == ["176"] and fuxi["--hidden"] == ["352"] and "--keep-every" not in fuxi
    assert fuxi["--steps"] == ["200000"] and fuxi["--train-years"] == ["100"] and fuxi["--test-years"] == ["2005"]


def test_a_smoke_run_is_short_and_kept_apart():
    item = campaign.job(1234, smoke=True)
    assert item["steps"] == 600 and Path(item["output"]).parent.name == "smoke"
    assert options(item["command"])["--val-every"] == ["300"] and item["tag"] == "y100_fuxi_s1234"


transfer = script("173_glorys_transfer")
LEVELS = [5, 15, 25, 35, 45, 55, 65, 85, 105, 125, 145, 165.1, 186.3, 222.6, 267.7, 326.9, 408.8, 527.7, 707.6, 984.7]


def test_the_transfer_scenes_are_159s():
    assert transfer.COHORT == "glorys12_test" and tuple(transfer.TEST_YEARS) == (2006, 2007)
    assert transfer.GLORYS_TRAIN_YEARS == 20 and "damped_persistence_factors" not in transfer.GLORYS
    assert transfer.MAIN.parent.name == "sealed" and transfer.MAIN.name == "results.json"


def test_a_checkpoint_on_another_grid_or_feature_set_is_refused():
    data = SimpleNamespace(grid_lat=np.arange(-89.5, 90.), grid_lon=np.arange(.5, 360.), x=torch.zeros(1, 56),
                         levels=np.asarray(LEVELS))
    checkpoint = {"config": {"grid": [-89.5, 1., .5, 1., 180, 360], "n_query_features": 58}, "training": {"tag": "t"},
                  "data": {"levels_m": LEVELS}}
    transfer.compatible(checkpoint, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "grid": [-89., 2., 1., 2., 90, 180]}}, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "n_query_features": 60}}, data)
    with pytest.raises(SystemExit, match="levels"):
        transfer.compatible({**checkpoint, "data": {"levels_m": LEVELS[:19] + [1000.]}}, data)


def test_the_score_is_101s_mean_over_variables_of_the_z_rmse():
    data = SimpleNamespace(levels=np.asarray(LEVELS), norm=SimpleNamespace(std={"TEMP": np.ones(20), "SALT": np.ones(20)}))
    target = np.array([[1., 2.], [1., 2.], [1., 2.], [1., np.nan]])
    collected = {0: {"mean": np.zeros((4, 2)), "target": target, "level": np.array([0, 1, 2, 3])}}
    assert transfer.macro_z(collected, data) == {"0": pytest.approx(1.5)}


def entry(tag, oi=.5, params=0):
    by_lead = {lead: {"all": {ch: {"model": {"rmse": .3}, "persisted_oi": {"rmse": oi}, "climatology": {"rmse": .8}}
                              for ch in ("TEMP", "SALT")}} for lead in ("0", "1", "3", "6")}
    by_lead = {lead: {**v, "cells": 729600} for lead, v in by_lead.items()}
    return {"tag": tag, "params": params, "by_lead": by_lead, "macro_z": {lead: .5 for lead in ("0", "1", "3", "6")}}


def test_every_entry_must_have_the_main_passs_baselines():
    main = {"entries": [entry("g_ka_s1234")]}
    transfer.same_baselines([entry("y100_fuxi_s1234")], main)
    with pytest.raises(SystemExit, match="differs from the main pass"):
        transfer.same_baselines([entry("y100_fuxi_s1234", oi=.5001)], main)


def test_registration_needs_the_main_pass_and_cesm2_trained_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "REGISTRATION", tmp_path / "registration.json")
    monkeypatch.setattr(transfer, "MAIN", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="main pass"):
        transfer.register([])
    main = tmp_path / "results.json"
    main.write_text("{}")
    monkeypatch.setattr(transfer, "MAIN", main)
    with pytest.raises(SystemExit, match="nothing to register"):
        transfer.register([])
    checkpoint = tmp_path / "best_seed1234.pt"
    torch.save({"training": {"cohort": "glorys12_uniform"}}, checkpoint)
    with pytest.raises(SystemExit, match="CESM2-trained"):
        transfer.register([str(checkpoint)])
    assert not (tmp_path / "registration.json").exists()


def test_the_transfer_is_scored_once(tmp_path, monkeypatch):
    done = tmp_path / "results.json"
    done.write_text("{}")
    monkeypatch.setattr(transfer, "RESULTS", done)
    with pytest.raises(SystemExit, match="scored once"):
        transfer.score("cpu")


def test_scoring_refuses_before_writing_and_records_every_entry(tmp_path, monkeypatch):
    registration, main = tmp_path / "registration.json", tmp_path / "main.json"
    registration.write_text(json.dumps({"entries": [{"family": "grid", "checkpoint": t} for t in ("a", "b")]}))
    main.write_text(json.dumps({"entries": [entry("g_ka_s1234")]}))
    for name, value in (("REGISTRATION", registration), ("RESULTS", tmp_path / "results.json"), ("MAIN", main)):
        monkeypatch.setattr(transfer, name, value)
    norm = SimpleNamespace(mean={"TEMP": np.zeros(20), "SALT": np.zeros(20)}, std={"TEMP": np.ones(20), "SALT": np.ones(20)})
    monkeypatch.setattr(transfer.SS, "verify", lambda record: None)
    monkeypatch.setattr(transfer, "glorys_scenes", lambda device, cache: cache.setdefault("glorys", SimpleNamespace(norm=norm)))
    monkeypatch.setattr(transfer, "load", lambda path: {})
    monkeypatch.setattr(transfer, "compatible", lambda checkpoint, data: None)
    ones = {k: {ch: torch.zeros(20) for ch in ("TEMP", "SALT")} for k in ("mean", "std")}

    def evaluate(path, oi):
        checkpoint = {"training": {"tag": f"y100_fuxi_{path}", "cohort": "cesm2_vlong_uniform"}, "seed": 1234, "step": 9,
                      "config": {"backbone": "fuxi", "kriging": "none"}, "normalization": ones}
        e = entry(checkpoint["training"]["tag"], oi=oi)
        return checkpoint, e["by_lead"], e["macro_z"], 17_508_596

    monkeypatch.setattr(transfer, "evaluate", lambda path, device, cache: evaluate(path, .5 if path == "a" else .6))
    with pytest.raises(SystemExit, match="differs from the main pass"):
        transfer.score("cpu")
    assert not (tmp_path / "results.json").exists() and not (tmp_path / "results.tmp").exists()
    monkeypatch.setattr(transfer, "evaluate", lambda path, device, cache: evaluate(path, .5))
    transfer.score("cpu")
    results = json.loads((tmp_path / "results.json").read_text())
    assert [e["params"] for e in results["entries"]] == [17_508_596] * 2 and "declared_extra_pass" in results


report = script("174_fuxi_transfer_report")


def test_a_verdict_needs_a_difference_beyond_the_larger_seed_sd():
    assert report.verdict([.70, .71, .69], [.75, .76, .74])["verdict"] == "KA better"
    assert report.verdict([.75, .76, .74], [.70, .71, .69])["verdict"] == "FuXi-style better"
    assert report.verdict([.70, .72, .68], [.71, .73, .69])["verdict"] == "no difference beyond the seed spread"


def test_the_tables_name_each_model_with_its_size(monkeypatch):
    monkeypatch.setattr(report, "g_ka_params", lambda entries: 9_422_894)
    results = {"entries": [entry("y100_x1375_s1234", params=17_730_974), entry("y100_fuxi_s1234", params=17_508_596)]}
    glorys = report.transfer(results, {"entries": [entry("g_ka_s1234")]})
    text = report.markdown({}, glorys)
    assert "KA-Ocean (17.7M)" in text and "FuXi-Ocean-style (17.5M)" in text and "GLORYS-trained) (9.4M)" in text


def test_the_report_can_run_before_anything_finished():
    text = report.markdown({}, None)
    assert "Waits until both models have all three seeds" in text and "Waits for `173_glorys_transfer.py --score`" in text
