"""The FuXi comparison: the queue's runs, the transfer scorer's refusals, the report's verdicts and labels."""
import importlib.util
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
    data = SimpleNamespace(grid_lat=np.arange(-89.5, 90.), grid_lon=np.arange(.5, 360.), x=torch.zeros(1, 56))
    checkpoint = {"config": {"grid": [-89.5, 1., .5, 1., 180, 360], "n_query_features": 58}, "training": {"tag": "t"}}
    transfer.compatible(checkpoint, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "grid": [-89., 2., 1., 2., 90, 180]}}, data)
    with pytest.raises(SystemExit, match="query features"):
        transfer.compatible({**checkpoint, "config": {**checkpoint["config"], "n_query_features": 60}}, data)


def test_the_score_is_101s_mean_over_variables_of_the_z_rmse():
    data = SimpleNamespace(levels=np.asarray(LEVELS), norm=SimpleNamespace(std={"TEMP": np.ones(20), "SALT": np.ones(20)}))
    target = np.array([[1., 2.], [1., 2.], [1., 2.], [1., np.nan]])
    collected = {0: {"mean": np.zeros((4, 2)), "target": target, "level": np.array([0, 1, 2, 3])}}
    assert transfer.macro_z(collected, data) == {"0": pytest.approx(1.5)}


def entry(tag, oi=.5, params=0):
    by_lead = {lead: {"all": {ch: {"model": {"rmse": .3}, "persisted_oi": {"rmse": oi}, "climatology": {"rmse": .8}}
                              for ch in ("TEMP", "SALT")}} for lead in ("0", "1", "3", "6")}
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
