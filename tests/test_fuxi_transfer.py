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
