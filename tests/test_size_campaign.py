"""The size study: the queue's runs, the curves' reader and figure, the report's decision."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = script("169_size_campaign")


def after(command, flag, n=1):
    i = command.index(flag)
    return command[i + 1:i + 1 + n]


def test_three_sizes_scale_every_channel_count():
    c = {arm: campaign.job(arm, 1234)["command"] for arm in campaign.FACTORS}
    assert after(c["y100_x1"], "--encoder-channels", 4) == ["64", "128", "192", "256"]
    assert after(c["y100_x1375"], "--state-channels") == ["176"] and after(c["y100_x1375"], "--hidden") == ["352"]
    assert after(c["y100_x175"], "--encoder-channels", 4) == ["112", "224", "336", "448"]
    assert after(c["y100_x175"], "--evolution-channels", 3) == ["112", "224", "336"]


def test_seeds_are_outermost():
    tags = [j["tag"] for j in campaign.jobs(list(campaign.FACTORS), [1234, 1235])]
    assert tags == ["y100_x1_s1234", "y100_x1375_s1234", "y100_x175_s1234",
                    "y100_x1_s1235", "y100_x1375_s1235", "y100_x175_s1235"]


def test_runs_train_100_years_for_200k_steps_chosen_on_2004_with_2005_held_out():
    command = campaign.job("y100_x1", 1234)["command"]
    assert after(command, "--cohort") == ["cesm2_vlong_uniform"] and after(command, "--train-years") == ["100"]
    assert after(command, "--steps") == ["200000"] and after(command, "--leads", 4) == ["0", "1", "3", "6"]
    assert after(command, "--val-years", 2) == ["2004", "--test-years"] and after(command, "--test-years") == ["2005"]
    assert after(command, "--kriging") == ["ka_flow"] and "--surface" in command
    assert after(command, "--keep-every") == ["5000"] and after(command, "--checkpoint-every") == ["500"]


def test_a_smoke_run_is_short_and_kept_apart():
    item = campaign.job("y100_x175", 1234, smoke=True)
    assert item["steps"] == 600 and Path(item["output"]).parent.name == "smoke"
    assert after(item["command"], "--val-every") == ["300"]
