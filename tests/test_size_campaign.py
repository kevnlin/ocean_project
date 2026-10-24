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


curves = script("171_size_curves")


def log(folder, records):
    folder.mkdir(parents=True)
    (folder / "train.log").write_text("".join(
        'validation {"step": %d, "training_mse": %r, "validation_mean_standardized_rmse": %r}\n' % r for r in records))


def test_curve_runs_are_the_size_studys_by_name(tmp_path):
    for name in ("y100_x1_s1234", "y100_x175_s1235", "smoke", "c_y154_s1234"):
        log(tmp_path / name, [(2000, 1.0, .9)])
    assert [(r["arm"], r["seed"]) for r in curves.runs(tmp_path)] == [("y100_x1", 1234), ("y100_x175", 1235)]


def test_a_best_step_in_the_last_tenth_is_still_improving(tmp_path):
    steps = range(2000, 200001, 2000)
    log(tmp_path / "y100_x1_s1234", [(s, 1.0, 1 - s / 1e6) for s in steps])                 # best at 200k
    log(tmp_path / "y100_x1375_s1234", [(s, 1.0, abs(s - 100000) / 1e6) for s in steps])   # best at 100k
    rows = {r["tag"]: r for r in curves.rows(curves.runs(tmp_path))}
    assert rows["y100_x1_s1234"]["best_step"] == 200000 and rows["y100_x1_s1234"]["still_improving"]
    assert rows["y100_x1375_s1234"]["best_step"] == 100000 and not rows["y100_x1375_s1234"]["still_improving"]
    assert rows["y100_x1_s1234"]["tail_change_pct"] == 0.0


def test_the_figure_is_drawn_from_unfinished_runs_too(tmp_path):
    log(tmp_path / "y100_x175_s1234", [(s, 2.0 / s ** .1, .8 + 1 / s ** .5) for s in range(2000, 40001, 2000)])
    path = tmp_path / "curves.png"
    curves.plot(curves.runs(tmp_path), path)
    assert path.stat().st_size > 10000


import json
import pytest

report = script("170_size_report")


def fake_summary(macro_by_year, seed=1234):
    year = lambda m: {"scores": {"macro_z": m, "TEMP": {"rmse_physical": m}, "SALT": {"rmse_physical": m / 10}},
                      "baseline_scores": {"TEMP": {"rmse_physical": 1.}, "SALT": {"rmse_physical": .1}}}
    by_lead = {lead: {"by_year": {y: year(m) for y, m in macro_by_year.items()}} for lead in ("0", "1", "3", "6")}
    return {"seed": seed, "completed_steps": 200000, "forecast": {"best": {"by_lead": by_lead}}}


def test_the_score_is_the_mean_over_leads_of_a_year():
    assert report.score(fake_summary({"2004": .8, "2005": .7}), "2005") == pytest.approx(.7)


def test_bigger_helps_only_beyond_the_seed_spread():
    scores = {"y100_x1": {"2004": [.80, .81, .79], "2005": [.80, .81, .79]},
              "y100_x1375": {"2004": [.78, .78, .78], "2005": [.78, .78, .78]},
              "y100_x175": {"2004": [.77, .78, .76], "2005": [.77, .78, .76]}}
    assert report.decide(scores)["bigger_helps"]                       # 0.03 lower, spread 0.01
    scores["y100_x175"]["2005"] = [.795, .80, .79]
    assert not report.decide(scores)["bigger_helps"]                   # 0.005 lower, within the spread


def test_the_best_size_is_chosen_on_2004_not_on_the_held_out_year():
    scores = {"y100_x1": {"2004": [.80], "2005": [.70]}, "y100_x1375": {"2004": [.75], "2005": [.79]},
              "y100_x175": {"2004": [.78], "2005": [.71]}}
    assert report.decide(scores)["best"] == "y100_x1375"


def test_report_reads_only_finished_runs(tmp_path):
    for name, steps in (("y100_x1_s1234", 200000), ("y100_x175_s1234", 150000)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary_seed1234.json").write_text(json.dumps({**fake_summary({"2004": .8}), "completed_steps": steps}))
    assert list(report.runs(tmp_path)) == ["y100_x1"]


def test_the_report_can_run_before_anything_finished():
    text = report.markdown({}, {}, None, {})
    assert "Waits until every size has all three seeds" in text and "by lead" not in text
