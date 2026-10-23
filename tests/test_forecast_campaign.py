"""The forecast campaign's run names by length, the report's choice of runs, and the training-curve reader."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/synthetic" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = script("95_forecast_campaign")
report = script("96_forecast_report")
curves = script("97_forecast_curves")


def test_runs_of_another_length_get_their_own_folders(tmp_path):
    default = campaign.job(4, 1234, tmp_path)
    longer = campaign.job(4, 1234, tmp_path, steps=100000)
    assert default["tag"] == "f_y4_s1234" and longer["tag"] == "k100_f_y4_s1234"
    assert Path(longer["output"]) == tmp_path / "k100_f_y4_s1234"
    command = longer["command"]
    assert command[command.index("--steps") + 1] == "100000"
    assert command[command.index("--tag") + 1] == "k100_f_y4_s1234"


def summary(folder, seed, steps):
    folder.mkdir()
    (folder / f"summary_seed{seed}.json").write_text(json.dumps(
        {"seed": seed, "completed_steps": steps, "forecast": {"best": None, "last": None}}))


def test_report_reads_only_the_runs_of_its_length(tmp_path):
    summary(tmp_path / "f_y4_s1234", 1234, 70000)
    summary(tmp_path / "k100_f_y4_s1234", 1234, 100000)
    summary(tmp_path / "k100_f_y40_s1235", 1235, 100000)
    seeds = lambda found: {years: [s["seed"] for s in runs] for years, runs in found.items()}
    assert seeds(report.runs(tmp_path)) == {4: [1234]}
    assert seeds(report.runs(tmp_path, steps=100000)) == {4: [1234], 40: [1235]}
    assert report.report_path(70000).name == "forecast_20261019"
    assert report.report_path(100000).name == "forecast_20261019_k100"
    assert "100,000 steps" in report.markdown({}, {}, steps=100000)


LOG = """{"tag": "k100_f_y4_s1234", "config": {}}
initial validation physical RMSE {"TEMP": 0.4, "SALT": 0.07}
step 1 train_mse 2.0 elapsed 1.0s
step 100 train_mse 1.0 elapsed 9.0s
validation {"step": 100, "training_mse": 1.2, "validation_mean_standardized_rmse": 0.9} *
validation {"step": 200, "training_mse": 0.5, "validation_mean_standardized_rmse": 0.97}
Traceback (most recent call last):
resumed complete training state at step 150; best step 100
step 200 train_mse 0.25 elapsed 30.0s
validation {"step": 200, "training_mse": 0.4, "validation_mean_standardized_rmse": 0.95}
"""


def test_curve_reader_takes_interval_losses_and_the_last_record_of_a_step(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LOG)
    train, validation = curves.parse(path)
    assert train == {100: 1.2, 200: 0.4}
    assert validation == {100: 0.9, 200: 0.95}


def test_training_loss_change_over_the_last_tenth():
    train = {step: 2.0 if step <= 90 else 1.0 for step in range(10, 101, 10)}
    assert curves.tail_change(train, 100) == -50.0
    assert curves.tail_change({step: 1.0 for step in range(10, 51, 10)}, 100) is None


def test_curve_runs_are_found_by_name_with_their_length(tmp_path):
    for name in ("f_y4_s1234", "k100_f_y40_s1235", "smoke"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "train.log").write_text("")
    (tmp_path / "f_y4_s1236").mkdir()
    found = [(run["years"], run["seed"], run["steps"]) for run in curves.runs(tmp_path)]
    assert found == [(4, 1234, 70000), (40, 1235, 100000)]


def test_campaign_selects_on_2004_and_holds_2005_out(tmp_path):
    command = campaign.job(4, 1234, tmp_path)["command"]
    years = lambda flag: command[command.index(flag) + 1:command.index(flag) + 2]
    assert years("--val-years") == ["2004"] and years("--test-years") == ["2005"]
    assert command[command.index("--val-years") + 2] == "--test-years"


def test_completion_rejects_a_run_selected_on_other_years(tmp_path):
    item = campaign.job(4, 1234, tmp_path)
    Path(item["output"]).mkdir()
    (Path(item["output"]) / "summary_seed1234.json").write_text(json.dumps(
        {"completed_steps": 70000, "seed": 1234, "train_years": 4, "forecast": {},
         "training": {"val_years": [2004, 2005]}}))
    import pytest
    with pytest.raises(ValueError):
        campaign.completed(item)


def test_climatology_is_the_rmse_of_a_zero_anomaly_by_lead_and_year(tmp_path):
    import numpy as np
    path = tmp_path / "forecast_seed1234.npz"
    # level 0 has scale 2 and offset 1: physical targets are 2 * z + 1
    np.savez(path, target=np.array([[0., 0.], [1., 1.], [np.nan, 1.]]), lead=np.array([0, 0, 3]),
             month=np.array([50, 62, 50]), level=np.array([0, 0, 0]),
             normalization_scale=np.array([[2., 2.]]), normalization_offset=np.array([[1., 1.]]))
    clim = report.climatology(path)
    assert clim[0]["TEMP"] == np.sqrt((1 ** 2 + 3 ** 2) / 2)
    assert clim[0]["by_year"]["2004"]["TEMP"] == 1.0 and clim[0]["by_year"]["2005"]["TEMP"] == 3.0
    assert clim[3]["SALT"] == 3.0 and np.isnan(clim[3]["TEMP"])


def test_report_warns_when_the_checkpoint_was_chosen_on_a_scored_year():
    table = {4: {"seeds": [1234], "best_step": [8000], "hours": (1.0, None),
                 0: {ch: (0.1, None) for ch in report.CH} | {f"oi_{ch}": 0.09 for ch in report.CH}
                    | {f"clim_{ch}": 0.7 for ch in report.CH} | {"by_year": {}}}}
    text = report.markdown(table, {}, selection=[2004, 2005])
    assert "Climatology" in text and "not a held-out result" in text
    assert "not a held-out result" not in report.markdown(table, {}, selection=[2004])


def test_report_is_named_after_its_output_folder(tmp_path):
    held_out = tmp_path / "forecast_20261021"
    assert report.report_path(70000, held_out).name == "forecast_20261021"
    assert report.report_path(100000, held_out).name == "forecast_20261021_k100"
