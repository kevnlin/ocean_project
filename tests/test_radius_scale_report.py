"""Scores of the radius / local-scale report, on hand-made arrays."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/synthetic/83_radius_scale_report.py"
SPEC = importlib.util.spec_from_file_location("radius_scale_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)

LEVEL_STD = {"TEMP": np.array([1., 2.]), "SALT": np.array([.1, .2])}


def arrays():
    target = np.zeros((4, 2))
    mean = np.array([[1., 1.], [1., 1.], [3., 3.], [3., 3.]])
    return {"mean": mean, "std": np.full((4, 2), 2.), "target": target,
            "level": np.array([0, 0, 1, 1])}


def test_rmse_is_pooled_in_physical_units():
    m = REPORT.run_metrics(arrays(), np.ones((4, 2)), LEVEL_STD)
    # errors 1, 1, 3, 3 in z; times the level's std 1, 1, 2, 2 -> 1, 1, 6, 6
    assert m["TEMP"]["rmse"] == pytest.approx(np.sqrt((1 + 1 + 36 + 36) / 4))
    assert m["SALT"]["rmse"] == pytest.approx(np.sqrt((.01 + .01 + .36 + .36) / 4))
    assert m["TEMP"]["n"] == 4


def test_local_rmse_divides_each_error_by_its_cells_scale():
    scale = np.array([[1., 1.], [1., 1.], [3., 3.], [3., 3.]])
    m = REPORT.run_metrics(arrays(), scale, LEVEL_STD)
    assert m["TEMP"]["rmse_local"] == pytest.approx(1.)      # 1/1, 1/1, 3/3, 3/3


def test_quiet_half_and_energetic_tenth_are_chosen_by_the_scale():
    a = {"mean": np.arange(1., 21.)[:, None].repeat(2, 1), "std": np.ones((20, 2)),
         "target": np.zeros((20, 2)), "level": np.zeros(20, int)}
    scale = np.arange(1., 21.)[:, None].repeat(2, 1)
    m = REPORT.run_metrics(a, scale, LEVEL_STD)
    assert m["TEMP"]["rmse_quiet_half"] == pytest.approx(np.sqrt(np.mean(np.arange(1., 11.) ** 2)))
    assert m["TEMP"]["rmse_energetic_tenth"] == pytest.approx(np.sqrt(np.mean(np.array([19., 20.]) ** 2)))


def test_missing_targets_are_left_out_per_variable():
    a = arrays()
    a["target"][2:, 0] = np.nan
    m = REPORT.run_metrics(a, np.ones((4, 2)), LEVEL_STD)
    assert m["TEMP"]["n"] == 2 and m["SALT"]["n"] == 4
    assert m["TEMP"]["rmse"] == pytest.approx(1.)


def test_coverage_is_the_share_of_errors_inside_one_sigma():
    a = arrays()                        # errors 1, 1, 3, 3 against sigma 2
    assert REPORT.run_metrics(a, np.ones((4, 2)), LEVEL_STD)["TEMP"]["coverage_68"] == .5
    a["std"] = None                     # OI has no predicted spread
    assert REPORT.run_metrics(a, np.ones((4, 2)), LEVEL_STD)["TEMP"]["coverage_68"] is None


def test_seeds_are_summarised_by_mean_and_spread():
    runs = [{"TEMP": {"rmse": 1.}}, {"TEMP": {"rmse": 3.}}]
    mean, sd, n = REPORT.over_seeds(runs, "TEMP", "rmse")
    assert (mean, n) == (2., 2) and sd == pytest.approx(np.std([1., 3.], ddof=1))
    assert REPORT.over_seeds(runs[:1], "TEMP", "rmse") == (1., None, 1)
    assert REPORT.over_seeds([{"TEMP": {"rmse": None}}], "TEMP", "rmse") == (None, None, 0)


def test_cells_show_the_spread_only_when_there_is_one():
    assert REPORT.cell((0.07842, 0.00031, 3), 4) == "0.0784 ± 0.0003"
    assert REPORT.cell((0.07842, None, 1), 4) == "0.0784"
    assert REPORT.cell((None, None, 0), 4) == "—"


def test_arms_are_described_in_words():
    assert REPORT.describe("r500") == "radius 500 km"
    assert REPORT.describe("qres_month_lossz") == "OI-error scale, cell × month, loss in z units"
    assert REPORT.describe("r500+qanom_cell") == "radius 500 km; anomaly scale, cell"
    assert REPORT.describe("res_month") == "OI-error scale, cell × month, each profile by its own entry (first design)"


def test_a_run_folder_names_its_arm_inputs_and_seed():
    assert REPORT.parse_folder("r750_argo_s1234") == ("r750", "argo", 1234)
    assert REPORT.parse_folder("res_month_lossz_surface_s1236") == ("res_month_lossz", "surface", 1236)
    assert REPORT.parse_folder("r500_qres_month_argo_s1235") == ("r500+qres_month", "argo", 1235)
    assert REPORT.parse_folder("l64_qres_cell_surface_s1234") == ("l64+qres_cell", "surface", 1234)
    assert REPORT.parse_folder("r500_l64_qres_cell_argo_s1234") == ("r500+l64+qres_cell", "argo", 1234)
    assert REPORT.describe("l64+qres_cell") == "64 latent slots; OI-error scale, cell"
    assert REPORT.parse_folder("controls") is None


def test_the_published_reference_is_read_from_the_committed_table():
    reference = REPORT.published_reference()
    assert reference["seeds"] == 3
    assert reference["TEMP"][0] == pytest.approx(0.0784, abs=1e-4)
    assert reference["SALT"][0] == pytest.approx(0.0200, abs=1e-4)
    assert 0 < reference["TEMP"][1] < 0.001


def test_the_control_trained_here_is_the_reference_row():
    assert REPORT.describe("control") == "no option (the reference model)"
    assert REPORT.ORDER[:2] == ("oi", "control")
