"""Real-data forecast scenes: months by split, evaluation months by lead, year -> split."""
import numpy as np

from ocean_tokenizer.real_forecast_scenes import RealGridForecastScenes
from ocean_tokenizer.real_global import REAL_SPLITS, real_eval_month
from test_real_global import fake_cohort


def scenes(leads=(0, 1, 3)):
    data = RealGridForecastScenes.__new__(RealGridForecastScenes)
    c = fake_cohort()
    c.year_split = np.where(c.month_index < 214, "train", "validation")
    data.c, data.leads, data.n_levels = c, tuple(leads), 3
    data.obs = {"TEMP": c.TEMP.copy(), "SALT": c.SALT.copy()}
    data.months_available = set(int(m) for m in np.unique(c.month_index))
    data._months_cache = {}
    data.splits = dict(REAL_SPLITS)                  # what __init__ sets without a train_split
    return data


def test_training_months_are_the_months_of_the_split_with_inputs_and_queries():
    data = scenes()
    assert data.months("train") == [200, 201]        # both have an input and a held-out float
    assert data.months("validation") == []           # 214 has only a held-out profile


def test_every_lead_scores_the_target_months_cells_from_its_own_input_month():
    data = scenes(leads=(0, 1))
    data.c.year_split[:] = "validation"
    evs = data.forecast_months([2016], max_cells=2)
    by = {(ev.month, ev.lead): ev for ev in evs}
    assert set(by) == {(200, 0), (201, 0), (201, 1)}           # 199 would be before the record
    reference = real_eval_month(data.c, data.obs, 201, 2, lead=1)
    assert np.array_equal(by[201, 1].source, reference["src"])
    assert np.array_equal(by[201, 1].profile, by[201, 0].profile) and np.array_equal(by[201, 1].level, by[201, 0].level)


def test_years_map_to_the_real_splits():
    data = scenes()
    assert [data.split_of_year(y) for y in (2016, 2020, 2021, 2022, 2023, 2024)] == \
        ["train", "train", "validation", "development", "development", None]


def test_a_shorter_training_window_keeps_the_last_years_of_the_training_split():
    data = scenes()
    c = data.c
    c.month_index = np.array([228, 228, 229, 229, 240, 240])            # 2019-01, 2019-02, 2020-01
    c.year = 2000 + c.month_index // 12
    c.year_split[:] = "train"
    c._by_month = {228: (0, 2), 229: (2, 4), 240: (4, 6)}
    data.months_available = {228, 229, 240}
    for years, expected in ((1, [240]), (2, [228, 229, 240]), (None, [228, 229, 240])):
        data.train_years, data._months_cache = years, {}
        assert data.months("train") == expected, years
