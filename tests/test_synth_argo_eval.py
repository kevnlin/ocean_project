"""The synthetic audit's evaluation sets, on a toy cohort (no data store)."""
import json
import os

import numpy as np
import pytest

from ocean_tokenizer.argo_obs import ArgoCohort
from ocean_tokenizer.synth_argo_eval import (EVAL_SEED, REFERENCE, eval_month, eval_set,
                                             identity_check)

L = 4                    # levels
N_IN, N_Q = 3, 2         # inputs and queries a month in the toy cohort


def _cohort(months=((48, "validation"), (49, "validation"), (60, "development"))):
    per = N_IN + N_Q
    P = per * len(months)
    rng = np.random.default_rng(0)
    c = ArgoCohort(
        region="toy", levels=np.array([5.0, 50.0, 200.0, 500.0]),
        month_index=np.repeat([m for m, _ in months], per),
        grid_y=np.zeros(P, int), grid_x=np.zeros(P, int),
        lat=rng.uniform(-60, 60, P), lon=rng.uniform(0, 360, P),
        wmo=np.array([f"S{i:04d}" for i in range(P)]),
        year=np.repeat([2000 + m // 12 for m, _ in months], per),
        year_split=np.repeat([s for _, s in months], per),
        float_split=np.tile(["cohort_float"] * N_IN + ["heldout_float"] * N_Q,
                            len(months)),
        TEMP=rng.normal(size=(P, L)), SALT=rng.normal(size=(P, L)),
        TEMP_ERR=np.zeros((P, L)), SALT_ERR=np.zeros((P, L)))
    c._by_month = {m: (i * per, (i + 1) * per) for i, (m, _) in enumerate(months)}
    return c, {"TEMP": c.TEMP, "SALT": c.SALT}


def test_cells_are_profile_major_and_carry_the_query_values():
    c, obs = _cohort()
    ev = eval_month(c, obs, 48, n_input=N_IN, n_query=N_Q)
    assert list(ev["src"]) == [0, 1, 2] and list(ev["tgt"]) == [3, 4]
    assert list(ev["prof"]) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(ev["lev"]) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert np.array_equal(ev["target"]["TEMP"], c.TEMP[[3, 4]].ravel())
    assert np.array_equal(ev["target"]["SALT"], c.SALT[[3, 4]].ravel())


def test_the_cap_keeps_the_cells_the_training_script_keeps():
    """62's make_sample: default_rng([seed, month]).choice(R * L, cap, replace=False)."""
    c, obs = _cohort()
    ev = eval_month(c, obs, 49, max_cells=5, n_input=N_IN, n_query=N_Q)
    pick = np.random.default_rng([EVAL_SEED, 49]).choice(N_Q * L, 5, replace=False)
    assert list(ev["prof"]) == list(pick // L) and list(ev["lev"]) == list(pick % L)
    assert np.array_equal(ev["target"]["TEMP"], c.TEMP[ev["tgt"]].ravel()[pick])
    # a cap above the month's cells changes nothing
    assert eval_month(c, obs, 49, max_cells=99, n_input=N_IN, n_query=N_Q)["prof"].size == 8


def test_a_month_with_fewer_profiles_is_refused():
    c, obs = _cohort()
    with pytest.raises(SystemExit, match="input parity violated in month 48"):
        eval_month(c, obs, 48, n_input=N_IN + 1, n_query=N_Q)
    with pytest.raises(SystemExit, match="input parity violated"):
        eval_month(c, obs, 48)          # the real cohort's 6,080 / 1,520


def test_eval_set_walks_the_months_of_a_split_in_order():
    c, obs = _cohort()
    kw = dict(n_input=N_IN, n_query=N_Q)
    assert [e["month"] for e in eval_set(c, obs, "validation", **kw)] == [48, 49]
    assert [e["month"] for e in eval_set(c, obs, "development", **kw)] == [60]
    assert [e["month"] for e in eval_set(c, obs, "validation", n_months=1, **kw)] == [48]


def _reference(tmp_path, n, clim):
    p = os.path.join(tmp_path, REFERENCE)
    os.makedirs(os.path.dirname(p))
    json.dump({"scores": {"development": {ch: {"n": n, "climatology_z": clim}
                                          for ch in ("TEMP", "SALT")}}}, open(p, "w"))


def test_identity_check_accepts_the_models_fingerprint(tmp_path):
    _reference(tmp_path, 100, 1.5)
    zero = {ch: {"n": 100, "climatology_z": 1.5 * (1 + 4e-6)} for ch in ("TEMP", "SALT")}
    rep = identity_check(str(tmp_path), "development", zero)
    assert rep["TEMP"]["reference_n"] == 100


@pytest.mark.parametrize("n, clim", [(99, 1.5), (100, 1.5001)])
def test_identity_check_refuses_other_cells_or_another_scale(tmp_path, n, clim):
    _reference(tmp_path, 100, 1.5)
    zero = {ch: {"n": n, "climatology_z": clim} for ch in ("TEMP", "SALT")}
    with pytest.raises(SystemExit, match="identity check failed on development TEMP"):
        identity_check(str(tmp_path), "development", zero)


# ------------------------------------------------------------------ another cohort file
def _write_cohort(root, name):
    """A cohort file in the layout of 41_synth_argo_cohort.py: one profile a year."""
    import xarray as xr
    year = np.array([2000, 2001, 2004, 2005])
    P = year.size
    val = np.arange(P * L, dtype="float32").reshape(P, L)
    ds = xr.Dataset(
        {"TEMP": (("profile", "level"), val + 10.0), "SALT": (("profile", "level"), val + 30.0),
         "CLIM_POS_TEMP": (("profile", "level"), np.full((P, L), 10.0, "float32")),
         "CLIM_POS_SALT": (("profile", "level"), np.full((P, L), 30.0, "float32")),
         "CLIM_CELL_TEMP": (("profile", "level"), np.full((P, L), 9.0, "float32")),
         "CLIM_CELL_SALT": (("profile", "level"), np.full((P, L), 29.0, "float32")),
         "TEMP_ERR": (("profile", "level"), np.zeros((P, L), "float32")),
         "SALT_ERR": (("profile", "level"), np.zeros((P, L), "float32")),
         "lat": ("profile", np.zeros(P)), "lon": ("profile", np.arange(P, dtype=float)),
         "grid_y": ("profile", np.zeros(P, int)), "grid_x": ("profile", np.zeros(P, int)),
         "month_index": ("profile", (year - 2000) * 12), "year": ("profile", year),
         "wmo": ("profile", np.array([f"S{i}" for i in range(P)])),
         "year_split": ("profile", np.array(["train"] * P)),
         "float_split": ("profile", np.array(["cohort_float"] * P))},
        coords={"level": np.array([5.0, 50.0, 200.0, 500.0])},
        attrs={"region": "synthetic", "grid": [180, 360]})
    path = os.path.join(root, "data", "synthetic_argo")
    os.makedirs(path, exist_ok=True)
    ds.to_netcdf(os.path.join(path, f"{name}.nc"))
    return val


def test_load_reads_a_named_cohort_file(tmp_path):
    from ocean_tokenizer.synth_argo_eval import load
    val = _write_cohort(str(tmp_path), "cesm2_other")
    c, norm, obs = load(str(tmp_path), cohort="cesm2_other")
    # the at-position anomaly, split by the audit's years
    assert np.allclose(c.TEMP, val) and np.allclose(c.SALT, val)
    assert list(c.year_split) == ["train", "train", "validation", "development"]
    assert np.allclose(obs["TEMP"][0], (val[0] - val[:2].mean(0)) / val[:2].std(0))


def test_load_without_a_name_reads_the_audits_cohort(tmp_path):
    from ocean_tokenizer.synth_argo_eval import load
    _write_cohort(str(tmp_path), "cesm2_uniform")
    c, _, _ = load(str(tmp_path))
    assert c.TEMP.shape == (4, L)


def test_named_cohort_uses_the_requested_climatology(tmp_path):
    from ocean_tokenizer.audit_tools import load_cohort
    val = _write_cohort(str(tmp_path), "cesm2_other")
    c, _ = load_cohort(str(tmp_path), "synthetic", anomaly="cell", cohort="cesm2_other")
    assert np.allclose(c.TEMP, val + 1.0) and np.allclose(c.SALT, val + 1.0)


@pytest.mark.parametrize("region", ["global", "gulf_stream"])
def test_named_cohort_cannot_override_real_data(tmp_path, region):
    from ocean_tokenizer.audit_tools import cohort_path, load_cohort
    for loader in (cohort_path, load_cohort):
        with pytest.raises(ValueError, match="only for region='synthetic'"):
            loader(str(tmp_path), region, cohort="cesm2_other")


@pytest.mark.parametrize("name", ["", "../cesm2_other", "/cesm2_other",
                                   "subdir\\cesm2_other", "cesm2_other.nc"])
def test_named_cohort_requires_a_file_stem(tmp_path, name):
    from ocean_tokenizer.synth_argo_eval import load
    with pytest.raises(ValueError, match="without a path or .nc suffix"):
        load(str(tmp_path), cohort=name)
