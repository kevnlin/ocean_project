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
