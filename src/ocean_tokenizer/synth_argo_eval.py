"""The synthetic Argo audit's fixed evaluation sets, defined once.

Every method compared on the synthetic cohort has to be scored on the cells
``62_sanity_train.py --region synthetic --ablation anomaly_exact`` scores: the
same cohort, at-position target and train-year normalisation, the same months,
every input profile of the month, the same query profiles and the same capped
validation draw. The baseline drivers build their evaluation sets here, so two
of them cannot drift apart, and each proves it matches the trained models with
:func:`identity_check` before it writes a number.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .argo_obs import ArgoNorm
from .audit_tools import load_cohort
from .point_baselines import CH

SPLITS = {"train": (2000, 2003), "validation": (2004, 2004), "development": (2005, 2005)}
#: profiles a month: inputs (``cohort_float``) and queries (``heldout_float``)
N_INPUT, N_QUERY = 6080, 1520
EVAL_CELLS = 8000       # 62_sanity_train.py --eval-cells default (validation cap)
EVAL_SEED = 20260918    # 62's build_eval seeds the cap with [EVAL_SEED, month]
#: the trained model whose stored cell counts and climatology error a zero
#: prediction must reproduce
REFERENCE = os.path.join("outputs", "audit", "synthetic", "syn_r500_g1",
                         "summary_seed1234.json")


def load(root, *, cohort=None):
    """``(cohort, norm, obs)``: the at-position anomaly cohort under the audit's
    splits, its train-year normalisation, and ``{channel: (P, L)}`` z-scores.

    ``cohort`` is the synthetic file name without ``.nc``; the audit's
    ``cesm2_uniform`` remains the default.
    """
    c, _ = load_cohort(root, "synthetic", SPLITS, anomaly="exact", cohort=cohort)
    norm = ArgoNorm.fit(c, "train")
    return c, norm, {ch: norm.z(ch, getattr(c, ch)) for ch in CH}


def eval_month(c, obs, m, max_cells=0, n_input=N_INPUT, n_query=N_QUERY):
    """One month's inputs and scored cells, as build_eval / make_sample in 62.

    Cells are laid out profile-major (cell ``i`` is query profile ``i // L`` at
    level ``i % L``); with ``max_cells`` the month keeps the cells 62 keeps.
    Exits if the month does not have ``n_input`` inputs and ``n_query`` queries:
    no method may be handed fewer Argo profiles than another.
    """
    src = c.month(m, float_split="cohort_float")
    tgt = c.month(m, float_split="heldout_float")
    if (src.size, tgt.size) != (n_input, n_query):
        raise SystemExit(f"input parity violated in month {m}: {src.size} inputs, "
                         f"{tgt.size} queries (expected {n_input} and {n_query})")
    R, L = tgt.size, c.levels.size
    prof = np.repeat(np.arange(R), L)        # query profile of each cell
    lev = np.tile(np.arange(L), R)           # level of each cell
    if max_cells and R * L > max_cells:
        pick = np.random.default_rng([EVAL_SEED, int(m)]).choice(
            R * L, max_cells, replace=False)
        prof, lev = prof[pick], lev[pick]
    return dict(month=int(m), src=src, tgt=tgt, prof=prof, lev=lev,
                target={ch: obs[ch][tgt][prof, lev] for ch in CH})


def eval_set(c, obs, split, max_cells=0, n_months=None, **counts):
    """Every month of a split (or its first ``n_months``), in order."""
    months = [int(m) for m in c.months_in(split)]
    return [eval_month(c, obs, m, max_cells, **counts)
            for m in (months[:n_months] if n_months else months)]


def identity_check(root, split, zero_scores):
    """A zero prediction must reproduce the trained model's stored fingerprint.

    ``zero_scores`` is the ``Scores.result()`` of predicting 0 on the split.
    Returns the compared values; exits if the cell count differs or the
    climatology error is off by more than 1e-5 (the model's sums were
    accumulated in float32, hence a tolerance and not equality).
    """
    ref = json.load(open(os.path.join(root, REFERENCE)))["scores"][split]
    rep = {}
    for ch in CH:
        rep[ch] = {"n": zero_scores[ch]["n"],
                   "climatology_z": zero_scores[ch]["climatology_z"],
                   "reference_n": ref[ch]["n"],
                   "reference_climatology_z": ref[ch]["climatology_z"]}
        if (rep[ch]["n"] != rep[ch]["reference_n"]
                or abs(rep[ch]["climatology_z"] / rep[ch]["reference_climatology_z"]
                       - 1.0) > 1e-5):
            raise SystemExit(f"identity check failed on {split} {ch}: {rep[ch]}")
    return rep
