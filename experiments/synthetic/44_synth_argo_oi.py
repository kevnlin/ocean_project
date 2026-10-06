"""Fixed baselines on the synthetic Argo cohort: climatology, nearest profile, OI.

The reference the synthetic audit lacked. Methods with no trainable parameters
are scored on exactly the cells the trained models are scored on
(`62_sanity_train.py --region synthetic --ablation anomaly_exact`): the same
cohort, at-position target and train-year normalisation, the same months, the
same 6,080 input profiles and 1,520 query profiles a month, the same 8,000-cell
validation draw, the same pooled metrics.

  climatology       zero anomaly (the J = 1 floor)
  nearest_profile   the nearest input profile's value at the same level
  oi                optimal interpolation (ocean_tokenizer.oi), level by level,
                    with (L_km, gamma, k) selected per variable and depth band
                    on the VALIDATION year; the test year is scored once
  oi_single         the same with one setting per variable (no band split)

Two guards run before any number is written:

* input parity -- every evaluated month hands every method all 6,080 input
  profiles and scores 1,520 query profiles, so no method sees fewer Argo
  profiles than another;
* identity -- a zero prediction reproduces the cell count and climatology
  error stored in a trained model's summary, so the cells and the
  normalisation are the model's.

Writes outputs/audit/synthetic/fixed_baselines/{summary,tuning}.json.

  .venv/bin/python experiments/synthetic/44_synth_argo_oi.py           # a few minutes, CPU
  .venv/bin/python experiments/synthetic/44_synth_argo_oi.py --smoke   # 1 month, prints only
"""
from __future__ import annotations

import os

# one BLAS thread per process: the work is many small solves spread over months
# by a process pool, and nested threading only oversubscribes a shared host
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, itertools, json, subprocess, sys, time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np

from ocean_tokenizer import synth_argo_eval
from ocean_tokenizer.audit_tools import cohort_path
from ocean_tokenizer.point_baselines import (BANDS, CH, Scores, band_of_levels,
                                             nearest_profile, oi_points, point_sweep)
from ocean_tokenizer.synth_argo_eval import (EVAL_CELLS, N_INPUT, N_QUERY, REFERENCE,
                                             SPLITS, identity_check)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AXES = ("L_km", "gamma", "k")
GRID = {"L_km": [150.0, 250.0, 400.0, 600.0, 900.0, 1500.0],
        "gamma": [0.01, 0.03, 0.1, 0.3],
        "k": [10, 20, 40]}
#: edge rule: when a selection sits on an end of an axis, the next value of
#: this ladder is added on that side and the selection is repeated
LADDER = {"L_km": {"lo": [100.0, 60.0], "hi": [2500.0, 4000.0]},
          "gamma": {"lo": [0.003, 0.001], "hi": [1.0, 3.0]},
          "k": {"lo": [5], "hi": [80]}}
BAND_NAMES = [b for b, _, _ in BANDS]

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=12, help="processes, one month each")
ap.add_argument("--out", default=None,
                help="default: outputs/audit/synthetic/fixed_baselines")
ap.add_argument("--smoke", action="store_true",
                help="one validation month, a 2 x 2 x 1 grid, no edge rule, "
                     "no identity check; prints and writes nothing")
args = ap.parse_args()
OUT = args.out or os.path.join(ROOT, "outputs", "audit", "synthetic", "fixed_baselines")
t0 = time.time()

# ------------------------------------------------------------------ data
# the cohort, its normalisation and the fixed evaluation sets are the audit's own
# (ocean_tokenizer.synth_argo_eval), shared with the other baseline drivers
c, norm, OBS = synth_argo_eval.load(ROOT)              # OBS: (P, L) z-scored anomaly
LEV = c.levels
BAND_OF = band_of_levels(LEV)                          # (L,) band name per level


def eval_set(split, max_cells=0, n_months=None):
    return synth_argo_eval.eval_set(c, OBS, split, max_cells, n_months)


# ------------------------------------------------------------------ methods
def predict(ev, method, setting=None):
    """{channel: (R, L)} analysis at the month's query profiles.

    Every method is given ALL of the month's input profiles. ``setting`` is,
    for OI, {channel: {band: (L_km, gamma, k)}}.
    """
    src, tgt = ev["src"], ev["tgt"]
    out = {}
    for ch in CH:
        p = np.zeros((tgt.size, LEV.size))
        if method != "climatology":
            for l in range(LEV.size):
                a = (c.lat[src], c.lon[src], OBS[ch][src, l], c.lat[tgt], c.lon[tgt])
                if method == "nearest_profile":
                    p[:, l] = nearest_profile(*a)
                else:
                    L_km, gamma, k = setting[ch][BAND_OF[l]]
                    p[:, l] = oi_points(*a, L_km=L_km, gamma=gamma, k=k)
        out[ch] = p
    return out


def _score_month(job):
    """Worker: one month's predictions at its scored cells."""
    ev, method, setting = job
    pred = predict(ev, method, setting)
    return {ch: pred[ch][ev["prof"], ev["lev"]] for ch in CH}


def score(evs, method, setting=None):
    s = Scores(LEV, norm.std)
    cells = POOL.map(_score_month, [(ev, method, setting) for ev in evs])
    for ev, cell in zip(evs, cells):
        for ch in CH:
            s.add(ch, cell[ch], ev["target"][ch], ev["lev"])
    return s.result()


# ------------------------------------------------------------------ tuning
def _sweep_month(job):
    """Worker: one month's squared error of every setting.

    Returns (n_settings, n_channels, n_bands). The k-NN geometry of a level is
    built once at the largest k and sliced for the smaller ones.
    """
    ev, combos = job
    kmax = max(k for _, _, k in combos)
    src, tgt = ev["src"], ev["tgt"]
    sse = np.zeros((len(combos), len(CH), len(BAND_NAMES)))
    for ci, ch in enumerate(CH):
        for l in range(LEV.size):
            cells = ev["lev"] == l
            rows, t = ev["prof"][cells], ev["target"][ch][cells]
            ok = np.isfinite(t)
            rows, t = rows[ok], t[ok]
            if rows.size == 0:
                continue
            # the analysis is needed only at the query profiles scored here
            need, inv = np.unique(rows, return_inverse=True)
            sweep, v = point_sweep(c.lat[src], c.lon[src], OBS[ch][src, l],
                                   c.lat[tgt][need], c.lon[tgt][need], kmax)
            subs = {}
            b = BAND_NAMES.index(BAND_OF[l])
            for i, (L_km, gamma, k) in enumerate(combos):
                if k not in subs:
                    subs[k] = sweep.sub_k(k)
                z = subs[k].analyse(v, L_km, gamma)[inv]
                sse[i, ci, b] += float(((z - t) ** 2).sum())
    return sse


def band_counts(evs):
    n = np.zeros((len(CH), len(BAND_NAMES)), dtype=int)
    for ev in evs:
        band = BAND_OF[ev["lev"]]
        for ci, ch in enumerate(CH):
            ok = np.isfinite(ev["target"][ch])
            for bi, b in enumerate(BAND_NAMES):
                n[ci, bi] += int((ok & (band == b)).sum())
    return n


def tune(evs, grid, extend=True):
    """Validation squared error of the whole grid, grown by the edge rule.

    Returns (axes, keys, sse, best, extensions): ``keys`` the settings as
    (L_km, gamma, k) tuples, ``sse`` (n_settings, n_channels, n_bands), and
    ``best[(ci, bi)]`` the selected setting of channel ci in band bi
    (``bi = None``: one setting for the whole column).
    """
    axes = {a: sorted(grid[a]) for a in AXES}
    ladder = {a: {d: list(v) for d, v in LADDER[a].items()} for a in AXES}
    done, extensions = {}, []
    while True:
        combos = [x for x in itertools.product(*(axes[a] for a in AXES))
                  if x not in done]
        if combos:
            tot = sum(POOL.map(_sweep_month, [(ev, combos) for ev in evs]))
            done.update({x: tot[i] for i, x in enumerate(combos)})
            print(f"  tuned {len(combos)} settings ({len(done)} in all, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        keys = sorted(done)
        sse = np.stack([done[x] for x in keys])
        best = {(ci, bi): keys[int(np.argmin(sse[:, ci, bi]))]
                for ci in range(len(CH)) for bi in range(len(BAND_NAMES))}
        best.update({(ci, None): keys[int(np.argmin(sse[:, ci, :].sum(axis=1)))]
                     for ci in range(len(CH))})
        grown = set()
        if extend:
            ends = {a: {"lo": axes[a][0], "hi": axes[a][-1]} for a in AXES}
            for (ci, bi), x in best.items():
                for ai, a in enumerate(AXES):
                    for d in ("lo", "hi"):
                        if x[ai] == ends[a][d] and ladder[a][d] and (a, d) not in grown:
                            new = ladder[a][d].pop(0)
                            axes[a] = sorted(axes[a] + [new])
                            grown.add((a, d))
                            extensions.append({
                                "axis": a, "value": new,
                                "for": f"{CH[ci]} "
                                       f"{BAND_NAMES[bi] if bi is not None else 'all bands'}"})
        if not grown:
            return axes, keys, sse, best, extensions


def rmse(sq, n):
    return float(np.sqrt(sq / n)) if n else float("nan")


# ------------------------------------------------------------------ run
POOL = get_context("fork").Pool(max(1, args.workers))
val = eval_set("validation", EVAL_CELLS, n_months=1 if args.smoke else None)
test = [] if args.smoke else eval_set("development")
print(f"synthetic cohort: {len(val)} validation months, {len(test)} test months, "
      f"{N_INPUT} inputs and {N_QUERY} queries a month, {LEV.size} levels "
      f"({time.time() - t0:.0f}s)", flush=True)


ident = {}
if not args.smoke:
    # a zero prediction must reproduce the trained model's stored fingerprint
    ident = {split: identity_check(ROOT, split, score(evs, "climatology"))
             for split, evs in (("validation", val), ("development", test))}
    print("identity check passed: zero prediction matches "
          f"{REFERENCE} on both splits", flush=True)

grid = ({"L_km": [250.0, 600.0], "gamma": [0.03, 0.1], "k": [10]}
        if args.smoke else GRID)
axes, keys, sse, best, extensions = tune(val, grid, extend=not args.smoke)
counts = band_counts(val)
ends = {a: (axes[a][0], axes[a][-1]) for a in AXES}


def describe(x, sq, n):
    return {"L_km": float(x[0]), "gamma": float(x[1]), "k": int(x[2]),
            "val_rmse_z": rmse(sq, n),
            "on_edge": [a for i, a in enumerate(AXES) if x[i] in ends[a]]}


sel_band = {ch: {b: best[(ci, bi)] for bi, b in enumerate(BAND_NAMES)}
            for ci, ch in enumerate(CH)}
sel_single = {ch: {b: best[(ci, None)] for b in BAND_NAMES}
              for ci, ch in enumerate(CH)}
oi_selection = {ch: {b: describe(best[(ci, bi)],
                                 sse[keys.index(best[(ci, bi)]), ci, bi], counts[ci, bi])
                     for bi, b in enumerate(BAND_NAMES)} for ci, ch in enumerate(CH)}
oi_single_selection = {ch: describe(best[(ci, None)],
                                    sse[keys.index(best[(ci, None)]), ci].sum(),
                                    counts[ci].sum()) for ci, ch in enumerate(CH)}

baselines = {}
for name, method, setting in (("climatology", "climatology", None),
                              ("nearest_profile", "nearest_profile", None),
                              ("oi", "oi", sel_band), ("oi_single", "oi", sel_single)):
    sc = {"validation": score(val, method, setting)}
    if test:
        sc["development"] = score(test, method, setting)
    baselines[name] = {"scores": sc}
    print(f"  scored {name} ({time.time() - t0:.0f}s)", flush=True)

# the tuned error and the scored error are the same cells: they must agree
for ch in CH:
    for b in BAND_NAMES:
        a = baselines["oi"]["scores"]["validation"][ch]["by_band_z"][b]
        if abs(a - oi_selection[ch][b]["val_rmse_z"]) > 1e-9:
            raise SystemExit(f"tuning and scoring disagree on {ch} {b}: "
                             f"{a} vs {oi_selection[ch][b]['val_rmse_z']}")

split = "validation" if args.smoke else "development"
print(f"\n{split}:  TEMP degC / SALT PSU / J TEMP / J SALT")
for name, b in baselines.items():
    s = b["scores"][split]
    print(f"  {name:16s} {s['TEMP']['rmse_physical']:.4f} / {s['SALT']['rmse_physical']:.4f}"
          f" / {s['TEMP']['J']:.3f} / {s['SALT']['J']:.3f}")
for ch in CH:
    for b in BAND_NAMES:
        o = oi_selection[ch][b]
        print(f"  OI {ch} {b:10s} L={o['L_km']:.0f} km gamma={o['gamma']} k={o['k']}"
              f" val rmse_z={o['val_rmse_z']:.4f}"
              + (f"  ON EDGE {o['on_edge']}" if o["on_edge"] else ""))
if args.smoke:
    print(f"smoke ok ({time.time() - t0:.0f}s), nothing written")
    raise SystemExit(0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


cpath = cohort_path(ROOT, "synthetic")
summary = {
    "tag": "fixed_baselines", "region": "synthetic", "target": "anomaly_exact",
    "splits": {k: list(v) for k, v in SPLITS.items()},
    "cohort": {"path": os.path.relpath(cpath, ROOT), "sha256": sha256(cpath)},
    "git_commit": git_commit(),
    "inputs_per_month": N_INPUT, "queries_per_month": N_QUERY,
    "eval_cells": EVAL_CELLS,
    "per_month": {name: {str(ev["month"]): {"inputs": int(ev["src"].size),
                                            "queries": int(ev["tgt"].size)}
                         for ev in evs}
                  for name, evs in (("validation", val), ("development", test))},
    "identity_check": {"reference": REFERENCE, **ident},
    "oi_selection": oi_selection, "oi_single_selection": oi_single_selection,
    "baselines": baselines, "runtime_s": time.time() - t0}
tuning = {
    "axes": axes, "bands": BAND_NAMES, "extensions": extensions,
    "cells": {ch: {b: int(counts[ci, bi]) for bi, b in enumerate(BAND_NAMES)}
              for ci, ch in enumerate(CH)},
    "grid": [{"L_km": float(x[0]), "gamma": float(x[1]), "k": int(x[2]),
              "rmse_z": {ch: {**{b: rmse(sse[i, ci, bi], counts[ci, bi])
                                 for bi, b in enumerate(BAND_NAMES)},
                              "all": rmse(sse[i, ci].sum(), counts[ci].sum())}
                         for ci, ch in enumerate(CH)}}
             for i, x in enumerate(keys)]}
os.makedirs(OUT, exist_ok=True)
for name, obj in (("summary.json", summary), ("tuning.json", tuning)):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
print(f"wrote {OUT}/summary.json and tuning.json ({time.time() - t0:.0f}s)")
