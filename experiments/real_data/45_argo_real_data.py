"""Track A driver — the registered rows on REAL Argo, scored on held-out floats.

This is the real-data analogue of `14_godas_dfs_d4rt.py`, and the difference is
the point of the whole exercise:

    14_godas_dfs_d4rt.py   input: random columns of the GODAS reanalysis
                           target: the same reanalysis
    45_argo_real_data.py   input: real QC'd Argo profiles (cohort floats)
                           target: real Argo measurements from floats the model
                                   has never seen, in any month

The second is not circular.  GODAS assimilates Argo, so predicting GODAS from
Argo columns is partly a test of whether the model can invert an assimilation;
predicting a held-out float's own measurements is a test of whether it can
reconstruct the ocean.

Packages implemented here (plan S4)
-----------------------------------
P0  real-data baseline table: every registered row plus the non-learned
    references, per lead, per depth band, with WMO- and month-clustered CIs.
P1  layout: natural / clustered / dispersed at equal profile count, and the DiD.
P2  redundancy: k = 1..32 across five duplication families, with the positive
    control that makes the rest interpretable.
P3  the thinning + superobbing control ladder (rows `thin_*`, `superob_*`).
P5  sparsity: 100/75/50/25/10 % of the month's real profiles.

Every package writes a signed `ResultArtifact` (protocol hash, git commit, data
manifest hashes, counts) so a Track B run can be compared against it mechanically
by `ocean_tokenizer.crosscheck` rather than by reading two markdown tables.

  .venv/bin/python experiments/45_argo_real_data.py --package P0 --smoke
  .venv/bin/python experiments/45_argo_real_data.py --package P0 \
      --rows dfs_expertlocal_cbottle,uniform_expertlocal_cbottle --seed 1234
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer import argo_experiments as E
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample, training_rows)
from ocean_tokenizer.clustered_ci import (accumulate, ci_rmse, ci_difference,
                                          ci_did, ranking, ClusterStats)
from ocean_tokenizer.godas_model import build_row, ROWS
from ocean_tokenizer.objective_interpolation import (ObjectiveInterpolation,
                                                     OISettings)
from ocean_tokenizer.reference_adapters import GriddedReference
from ocean_tokenizer.losses import CBottleMaskedLoss

CHANNELS = P.CHANNELS
#: Four bands. The deepest runs to 1401 m so the 1400.5 m level falls INSIDE
#: it rather than off the end of the table.
DEPTH_BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
               ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))

ap = argparse.ArgumentParser()
ap.add_argument("--package", default="P0", choices=["P0", "P1", "P2", "P3", "P5"])
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--rows", default="dfs_expertlocal_cbottle,uniform_expertlocal_cbottle,count_expertlocal_cbottle")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--validation-interval", type=int, default=500)
ap.add_argument("--n-profiles", type=int, default=24)
ap.add_argument("--queries", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--leads", default="0,1,3,6")
ap.add_argument("--split-protocol", default="main",
                choices=list(P.SPLIT_PROTOCOLS),
                help="'ecco_overlap' shifts the eras inside ECCO V4r4's "
                     "coverage so ECCO can be scored; a SECONDARY protocol, "
                     "never merged into the main headline table")
ap.add_argument("--eval-split", default="development",
                choices=["validation", "development", "holdout"])
ap.add_argument("--n-boot", type=int, default=10000)
ap.add_argument("--cohort", default=None)
ap.add_argument("--reference-suffix", default="",
                help="'_deep' selects the EN4/ECCO cut that reaches 1450 m, "
                     "which the 700-1400 m band needs")
ap.add_argument("--output", default=None)
ap.add_argument("--checkpoint-dir", default=None,
                help="where trained rows live. Defaults to this package's own "
                     "output dir, falling back to the P0 dir for the region.")
ap.add_argument("--region-kernel", action="store_true",
                help="size the DFS support kernel by THIS region's physical box "
                     "instead of the historical Gulf Stream one. 51 deg of "
                     "longitude is 4,499 km at the Gulf Stream and 5,671 km at "
                     "the equator, so the default makes equatorial correlation "
                     "lengths ~26%% too short. Opt-in because it changes the "
                     "function a checkpoint was fitted to: only use it for rows "
                     "TRAINED with it.")
ap.add_argument("--require-checkpoints", action="store_true",
                help="refuse to train: every requested row must already have a "
                     "checkpoint. P7 opens a one-shot holdout and forbids "
                     "retuning, so a row without a frozen checkpoint must abort "
                     "the run rather than quietly train a fresh model during "
                     "the open.")
ap.add_argument("--retrain", action="store_true",
                help="train even when a checkpoint exists. P1/P5 are "
                     "EVALUATION-time manipulations of an already-trained row "
                     "-- retraining per package would compare different models "
                     "under different layouts, which is not the experiment.")
ap.add_argument("--freeze-record", default=None,
                help="P7 freeze record. When given, every checkpoint loaded "
                     "must hash to what the record pinned, or the run aborts. "
                     "The freeze's own before/after comparison cannot catch a "
                     "freeze and an open that resolve the same FILENAME to "
                     "different FILES -- it re-derives both sides over one "
                     "directory -- so the guarantee has to be enforced where "
                     "the bytes are actually read.")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COHORT = args.cohort or os.path.join(ROOT, "data", "argo_cohort",
                                     f"{args.region}.nc")
OUT = args.output or os.path.join(ROOT, "outputs", f"argo_{args.package}_{args.region}")
os.makedirs(OUT, exist_ok=True)
rows_req = [r.strip() for r in args.rows.split(",") if r.strip()]
for r in rows_req:
    if r not in ROWS:
        raise SystemExit(f"unknown row {r!r}; registered: {ROWS}")
if args.smoke:
    args.steps, args.validation_interval = 30, 15
    args.queries, args.n_boot = 128, 500
leads = [int(x) for x in args.leads.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()

results_early_warnings: list = []
warnings_: list = results_early_warnings
print(f"argo real-data driver  package={args.package} region={args.region} "
      f"rows={rows_req} seed={args.seed} device={dev} smoke={args.smoke}",
      flush=True)

# ------------------------------------------------------------------ data
c = ArgoCohort.load(COHORT)
if args.split_protocol != "main":
    c.apply_splits(P.SPLIT_PROTOCOLS[args.split_protocol])
    warnings_.append(
        f"split_protocol={args.split_protocol} "
        f"({P.SPLIT_PROTOCOLS[args.split_protocol]}) — a SECONDARY protocol; "
        f"these numbers must not be merged into the main headline table")
    print(f"  split protocol: {args.split_protocol} "
          f"{P.SPLIT_PROTOCOLS[args.split_protocol]}", flush=True)
norm = ArgoNorm.fit(c, "train")
cfg_eval = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                         train=False)
cfg_train = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                          train=True)


def eligible(split: str, lead: int = 0) -> np.ndarray:
    """Source months that can actually produce a sample for this split.

    A month qualifies only if it has cohort floats to feed in AND held-out
    floats reporting at t+lead.  Silently scoring months with no held-out
    target would quietly shrink the cohort and make two tracks' 'same' month
    lists differ.
    """
    out = []
    for m in c.months_in(split):
        if (c.month(m, float_split="cohort_float").size
                and c.month(m + lead, float_split="heldout_float").size):
            out.append(int(m))
    return np.array(sorted(out), dtype=int)


MAX_LEAD_REQUESTED = 0        # set after the lead list is parsed


elig = {s: eligible(s) for s in ("train", "validation", "development", "holdout")}
print("  " + "  ".join(f"{k}: {len(v)} eligible months" for k, v in elig.items()),
      flush=True)
if elig["train"].size == 0:
    raise SystemExit("no eligible training month — check the cohort splits")

LEVELS = c.levels
band_of_level = []
for d in LEVELS:
    for name, lo, hi in DEPTH_BANDS:
        if (lo < d <= hi) or (d <= LEVELS.min() and lo <= 0):
            band_of_level.append(name)
            break
    else:
        band_of_level.append(DEPTH_BANDS[-1][0])
band_of_level = np.array(band_of_level)


# --------------------------------------------------------------- scoring
def _residuals(pred: torch.Tensor, s: dict) -> tuple[np.ndarray, ...]:
    """Per-query squared error plus the labels every CI and table needs."""
    e = ((pred - s["target"]) ** 2).detach().cpu().numpy()
    m = s["target_mask"].detach().cpu().numpy()
    wmo = np.asarray(s["target_wmo"])
    # z index of each query, recovered from the normalised depth coordinate
    zq = s["query"][:, 2].detach().cpu().numpy()
    zi = np.rint(zq * max(LEVELS.size - 1, 1)).astype(int).clip(0, LEVELS.size - 1)
    return e, m, wmo, zi


def score_model(model, months: np.ndarray, lead: int, seed: int,
                row_fn=None) -> dict:
    """Evaluate over months, keeping per-cluster sufficient statistics.

    ``row_fn(month, rng) -> (profile_rows, target_rows)`` lets a package change
    WHICH observations go in without touching how the sample is built.
    """
    is_oi = isinstance(model, ObjectiveInterpolation)
    if not is_oi and hasattr(model, "eval"):
        model.eval()
    acc = {ch: {"wmo": ([], [], []), "source_month": ([], [], [])}
           for ch in CHANNELS}
    band_acc = {ch: {b: [0.0, 0.0] for b, _, _ in DEPTH_BANDS} for ch in CHANNELS}
    n_targets = n_months = 0
    wmos_seen: set = set()
    for m in months:
        rng = np.random.default_rng([seed, int(m), lead])
        pr, tr = (row_fn(int(m), rng) if row_fn else (None, None))
        s = build_argo_sample(c, norm, int(m), cfg=cfg_eval, rng=rng, lead=lead,
                              profile_rows=pr, target_rows=tr)
        if s is None or s["target"].shape[0] == 0:
            continue
        s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            if is_oi:
                live = s["mask"] & s["value_mask"].all(dim=-1)
                pred = model(s["query"],
                             torch.zeros(s["query"].shape[0],
                                         dtype=s["query"].dtype, device=dev),
                             s["coord"][live],
                             s["value"][live].to(s["query"].dtype),
                             s["noise_density"][live]).to(torch.float32)
            else:
                pred = model(s)
        e, msk, wmo, zi = _residuals(pred, s)
        n_targets += int(msk.sum()); n_months += 1
        wmos_seen |= set(np.unique(wmo).tolist())
        for j, ch in enumerate(CHANNELS):
            ok = msk[:, j]
            if not ok.any():
                continue
            acc[ch]["wmo"][0].append(wmo[ok])
            acc[ch]["wmo"][1].append(e[ok, j])
            acc[ch]["source_month"][0].append(np.full(int(ok.sum()), int(m)))
            acc[ch]["source_month"][1].append(e[ok, j])
            eok = e[ok, j]
            fin = np.isfinite(eok)
            bl = band_of_level[zi[ok]]
            for bname, _, _ in DEPTH_BANDS:
                sel = (bl == bname) & fin
                if sel.any():
                    band_acc[ch][bname][0] += float(eok[sel].sum())
                    band_acc[ch][bname][1] += float(sel.sum())
    if not is_oi and hasattr(model, "train"):
        model.train()

    out = {"channels": {}, "n_targets": n_targets, "n_months": n_months,
           "n_wmos": len(wmos_seen), "lead": lead}
    for ch in CHANNELS:
        entry = {"clusters": {}}
        for unit in ("wmo", "source_month"):
            labs, errs, _ = acc[ch][unit]
            if not labs:
                entry["clusters"][unit] = None
                continue
            st = accumulate(np.concatenate(labs), np.concatenate(errs),
                            cluster_unit=unit, channel=ch)
            entry["clusters"][unit] = st
        entry["by_band"] = {
            b: (float(np.sqrt(v[0] / v[1])) if v[1] > 0 else float("nan"))
            for b, v in band_acc[ch].items()}
        out["channels"][ch] = entry
    return out


def stats_to_json(sc: dict, n_boot: int, seed: int,
                  keep_cluster_stats: bool = False) -> dict:
    """Point estimates, both clustered CIs, and the band table — JSON-able.

    ``keep_cluster_stats`` also stores the per-cluster (id, se, n) sufficient
    statistics.  They are what a CI is computed FROM, so persisting them lets
    the seed-aware interval be re-derived afterwards without re-running the
    model, and lets a cross-check compare two tracks at the level of the
    residuals rather than only at the level of the final RMSE — where a
    compensating pair of errors can agree by accident.
    """
    j = {"n_targets": sc["n_targets"], "n_months": sc["n_months"],
         "n_wmos": sc["n_wmos"], "lead": sc["lead"], "channels": {}}
    for ch, e in sc["channels"].items():
        d = {"by_band": e["by_band"], "ci": {}}
        for unit, st in e["clusters"].items():
            if st is None:
                d["ci"][unit] = None; continue
            d["rmse"] = st.rmse()
            iv = ci_rmse(st, n_boot=n_boot, seed=seed,
                         kind="bca" if st.n_clusters < 20 else "percentile")
            d["ci"][unit] = iv.to_dict()
            if keep_cluster_stats:
                d.setdefault("cluster_stats", {})[unit] = {
                    "ids": [str(x) for x in st.cluster_ids],
                    "se": [float(x) for x in st.se],
                    "n": [float(x) for x in st.n]}
        j["channels"][ch] = d
    return j


def macro_score(sc: dict) -> float:
    """Mean over channels of pooled RMSE — the selection score, as in GODAS."""
    vals = []
    for ch, e in sc["channels"].items():
        st = e["clusters"]["source_month"]
        vals.append(st.rmse() if st is not None else np.nan)
    return float(np.nanmean(vals)) if vals else float("nan")


# ------------------------------------------------------- non-learned rows
class TrainClimatology:
    """Zero anomaly in z space — the plan's train-climatology floor."""
    def __call__(self, s):
        return torch.zeros(s["query"].shape[0], len(CHANNELS),
                           dtype=torch.float32, device=s["query"].device)
    def eval(self): pass
    def train(self): pass


#: distances within this of the minimum count as tied. The coordinate box is
#: normalised to [0, 1], so 1e-12 is far below any real separation and far
#: above float noise.
TIE_EPS = 1e-12


def nearest_index(q3: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """Index of the nearest observation, with ties broken by lowest index.

    Ties are resolved EXPLICITLY.  Query and observation coordinates both live
    on the discrete grid (cell index, level index), so exact ties for "nearest"
    are common -- 16 of 512 queries in a typical month.  Two mathematically
    equivalent distance formulations then disagree in the last ulp, `argmin`
    picks differently, and the two tracks' RMSE differed by 0.3 % (0.8075 vs
    0.8046 over 201 of 18,432 queries) with neither being wrong.  That is a gap
    in the BASELINE'S DEFINITION, not a coding error, so it is closed here: among
    observations within `TIE_EPS` of the minimum distance, take the lowest token
    index.  Deterministic, and identical in both tracks.
    """
    d = torch.cdist(q3.to(torch.float64), c3.to(torch.float64))
    tied = d <= (d.min(dim=1, keepdim=True).values + TIE_EPS)
    return tied.float().argmax(dim=1)          # first True == lowest index


class SourcePersistence:
    """Nearest input profile's value at the query depth — 'source persistence'.

    The honest cheap baseline for a real-observation test: the field has not
    changed since the nearest float measured it.  Beating it is a low bar; not
    beating it would be decisive.
    """
    def __call__(self, s):
        q = s["query"]; co = s["coord"]
        live = s["mask"] & s["value_mask"].all(dim=-1)
        if not bool(live.any()):
            return torch.zeros(q.shape[0], len(CHANNELS), dtype=torch.float32,
                               device=q.device)
        idx = nearest_index(q[:, :3], co[live][:, :3])
        return s["value"][live][idx].to(torch.float32)
    def eval(self): pass
    def train(self): pass


class GriddedRow:
    """EN4 / ECCO scored at exactly the queries every other row is scored on.

    The product is interpolated to each held-out float's real position and
    depth, then z-scored with the SAME train-only statistics the model targets
    use, so its RMSE is directly comparable and not an artefact of units.

    Both products assimilate Argo, including the floats being scored. They are
    upper references that have seen the answer, not peers; the report says so.
    """
    def __init__(self, ref, norm):
        self.ref, self.norm = ref, norm
        self.n_query = self.n_covered = 0

    def __call__(self, s):
        lat = np.asarray(s["target_lat"]); lon = np.asarray(s["target_lon"])
        lev = np.asarray(s["target_level"]); m = int(s["target_month"])
        out = np.full((lat.size, len(CHANNELS)), np.nan)
        if self.ref.covers(m):
            uniq_lev = self.ref_levels
            T, S = self.ref.predict(m, lat, lon, uniq_lev)
            # each query wants its OWN level; pick the matching column
            col = np.searchsorted(uniq_lev, lev).clip(0, uniq_lev.size - 1)
            r = np.arange(lat.size)
            out[:, 0] = (T[r, col] - self.norm.mean["TEMP"][col]) / self.norm.std["TEMP"][col]
            out[:, 1] = (S[r, col] - self.norm.mean["SALT"][col]) / self.norm.std["SALT"][col]
        # A query the product cannot reach (below its deepest wet level, or in
        # a cell its land mask kills) is returned as NaN, NOT as zero anomaly.
        # Zero-filling scored the CLIMATOLOGY under the product's name wherever
        # it had no answer -- which penalises a gridded product for its mask and
        # quietly mixes two different estimators into one row. NaNs are dropped
        # by the accumulator, so the row reads "RMSE where the product has an
        # answer", and `reference_coverage` records what fraction that was.
        self.n_query += out.shape[0]
        self.n_covered += int(np.isfinite(out[:, 0]).sum())
        return torch.as_tensor(out, dtype=torch.float32,
                               device=s["query"].device)

    ref_levels = None
    def eval(self): pass
    def train(self): pass


class WOAClimatology:
    """WOA23 at each query's OWN (lat, lon, depth, calendar month).

    `train_climatology` predicts zero anomaly, and the anomaly reference is
    `ArgoNorm` -- a per-depth-level mean POOLED over every location and month in
    the training era. That is the region's mean vertical profile: no spatial
    structure, no seasonal cycle. It is a weak baseline, and normalising J by it
    inflates every skill number relative to what the literature reports.

    This row is the defensible denominator: an observational climatology that
    varies with position and season, which is what WOA23 is and what published
    reconstruction skill is normally measured against. Both rows are kept so the
    difference between them is visible rather than assumed.
    """

    def __init__(self, woa_phys, box, levels, norm, grid_shape):
        self.w = woa_phys                    # (12, 2, L, NY, NX) physical units
        self.box, self.levels, self.norm = box, levels, norm
        self.NY, self.NX = grid_shape

    def __call__(self, s):
        lat = np.asarray(s["target_lat"]); lon = np.asarray(s["target_lon"])
        lev = np.asarray(s["target_level"]); cm = int(s["target_month"]) % 12
        (la0, la1), (lo0, lo1) = self.box["lat"], self.box["lon"]
        gy = np.clip(((lat - la0) / (la1 - la0) * self.NY).astype(int), 0, self.NY - 1)
        gx = np.clip(((lon - lo0) / (lo1 - lo0) * self.NX).astype(int), 0, self.NX - 1)
        li = np.abs(self.levels[None, :] - lev[:, None]).argmin(axis=1)
        out = np.full((lat.size, len(CHANNELS)), np.nan)
        for j, ch in enumerate(CHANNELS):
            v = self.w[cm, j, li, gy, gx]
            out[:, j] = (v - self.norm.mean[ch][li]) / self.norm.std[ch][li]
        return torch.as_tensor(out, dtype=torch.float32,
                               device=s["query"].device)

    def eval(self): pass
    def train(self): pass


def woa_on_region(region: str, levels: np.ndarray, grid_shape) -> np.ndarray:
    """WOA23 interpolated onto a region box's cell centres and the cohort levels."""
    import xarray as xr
    box = P.REGIONS[region]
    NY, NX = grid_shape
    la0, la1 = box["lat"]; lo0, lo1 = box["lon"]
    lat_c = la0 + (np.arange(NY) + 0.5) * (la1 - la0) / NY
    lon_c = lo0 + (np.arange(NX) + 0.5) * (lo1 - lo0) / NX
    src = xr.open_zarr(os.path.join(ROOT, "data", "woa23_standard.zarr"))
    src = src.assign_coords(lon=(src.lon % 360.0)).sortby("lon")
    out = np.empty((12, len(CHANNELS), levels.size, NY, NX), "float32")
    for j, ch in enumerate(CHANNELS):
        di = src[ch].interp(lat=lat_c, lon=lon_c, depth=levels, method="linear",
                            kwargs={"fill_value": None}).values
        out[:, j] = np.asarray(di, "float32")
    return out


NONLEARNED = {"train_climatology": TrainClimatology,
              "source_persistence": SourcePersistence}


# ------------------------------------------------------------- training
CKPT_DIRS = [d for d in (args.checkpoint_dir, OUT,
                         os.path.join(ROOT, "outputs", f"argo_P0_{args.region}"))
             if d]


FROZEN_CKPTS: dict = {}
if args.freeze_record:
    FROZEN_CKPTS = json.load(open(args.freeze_record)).get("checkpoints", {})
    print(f"  freeze record: {len(FROZEN_CKPTS)} checkpoints pinned; every load "
          f"will be verified against it", flush=True)


def regime() -> dict:
    """The data regime a checkpoint is only valid within.

    `split_protocol` belongs here for the same reason the depth grid does. The
    `ecco_overlap` protocol evaluates on 2015-2017, which sits INSIDE the main
    protocol's 2000-2018 training era -- so loading a main-protocol checkpoint
    for an ecco_overlap run scores a model on months it trained on. The floats
    stay held out (the cohort is WMO-disjoint), so it is not a hard leak, but it
    is in-sample in time and optimistically biased. It happened, and the guard
    did not catch it because it only compared levels and lead.
    """
    return {"n_levels": int(c.levels.size),
            "max_level_m": round(float(c.levels.max()), 1),
            "max_lead": int(max(leads)),
            "split_protocol": args.split_protocol,
            # Input density belongs here too: a model fitted with 24 profiles
            # per month has never seen the token count, or the redundancy, that
            # 128 profiles present. Scoring one at the other's density is the
            # same silent substitution the depth grid and the split protocol
            # already guard against.
            "n_profiles": int(args.n_profiles)}


def regime_matches(path: str) -> bool:
    """Refuse a checkpoint trained on a different depth grid or lead horizon.

    Reuse ACROSS PACKAGES is deliberate: P1 and P5 manipulate the input at
    evaluation time and must score the model P0 registered. But a checkpoint is
    interchangeable only within one data regime. Loading a 16-level, lead-0..3
    model and scoring it on the 23-level cohort at lead 6 runs without error
    and means nothing — the model never saw water below 949 m, nor any lead
    past 3. That happened here and produced a plausible-looking table, which is
    the dangerous kind of wrong.

    A checkpoint with no sidecar predates this guard and is accepted only when
    the current regime matches the 16-level / lead-3 defaults it must have been
    trained under.
    """
    want = regime()
    side = path.replace(".pt", ".regime.json")
    if os.path.exists(side):
        got = json.load(open(side))
        # a sidecar written before a key existed cannot speak to it; compare
        # only what it actually recorded, and treat the rest as unconstrained
        keys = [k for k in want if k in got]
        if keys and all(got[k] == want[k] for k in keys):
            return True
        print(f"  skipping {os.path.relpath(path, ROOT)}: trained on "
              f"{got.get('n_levels')} levels / lead {got.get('max_lead')}, "
              f"need {want['n_levels']} / lead {want['max_lead']}", flush=True)
        return False
    legacy = {"n_levels": 16, "max_level_m": 949.0, "max_lead": 3,
              "split_protocol": "main", "n_profiles": 24}
    if all(legacy[k] == want[k] for k in legacy):
        return True
    print(f"  skipping {os.path.relpath(path, ROOT)}: no regime sidecar; the "
          f"legacy regime (16 levels / lead 3) does not match the current "
          f"{want['n_levels']} levels / lead {want['max_lead']}", flush=True)
    return False


def find_checkpoint(row: str, seed: int) -> str | None:
    """First matching checkpoint on the search path.

    When a freeze record is in force, a candidate whose bytes do not match the
    pin is passed over rather than returned.  `outputs/argo_P7_<region>/` can
    hold models an ABANDONED earlier open trained under exactly these
    filenames, and that directory precedes the checkpoint directory here, so
    name-order resolution silently scores a model that was never frozen.
    Resolving by hash makes the pin decide which file is meant.
    """
    name = f"{row}_s{seed}.pt"
    path, decoys = P.resolve_pinned_checkpoint(name, CKPT_DIRS,
                                               FROZEN_CKPTS.get(name))
    for dc in decoys:
        warnings_.append(
            f"{os.path.relpath(dc, ROOT)} shares the name of a frozen "
            f"checkpoint but not its bytes; skipped in favour of the file the "
            f"freeze record pins")
    if path is not None and not regime_matches(path):
        return None
    return path


def train_row(row: str, seed: int) -> tuple[torch.nn.Module, dict]:
    """Load the registered row if it is already trained, else train it.

    P1 (layout) and P5 (sparsity) manipulate the INPUT at evaluation time; the
    model under test is the same one P0 registered.  Training a fresh model per
    package would compare different models under different layouts and confound
    the layout effect with seed variance -- which, measured on this task, is an
    order of magnitude larger than the layout effect itself.
    """
    ck = None if args.retrain else find_checkpoint(row, seed)
    if ck is not None and FROZEN_CKPTS:
        name = f"{row}_s{seed}.pt"
        pin, got = FROZEN_CKPTS.get(name), P.sha256_file(ck)
        if pin is None:
            raise SystemExit(
                f"--freeze-record: {row!r} is about to be scored but the freeze "
                f"record pins no checkpoint for it. Scoring an unregistered row "
                f"inside a one-shot holdout is the selection P7 forbids.")
        if pin != got:
            raise SystemExit(
                f"--freeze-record: {os.path.relpath(ck, ROOT)} hashes to "
                f"{got[:16]} but the freeze pinned {pin[:16]}. The model about "
                f"to be scored is not the model that was registered.")
    if ck is None and args.require_checkpoints:
        raise SystemExit(
            f"--require-checkpoints: no checkpoint for {row!r} seed {seed} in "
            f"{CKPT_DIRS}. Training it here would mean the model scored was "
            f"never frozen.")
    if ck is not None:
        model = build_row(row, region=(args.region if args.region_kernel else None)).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()
        print(f"  loaded {os.path.relpath(ck, ROOT)}", flush=True)
        return model, {"loaded_from": os.path.relpath(ck, ROOT),
                       "checkpoint_sha256": P.sha256_file(ck), "trained": False}
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_row(row, region=(args.region if args.region_kernel else None)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = CBottleMaskedLoss()
    rng = np.random.default_rng([seed, 1])
    best, best_state, hist = np.inf, None, []
    tr_months = elig["train"]
    step = 0
    while step < args.steps:
        m = int(rng.choice(tr_months))
        lead = int(rng.choice(leads))
        pr, tr = training_rows(c, m, lead, args.n_profiles, rng)
        if pr.size == 0 or tr.size == 0:
            continue
        s = build_argo_sample(c, norm, m, cfg=cfg_train, rng=rng, lead=lead,
                              profile_rows=pr, target_rows=tr)
        if s is None or s["target"].shape[0] == 0:
            continue
        s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        pred = model(s)
        loss = loss_fn(pred, s["target"], s["target_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % args.validation_interval == 0 or step == args.steps:
            sc = score_model(model, elig["validation"], 0, seed)
            j = macro_score(sc)
            hist.append({"step": step, "loss": float(loss), "val_macro": j})
            star = ""
            if np.isfinite(j) and j < best:
                best = j
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                star = " *best*"
            print(f"  step {step:5d}  loss {float(loss):.4f}  val {j:.4f}{star}",
                  flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_macro": float(best), "trained": True}


# ------------------------------------------------------------- packages
results: dict = {}
counts: dict = {}


def run_P0():
    """The real-data baseline table."""
    trained = {}
    for row in rows_req:
        print(f"\n=== {row} ===", flush=True)
        model, meta = train_row(row, args.seed)
        trained[row] = model
        results.setdefault("training", {})[row] = meta
        if meta.get("trained"):
            ck_ = os.path.join(OUT, f"{row}_s{args.seed}.pt")
            torch.save(model.state_dict(), ck_)
            # a checkpoint is only interchangeable within its data regime
            with open(ck_.replace(".pt", ".regime.json"), "w") as _f:
                json.dump(regime(), _f, indent=1)
    for name, cls in NONLEARNED.items():
        trained[name] = cls()
    try:
        wp = woa_on_region(args.region, LEVELS, c.grid)
        trained["woa_climatology"] = WOAClimatology(
            wp, P.REGIONS[args.region], LEVELS, norm, c.grid)
        print(f"  woa_climatology: WOA23 on {c.grid} x {LEVELS.size} levels",
              flush=True)
    except Exception as e:
        warnings_.append(f"woa_climatology unavailable: {type(e).__name__}: {e}")
    trained["objective_interpolation"] = ObjectiveInterpolation(OISettings()).to(dev)
    for name, factory in (("en4", GriddedReference.en4),
                          ("ecco", GriddedReference.ecco)):
        try:
            # Fall back per product. The deep (1450 m) re-cut succeeded for
            # EN4 -- an open Met Office download -- but failed for ECCO, whose
            # PO.DAAC fetch needs Earthdata credentials that are no longer
            # valid. Rather than drop ECCO entirely, use whatever cut exists
            # and record which one: the shallower file simply cannot answer
            # queries below 1000 m, and those are dropped and counted in
            # `query_coverage` instead of being filled.
            rdir = os.path.join(ROOT, "data", "reference")
            ref, used = None, None
            for cand in (args.region + args.reference_suffix, args.region):
                try:
                    ref = factory(rdir, cand); used = cand; break
                except Exception:
                    continue
            if ref is None:
                raise FileNotFoundError(
                    f"no {name} files for {args.region}"
                    f"[{args.reference_suffix}] or {args.region}")
            if used != args.region + args.reference_suffix:
                warnings_.append(
                    f"{name}: the '{args.reference_suffix}' cut is absent, so "
                    f"the shallower '{used}' files were used; queries below "
                    f"its deepest level are dropped, see query_coverage")
        except Exception as e:
            warnings_.append(f"{name} unavailable: {type(e).__name__}: {e}")
            continue
        covered = [m for lead in leads
                   for m in eligible(args.eval_split, lead) if ref.covers(int(m))]
        if not covered:
            # ECCO V4r4 ends 2017.  Scoring it anyway makes every query fall
            # back to zero anomaly, i.e. it silently reports the CLIMATOLOGY
            # FLOOR under ECCO's name -- a number that looks like a result and
            # is not one.  The row is dropped and the absence is recorded.
            warnings_.append(
                f"{name} does not cover any {args.eval_split} month "
                f"(product range {ref.time[0]}..{ref.time[-1]}); row omitted "
                f"rather than scored as the climatology floor")
            print(f"  {name}: no coverage of {args.eval_split} — row omitted",
                  flush=True)
            continue
        row = GriddedRow(ref, norm)
        row.ref_levels = LEVELS
        trained[name] = row
        results.setdefault("reference_coverage", {})[name] = {
            "months_covered": len(covered), "files_used": used,
            "max_depth_m": float(ref.depth.max()),
            "product_range": [str(ref.time[0]), str(ref.time[-1])]}

    per_row, cl_stats = {}, {}
    for name, model in trained.items():
        per_lead = {}
        for lead in leads:
            months = eligible(args.eval_split, lead)
            sc = score_model(model, months, lead, args.seed)
            per_lead[f"lead{lead}"] = stats_to_json(
                sc, args.n_boot, args.seed, keep_cluster_stats=(lead == 0))
            if lead == 0:
                cl_stats[name] = sc
        per_row[name] = per_lead
        if isinstance(model, GriddedRow) and model.n_query:
            results.setdefault("reference_coverage", {}).setdefault(name, {})[
                "query_coverage"] = model.n_covered / model.n_query
        r0 = per_lead["lead0"]["channels"]
        print(f"  {name:32s} TEMP {r0['TEMP'].get('rmse', float('nan')):.4f}  "
              f"SALT {r0['SALT'].get('rmse', float('nan')):.4f}  "
              f"(n_wmo={per_lead['lead0']['n_wmos']})", flush=True)
    results["rows"] = per_row

    # the S6 conclusion quantities
    if "dfs_expertlocal_cbottle" in cl_stats and "uniform_expertlocal_cbottle" in cl_stats:
        for ch in CHANNELS:
            a = cl_stats["dfs_expertlocal_cbottle"]["channels"][ch]["clusters"]["wmo"]
            b = cl_stats["uniform_expertlocal_cbottle"]["channels"][ch]["clusters"]["wmo"]
            if a is None or b is None:
                continue
            a.method, b.method = "dfs", "uniform"
            d = ci_difference(a, b, n_boot=args.n_boot, seed=args.seed)
            results.setdefault("dfs_minus_uniform", {})[ch] = d.to_dict()
        # the plan compares one headline; TEMP is the primary channel
        if "TEMP" in results.get("dfs_minus_uniform", {}):
            results["dfs_minus_uniform"] = {
                **results["dfs_minus_uniform"],
                "point": results["dfs_minus_uniform"]["TEMP"]["point"],
                "excludes_zero": results["dfs_minus_uniform"]["TEMP"]["excludes_zero"]}
    rk = []
    for name, sc in cl_stats.items():
        st = sc["channels"]["TEMP"]["clusters"]["wmo"]
        if st is not None:
            st.method = name
            rk.append(st)
    if rk:
        results["ranking"] = ranking(rk)

    first = next(iter(per_row.values()))["lead0"]
    counts["split_protocol"] = args.split_protocol
    counts.update(targets=first["n_targets"], wmos=first["n_wmos"],
                  months=first["n_months"],
                  eval_split=args.eval_split, n_profiles=args.n_profiles)
    from ocean_tokenizer.crosscheck import build_comparable
    results["comparable"] = build_comparable(
        rmse={r: {ch: v["lead0"]["channels"][ch].get("rmse")
                  for ch in CHANNELS} for r, v in per_row.items()},
        per_seed_rmse={r: {ch: [v["lead0"]["channels"][ch].get("rmse")]
                           for ch in CHANNELS} for r, v in per_row.items()},
        n_wmos=first["n_wmos"], n_targets=first["n_targets"],
        n_months=first["n_months"],
        dfs_minus_uniform={
            ch: {k: results["dfs_minus_uniform"][ch][k]
                 for k in ("point", "lo", "hi", "excludes_zero")}
            for ch in CHANNELS
            if isinstance(results.get("dfs_minus_uniform", {}).get(ch), dict)} or None,
        ranking=[m for m, _ in results.get("ranking", [])] or None)


def run_P1():
    """Layout: natural / clustered / dispersed at equal count, plus the DiD."""
    months = eligible(args.eval_split, 0)
    ver = {}
    for m in months[:12]:
        ver[int(m)] = E.layout_report(c, int(m), args.n_profiles, args.seed)
    results["layout_verification"] = ver
    counts["layout_months"] = int(months.size)

    trained = {}
    for row in rows_req:
        print(f"\n=== {row} ===", flush=True)
        model, meta = train_row(row, args.seed)
        trained[row] = model
        results.setdefault("training", {})[row] = meta
        if meta.get("trained"):
            ck_ = os.path.join(OUT, f"{row}_s{args.seed}.pt")
            torch.save(model.state_dict(), ck_)
            # a checkpoint is only interchangeable within its data regime
            with open(ck_.replace(".pt", ".regime.json"), "w") as _f:
                json.dump(regime(), _f, indent=1)
    trained["objective_interpolation"] = ObjectiveInterpolation(OISettings()).to(dev)

    per = {}
    stats = {}
    for name, model in trained.items():
        per[name] = {}
        for lay in E.LAYOUTS:
            fn = (lambda lay: (lambda m, rng: (
                E.layout(c, m, args.n_profiles, lay, args.seed), None)))(lay)
            sc = score_model(model, months, 0, args.seed, row_fn=fn)
            per[name][lay] = stats_to_json(sc, args.n_boot, args.seed)
            stats[(name, lay)] = sc
        gaps = {}
        for ch in CHANNELS:
            a = stats[(name, "clustered")]["channels"][ch]["clusters"]["wmo"]
            b = stats[(name, "dispersed")]["channels"][ch]["clusters"]["wmo"]
            if a is not None and b is not None:
                a.method, b.method = f"{name}_clustered", f"{name}_dispersed"
                gaps[ch] = ci_difference(a, b, n_boot=args.n_boot,
                                         seed=args.seed).to_dict()
        per[name]["layout_gap"] = gaps
        print(f"  {name:32s} gap(TEMP) = "
              f"{gaps.get('TEMP', {}).get('point', float('nan')):+.4f}", flush=True)
    results["layouts"] = per

    d, u = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    if all(k in trained for k in (d, u)):
        did = ci_did(stats[(d, "clustered")]["channels"]["TEMP"]["clusters"]["wmo"],
                     stats[(d, "dispersed")]["channels"]["TEMP"]["clusters"]["wmo"],
                     stats[(u, "clustered")]["channels"]["TEMP"]["clusters"]["wmo"],
                     stats[(u, "dispersed")]["channels"]["TEMP"]["clusters"]["wmo"],
                     n_boot=args.n_boot, seed=args.seed)
        results["layout_did"] = did.to_dict()
        print(f"\n  DiD = gap(Uniform) - gap(DFS) = {did}", flush=True)
    from ocean_tokenizer.crosscheck import build_comparable
    lay0 = next(iter(per.values()))["natural"]
    results["comparable"] = build_comparable(
        rmse={f"{r}|{lay}": {ch: v[lay]["channels"][ch].get("rmse")
                             for ch in CHANNELS}
              for r, v in per.items() for lay in E.LAYOUTS if lay in v},
        n_wmos=lay0["n_wmos"], n_targets=lay0["n_targets"],
        n_months=lay0["n_months"],
        layout_did=({k: results["layout_did"][k]
                     for k in ("point", "lo", "hi", "excludes_zero")}
                    if "layout_did" in results else None),
        ranking=sorted(per, key=lambda r: per[r]["natural"]["channels"]["TEMP"]
                       .get("rmse", float("inf"))) or None)


def run_P5():
    """Sparsity stress on real coverage."""
    months = eligible(args.eval_split, 0)
    trained = {}
    for row in rows_req:
        print(f"\n=== {row} ===", flush=True)
        model, meta = train_row(row, args.seed)
        trained[row] = model
        results.setdefault("training", {})[row] = meta
        if meta.get("trained"):
            ck_ = os.path.join(OUT, f"{row}_s{args.seed}.pt")
            torch.save(model.state_dict(), ck_)
            # a checkpoint is only interchangeable within its data regime
            with open(ck_.replace(".pt", ".regime.json"), "w") as _f:
                json.dump(regime(), _f, indent=1)
    trained["objective_interpolation"] = ObjectiveInterpolation(OISettings()).to(dev)

    per = {}
    for name, model in trained.items():
        per[name] = {}
        for frac in E.SPARSITY_FRACTIONS:
            # Thin the REGISTERED input set, not the month's full census.
            # A month holds ~300 cohort profiles but the rows were trained on
            # n_profiles = 24; thinning the census made "100%" a 12x token
            # increase over training, so the sweep measured distribution shift
            # rather than sparsity. Selecting first, then thinning, keeps 100%
            # equal to the training regime and makes each step a pure removal.
            fn = (lambda f: (lambda m, rng: (
                E.thin(c, E.layout(c, m, args.n_profiles, "natural", args.seed),
                       f, args.seed, m), None)))(frac)
            sc = score_model(model, months, 0, args.seed, row_fn=fn)
            per[name][f"{int(frac*100)}pct"] = stats_to_json(sc, args.n_boot, args.seed)
        deg = {k: v["channels"]["TEMP"].get("rmse", float("nan"))
               for k, v in per[name].items()}
        print(f"  {name:32s} " + "  ".join(f"{k}={v:.4f}" for k, v in deg.items()),
              flush=True)
    results["sparsity"] = per
    counts["sparsity_months"] = int(months.size)
    from ocean_tokenizer.crosscheck import build_comparable
    f0 = next(iter(per.values()))["100pct"]
    results["comparable"] = build_comparable(
        rmse={f"{r}|{frac}": {ch: v[frac]["channels"][ch].get("rmse")
                              for ch in CHANNELS}
              for r, v in per.items() for frac in v},
        n_wmos=f0["n_wmos"], n_targets=f0["n_targets"], n_months=f0["n_months"],
        ranking=sorted(per, key=lambda r: per[r]["100pct"]["channels"]["TEMP"]
                       .get("rmse", float("inf"))) or None)


RUN = {"P0": run_P0, "P1": run_P1, "P3": run_P0, "P5": run_P5}
if args.package == "P2":
    raise SystemExit("P2 runs from experiments/46_argo_redundancy.py")
RUN[args.package]()

art = P.ResultArtifact(
    package=args.package, track="A", region=args.region, results=results,
    seeds=[args.seed], command=" ".join(sys.argv),
    counts=counts, warnings=warnings_,
    notes=("Input: real Argo cohort floats. Target: real measurements from "
           "WMO-disjoint held-out floats. No held-out float appears in any "
           "training month.")).finalize(ROOT)
path = os.path.join(OUT, f"artifact_{args.package}_seed{args.seed}.json")
sha = art.write(path)
print(f"\nartifact: {path}\n  sha256 {sha}\n  protocol {art.protocol_hash_[:16]}"
      f"\ntotal {time.time()-t0:.0f}s")
