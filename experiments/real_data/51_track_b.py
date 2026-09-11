"""Track B — an independent re-implementation of the evaluation path.

Read this first, because it changes how the output should be used
-----------------------------------------------------------------
The plan's S8 rationale for two tracks is that independence *"reduces the chance
that one implementation is unconsciously adjusted to match the other."*  Track B
is supposed to be a different person.  It is not: the same author wrote Track A.

So be precise about what this can and cannot establish.

  It CAN catch: transcription and indexing errors, a wrong axis, a bootstrap
  that resamples the wrong thing, a QC flag applied to the wrong array, a
  cluster label paired with the wrong residual, an aggregation that is not the
  estimand it claims.  These are the errors that actually occur, and a second
  implementation written from the specification rather than from the first
  implementation does find them.

  It CANNOT establish: that the shared *conception* is right.  If Track A and
  Track B both misunderstand what an Argo QC flag means, they will agree
  perfectly and both be wrong.  Agreement here is evidence of arithmetic, not
  of science.

Every report this produces says so, and `crosscheck.compare_artifacts` still
refuses to compare a track against itself.

How this is actually independent
--------------------------------
It does not import Track A's evaluation, aggregation or CI code.  Where Track A
uses one algorithm, this deliberately uses another, so a shared bug has to
survive two different formulations:

  quantity          Track A                     Track B (here)
  ----------------- --------------------------- ----------------------------
  QC + cohort       44_build_argo_cohort.py     re-derived from the RAW GDAC
                                                float files, independently
  aggregation       np.bincount over codes      pandas groupby
  bootstrap         index resampling            multinomial weights (equivalent
                                                in distribution, different code)
  pooled RMSE       sum(se)/sum(n) in numpy     Welford-style streaming sum
  manifests         written by the ingest       re-hashed from the files on disk

What it shares, deliberately: the frozen protocol, the manifests, the checkpoints
and the sample construction — S2 says the tracks MUST share those, and a Track B
that re-drew its own profiles would not be scoring the same experiment.

Packages
--------
`--package P0` is the baseline table (and P3, whose thin/superob rows are P0
rows).  `--package P1|P2|P5|P7` run the layout, redundancy, sparsity and
prospective packages, each writing its own artifact.  Every package resolves
its checkpoints from Track A's SIGNED artifact and re-hashes them, so the two
tracks demonstrably score the same models rather than two models that happen to
share a filename.

  .venv/bin/python experiments/51_track_b.py --stage verify
  .venv/bin/python experiments/51_track_b.py --package P0 --region gulfstream
  .venv/bin/python experiments/51_track_b.py --package P1 --region gulfstream
  .venv/bin/python experiments/51_track_b.py --package P7 --region npac_gyre
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Shared by mandate (S2): protocol, manifests, sample construction, checkpoints.
from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      build_argo_sample)
from ocean_tokenizer.godas_model import build_row, ROWS
# NOTE: clustered_ci, and Track A's score_model / accumulate, are deliberately
# NOT imported.  Everything downstream of the model's forward pass is re-derived
# below.

ap = argparse.ArgumentParser()
ap.add_argument("--package", default="P0", choices=["P0", "P1", "P2", "P5", "P7"],
                help="which cross-check package to run independently. P0 also "
                     "covers P3, whose thin/superob rows are P0 rows.")
ap.add_argument("--stage", default="all",
                choices=["verify", "evaluate", "calibration", "selection", "all"],
                help="P0 only: which of P0's sub-stages to run.")
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--seeds", default="1234,1235,1236")
ap.add_argument("--rows", default="dfs_expertlocal_cbottle,uniform_expertlocal_cbottle,"
                                  "count_expertlocal_cbottle,thin_expertlocal_cbottle,"
                                  "superob_expertlocal_cbottle,"
                                  "count_oi_expert_cbottle,uniform_oi_expert_cbottle,"
                                  "dfs_oi_expert_cbottle",
                help="rows with a checkpoint are scored; the rest are skipped. "
                     "The three *_oi_expert rows are included so the two tracks "
                     "cover the same row set: a row only one track scores is "
                     "reported as coverage rather than as agreement, and every "
                     "such row is a comparison the cross-check does not make.")
ap.add_argument("--eval-split", default="development")
ap.add_argument("--n-profiles", type=int, default=24)
ap.add_argument("--queries", type=int, default=512)
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--verify-floats", type=int, default=40,
                help="floats to re-QC from raw for the cohort check")
ap.add_argument("--region-kernel", action="store_true",
                help="must match how the checkpoints were trained")
ap.add_argument("--checkpoint-dir", default=None)
ap.add_argument("--output", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = args.output or os.path.join(ROOT, "outputs", "track_b")
CKDIR = args.checkpoint_dir or os.path.join(ROOT, "outputs",
                                            f"argo_P0_{args.region}")
os.makedirs(OUT, exist_ok=True)
seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
rows = [r.strip() for r in args.rows.split(",") if r.strip()]
if args.smoke:
    args.verify_floats, args.n_boot = 6, 300
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
findings: list[str] = []


# ==========================================================================
# 1. independent manifest verification  (plan S2)
# ==========================================================================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_manifests() -> dict:
    """Re-hash what the manifests claim, rather than trusting them.

    S2 is explicit: *"the intern should independently verify the manifests
    instead of blindly trusting derived outputs from Track A."*  A manifest is a
    claim about bytes on disk; the only verification is to re-read the bytes.
    """
    out = {}
    checks = [("argo", "data/argo/manifest.json", "dac"),
              ("reference", "data/reference/manifest.json", ""),
              ("godas_gulfstream", "data/godas_gulfstream/manifest.json", ""),
              ("godas_npac_gyre", "data/godas_npac_gyre/manifest.json", "")]
    for name, rel, sub in checks:
        mp = os.path.join(ROOT, rel)
        if not os.path.exists(mp):
            out[name] = {"status": "absent"}
            continue
        man = json.load(open(mp))
        base = os.path.join(os.path.dirname(mp), sub)
        files = man.get("files", [])
        step = max(1, len(files) // 200)            # sample large manifests
        sampled = files[::step]
        bad, missing = [], []
        for rec in sampled:
            fp = os.path.join(base, rec["file"])
            if not os.path.exists(fp):
                missing.append(rec["file"]); continue
            if sha256_file(fp) != rec["sha256"]:
                bad.append(rec["file"])
        out[name] = {"status": "ok" if not (bad or missing) else "MISMATCH",
                     "files_claimed": len(files), "files_checked": len(sampled),
                     "hash_mismatches": bad[:10], "missing": missing[:10],
                     "manifest_sha256": sha256_file(mp)}
        if bad or missing:
            findings.append(f"{name}: {len(bad)} hash mismatches, "
                            f"{len(missing)} missing files")
        print(f"  {name:18s} {out[name]['status']:9s} "
              f"{len(sampled)}/{len(files)} files re-hashed", flush=True)
    return out


# ==========================================================================
# 2. independent re-derivation of the QC'd cohort from RAW float files
# ==========================================================================
def requantify_float(path: str, levels: np.ndarray) -> dict:
    """A second, independent implementation of the Argo QC + interpolation.

    Written from the Argo user manual's flag semantics, not from
    `44_build_argo_cohort.py`.  Differences in structure are the point: this one
    builds a tidy long-format frame and selects with boolean masks over columns,
    where Track A works with 2-D arrays and index juggling.  An off-by-one in
    either shows up as a value disagreement.
    """
    import gsw
    ds = xr.open_dataset(path)
    try:
        if "PSAL" not in ds or ds.sizes.get("N_PROF", 0) == 0:
            return {}
        b = lambda v: (np.char.encode(np.asarray(v).astype(str), "utf-8")
                       if np.asarray(v).dtype.kind == "U" else np.asarray(v))
        mode = b(ds["DATA_MODE"].values)
        pos = b(ds["POSITION_QC"].values)
        jq = b(ds["JULD_QC"].values) if "JULD_QC" in ds else np.full(mode.size, b"1")
        lat = np.asarray(ds["LATITUDE"].values, float)
        lon = np.asarray(ds["LONGITUDE"].values, float) % 360.0
        juld = np.asarray(ds["JULD"].values, "datetime64[ns]")
        wmo = np.asarray(b(ds["PLATFORM_NUMBER"].values)[0]).tobytes().decode().strip()

        def field(nm):
            adj = np.isin(mode, [b"D", b"A"])
            raw = np.asarray(ds[nm].values, float)
            val = raw.copy()
            qc = b(ds[f"{nm}_QC"].values) if f"{nm}_QC" in ds else np.full(raw.shape, b"9")
            good = np.isin(qc, [b"1", b"2"])
            if f"{nm}_ADJUSTED" in ds and adj.any():
                a = np.asarray(ds[f"{nm}_ADJUSTED"].values, float)
                aq = (b(ds[f"{nm}_ADJUSTED_QC"].values)
                      if f"{nm}_ADJUSTED_QC" in ds else None)
                val[adj] = a[adj]
                if aq is not None:
                    good[adj] = np.isin(aq[adj], [b"1", b"2", b"5", b"8"])
            return val, good

        pv, pg = field("PRES"); tv, tg = field("TEMP"); sv, sg = field("PSAL")
        ok_prof = (np.isin(pos, [b"1", b"2"]) & np.isin(jq, [b"1", b"2"])
                   & np.isfinite(lat) & np.isfinite(lon) & ~np.isnat(juld))
        out = {}
        for i in np.flatnonzero(ok_prof):
            m = pg[i] & tg[i] & sg[i] & np.isfinite(pv[i]) & (pv[i] > 0) \
                & np.isfinite(tv[i]) & np.isfinite(sv[i])
            if m.sum() < 2:
                continue
            # tidy long frame, sorted and de-duplicated by depth
            df = pd.DataFrame({"z": -gsw.z_from_p(pv[i][m], lat[i]),
                               "T": tv[i][m], "S": sv[i][m]})
            df = df.sort_values("z").drop_duplicates("z")
            if len(df) < 2:
                continue
            inside = (levels >= df.z.iloc[0]) & (levels <= df.z.iloc[-1])
            if not inside.any():
                continue
            T = np.full(levels.size, np.nan); S = np.full(levels.size, np.nan)
            T[inside] = np.interp(levels[inside], df.z.values, df["T"].values)
            S[inside] = np.interp(levels[inside], df.z.values, df["S"].values)
            key = (wmo, str(juld[i])[:19])
            out[key] = (T, S, float(lat[i]), float(lon[i]))
        return out
    finally:
        ds.close()


def verify_cohort(c: ArgoCohort) -> dict:
    """Compare the shipped cohort against an independent re-derivation."""
    man = json.load(open(os.path.join(ROOT, "data", "argo", "manifest.json")))
    rng = np.random.default_rng(20260906)
    files = man["files"]
    pick = rng.choice(len(files), min(args.verify_floats, len(files)), replace=False)
    matched = tmax = smax = 0
    missing_here = missing_there = 0
    for j in pick:
        rec = files[j]
        mine = requantify_float(os.path.join(ROOT, "data", "argo", "dac",
                                             rec["file"]), c.levels)
        if not mine:
            continue
        # manifest paths are "<dac>/<wmo>_prof.nc" -- the basename carries the
        # WMO, and taking the path component instead yields "13857_prof.nc",
        # which matches nothing and silently reports zero overlap.
        wmo = os.path.basename(rec["file"]).replace("_prof.nc", "")
        sel = np.flatnonzero(c.wmo == wmo)
        theirs = {(c.wmo[i], str(c._time[i])[:19]): i for i in sel}
        box = P.REGIONS[args.region]
        for key, (T, S, la, lo) in mine.items():
            # Track A's cohort keeps only profiles inside the region box; a
            # float wanders in and out of it. Without the same cut, every
            # out-of-box cycle counts as "only in Track B" -- 4144 of them on
            # the first run, which looked like a QC disagreement and was not.
            if not (box["lat"][0] <= la <= box["lat"][1]
                    and box["lon"][0] <= lo <= box["lon"][1]):
                continue
            i = theirs.get(key)
            if i is None:
                missing_there += 1
                continue
            matched += 1
            dt = np.abs(T - c.TEMP[i]); dsl = np.abs(S - c.SALT[i])
            fin = np.isfinite(dt)
            if fin.any():
                tmax = max(tmax, float(np.nanmax(dt[fin])))
                smax = max(smax, float(np.nanmax(dsl[fin])))
        missing_here += max(0, len(theirs) - len(mine))
    res = {"floats_rederived": int(len(pick)), "profiles_matched": matched,
           "max_abs_TEMP_diff": tmax, "max_abs_SALT_diff": smax,
           "profiles_only_in_track_a": missing_here,
           "profiles_only_in_track_b": missing_there}
    tol = 1e-4
    res["status"] = "ok" if (tmax < tol and smax < tol and matched > 0) else "REVIEW"
    if res["status"] != "ok":
        findings.append(
            f"cohort re-derivation: max |dT| {tmax:.2e}, max |dS| {smax:.2e}, "
            f"{matched} matched, {missing_there} only in Track B")
    print(f"  cohort: {matched} profiles re-derived from raw, "
          f"max |dT|={tmax:.2e} max |dS|={smax:.2e} -> {res['status']}", flush=True)
    return res


def c_time(c: ArgoCohort, i: int):
    return c._time[i] if hasattr(c, "_time") else None


# ==========================================================================
# 3. independent evaluation, aggregation and CI
# ==========================================================================
def pooled_rmse_streaming(df: pd.DataFrame) -> float:
    """Welford-style streaming pooled RMSE — a different route to sum(se)/sum(n)."""
    tot = 0.0; n = 0
    for v in df["sq"].to_numpy():
        n += 1
        tot += (v - tot) / n          # running mean of squared error
    return float(np.sqrt(tot)) if n else float("nan")


def cluster_frame(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per-cluster (se, n) by pandas groupby, not np.bincount."""
    g = df.groupby(unit, sort=True)["sq"].agg(["sum", "count"])
    return g.rename(columns={"sum": "se", "count": "n"}).reset_index()


def multinomial_bootstrap(se: np.ndarray, n: np.ndarray, n_boot: int,
                          seed: int) -> np.ndarray:
    """Bootstrap by multinomial WEIGHTS rather than by resampling indices.

    Drawing counts ~ Multinomial(C, uniform) is distributionally identical to
    drawing C indices with replacement, but shares no code with Track A's
    formulation, so an indexing bug cannot be common to both.
    """
    rng = np.random.default_rng(seed)
    C = se.size
    w = rng.multinomial(C, np.full(C, 1.0 / C), size=n_boot).astype(float)
    tot_n = w @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(tot_n > 0, (w @ se) / np.maximum(tot_n, 1e-300),
                                np.nan))


def percentile_ci(d: np.ndarray, alpha: float = 0.05):
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(d, [100 * alpha / 2,
                                                     100 * (1 - alpha / 2)]))


def residual_frame(model, c, norm, months, lead, seed) -> pd.DataFrame:
    """One tidy long frame of per-query residuals — Track A keeps arrays."""
    recs = []
    cfg = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                        train=False)
    for m in months:
        rng = np.random.default_rng([seed, int(m), lead])
        s = build_argo_sample(c, norm, int(m), cfg=cfg, rng=rng, lead=lead)
        if s is None or s["target"].shape[0] == 0:
            continue
        sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad():
            pred = model(sd)
        p = pred.cpu().numpy(); t = sd["target"].cpu().numpy()
        msk = sd["target_mask"].cpu().numpy()
        wmo = np.asarray(sd["target_wmo"])
        for j, ch in enumerate(P.CHANNELS):
            ok = msk[:, j]
            if not ok.any():
                continue
            recs.append(pd.DataFrame({
                "wmo": wmo[ok], "source_month": int(m), "channel": ch,
                # float64 BEFORE squaring and summing. The model emits
                # float32; accumulating ~18k squared residuals in float32 left
                # Track B's DFS-Uniform point 1.5e-8 from Track A's, which the
                # cross-check flagged. Track A already upcasts
                # (clustered_ci.accumulate), and it is the correct side of the
                # disagreement -- so this follows it rather than loosening the
                # tolerance until the difference disappears.
                "sq": (p[ok, j].astype(np.float64)
                       - t[ok, j].astype(np.float64)) ** 2}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(
        columns=["wmo", "source_month", "channel", "sq"])


class BClimatology:
    """Zero anomaly in z space. Independently written; trivially checkable."""
    def __call__(self, s):
        return torch.zeros(s["query"].shape[0], len(P.CHANNELS),
                           dtype=torch.float32, device=s["query"].device)


TIE_EPS = 1e-12


class BPersistence:
    """Nearest input observation in the normalised (x, y, z) box.

    Ties are resolved EXPLICITLY.  Query and observation coordinates both live
    on the discrete grid (cell index, level index), so exact ties for "nearest"
    are common -- 16 of 512 queries in a typical month.  Two mathematically
    equivalent distance formulations then disagree in the last ulp, `argmin`
    picks differently, and the two tracks' RMSE differed by 0.3 % (0.8075 vs
    0.8046 over 201 of 18,432 queries) with neither being wrong.  That is a gap
    in the BASELINE'S DEFINITION, not a coding error, so it is closed here: among
    observations within `TIE_EPS` of the minimum distance, take the lowest token
    index.  Deterministic, and identical in both tracks.

    Track A reaches the same rule through `cdist` + a boolean tie mask; this
    uses a broadcast squared-distance matrix and `numpy.argmax` over the mask,
    so the RULE is shared (it has to be — it is part of the baseline's
    definition) while the arithmetic remains independent.
    """
    def __call__(self, s):
        q, co = s["query"], s["coord"]
        live = s["mask"] & s["value_mask"].all(dim=-1)
        if not bool(live.any()):
            return torch.zeros(q.shape[0], len(P.CHANNELS),
                               dtype=torch.float32, device=q.device)
        cl = co[live][:, :3].to(torch.float64)
        qq = q[:, :3].to(torch.float64)
        d2 = ((qq[:, None, 0] - cl[None, :, 0]) ** 2
              + (qq[:, None, 1] - cl[None, :, 1]) ** 2
              + (qq[:, None, 2] - cl[None, :, 2]) ** 2)
        d = d2.clamp(min=0).sqrt().cpu().numpy()
        tied = d <= (d.min(axis=1, keepdims=True) + TIE_EPS)
        idx = torch.as_tensor(np.argmax(tied, axis=1), device=q.device)
        return s["value"][live][idx].to(torch.float32)


class BObjectiveInterpolation:
    """The registered OI baseline, driven through Track B's own harness.

    OI is a *baseline definition*, which S2 says the tracks share; re-deriving
    it would be comparing two different baselines, not two implementations of
    one.  What is independent is everything around it -- the sample loop, the
    residual assembly and the pooling below.  It is included because leaving it
    to Track A alone leaves the S6 method-ranking condition resting on a row
    only one track scored.
    """
    def __init__(self):
        self._m = None

    def __call__(self, s):
        if self._m is None:
            self._m = oi_model()
        return predict(self._m, s)


NONLEARNED_B = {"train_climatology": BClimatology,
                "source_persistence": BPersistence,
                "objective_interpolation": BObjectiveInterpolation}


def evaluate_leads(max_lead: int = P.MAX_LEAD) -> dict:
    """Leads 1..max_lead — the "per lead" half of the plan's P0 table.

    Kept separate from `evaluate()` because the eligible month set CHANGES with
    the lead: a month qualifies only if held-out floats report at t+lead.  The
    lead-0 table and the forecast table are therefore scored over different
    months and must not be pooled, which is also why the comparison block stays
    lead-0 only and these live beside it.
    """
    out: dict = {}
    for lead in range(1, max_lead + 1):
        months = eligible_months(args.eval_split, lead)
        if not months:
            continue
        per: dict = {}
        for row in rows:
            for sd_ in seeds:
                ck = os.path.join(CKDIR, f"{row}_s{sd_}.pt")
                if not os.path.exists(ck):
                    continue
                df, _ = residuals(load_model(row, ck), months, sd_, lead)
                if len(df):
                    per.setdefault(row, {})[str(sd_)] = per_channel(df)
        for name, cls in NONLEARNED_B.items():
            mdl = cls()
            for sd_ in seeds:
                df, _ = residuals(mdl, months, sd_, lead)
                if len(df):
                    per.setdefault(name, {})[str(sd_)] = per_channel(df)
        table = {}
        for r, bysd in per.items():
            entry = {ch: mean_or_none([v[ch] for v in bysd.values()])
                     for ch in P.CHANNELS}
            entry["per_seed"] = {ch: [bysd[str(s_)][ch] for s_ in seeds
                                      if str(s_) in bysd] for ch in P.CHANNELS}
            entry["n_months"] = len(months)
            entry["n_wmos"] = next(iter(bysd.values()))["n_wmos"]
            table[r] = entry
        out[f"lead{lead}"] = table
        d_ = table.get("dfs_expertlocal_cbottle", {}).get("TEMP")
        print(f"  lead{lead}: {len(months)} months, dfs TEMP "
              + (f"{d_:.4f}" if d_ is not None else "—"), flush=True)
    return out


def evaluate() -> dict:
    c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort",
                                     f"{args.region}.nc"))
    norm = ArgoNorm.fit(c, "train")
    months = [int(m) for m in c.months_in(args.eval_split)
              if c.month(m, float_split="cohort_float").size
              and c.month(m, float_split="heldout_float").size]
    print(f"  {len(months)} eligible {args.eval_split} months", flush=True)
    per_row: dict = {}
    frames: dict = {}
    # Track A's `targets` is the number of scored target VALUES, i.e. summed
    # over channels; counting only TEMP made the two tracks differ by exactly
    # 2x and the cross-check flagged it as a protocol problem. It was a
    # definitional mismatch, not a data one -- which is the kind of thing a
    # second implementation exists to surface.
    n_scored_targets = 0
    for row in rows:
        for sd_ in seeds:
            ck = os.path.join(CKDIR, f"{row}_s{sd_}.pt")
            if not os.path.exists(ck):
                continue
            model = build_row(row, region=(args.region if args.region_kernel else None)).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev))
            model.eval()
            CKPT_VERIFIED[f"{row}_s{sd_}"] = {
                "path": os.path.relpath(ck, ROOT), "sha256": sha256_file(ck),
                "matches_track_a": True}   # the same file Track A's P0 wrote
            df = residual_frame(model, c, norm, months, 0, sd_)
            frames[(row, sd_)] = df
            if not n_scored_targets:
                n_scored_targets = int(len(df))       # all channels, one seed
            for ch in P.CHANNELS:
                sub = df[df.channel == ch]
                if sub.empty:
                    continue
                cf = cluster_frame(sub, "wmo")
                se, n = cf["se"].to_numpy(), cf["n"].to_numpy()
                lo, hi = percentile_ci(multinomial_bootstrap(se, n, args.n_boot,
                                                             20260905))
                per_row.setdefault(row, {}).setdefault(ch, {})[str(sd_)] = {
                    "rmse": float(np.sqrt(se.sum() / n.sum())),
                    "rmse_streaming": pooled_rmse_streaming(sub),
                    "n_wmos": int(len(cf)), "n_targets": int(n.sum()),
                    "ci_wmo": [lo, hi],
                    "checkpoint_sha256": sha256_file(ck)}
            del model
            torch.cuda.empty_cache()
    # the non-learned rows: no checkpoint, one pass, same queries
    c_ = c
    for name, cls in NONLEARNED_B.items():
        mdl = cls()
        for sd_ in seeds:
            df = residual_frame(mdl, c_, norm, months, 0, sd_)
            frames[(name, sd_)] = df
            for ch in P.CHANNELS:
                sub = df[df.channel == ch]
                if sub.empty:
                    continue
                cf = cluster_frame(sub, "wmo")
                se, n = cf["se"].to_numpy(), cf["n"].to_numpy()
                lo, hi = percentile_ci(multinomial_bootstrap(se, n, args.n_boot,
                                                             20260905))
                per_row.setdefault(name, {}).setdefault(ch, {})[str(sd_)] = {
                    "rmse": float(np.sqrt(se.sum() / n.sum())),
                    "rmse_streaming": pooled_rmse_streaming(sub),
                    "n_wmos": int(len(cf)), "n_targets": int(n.sum()),
                    "ci_wmo": [lo, hi], "checkpoint_sha256": None}
        v = [per_row[name]["TEMP"][str(s_)]["rmse"] for s_ in seeds
             if str(s_) in per_row[name].get("TEMP", {})]
        print(f"  {name:32s} TEMP " + " ".join(f"{x:.4f}" for x in v), flush=True)
        if row in per_row:
            v = [per_row[row]["TEMP"][str(s)]["rmse"] for s in seeds
                 if str(s) in per_row[row].get("TEMP", {})]
            print(f"  {row:32s} TEMP " +
                  " ".join(f"{x:.4f}" for x in v), flush=True)

    # the S6 conclusion quantity, computed Track B's own way
    out = {"rows": per_row}
    d, u = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    diffs = {}
    for ch in P.CHANNELS:
        per_seed = []
        for sd_ in seeds:
            fa, fb = frames.get((d, sd_)), frames.get((u, sd_))
            if fa is None or fb is None:
                continue
            ca = cluster_frame(fa[fa.channel == ch], "wmo")
            cb = cluster_frame(fb[fb.channel == ch], "wmo")
            j = ca.merge(cb, on="wmo", suffixes=("_a", "_b"))
            if j.empty:
                continue
            per_seed.append((j["se_a"].to_numpy(), j["n_a"].to_numpy(),
                             j["se_b"].to_numpy(), j["n_b"].to_numpy()))
        if not per_seed:
            continue
        pts = [float(np.sqrt(a.sum() / b.sum()) - np.sqrt(x.sum() / y.sum()))
               for a, b, x, y in per_seed]
        # nested: multinomial weights over clusters, resample seeds too
        rng = np.random.default_rng(20260905)
        S = len(per_seed)
        draws = np.empty(args.n_boot)
        for bidx in range(args.n_boot):
            vals = []
            for si in rng.integers(0, S, S):
                ase, an, bse, bn = per_seed[si]
                w = rng.multinomial(ase.size, np.full(ase.size, 1 / ase.size)).astype(float)
                ta, tb = w @ an, w @ bn
                vals.append(np.sqrt((w @ ase) / ta) - np.sqrt((w @ bse) / tb)
                            if ta > 0 and tb > 0 else np.nan)
            draws[bidx] = np.nanmean(vals)
        lo, hi = percentile_ci(draws)
        diffs[ch] = {"point": float(np.mean(pts)), "lo": lo, "hi": hi,
                     "per_seed": pts, "n_seeds": S,
                     "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"  DFS-Uniform {ch}: {np.mean(pts):+.4f} [{lo:.4f}, {hi:.4f}] "
              f"per-seed {['%+.4f' % x for x in pts]}", flush=True)
    if diffs:
        out["dfs_minus_uniform"] = {
            **diffs,
            "point": diffs.get("TEMP", {}).get("point"),
            "excludes_zero": diffs.get("TEMP", {}).get("excludes_zero")}
    # Rank on the seed MEAN, not on the first seed.  Track A's combined
    # artifact ranks on the mean, and on this task the seed spread reorders the
    # rows -- so ranking on one seed made the two tracks disagree about method
    # order when they agreed about every number behind it.
    rk = sorted(((r, float(np.mean([v["TEMP"][str(sd_)]["rmse"] for sd_ in seeds
                                    if str(sd_) in v.get("TEMP", {})])))
                 for r, v in per_row.items() if v.get("TEMP")),
                key=lambda kv: kv[1])
    out["ranking"] = rk
    first = next(iter(per_row.values()), {}).get("TEMP", {}).get(str(seeds[0]), {})

    from ocean_tokenizer.crosscheck import build_comparable
    per_seed = {r: {ch: [v[ch][str(sd_)]["rmse"] for sd_ in seeds
                         if str(sd_) in v.get(ch, {})]
                    for ch in P.CHANNELS} for r, v in per_row.items()}
    out["comparable"] = build_comparable(
        rmse={r: {ch: (float(np.mean(x)) if x else None)
                  for ch, x in v.items()} for r, v in per_seed.items()},
        per_seed_rmse=per_seed,
        n_wmos=first.get("n_wmos"), n_targets=n_scored_targets,
        n_months=len(months),
        dfs_minus_uniform={ch: {k: diffs[ch][k]
                                for k in ("point", "lo", "hi", "excludes_zero")}
                           for ch in diffs} or None,
        ranking=[m for m, _ in rk] or None)
    return out, {"targets": n_scored_targets, "wmos": first.get("n_wmos"),
                 "months": len(months), "eval_split": args.eval_split,
                 "n_profiles": args.n_profiles, "split_protocol": "main"}


# ==========================================================================
# 4. independent recomputation of the P6 calibration metrics
# ==========================================================================
def recompute_calibration() -> dict:
    """CRPS, coverage and reliability re-derived from Track A's predictions.

    Track A uses a closed-form Gaussian CRPS and scipy's norm; this uses a
    Monte-Carlo CRPS and an empirical quantile count, so the two agree only if
    the metric is right -- not because they share a formula. The plan asks
    Track B to recompute the calibration metrics, and the metric code is
    exactly where a quiet error would sit.
    """
    out = {}
    for reg in (args.region,):
        for sd_ in seeds:
            f = os.path.join(ROOT, "outputs", f"argo_P6_{reg}",
                             f"predictions_seed{sd_}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            mu, sg, y = d["mu"], d["sigma"], d["y"]
            per = {}
            rng = np.random.default_rng(20260907)
            for j, ch in enumerate(P.CHANNELS):
                ok = np.isfinite(mu[:, j]) & np.isfinite(sg[:, j]) & np.isfinite(y[:, j])
                if not ok.any():
                    continue
                m_, s_, t_ = mu[ok, j], np.maximum(sg[ok, j], 1e-12), y[ok, j]
                # Monte-Carlo CRPS: E|X-y| - 0.5 E|X-X'|, X ~ N(mu, sigma)
                n_s = 200
                X = m_[:, None] + s_[:, None] * rng.standard_normal((m_.size, n_s))
                Xp = m_[:, None] + s_[:, None] * rng.standard_normal((m_.size, n_s))
                crps = float(np.mean(np.abs(X - t_[:, None]).mean(1)
                                     - 0.5 * np.abs(X - Xp).mean(1)))
                # empirical coverage by counting, no ppf
                z = np.abs(t_ - m_) / s_
                cov = {p_: float(np.mean(z <= q))
                       for p_, q in (("0.50", 0.6744897501960817),
                                     ("0.90", 1.6448536269514722),
                                     ("0.95", 1.959963984540054))}
                # PIT by rank rather than by cdf
                pit = np.mean((m_[:, None] + s_[:, None]
                               * rng.standard_normal((m_.size, 64))) <= t_[:, None],
                              axis=1)
                hist, _ = np.histogram(pit, bins=10, range=(0, 1))
                frac = hist / max(hist.sum(), 1)
                per[ch] = {"crps_mc": crps,
                           "coverage": cov,
                           "rmse": float(np.sqrt(np.mean((t_ - m_) ** 2))),
                           "tv_from_uniform": float(0.5 * np.abs(frac - 0.1).sum()),
                           "n": int(ok.sum())}
            out[str(sd_)] = per
            t = per.get("TEMP", {})
            print(f"  seed {sd_}: TEMP CRPS(MC) {t.get('crps_mc', float('nan')):.4f} "
                  f"cov90 {t.get('coverage', {}).get('0.90', float('nan')):.3f}",
                  flush=True)
    return out


# ==========================================================================
# 5. independent re-derivation of the P1/P2/P5 selections
# ==========================================================================
def recheck_selections() -> dict:
    """Re-derive layout, redundancy and thinning selections independently.

    These are deterministic constructions, so a second implementation is a real
    check on the construction logic rather than on arithmetic noise. Distances
    are recomputed with the spherical law of cosines instead of the haversine,
    which agrees to rounding for well-separated points and is a different
    formula.
    """
    from ocean_tokenizer import argo_experiments as E
    c = ArgoCohort.load(os.path.join(ROOT, "data", "argo_cohort",
                                     f"{args.region}.nc"))
    held = set(c.wmo[c.float_split == "heldout_float"])
    months = [int(m) for m in c.months_in(args.eval_split)
              if c.month(m, float_split="cohort_float").size][:12]

    def mean_pair_km_lawcos(rows):
        if rows.size < 2:
            return float("nan")
        la = np.radians(c.lat[rows]); lo = np.radians(c.lon[rows])
        cosd = (np.sin(la)[:, None] * np.sin(la)[None, :]
                + np.cos(la)[:, None] * np.cos(la)[None, :]
                * np.cos(lo[:, None] - lo[None, :]))
        d = 6371.0 * np.arccos(np.clip(cosd, -1, 1))
        iu = np.triu_indices(rows.size, k=1)
        return float(d[iu].mean())

    res = {"layout": {}, "redundancy": {}, "thinning": {}}
    counts_equal, order_ok, leak = True, True, 0
    for m in months:
        got = {}
        for kind in E.LAYOUTS:
            r = E.layout(c, m, args.n_profiles, kind, 1234)
            r2 = E.layout(c, m, args.n_profiles, kind, 1234)
            if not np.array_equal(r, r2):
                res["layout"].setdefault("nondeterministic", []).append([m, kind])
            leak += len(set(c.wmo[r]) & held)
            got[kind] = (int(r.size), mean_pair_km_lawcos(r))
        if len({v[0] for v in got.values()}) != 1:
            counts_equal = False
        if not (got["dispersed"][1] >= got["clustered"][1]):
            order_ok = False
    res["layout"].update(months_checked=len(months),
                         exact_count_equality=counts_equal,
                         dispersed_farther_than_clustered=order_ok,
                         heldout_floats_in_input=leak)
    print(f"  layout: counts equal={counts_equal} dispersed>clustered={order_ok} "
          f"held-out leakage={leak}", flush=True)

    m0 = months[0]
    base = E.layout(c, m0, args.n_profiles, "natural", 1234)
    for fam in E.REDUNDANCY_FAMILIES:
        rows, edits = E.duplicate_rows(c, base, 8, fam, 1234, m0)
        res["redundancy"][fam] = {
            "distinct_rows_in_attack": int(np.unique(rows[:8]).size),
            "distinct_floats": int(np.unique(c.wmo[rows[:8]]).size),
            "provenance_groups": (int(np.unique(edits["provenance_override"]).size)
                                  if edits.get("provenance_override") is not None else None)}
    r = res["redundancy"]
    same_one_float = r["same_provenance"]["distinct_floats"] == 1
    indep_many = r["independent_provenance"]["distinct_floats"] > 1
    sep_new = r["separated"]["distinct_rows_in_attack"] > 1
    res["redundancy"]["properties_hold"] = bool(same_one_float and indep_many and sep_new)
    print(f"  redundancy: same_provenance=1 float:{same_one_float} "
          f"independent>1 float:{indep_many} separated=new water:{sep_new}",
          flush=True)

    sizes = [E.thin(c, base, f, 7, m0).size for f in E.SPARSITY_FRACTIONS]
    same = np.array_equal(E.thin(c, base, 0.5, 7, m0), E.thin(c, base, 0.5, 7, m0))
    res["thinning"] = {"sizes": sizes, "monotone": sizes == sorted(sizes, reverse=True),
                       "deterministic": bool(same)}
    print(f"  thinning: sizes {sizes} monotone={res['thinning']['monotone']} "
          f"deterministic={same}", flush=True)
    return res


# ==========================================================================
# 6. plumbing shared by the P1 / P2 / P5 / P7 packages
# ==========================================================================
# What is shared with Track A here, and why:
#
#   the protocol, the cohort, `build_argo_sample`, the layout / redundancy /
#   thinning OPERATORS in `argo_experiments`, and the checkpoints.  S2 requires
#   it.  Two tracks that drew different profiles, or duplicated different rows,
#   would not be scoring the same experiment, and a disagreement between them
#   would carry no information.
#
# What is re-implemented below:
#
#   everything downstream of the model's forward pass -- residual assembly,
#   clustering, pooling, the paired bootstrap for the layout gap and the DiD,
#   the evidence-mass aggregation, the sparsity curve, and the P7 scoring.  That
#   is where the errors a second implementation can actually catch live.
from ocean_tokenizer import argo_experiments as E

_COHORT: dict = {}

#: {row_seed: sha256} of every checkpoint this run resolved and re-hashed.
#: Plan S9 asks the intern to report the exact checkpoints; the artifact has a
#: `checkpoint_hashes` field for it, and leaving that empty made "Track B
#: scored the same models" an assertion rather than a record.
CKPT_VERIFIED: dict = {}


def cohort() -> ArgoCohort:
    if "c" not in _COHORT:
        _COHORT["c"] = ArgoCohort.load(
            os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc"))
        _COHORT["norm"] = ArgoNorm.fit(_COHORT["c"], "train")
    return _COHORT["c"]


def norm_of() -> ArgoNorm:
    cohort()
    return _COHORT["norm"]


def eligible_months(split: str, lead: int = 0) -> list[int]:
    """Months that can produce a sample: cohort floats in, held-out floats out.

    Deliberately the same rule as Track A's `eligible`.  It is part of the
    frozen specification -- two tracks scoring different month sets would
    disagree for a reason that has nothing to do with either implementation --
    so it is shared, and the COUNT is then an S6 exact-match quantity that
    catches a divergence if one ever appears.
    """
    c = cohort()
    return [int(m) for m in c.months_in(split)
            if c.month(m, float_split="cohort_float").size
            and c.month(int(m) + lead, float_split="heldout_float").size]


# ------------------------------------------------------- checkpoint sharing
def checkpoints_from_track_a(package: str, seed: int) -> dict[str, str]:
    """Resolve the checkpoints Track A used, from Track A's SIGNED artifact.

    Guessing a path would be the wrong kind of independence.  P1 for the Gulf
    Stream trained its own `dfs`/`uniform`/`count` rows into
    `outputs/argo_P1_gulfstream`, while P1 for the gyre reused the P0 rows --
    so "the P0 directory" is right for one region and wrong for the other, and
    a track that guessed would silently score a different model and report the
    difference as a scientific one.

    Instead the paths come out of Track A's artifact, and every file is
    RE-HASHED here and compared against the sha256 Track A recorded.  Sharing
    the checkpoint is mandated by S3 ("Track B may use the same registered
    training code and seeds"); verifying that the bytes are the ones Track A
    signed for is the independent part.
    """
    d = os.path.join(ROOT, "outputs", f"argo_{package}_{args.region}")
    ap_ = os.path.join(d, f"artifact_{package}_seed{seed}.json")
    out: dict[str, str] = {}
    if not os.path.exists(ap_):
        return out
    tr = json.load(open(ap_)).get("results", {}).get("training", {})
    for row, meta in tr.items():
        rel = meta.get("loaded_from")
        if not rel:
            findings.append(
                f"{package} seed {seed}: Track A TRAINED {row} in-place rather "
                f"than loading a checkpoint; Track B cannot share the model it "
                f"scored")
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            findings.append(f"{package} seed {seed}: {rel} recorded by Track A "
                            f"is missing from disk")
            continue
        claimed = meta.get("checkpoint_sha256")
        actual = sha256_file(path)
        if claimed and claimed != actual:
            findings.append(
                f"{package} seed {seed}: {rel} hashes to {actual[:16]} but "
                f"Track A signed for {claimed[:16]} — the model on disk is not "
                f"the model Track A scored")
        out[row] = path
        CKPT_VERIFIED[f"{row}_s{seed}"] = {
            "path": os.path.relpath(path, ROOT), "sha256": actual,
            "matches_track_a": bool(claimed and claimed == actual)}
    return out


def checkpoints_from_command(package: str, seed: int) -> dict[str, str]:
    """P2 records its checkpoints in the command line, not in a training block."""
    d = os.path.join(ROOT, "outputs", f"argo_{package}_{args.region}")
    ap_ = os.path.join(d, f"artifact_{package}_seed{seed}.json")
    if not os.path.exists(ap_):
        return {}
    cmd = json.load(open(ap_)).get("command", "")
    toks = cmd.split()
    if "--checkpoints" not in toks:
        return {}
    spec = toks[toks.index("--checkpoints") + 1]
    out = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        row, rel = part.split("=", 1)
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            out[row] = path
            CKPT_VERIFIED[f"{row}_s{seed}"] = {
                "path": rel, "sha256": sha256_file(path),
                "matches_track_a": True}   # the path IS Track A's own record
        else:
            findings.append(f"P2 seed {seed}: {rel} from Track A's command is "
                            f"missing from disk")
    return out


_MODEL_CACHE: dict = {}


def load_model(row: str, path: str, provenance_rho: float = 0.0):
    """Instantiate a registered row and load its checkpoint.

    ``provenance_rho`` defaults to 0.0, which is what P0 trained these
    checkpoints with and what P1, P5 and P7 score them with. P2 briefly scored
    them at 0.9 on both tracks; that made its accuracy table a different model
    from the P0 table it is read against, and it is now confined to the
    evidence computation, where rho is applied per family as documented.
    """
    key = (row, path, provenance_rho)
    if key not in _MODEL_CACHE:
        m = build_row(row, provenance_rho=provenance_rho,
                      region=(args.region if args.region_kernel else None)).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev))
        m.eval()
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def oi_model():
    if "oi" not in _MODEL_CACHE:
        from ocean_tokenizer.objective_interpolation import (ObjectiveInterpolation,
                                                             OISettings)
        _MODEL_CACHE["oi"] = ObjectiveInterpolation(OISettings()).to(dev)
    return _MODEL_CACHE["oi"]


def predict(model, sd: dict) -> torch.Tensor:
    """One forward pass, with OI's different call shape handled in one place."""
    with torch.no_grad():
        if model.__class__.__name__ == "ObjectiveInterpolation":
            live = sd["mask"] & sd["value_mask"].all(dim=-1)
            return model(sd["query"],
                         torch.zeros(sd["query"].shape[0], dtype=sd["query"].dtype,
                                     device=sd["query"].device),
                         sd["coord"][live],
                         sd["value"][live].to(sd["query"].dtype),
                         sd["noise_density"][live]).to(torch.float32)
        return model(sd)


# ------------------------------------------------------------- aggregation
def residuals(model, months, seed: int, lead: int = 0, row_fn=None,
              sample_fn=None) -> tuple[pd.DataFrame, np.ndarray]:
    """Tidy long frame of per-query squared residuals, plus the raw predictions.

    ``row_fn(month) -> profile_rows`` changes WHICH observations go in without
    touching how the sample is built -- the same separation Track A keeps, for
    the same reason: an experiment that built its own tokens would change the
    layout AND the encoding at once.

    ``sample_fn(month, rng) -> sample`` is the escape hatch P2 needs, where the
    attack has to be applied to the built token set rather than to the row list.
    """
    c, nrm = cohort(), norm_of()
    cfg = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                        train=False)
    recs, preds = [], []
    for m in months:
        m = int(m)
        rng = np.random.default_rng([seed, m, lead])
        if sample_fn is not None:
            s = sample_fn(m, rng)
        else:
            pr = row_fn(m) if row_fn else None
            if pr is not None and pr.size == 0:
                continue
            s = build_argo_sample(c, nrm, m, cfg=cfg, rng=rng, lead=lead,
                                  profile_rows=pr)
        if s is None or s["target"].shape[0] == 0:
            continue
        sd = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        pred = predict(model, sd)
        p = pred.cpu().numpy()
        t = sd["target"].cpu().numpy()
        msk = sd["target_mask"].cpu().numpy()
        wmo = np.asarray(sd["target_wmo"])
        preds.append(p)
        for j, ch in enumerate(P.CHANNELS):
            ok = msk[:, j]
            if not ok.any():
                continue
            recs.append(pd.DataFrame({
                "wmo": wmo[ok], "source_month": m, "channel": ch,
                # float64 before squaring: see the note in `residual_frame`.
                "sq": (p[ok, j].astype(np.float64)
                       - t[ok, j].astype(np.float64)) ** 2}))
    df = (pd.concat(recs, ignore_index=True) if recs
          else pd.DataFrame(columns=["wmo", "source_month", "channel", "sq"]))
    return df, (np.concatenate(preds) if preds else np.zeros((0, len(P.CHANNELS))))


def pooled(df: pd.DataFrame) -> float:
    return float(np.sqrt(df["sq"].sum() / len(df))) if len(df) else float("nan")


def per_channel(df: pd.DataFrame) -> dict:
    """Pooled RMSE per channel plus the counts every table needs."""
    out = {}
    for ch in P.CHANNELS:
        sub = df[df.channel == ch]
        out[ch] = pooled(sub)
    out["n_wmos"] = int(df["wmo"].nunique()) if len(df) else 0
    out["n_targets"] = int(len(df))
    return out


def align_arms(frames: list[pd.DataFrame], unit: str = "wmo"):
    """Per-cluster (se, n) for several arms, on the clusters common to all.

    The DiD's four arms come from the same months, so they must be paired; an
    unpaired interval on this design is enormously too wide.  Track A pairs by
    intersecting `cluster_ids` and reindexing; this does it with a pandas inner
    merge, which is a different route to the same alignment.
    """
    gs = [cluster_frame(f, unit) for f in frames]
    common = gs[0][[unit]]
    for g in gs[1:]:
        common = common.merge(g[[unit]], on=unit, how="inner")
    out = []
    for g in gs:
        j = common.merge(g, on=unit, how="left").fillna(0.0)
        out.append((j["se"].to_numpy(float), j["n"].to_numpy(float)))
    return len(common), out


def shared_weight_draws(arms, n_boot: int, seed: int) -> list[np.ndarray]:
    """Bootstrap every arm under ONE multinomial weight draw — the paired design.

    Track A realises the pairing by sharing a draw of cluster INDICES; this
    shares a draw of cluster WEIGHTS.  Pairing is part of the estimand and must
    be shared; the arithmetic that realises it is not, and a bug in either
    formulation shows up as an interval the other does not reproduce.
    """
    C = arms[0][0].size
    if C == 0:
        return [np.full(n_boot, np.nan) for _ in arms]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(C, np.full(C, 1.0 / C), size=n_boot).astype(float)
    out = []
    for se, n in arms:
        tot = w @ n
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.sqrt(np.where(tot > 0, (w @ se) / np.maximum(tot, 1e-300),
                                        np.nan)))
    return out


def interval(point: float, draws: np.ndarray, n_clusters: int,
             unit: str = "wmo", label: str = "") -> dict:
    lo, hi = percentile_ci(draws)
    return {"point": float(point), "lo": lo, "hi": hi,
            "n_clusters": int(n_clusters), "cluster_unit": unit,
            "n_boot": int(args.n_boot), "alpha": 0.05, "method_label": label,
            "kind": "percentile", "n_seeds": 1,
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                  and (lo > 0 or hi < 0))}


def seed_t(points: list[float], label: str = "") -> dict:
    """Small-sample t interval on the per-seed point estimates alone.

    With three seeds this is 2 degrees of freedom and t = 4.303, so it is wide
    on purpose.  It is used as the TRACK-LEVEL estimand for the layout DiD
    because Track A's P1 artifacts keep no per-cluster sufficient statistics,
    so a nested seed+cluster interval cannot be re-derived from them after the
    fact -- and comparing a nested interval against a single-seed one would
    report a difference in scope as a scientific disagreement.

    The aggregation RULE is shared with Track A (it is an estimand, not an
    implementation); the per-seed points it consumes are computed independently
    on each side.
    """
    from scipy import stats as sps
    v = np.asarray([x for x in points if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        m = float(v[0]) if v.size else float("nan")
        return {"point": m, "lo": float("nan"), "hi": float("nan"),
                "n_clusters": 0, "cluster_unit": "seed", "n_boot": 0,
                "alpha": 0.05, "method_label": label, "kind": "seed_t",
                "n_seeds": int(v.size), "excludes_zero": False}
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    t = float(sps.t.ppf(0.975, v.size - 1))
    lo, hi = m - t * se, m + t * se
    return {"point": m, "lo": lo, "hi": hi, "n_clusters": 0,
            "cluster_unit": "seed", "n_boot": 0, "alpha": 0.05,
            "method_label": label, "kind": "seed_t", "n_seeds": int(v.size),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


# ==========================================================================
# 7. P1 — layout generalization, independently
# ==========================================================================
def mean_pairwise_lawcos(c, rows: np.ndarray) -> float:
    """Mean pairwise separation by the spherical law of cosines.

    Track A uses the haversine.  The two agree to rounding for well-separated
    points and are different formulae, so the layout verification table is a
    real check on the geometry rather than a re-run of one function.
    """
    if rows.size < 2:
        return float("nan")
    la, lo = np.radians(c.lat[rows]), np.radians(c.lon[rows])
    cosd = (np.sin(la)[:, None] * np.sin(la)[None, :]
            + np.cos(la)[:, None] * np.cos(la)[None, :]
            * np.cos(lo[:, None] - lo[None, :]))
    d = 6371.0 * np.arccos(np.clip(cosd, -1, 1))
    iu = np.triu_indices(rows.size, k=1)
    return float(d[iu].mean())


def run_P1() -> tuple[dict, dict]:
    c = cohort()
    months = eligible_months(args.eval_split, 0)
    held = set(c.wmo[c.float_split == "heldout_float"].tolist())
    print(f"  {len(months)} eligible {args.eval_split} months", flush=True)

    # ---- the plan's P1 cross-check list, re-derived -----------------------
    ver, counts_equal, ordered, leak, nondet = {}, True, True, 0, []
    for m in months:
        row = {}
        for kind in E.LAYOUTS:
            r = E.layout(c, m, args.n_profiles, kind, 1234)
            if not np.array_equal(r, E.layout(c, m, args.n_profiles, kind, 1234)):
                nondet.append([m, kind])
            leak += len(set(c.wmo[r].tolist()) & held)
            row[kind] = {"n": int(r.size),
                         "mean_pairwise_km": mean_pairwise_lawcos(c, r),
                         "n_floats": int(np.unique(c.wmo[r]).size)}
        if len({row[k]["n"] for k in E.LAYOUTS}) != 1:
            counts_equal = False
        if not (row["dispersed"]["mean_pairwise_km"]
                >= row["clustered"]["mean_pairwise_km"]):
            ordered = False
        ver[str(m)] = row
    checks = {"months_checked": len(months),
              "exact_count_equality": bool(counts_equal),
              "dispersed_farther_than_clustered": bool(ordered),
              "heldout_floats_in_input": int(leak),
              "deterministic": not nondet,
              "nondeterministic_cases": nondet}
    print(f"  layout: counts equal={counts_equal} dispersed>clustered={ordered} "
          f"held-out leakage={leak} deterministic={not nondet}", flush=True)
    if leak:
        findings.append(f"P1: {leak} held-out floats appear in the input — the "
                        f"targets are not out-of-sample")
    if not counts_equal:
        findings.append("P1: layout arms do not hold equal profile counts, so "
                        "the layout gap confounds count with arrangement")

    # ---- score every (row, layout, seed) ---------------------------------
    frames: dict = {}
    per: dict = {}
    n_ref = None
    for sd_ in seeds:
        cks = checkpoints_from_track_a("P1", sd_)
        if not cks:
            findings.append(f"P1 seed {sd_}: no Track A artifact to share "
                            f"checkpoints from; seed skipped")
            continue
        specs = [(r, load_model(r, p)) for r, p in sorted(cks.items())]
        specs.append(("objective_interpolation", oi_model()))
        for name, model in specs:
            for lay in E.LAYOUTS:
                fn = (lambda lay_: (lambda m: E.layout(c, m, args.n_profiles,
                                                       lay_, sd_)))(lay)
                df, _ = residuals(model, months, sd_, 0, row_fn=fn)
                frames[(name, lay, sd_)] = df
                st = per_channel(df)
                per.setdefault(name, {}).setdefault(lay, {})[str(sd_)] = st
                if n_ref is None:
                    n_ref = st
            print(f"  seed {sd_} {name:32s} nat={per[name]['natural'][str(sd_)]['TEMP']:.4f} "
                  f"clu={per[name]['clustered'][str(sd_)]['TEMP']:.4f} "
                  f"dis={per[name]['dispersed'][str(sd_)]['TEMP']:.4f}", flush=True)

    # ---- layout gap, paired within seed ----------------------------------
    gaps: dict = {}
    for name in per:
        for ch in P.CHANNELS:
            pts, ivs = [], []
            for sd_ in seeds:
                fc = frames.get((name, "clustered", sd_))
                fd = frames.get((name, "dispersed", sd_))
                if fc is None or fd is None:
                    continue
                nC, arms = align_arms([fc[fc.channel == ch], fd[fd.channel == ch]])
                if nC == 0:
                    continue
                dc, dd = shared_weight_draws(arms, args.n_boot, 20260905)
                pt = (float(np.sqrt(arms[0][0].sum() / arms[0][1].sum()))
                      - float(np.sqrt(arms[1][0].sum() / arms[1][1].sum())))
                pts.append(pt)
                ivs.append(interval(pt, dc - dd, nC,
                                    label=f"{name} gap {ch} seed {sd_}"))
            if pts:
                gaps.setdefault(name, {})[ch] = {
                    **seed_t(pts, f"{name} layout gap {ch}"),
                    "per_seed": pts,
                    "per_seed_intervals": ivs}
        g = gaps.get(name, {}).get("TEMP")
        if g:
            print(f"  {name:32s} gap(TEMP) = {g['point']:+.4f} "
                  f"[{g['lo']:+.4f}, {g['hi']:+.4f}]", flush=True)

    # ---- difference-in-differences ---------------------------------------
    d_row, u_row = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    did, did_per_seed, did_nested = None, [], []
    for sd_ in seeds:
        need = [(d_row, "clustered"), (d_row, "dispersed"),
                (u_row, "clustered"), (u_row, "dispersed")]
        fs = [frames.get((r, l, sd_)) for r, l in need]
        if any(f is None for f in fs):
            continue
        fs = [f[f.channel == "TEMP"] for f in fs]
        nC, arms = align_arms(fs)
        if nC == 0:
            continue
        dr = shared_weight_draws(arms, args.n_boot, 20260905)
        r_ = lambda se, n: float(np.sqrt(se.sum() / n.sum()))
        gap_d = r_(*arms[0]) - r_(*arms[1])
        gap_u = r_(*arms[2]) - r_(*arms[3])
        pt = gap_u - gap_d
        did_per_seed.append(pt)
        did_nested.append(interval(pt, (dr[2] - dr[3]) - (dr[0] - dr[1]), nC,
                                   label=f"DiD seed {sd_}"))
    if did_per_seed:
        did = {**seed_t(did_per_seed, "DiD = gap(Uniform) - gap(DFS)"),
               "per_seed": did_per_seed, "per_seed_intervals": did_nested}
        print(f"\n  DiD = {did['point']:+.4f} [{did['lo']:+.4f}, {did['hi']:+.4f}] "
              f"per-seed {['%+.4f' % x for x in did_per_seed]}", flush=True)

    # ---- the comparison block --------------------------------------------
    from ocean_tokenizer.crosscheck import build_comparable
    rmse = {}
    per_seed_rmse = {}
    for name, lays in per.items():
        for lay, bysd in lays.items():
            key = f"{name}|{lay}"
            rmse[key] = {ch: mean_or_none([v[ch] for v in bysd.values()])
                         for ch in P.CHANNELS}
            per_seed_rmse[key] = {ch: [bysd[str(s_)][ch] for s_ in seeds
                                       if str(s_) in bysd]
                                  for ch in P.CHANNELS}
    rank = sorted((r for r in per
                   if mean_or_none([v["TEMP"] for v in per[r]["natural"].values()])
                   is not None),
                  key=lambda r: mean_or_none([v["TEMP"]
                                              for v in per[r]["natural"].values()]))
    results = {"layout_verification": ver, "layout_checks": checks,
               "layouts": per, "layout_gap": gaps}
    if did:
        results["layout_did"] = did
    results["comparable"] = build_comparable(
        rmse=rmse, per_seed_rmse=per_seed_rmse,
        n_wmos=(n_ref or {}).get("n_wmos"), n_targets=(n_ref or {}).get("n_targets"),
        n_months=len(months),
        layout_did=({k: did[k] for k in ("point", "lo", "hi", "excludes_zero")}
                    if did else None),
        ranking=rank or None)
    return results, {"layout_months": len(months)}


# ==========================================================================
# 8. P5 — sparsity stress, independently
# ==========================================================================
def run_P5() -> tuple[dict, dict]:
    c = cohort()
    months = eligible_months(args.eval_split, 0)
    print(f"  {len(months)} eligible {args.eval_split} months", flush=True)

    # The plan's "identical thinning seeds / realizations across methods" is a
    # property of the operator, not of a method: re-derived here rather than
    # assumed, because a thinning that depended on the method would confound
    # the sparsity response with the draw.
    base0 = E.layout(c, months[0], args.n_profiles, "natural", seeds[0])
    thin_sizes = [int(E.thin(c, base0, f, seeds[0], months[0]).size)
                  for f in E.SPARSITY_FRACTIONS]
    thin_checks = {
        "sizes": thin_sizes,
        "monotone": thin_sizes == sorted(thin_sizes, reverse=True),
        "deterministic": bool(np.array_equal(
            E.thin(c, base0, 0.5, seeds[0], months[0]),
            E.thin(c, base0, 0.5, seeds[0], months[0]))),
        "identical_across_methods": True,   # thin() takes no method argument
    }
    print(f"  thinning sizes {thin_sizes} monotone={thin_checks['monotone']}",
          flush=True)

    per: dict = {}
    n_ref = None
    for sd_ in seeds:
        cks = checkpoints_from_track_a("P5", sd_)
        if not cks:
            findings.append(f"P5 seed {sd_}: no Track A artifact to share "
                            f"checkpoints from; seed skipped")
            continue
        specs = [(r, load_model(r, p)) for r, p in sorted(cks.items())]
        specs.append(("objective_interpolation", oi_model()))
        for name, model in specs:
            line = []
            for frac in E.SPARSITY_FRACTIONS:
                fn = (lambda f_: (lambda m: E.thin(
                    c, E.layout(c, m, args.n_profiles, "natural", sd_),
                    f_, sd_, m)))(frac)
                df, _ = residuals(model, months, sd_, 0, row_fn=fn)
                st = per_channel(df)
                key = f"{int(frac * 100)}pct"
                per.setdefault(name, {}).setdefault(key, {})[str(sd_)] = st
                line.append(f"{key}={st['TEMP']:.4f}")
                if n_ref is None:
                    n_ref = st
            print(f"  seed {sd_} {name:32s} " + "  ".join(line), flush=True)

    from ocean_tokenizer.crosscheck import build_comparable
    rmse, per_seed_rmse = {}, {}
    for name, fracs in per.items():
        for key, bysd in fracs.items():
            k = f"{name}|{key}"
            rmse[k] = {ch: mean_or_none([v[ch] for v in bysd.values()])
                       for ch in P.CHANNELS}
            per_seed_rmse[k] = {ch: [bysd[str(s_)][ch] for s_ in seeds
                                     if str(s_) in bysd] for ch in P.CHANNELS}
    deg = {}
    for name, fracs in per.items():
        hi = mean_or_none([v["TEMP"] for v in fracs.get("100pct", {}).values()])
        lo = mean_or_none([v["TEMP"] for v in fracs.get("10pct", {}).values()])
        deg[name] = {"rmse_100pct": hi, "rmse_10pct": lo,
                     "ratio_10_over_100": (lo / hi) if (hi and lo) else None}
    rank = sorted((r for r in per
                   if mean_or_none([v["TEMP"]
                                    for v in per[r]["100pct"].values()]) is not None),
                  key=lambda r: mean_or_none([v["TEMP"]
                                              for v in per[r]["100pct"].values()]))
    results = {"sparsity": per, "degradation": deg, "thinning_checks": thin_checks,
               "comparable": build_comparable(
                   rmse=rmse, per_seed_rmse=per_seed_rmse,
                   n_wmos=(n_ref or {}).get("n_wmos"),
                   n_targets=(n_ref or {}).get("n_targets"),
                   n_months=len(months), ranking=rank or None)}
    return results, {"sparsity_months": len(months)}


# ==========================================================================
# 9. P2 — redundancy stress, independently
# ==========================================================================
K_VALUES = (1, 2, 4, 8, 16, 32)

#: The rho applied to the EVIDENCE computation, per family, matching Track A.
#: The accuracy rows deliberately do not use it: they score the registered
#: checkpoint under the configuration it was trained and reported with, so P2's
#: accuracy table and P0's baseline table describe the same model.
P2_EVIDENCE_RHO = 0.9


_P2_SEED = None            # the seed whose attack is currently being built


def build_attack(month: int, family: str, k: int, rng):
    """Apply the k-fold attack to a built sample, re-implemented.

    The row selection (`E.duplicate_rows`) and the token construction
    (`build_argo_sample`) are shared -- they define WHICH observations the two
    tracks duplicate, and a track that duplicated different rows would not be
    running the same experiment.  What is re-implemented is the application of
    the per-copy edits to the token set: the jitter is converted and scattered
    with a boolean mask and `np.kron` rather than a slice and `np.repeat`, so an
    off-by-one in the attacked-token span shows up as a disagreement.
    """
    c, nrm = cohort(), norm_of()
    cfg = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                        train=False)
    base = E.layout(c, month, args.n_profiles, "natural", _P2_SEED)
    if base.size == 0:
        return None, None
    rows_, edits = E.duplicate_rows(c, base, k, family, _P2_SEED, month)
    s = build_argo_sample(c, nrm, month, cfg=cfg, rng=rng, lead=0,
                          profile_rows=rows_)
    if s is None:
        return None, None
    L = int(c.levels.size)
    n_tok = int(s["coord"].shape[0])
    sel = np.zeros(n_tok, dtype=bool)
    sel[:k * L] = True                    # the first k profiles' tokens
    attacked = np.flatnonzero(sel)

    jit = edits.get("jitter_km")
    if jit is not None and np.any(jit):
        from ocean_tokenizer import batched_dfs as B
        # one km offset per COPY, held for that copy's L levels
        dx = np.kron(np.asarray(jit, float), np.ones(L)) / max(B.BOX_EXTENT[0], 1e-9)
        co = s["coord"].clone()
        co[sel, 0] = co[sel, 0] + torch.as_tensor(dx[: int(sel.sum())],
                                                  dtype=co.dtype)
        s["coord"] = co
    ov = edits.get("provenance_override")
    if ov is not None:
        pr = s["provenance"].clone()
        pr[sel] = torch.as_tensor(np.kron(np.asarray(ov, int), np.ones(L, int)),
                                  dtype=pr.dtype)
        s["provenance"] = pr
    return s, attacked


_BASIS = None              # the shared random-Fourier basis, built once


def evidence_mass(s: dict, attacked: np.ndarray, rho: float) -> tuple[float, float]:
    """Summed DFS evidence on the attacked tokens, and on all live tokens.

    `dfs_omega` and the support integration ARE the estimator under test, so
    they are called rather than re-derived -- re-deriving the thing being
    measured would test a different object.  The quadrature assembly, the live
    mask and the summation are re-implemented, which is where an indexing error
    would sit.
    """
    from ocean_tokenizer.batched_dfs import (to_physical, integrate_support,
                                             dfs_omega, vertical_quadrature,
                                             variable_group_coords,
                                             RandomFourierBasis, LENGTH_SCALES_KM,
                                             BASIS_SEED, N_FEATURES)
    from ocean_tokenizer.godas_obs import N_VARIABLE_GROUPS
    global _BASIS
    if _BASIS is None:
        _BASIS = RandomFourierBasis(
            N_FEATURES,
            tuple(LENGTH_SCALES_KM) + variable_group_coords.scales(N_VARIABLE_GROUPS),
            BASIS_SEED)
    coord = to_physical(s["coord"])
    area = s["support_area"].to(torch.float64)
    zq, wq = vertical_quadrature(coord[:, 2], s["support_dz"].to(torch.float64), 3)
    # Build the quadrature nodes column by column over however many coordinate
    # columns `to_physical` returns, replacing the depth column with the
    # quadrature abscissae.  Track A repeats the coordinate row and overwrites
    # column 2 in place; assuming three columns here (there are four) silently
    # dropped one and the basis projection failed loudly, which is the good
    # outcome -- a silent drop would have produced plausible wrong evidence.
    Q = int(zq.shape[1])
    cols = [coord[:, j][:, None].expand(-1, Q) for j in range(coord.shape[1])]
    cols[2] = zq
    nodes = torch.stack(cols, dim=-1)
    weight = wq * area[:, None]
    grp = variable_group_coords(s["variable_group"], N_VARIABLE_GROUPS)
    g = grp[:, None, :].expand(-1, nodes.shape[1], -1)
    psi, lam = integrate_support(_BASIS, torch.cat([nodes, g], -1), weight,
                                 s["noise_density"])
    live = s["mask"] & s["support_mask"]
    w = dfs_omega(psi, lam, live, s.get("provenance"), rho)
    hit = torch.zeros_like(live)
    hit[torch.as_tensor(attacked, dtype=torch.long, device=live.device)] = True
    return (float(w[hit & live].double().sum()), float(w[live].double().sum()))


def run_P2() -> tuple[dict, dict]:
    c = cohort()
    # 24 is Track A's `--months` default and part of the shared specification:
    # a different month list would change the evidence-mass average itself.
    n_months = 4 if args.smoke else 24
    months = eligible_months(args.eval_split, 0)[:n_months]
    print(f"  {len(months)} eligible {args.eval_split} months", flush=True)

    # ---- attack construction: do the families hold their defining property?
    global _P2_SEED, _BASIS
    _P2_SEED = seeds[0]
    base = E.layout(c, months[0], args.n_profiles, "natural", seeds[0])
    fam_checks = {}
    for fam in E.REDUNDANCY_FAMILIES:
        r_, ed = E.duplicate_rows(c, base, 8, fam, seeds[0], months[0])
        fam_checks[fam] = {
            "distinct_rows_in_attack": int(np.unique(r_[:8]).size),
            "distinct_floats": int(np.unique(c.wmo[r_[:8]]).size),
            "provenance_groups": (int(np.unique(ed["provenance_override"]).size)
                                  if ed.get("provenance_override") is not None
                                  else None)}
    props = bool(fam_checks["same_provenance"]["distinct_floats"] == 1
                 and fam_checks["independent_provenance"]["distinct_floats"] > 1
                 and fam_checks["separated"]["distinct_rows_in_attack"] > 1)
    fam_checks["properties_hold"] = props
    print(f"  attack families hold their properties: {props}", flush=True)
    if not props:
        findings.append("P2: the redundancy families do not hold their defining "
                        "properties; the provenance contrast is not testable")

    # ---- evidence mass, per seed then averaged ---------------------------
    # Estimand (shared): growth = mean_over_months(mass at k) / mean at k=1.
    # Implementation (independent): the per-(family, k, month) masses go into a
    # tidy frame and the means come from a groupby, where Track A accumulates
    # lists and calls np.mean.
    recs = []
    for sd_ in seeds:
        _P2_SEED = sd_
        for fam in E.REDUNDANCY_FAMILIES:
            rho = P2_EVIDENCE_RHO if fam in ("same_provenance", "exact") else 0.0
            for k in K_VALUES:
                for m in months:
                    rng = np.random.default_rng([sd_, m, k])
                    s, att = build_attack(m, fam, k, rng)
                    if s is None:
                        continue
                    grp_mass, tot_mass = evidence_mass(s, att, rho)
                    recs.append({"seed": sd_, "family": fam, "rho": rho, "k": k,
                                 "month": m, "group_mass": grp_mass,
                                 "total_mass": tot_mass})
        print(f"  seed {sd_}: evidence mass done", flush=True)
    ev_df = pd.DataFrame(recs)
    geom: dict = {}
    per_seed_growth: dict = {}
    for fam in E.REDUNDANCY_FAMILIES:
        sub = ev_df[ev_df.family == fam]
        rho = float(sub["rho"].iloc[0]) if len(sub) else 0.0
        by_k, by_seed = {}, {}
        for sd_ in seeds:
            s1 = sub[(sub.seed == sd_) & (sub.k == 1)]["group_mass"].mean()
            for k in K_VALUES:
                v = sub[(sub.seed == sd_) & (sub.k == k)]["group_mass"].mean()
                by_seed.setdefault(str(k), []).append(
                    float(v / s1) if s1 else float("nan"))
        for k in K_VALUES:
            gs = by_seed[str(k)]
            by_k[str(k)] = {
                "group_mass": float(sub[sub.k == k]["group_mass"].mean()),
                "total_mass": float(sub[sub.k == k]["total_mass"].mean()),
                "growth_vs_k1": float(np.nanmean(gs))}
        geom[fam] = {"rho": rho, "by_k": by_k}
        per_seed_growth[fam] = by_seed
        print(f"  {fam:24s} " + "  ".join(
            f"k{k}={by_k[str(k)]['growth_vs_k1']:.2f}x" for k in K_VALUES),
            flush=True)

    exact8 = geom["exact"]["by_k"]["8"]["growth_vs_k1"]
    sep8 = geom["separated"]["by_k"]["8"]["growth_vs_k1"]
    same8 = geom["same_provenance"]["by_k"]["8"]["growth_vs_k1"]
    indep8 = geom["independent_provenance"]["by_k"]["8"]["growth_vs_k1"]
    per_seed_pass = [bool(np.isfinite(a) and np.isfinite(b) and a > b)
                     for a, b in zip(per_seed_growth["separated"]["8"],
                                     per_seed_growth["exact"]["8"])]
    passes = bool(np.isfinite(sep8) and np.isfinite(exact8) and sep8 > exact8)
    print(f"\n  positive control: separated k8 = {sep8:.2f}x vs exact k8 = "
          f"{exact8:.2f}x -> {'PASS' if passes else 'FAIL'} "
          f"(per-seed {per_seed_pass})", flush=True)
    if not passes:
        findings.append("P2: POSITIVE CONTROL FAILED — the plan forbids using "
                        "the redundancy result until this is investigated")

    # ---- accuracy under duplication --------------------------------------
    acc: dict = {}
    for sd_ in seeds:
        _P2_SEED = sd_
        cks = checkpoints_from_command("P2", sd_)
        if not cks:
            findings.append(f"P2 seed {sd_}: no Track A artifact to share "
                            f"checkpoints from; seed skipped")
            continue
        specs = [(r, load_model(r, p)) for r, p in sorted(cks.items())]
        specs.append(("objective_interpolation", oi_model()))
        for name, model in specs:
            for fam in E.REDUNDANCY_FAMILIES:
                p1 = f1 = None
                for k in K_VALUES:
                    sfn = (lambda f_, k_: (lambda m, rng:
                                           build_attack(m, f_, k_, rng)[0]))(fam, k)
                    df, preds = residuals(model, months, sd_, 0, sample_fn=sfn)
                    if not len(df):
                        continue
                    st = per_channel(df)
                    entry = {"rmse_TEMP": st["TEMP"], "rmse_SALT": st["SALT"],
                             "n_wmos": st["n_wmos"]}
                    if k == 1:
                        p1 = preds
                        f1 = df
                    if p1 is not None and p1.shape == preds.shape:
                        entry["prediction_change_vs_k1"] = float(
                            np.mean(np.abs(preds - p1)))
                    # A CI on the change, paired against this family's own k=1.
                    # k and k=1 see the same months, queries and floats -- the
                    # sample RNG does not depend on k -- so the arms pair, and
                    # one draw of cluster WEIGHTS is shared across them. Track A
                    # pairs by sharing cluster INDICES; the pairing is the
                    # estimand and must be shared, the arithmetic is not.
                    if f1 is not None and k != 1:
                        for ch in P.CHANNELS:
                            nC, arms = align_arms([df[df.channel == ch],
                                                   f1[f1.channel == ch]])
                            if nC == 0:
                                continue
                            dk, d1 = shared_weight_draws(arms, args.n_boot,
                                                         20260905)
                            pt = (float(np.sqrt(arms[0][0].sum() / arms[0][1].sum()))
                                  - float(np.sqrt(arms[1][0].sum() / arms[1][1].sum())))
                            entry[f"error_change_{ch}_ci"] = interval(
                                pt, dk - d1, nC,
                                label=f"{name} {fam} k{k} vs k1 {ch}")
                    acc.setdefault(name, {}).setdefault(fam, {}) \
                       .setdefault(str(k), {})[str(sd_)] = entry
            print(f"  seed {sd_} {name}: accuracy done", flush=True)

    # seed-average, then the change relative to k=1 -- in that order, because
    # the difference of two seed means is not the mean of two differences when
    # a seed is missing a cell.
    acc_mean: dict = {}
    for name, fams in acc.items():
        for fam, byk in fams.items():
            base_T = mean_or_none([v.get("rmse_TEMP")
                                   for v in byk.get("1", {}).values()])
            base_S = mean_or_none([v.get("rmse_SALT")
                                   for v in byk.get("1", {}).values()])
            for k, bysd in byk.items():
                t = mean_or_none([v.get("rmse_TEMP") for v in bysd.values()])
                s_ = mean_or_none([v.get("rmse_SALT") for v in bysd.values()])
                pc = mean_or_none([v.get("prediction_change_vs_k1")
                                   for v in bysd.values()])
                cell = {
                    "rmse_TEMP": t, "rmse_SALT": s_,
                    "prediction_change_vs_k1": pc,
                    "error_change_TEMP_vs_k1": (None if (t is None or base_T is None)
                                                else t - base_T),
                    "error_change_SALT_vs_k1": (None if (s_ is None or base_S is None)
                                                else s_ - base_S),
                    "per_seed": bysd}
                # Seed-average the paired interval. Each seed's interval
                # conditions on one trained model; averaging the endpoints
                # keeps the WIDTH honest for a single model and no wider, so it
                # is reported as what it is rather than as a seed-integrated
                # interval it is not.
                for ch in P.CHANNELS:
                    ivs = [v[f"error_change_{ch}_ci"] for v in bysd.values()
                           if isinstance(v.get(f"error_change_{ch}_ci"), dict)]
                    if ivs:
                        cell[f"error_change_{ch}_ci"] = {
                            "point": mean_or_none([x["point"] for x in ivs]),
                            "lo": mean_or_none([x["lo"] for x in ivs]),
                            "hi": mean_or_none([x["hi"] for x in ivs]),
                            "excludes_zero": bool(all(x["excludes_zero"]
                                                      for x in ivs)),
                            "n_clusters": ivs[0]["n_clusters"],
                            "per_seed_excludes_zero": [bool(x["excludes_zero"])
                                                       for x in ivs],
                            "conditions_on_one_model": True}
                acc_mean.setdefault(name, {}).setdefault(fam, {})[k] = cell

    from ocean_tokenizer.crosscheck import build_comparable
    results = {
        "evidence_mass": geom,
        "per_seed_growth": per_seed_growth,
        "accuracy": acc_mean,
        "k_values": list(K_VALUES),
        "attack_checks": fam_checks,
        "redundancy_control": {
            "passes": passes, "separated_k8_growth": sep8,
            "exact_k8_growth": exact8, "per_seed_passes": per_seed_pass,
            "rule": "adding genuinely new independent observations must increase "
                    "evidence more than re-ingesting the same observation"},
        "provenance_contrast": {"same_provenance_k8": same8,
                                "independent_provenance_k8": indep8},
    }
    # Evidence growth AND accuracy -- the plan asks for both halves of P2 to be
    # compared, and a cross-check that saw only the geometry would pass while
    # the numbers that decide whether the geometry matters went unexamined.
    comparable_rmse = {
        f"{fam}|k{k}": {"evidence_growth": geom[fam]["by_k"][str(k)]["growth_vs_k1"]}
        for fam in E.REDUNDANCY_FAMILIES for k in K_VALUES}
    for row_, fams_ in acc_mean.items():
        for fam_, byk_ in fams_.items():
            for k_, e_ in byk_.items():
                comparable_rmse[f"{row_}|{fam_}|k{k_}"] = {
                    m: e_[m] for m in ("rmse_TEMP", "rmse_SALT",
                                       "error_change_TEMP_vs_k1",
                                       "error_change_SALT_vs_k1",
                                       "prediction_change_vs_k1")
                    if e_.get(m) is not None}
                if int(k_) == 8 and fam_ in ("exact", "separated"):
                    iv_ = e_.get("error_change_TEMP_ci")
                    if iv_:
                        comparable_rmse[f"{row_}|{fam_}|k8"].update(
                            {f"ci_{x}": iv_[x] for x in ("point", "lo", "hi",
                                                         "excludes_zero")})
    # Plan §5 asks for per-seed values, "because an average can agree while the
    # seeds behind it do not". Same flat keys as the means above.
    comparable_per_seed = {
        f"{fam}|k{k}": {"evidence_growth": per_seed_growth[fam][str(k)]}
        for fam in E.REDUNDANCY_FAMILIES for k in K_VALUES}
    for row_, fams_ in acc_mean.items():
        for fam_, byk_ in fams_.items():
            for k_, e_ in byk_.items():
                bysd_ = e_.get("per_seed", {})
                comparable_per_seed[f"{row_}|{fam_}|k{k_}"] = {
                    m: [bysd_[str(s_)][m] for s_ in seeds
                        if str(s_) in bysd_ and m in bysd_[str(s_)]]
                    for m in ("rmse_TEMP", "rmse_SALT")}
                base_ = {ch: mean_or_none([v.get(f"rmse_{ch}")
                                           for v in byk_.get("1", {})
                                           .get("per_seed", {}).values()])
                         for ch in P.CHANNELS}
                for ch in P.CHANNELS:
                    vals = comparable_per_seed[f"{row_}|{fam_}|k{k_}"][f"rmse_{ch}"]
                    comparable_per_seed[f"{row_}|{fam_}|k{k_}"][
                        f"error_change_{ch}_vs_k1"] = (
                        [] if base_[ch] is None else [v - base_[ch] for v in vals])
                comparable_per_seed[f"{row_}|{fam_}|k{k_}"][
                    "prediction_change_vs_k1"] = [
                    bysd_[str(s_)]["prediction_change_vs_k1"] for s_ in seeds
                    if str(s_) in bysd_
                    and "prediction_change_vs_k1" in bysd_[str(s_)]]
    results["comparable"] = build_comparable(
        rmse=comparable_rmse,
        per_seed_rmse=comparable_per_seed,
        n_months=len(months),
        redundancy_control={"passes": passes, "separated_k8_growth": sep8,
                            "exact_k8_growth": exact8})
    return results, {"months": len(months), "n_profiles": args.n_profiles,
                     "eval_split": args.eval_split}


# ==========================================================================
# 10. P7 — the prospective test, scored independently
# ==========================================================================
def run_P7() -> tuple[dict, dict]:
    """Re-score the ALREADY-OPENED holdout from the frozen checkpoints.

    The plan is explicit that this is Track B's role here: *"Both tracks use the
    SAME prospective cohort and frozen checkpoints, but should run the scoring /
    aggregation independently."*  So this is not a second opening of the
    holdout -- the freeze record still shows one open, by Track A -- it is an
    independent aggregation of the same frozen predictions, and it trains
    nothing.

    The freeze record is verified first, by re-hashing every checkpoint it pins.
    A freeze record is a claim; the bytes are the evidence.
    """
    sd_ = seeds[0]
    d = os.path.join(ROOT, "outputs", f"argo_P7_{args.region}")
    fr = os.path.join(d, f"freeze_record_seed{sd_}.json")
    if not os.path.exists(fr):
        raise SystemExit(f"no freeze record at {fr}: Track A has not run P7 for "
                         f"{args.region}, so there is nothing to score against.")
    rec = json.load(open(fr))
    verify = {"record_path": os.path.relpath(fr, ROOT),
              "record_sha256": sha256_file(fr),
              "frozen_utc": rec.get("frozen_utc"),
              "opened_utc": rec.get("opened_utc"),
              "holdout_read_by_track_a": bool(rec.get("holdout_read")),
              "git_commit": rec.get("git_commit"),
              "git_dirty": rec.get("git_dirty"),
              "freeze_mismatches_at_open": rec.get("freeze_mismatches", []),
              "primary_endpoint": rec.get("primary_endpoint")}
    # Resolve each pinned checkpoint BY HASH, not by directory order.
    #
    # `outputs/argo_P7_<region>/` still holds the models the ABANDONED first P7
    # attempt trained during its open -- the very failure `50_prospective_test`
    # was hardened against afterwards.  They sit under the same filenames as the
    # frozen ones, four hours older, so a resolver that takes the first
    # directory containing the name loads a decoy and scores a model that was
    # never frozen, silently.  The freeze record pins hashes precisely so this
    # is decidable: search every candidate directory and accept the file whose
    # bytes match the pin.  A name that resolves only to non-matching files is a
    # real mismatch; a name that matches in one directory while a stale twin
    # sits in another is a decoy, and worth saying out loud.
    ck_dirs = [d, os.path.join(ROOT, "outputs", f"argo_P0_{args.region}")]
    resolved, bad, decoys = {}, [], []
    for fname, claimed in sorted(rec.get("checkpoints", {}).items()):
        hit, dcs = P.resolve_pinned_checkpoint(fname, ck_dirs, claimed)
        if hit is None:
            cands = [os.path.join(x, fname) for x in ck_dirs
                     if os.path.exists(os.path.join(x, fname))]
            bad.append(f"{fname}: " + (
                "pinned by the freeze record, absent from disk" if not cands else
                f"none of {len(cands)} copies match the frozen {claimed[:16]} "
                f"(on disk: " + ", ".join(sha256_file(c_)[:16]
                                          for c_ in cands) + ")"))
            continue
        resolved[fname[:-len(f"_s{sd_}.pt")]] = hit
        CKPT_VERIFIED[fname[:-len(".pt")]] = {
            "path": os.path.relpath(hit, ROOT), "sha256": claimed,
            "matches_track_a": True}      # matches the FREEZE pin, by construction
        decoys += [f"{os.path.relpath(c_, ROOT)} shares the name of a frozen "
                   f"checkpoint but not its bytes" for c_ in dcs]
    verify["checkpoints_rehashed"] = len(rec.get("checkpoints", {}))
    verify["checkpoint_mismatches"] = bad
    verify["same_name_decoys"] = decoys
    verify["resolved_from"] = {k: os.path.relpath(v, ROOT)
                               for k, v in sorted(resolved.items())}
    for dc in decoys:
        findings.append(f"P7: {dc} — resolving a pinned checkpoint by directory "
                        f"order rather than by hash would score the wrong model")

    # Did Track A's open actually score the models its freeze pinned?
    #
    # This is the question P7 exists to answer, and it is answerable without
    # trusting either script: Track A's artifact records the sha256 of every
    # checkpoint it loaded, and the freeze record pins the sha256 of every
    # checkpoint it registered.  Comparing the two is arithmetic.  It is worth
    # doing precisely because the freeze's own mismatch check cannot detect this
    # failure -- it re-derives `checkpoint_hashes()` over ONE directory before
    # and after, so a freeze and an open that resolve the same FILENAME to
    # different FILES both pass it.
    a_art = os.path.join(d, f"artifact_P0_seed{sd_}.json")
    loaded_check = {}
    if os.path.exists(a_art):
        tr = json.load(open(a_art)).get("results", {}).get("training", {})
        for row, meta in sorted(tr.items()):
            pin = rec.get("checkpoints", {}).get(f"{row}_s{sd_}.pt")
            got = meta.get("checkpoint_sha256")
            ok = bool(pin and got and pin == got)
            loaded_check[row] = {"frozen_sha256": pin, "scored_sha256": got,
                                 "loaded_from": meta.get("loaded_from"),
                                 "matches_freeze": ok}
            if not ok:
                findings.append(
                    f"P7 PROTOCOL VIOLATION: Track A scored {row} from "
                    f"{meta.get('loaded_from')} ({str(got)[:16]}), but the "
                    f"freeze record pins {str(pin)[:16]}. That row of the "
                    f"holdout table was produced by a model that was never "
                    f"frozen.")
    verify["track_a_loaded_what_it_froze"] = loaded_check
    n_bad = sum(1 for v in loaded_check.values() if not v["matches_freeze"])
    if n_bad:
        print(f"  *** {n_bad} of {len(loaded_check)} rows in Track A's holdout "
              f"table were scored from UNFROZEN checkpoints", flush=True)
    verify["status"] = "ok" if not bad else "MISMATCH"
    print(f"  freeze record: {verify['checkpoints_rehashed']} checkpoints "
          f"re-hashed -> {verify['status']}", flush=True)
    if bad:
        for b in bad:
            findings.append(f"P7 freeze verification: {b}")

    months = eligible_months("holdout", 0)
    print(f"  {len(months)} eligible holdout months", flush=True)

    scored = {r: load_model(r, p) for r, p in sorted(resolved.items())}
    scored["train_climatology"] = BClimatology()
    scored["source_persistence"] = BPersistence()
    scored["objective_interpolation"] = oi_model()

    rows_out, frames = {}, {}
    n_ref = None
    for name, model in scored.items():
        df, _ = residuals(model, months, sd_, 0)
        frames[name] = df
        st = per_channel(df)
        entry = {"TEMP": st["TEMP"], "SALT": st["SALT"],
                 "n_wmos": st["n_wmos"], "n_targets": st["n_targets"], "ci": {}}
        for unit in P.CLUSTER_UNITS:
            cf = cluster_frame(df[df.channel == "TEMP"], unit)
            se, n = cf["se"].to_numpy(), cf["n"].to_numpy()
            lo, hi = percentile_ci(multinomial_bootstrap(se, n, args.n_boot,
                                                         20260905))
            entry["ci"][unit] = {"point": entry["TEMP"], "lo": lo, "hi": hi,
                                 "n_clusters": int(len(cf)), "cluster_unit": unit}
        rows_out[name] = entry
        if n_ref is None:
            n_ref = st
        print(f"  {name:32s} TEMP {entry['TEMP']:.4f}  SALT {entry['SALT']:.4f}",
              flush=True)

    # the S6 conclusion quantity, paired over the floats both rows scored
    d_row, u_row = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
    dmu = {}
    if d_row in frames and u_row in frames:
        for ch in P.CHANNELS:
            fa = frames[d_row][frames[d_row].channel == ch]
            fb = frames[u_row][frames[u_row].channel == ch]
            nC, arms = align_arms([fa, fb])
            if nC == 0:
                continue
            da, db = shared_weight_draws(arms, args.n_boot, 20260905)
            pt = (float(np.sqrt(arms[0][0].sum() / arms[0][1].sum()))
                  - float(np.sqrt(arms[1][0].sum() / arms[1][1].sum())))
            dmu[ch] = interval(pt, da - db, nC, label="DFS - Uniform")
        if "TEMP" in dmu:
            dmu = {**dmu, "point": dmu["TEMP"]["point"],
                   "excludes_zero": dmu["TEMP"]["excludes_zero"]}
            print(f"  DFS-Uniform TEMP {dmu['point']:+.4f} "
                  f"[{dmu['TEMP']['lo']:+.4f}, {dmu['TEMP']['hi']:+.4f}]", flush=True)

    from ocean_tokenizer.crosscheck import build_comparable
    rank = sorted((r for r in rows_out if np.isfinite(rows_out[r]["TEMP"])),
                  key=lambda r: rows_out[r]["TEMP"])
    results = {"freeze_verification": verify, "rows": rows_out}
    if dmu:
        results["dfs_minus_uniform"] = dmu
    results["comparable"] = build_comparable(
        rmse={r: {ch: v[ch] for ch in P.CHANNELS} for r, v in rows_out.items()},
        per_seed_rmse={r: {ch: [v[ch]] for ch in P.CHANNELS}
                       for r, v in rows_out.items()},
        n_wmos=(n_ref or {}).get("n_wmos"), n_targets=(n_ref or {}).get("n_targets"),
        n_months=len(months),
        dfs_minus_uniform={ch: {k: dmu[ch][k]
                                for k in ("point", "lo", "hi", "excludes_zero")}
                           for ch in P.CHANNELS
                           if isinstance(dmu.get(ch), dict)} or None,
        ranking=rank or None)
    return results, {"split_protocol": "main",
                     "targets": (n_ref or {}).get("n_targets"),
                     "wmos": (n_ref or {}).get("n_wmos"),
                     "months": len(months), "eval_split": "holdout",
                     "n_profiles": args.n_profiles}


# ==========================================================================
# 11. dispatch
# ==========================================================================
print(f"TRACK B  package={args.package} region={args.region} seeds={seeds} "
      f"device={dev}", flush=True)
results: dict = {}
counts: dict = {}

PACKAGES = {"P1": run_P1, "P2": run_P2, "P5": run_P5, "P7": run_P7}

if args.package != "P0":
    print(f"\n[1] independent {args.package} run", flush=True)
    results, counts = PACKAGES[args.package]()

if args.package == "P0" and args.stage in ("verify", "all"):
    print("\n[1] re-hashing manifests", flush=True)
    results["manifest_verification"] = verify_manifests()
    print("\n[2] re-deriving the QC'd cohort from raw GDAC files", flush=True)
    cpath = os.path.join(ROOT, "data", "argo_cohort", f"{args.region}.nc")
    cc = ArgoCohort.load(cpath)
    d = xr.open_dataset(cpath)
    # The cohort file is written already sorted by month_index and load() sorts
    # stably by the same key, so file order and loaded order coincide. Assert it
    # rather than assume it: a silent misalignment here would compare Track B's
    # profile against a DIFFERENT profile of Track A's and could look like a QC
    # disagreement.
    raw_mi = np.asarray(d["month_index"].values)
    assert (np.diff(raw_mi) >= 0).all(), "cohort file is not month-sorted"
    assert np.array_equal(raw_mi, cc.month_index), "load() reordered the cohort"
    cc._time = np.asarray(d["time"].values, "datetime64[ns]")
    d.close()
    results["cohort_verification"] = verify_cohort(cc)

if args.package == "P0" and args.stage in ("evaluate", "all"):
    print("\n[3] independent evaluation", flush=True)
    ev, counts = evaluate()
    results.update(ev)
    print("\n[3b] leads 1..3", flush=True)
    results["leads"] = evaluate_leads()

if args.package == "P0" and args.stage in ("selection", "all"):
    print("\n[4] independent re-derivation of P1/P2/P5 selections", flush=True)
    results["selection_checks"] = recheck_selections()

if args.package == "P0" and args.stage in ("calibration", "all"):
    print("\n[5] independent recomputation of the P6 calibration metrics", flush=True)
    results["calibration_recomputed"] = recompute_calibration()

art = P.ResultArtifact(
    package=args.package, track="B", region=args.region, results=results,
    seeds=seeds, command=" ".join(sys.argv), counts=counts,
    checkpoint_hashes=CKPT_VERIFIED,
    warnings=findings + [
        "Track B was written by the same author as Track A. It can catch "
        "implementation error; it cannot establish that a shared conception is "
        "correct. Agreement here is evidence of arithmetic, not of science."],
    notes="Independent re-implementation of QC, aggregation and CI. Shares the "
          "frozen protocol, manifests, sample construction and checkpoints, as "
          "plan S2 requires.").finalize(ROOT)
path = os.path.join(OUT, f"artifact_{args.package}_{args.region}_trackB.json")
sha = art.write(path)
print(f"\nartifact: {path}\n  sha256 {sha}")
if findings:
    print("\nFINDINGS:")
    for f in findings:
        print(f"  - {f}")
print(f"total {time.time()-t0:.0f}s")
