"""Main tables on the Perceiver-IO family: DFS-Attention + D4RT query decoder.

`45_argo_real_data.py` runs the `GodasRowModel` line (233,859 params). This
runs the architecture the global reconstruction figure uses -- `fusion.
D4RTFusion`: DFS-Attention over a Perceiver-IO latent bottleneck with the D4RT
causal space-time query decoder -- on the same regional Argo cohorts, so
Tables 1-4 describe that model instead.

The matched triple
------------------
All three learned rows are the SAME network with the SAME parameter count; only
the observation-mass rule differs, so a gap between them is attributable to that
rule rather than to capacity, depth, or optimisation.

    dfs_...      measured DFS evidence, conservative transport
    uniform_...  tau = 1, same conservative transport -- duplicates still
                 compete for a fixed slot budget
    count_...    softmax over TOKENS rather than slots, tau ignored -- token
                 multiplicity feeds straight through (the Perceiver rule)

`uniform` and `count` are different claims and the distinction is easy to blur:
handing unit tau to the conservative transport would make `count` a copy of
`uniform`. It is verified behaviourally instead -- given an evidence estimate
that collapses under duplication, conservative total mass holds at 1.00x through
k=8 while count grows 1.18x.

Inputs are PROFILES ONLY. The figure model also ingests surface, WOA and SSH
streams; adding them here would change more than the fusion rule and would
confound the contrast the tables exist to show. The encoders are still built, so
the parameter count stays identical across the three rows.

Ground truth is held-out real Argo: every target is a measurement from a
WMO-disjoint float that appears in no training month. Evaluation observations
are drawn by the same seeded rule `45_argo_real_data.py` uses, so the two model
families are scored on identical inputs at identical queries.

  .venv/bin/python experiments/real_data/58_argo_fusion_tables.py --region gulfstream \
      --seed 1234 --n-profiles 128
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, ArgoObsConfig,
                                      _select_profiles, build_argo_sample,
                                      training_rows)
from ocean_tokenizer.clustered_ci import accumulate, ci_rmse, ci_difference
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.objective_interpolation import (ObjectiveInterpolation,
                                                     OISettings)
from ocean_tokenizer.reference_adapters import GriddedReference

CHANNELS = P.CHANNELS
DEPTH_BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
               ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))
#: table row key -> fusion variant. The keys are the ones `57_main_tables.py`
#: already reads, so Tables 1-4 regenerate with no change to the generator; the
#: variant actually built is recorded in `results["model"]`.
ROWS = {"dfs_expertlocal_cbottle": "d4rt",
        "uniform_expertlocal_cbottle": "d4rt_uniform",
        "count_expertlocal_cbottle": "d4rt_count"}

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="gulfstream", choices=list(P.REGIONS))
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n-profiles", type=int, default=128)
ap.add_argument("--leads", default="0,1,3,6")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--val-every", type=int, default=500)
ap.add_argument("--queries", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--warmup", type=int, default=300)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-latent", type=int, default=32)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=2)
ap.add_argument("--n-dec-blocks", type=int, default=2)
ap.add_argument("--query-chunk", type=int, default=2048)
ap.add_argument("--split-protocol", default="main",
                choices=sorted(P.SPLIT_PROTOCOLS))
ap.add_argument("--eval-split", default="development")
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--cohort", default=None)
ap.add_argument("--reference-suffix", default="_deep")
ap.add_argument("--suffix", default="_fusion")
ap.add_argument("--output", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COHORT = args.cohort or os.path.join(ROOT, "data", "argo_cohort",
                                     f"{args.region}_ext.nc")
OUT = args.output or os.path.join(ROOT, "outputs",
                                  f"argo_P0_{args.region}{args.suffix}")
os.makedirs(OUT, exist_ok=True)
if args.smoke:
    args.steps, args.val_every, args.n_boot = 60, 30, 300
leads = [int(x) for x in args.leads.split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
warnings_: list = []
results: dict = {}

c = ArgoCohort.load(COHORT)
if args.split_protocol != "main":
    # Re-label the YEAR split only.  `apply_splits` deliberately leaves
    # `float_split` alone: the held-out cohort is drawn from WMO ids and is
    # independent of the era, so shifting the era cannot change which floats
    # are held out.  This must happen before `ArgoNorm.fit(..., "train")` or
    # the normalisation would be fitted on the main protocol's training era
    # and would have seen the secondary protocol's evaluation years.
    c.apply_splits(P.SPLIT_PROTOCOLS[args.split_protocol])
    warnings_.append(
        f"split_protocol={args.split_protocol} "
        f"({P.SPLIT_PROTOCOLS[args.split_protocol]}) — a SECONDARY protocol, "
        f"never to be merged into the main headline table")
    print(f"  split protocol: {args.split_protocol} "
          f"{P.SPLIT_PROTOCOLS[args.split_protocol]}", flush=True)
norm = ArgoNorm.fit(c, "train")
LEVELS = c.levels
print(f"fusion tables  region={args.region} seed={args.seed} "
      f"n_profiles={args.n_profiles} leads={leads} device={dev}\n"
      f"  cohort {c.TEMP.shape[0]:,} profiles, {LEVELS.size} levels to "
      f"{LEVELS.max():.0f} m", flush=True)

band_of_level = []
for d in LEVELS:
    for name, lo, hi in DEPTH_BANDS:
        if (lo < d <= hi) or (d <= LEVELS.min() and lo <= 0):
            band_of_level.append(name); break
    else:
        band_of_level.append(DEPTH_BANDS[-1][0])
band_of_level = np.array(band_of_level)

cfg_train = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=args.queries,
                          train=True, max_lead=max(leads),
                          modality_dropout=0.0)
cfg_eval = ArgoObsConfig(n_profiles=args.n_profiles, n_queries=0,
                         train=False, max_lead=max(leads))


def eligible(split: str, lead: int = 0) -> np.ndarray:
    """Months of ``split`` that have both an input and a held-out target."""
    return np.array([int(m) for m in c.months_in(split)
                     if c.month(m, float_split="cohort_float").size
                     and c.month(m + lead, float_split="heldout_float").size],
                    dtype=int)


elig = {k: eligible(k) for k in ("train", "validation", args.eval_split)}
print("  months  " + "  ".join(f"{k}={v.size}" for k, v in elig.items()),
      flush=True)
if elig["train"].size == 0:
    raise SystemExit("no eligible training month — check the cohort splits")


# ------------------------------------------------------------- the model row
class FusionRow(torch.nn.Module):
    """Adapts `D4RTFusion` to the sample dict every other row is scored on.

    The sample is built by `build_argo_sample`, exactly as in
    `45_argo_real_data.py`, so the observations, the targets, the query set and
    the cluster labels are shared with the other family rather than rebuilt.
    This module only re-expresses the SELECTED profiles in the fusion family's
    `obs` layout and asks for predictions at the queries' real coordinates.
    """

    def __init__(self, variant: str, seed: int):
        super().__init__()

        class _Grid:      # the builder needs only the vertical grid
            depth = LEVELS
        self.net = build_fusion_model(
            variant, _Grid(), d_model=args.d_model, n_latent=args.n_latent,
            n_heads=args.n_heads, n_self_blocks=args.n_self_blocks, seed=seed,
            n_dec_blocks=args.n_dec_blocks, max_lead=max(leads),
            query_chunk=args.query_chunk)
        self.variant = variant

    def _obs(self, s: dict) -> dict:
        rows = s["_prof_rows"]
        # NaN, not 0: `ProfileEncoder` derives per-level validity from
        # isfinite(prof), so zero-filling would present a missing level as a
        # measured zero anomaly and hand the model evidence that does not exist.
        prof = np.stack([norm.z("TEMP", c.TEMP[rows]),
                         norm.z("SALT", c.SALT[rows])], axis=1)   # (K,2,L)
        cm = int(s["t_src"]) % 12 + 1
        t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
        return {"profiles": dict(prof=t(prof)[None],
                                 lat=t(c.lat[rows])[None],
                                 lon=t(c.lon[rows])[None],
                                 month=t([cm], torch.long))}

    def _query(self, s: dict) -> torch.Tensor:
        cm = float(int(s["target_month"]) % 12 + 1)
        lat = np.asarray(s["target_lat"], "float32")
        q = np.stack([lat, np.asarray(s["target_lon"], "float32"),
                      np.asarray(s["target_level"], "float32"),
                      np.full(lat.size, cm, "float32")], -1)
        return torch.as_tensor(q, device=dev)[None]

    def forward(self, s: dict) -> torch.Tensor:
        q = self._query(s)
        lead = torch.full(q.shape[:2], int(s["lead"]), dtype=torch.long,
                          device=dev)
        return self.net(self._obs(s), q, lead=lead)[0]


# --------------------------------------------------------------- non-learned
class TrainClimatology:
    """Zero anomaly in z space — the plan's train-climatology floor."""

    def __call__(self, s):
        return torch.zeros_like(s["target"])

    def eval(self): pass
    def train(self): pass


class GriddedRow:
    """EN4 / ECCO scored at exactly the queries every other row is scored on.

    Interpolated to each held-out float's real position and depth, then z-scored
    with the SAME train-only statistics the targets use, so the RMSE is directly
    comparable. A query the product cannot reach is NaN, never zero anomaly:
    zero-filling would score the climatology floor under the product's name.

    Both products assimilate Argo, including the floats being scored. They are
    external references that have seen the answer, not peers.
    """
    ref_levels = None

    def __init__(self, ref, norm):
        self.ref, self.norm = ref, norm
        self.n_query = self.n_covered = 0

    def __call__(self, s):
        lat = np.asarray(s["target_lat"]); lon = np.asarray(s["target_lon"])
        lev = np.asarray(s["target_level"]); m = int(s["target_month"])
        out = np.full((lat.size, len(CHANNELS)), np.nan)
        if self.ref.covers(m):
            ul = self.ref_levels
            T, S = self.ref.predict(m, lat, lon, ul)
            col = np.searchsorted(ul, lev).clip(0, ul.size - 1)
            r = np.arange(lat.size)
            for j, (ch, v) in enumerate((("TEMP", T), ("SALT", S))):
                out[:, j] = ((v[r, col] - self.norm.mean[ch][col])
                             / self.norm.std[ch][col])
        self.n_query += out.shape[0]
        self.n_covered += int(np.isfinite(out[:, 0]).sum())
        return torch.as_tensor(out, dtype=torch.float32, device=dev)

    def eval(self): pass
    def train(self): pass


# -------------------------------------------------------------------- scoring
def eval_sample(month: int, lead: int, seed: int):
    """One evaluation sample, with the chosen input rows attached.

    The rows are drawn with a fresh `default_rng([seed, month, lead])` and then
    passed in explicitly. With ``cfg.train=False`` `_select_profiles` is the
    first consumer of that generator inside `build_argo_sample`, so this picks
    the identical set the default path would have picked — the fusion family is
    scored on the same observations as `45_argo_real_data.py`, not merely on a
    same-sized draw.
    """
    pr = _select_profiles(c, c.month(month, float_split="cohort_float"),
                          cfg_eval.n_profiles,
                          np.random.default_rng([seed, month, lead]))
    if pr.size == 0:
        return None
    s = build_argo_sample(c, norm, month, cfg=cfg_eval,
                          rng=np.random.default_rng([seed, month, lead]),
                          lead=lead, profile_rows=pr)
    if s is None or s["target"].shape[0] == 0:
        return None
    s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
    s["_prof_rows"] = pr
    return s


def _residuals(pred: torch.Tensor, s: dict):
    e = ((pred - s["target"]) ** 2).detach().cpu().numpy()
    m = s["target_mask"].detach().cpu().numpy()
    wmo = np.asarray(s["target_wmo"])
    zq = s["query"][:, 2].detach().cpu().numpy()
    zi = np.rint(zq * max(LEVELS.size - 1, 1)).astype(int).clip(
        0, LEVELS.size - 1)
    return e, m, wmo, zi


def score_model(model, months: np.ndarray, lead: int, seed: int) -> dict:
    is_oi = isinstance(model, ObjectiveInterpolation)
    if hasattr(model, "eval"):
        model.eval()
    acc = {ch: {"wmo": ([], []), "source_month": ([], [])} for ch in CHANNELS}
    band_acc = {ch: {b: [0.0, 0.0] for b, _, _ in DEPTH_BANDS}
                for ch in CHANNELS}
    n_targets = n_months = 0
    wmos_seen: set = set()
    for m in months:
        s = eval_sample(int(m), lead, seed)
        if s is None:
            continue
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
            eok = e[ok, j]; fin = np.isfinite(eok); bl = band_of_level[zi[ok]]
            for bname, _, _ in DEPTH_BANDS:
                sel = (bl == bname) & fin
                if sel.any():
                    band_acc[ch][bname][0] += float(eok[sel].sum())
                    band_acc[ch][bname][1] += float(sel.sum())
    if hasattr(model, "train"):
        model.train()

    out = {"channels": {}, "n_targets": n_targets, "n_months": n_months,
           "n_wmos": len(wmos_seen), "lead": lead}
    for ch in CHANNELS:
        entry = {"clusters": {}}
        for unit in ("wmo", "source_month"):
            labs, errs = acc[ch][unit]
            entry["clusters"][unit] = (
                None if not labs else
                accumulate(np.concatenate(labs), np.concatenate(errs),
                           cluster_unit=unit, channel=ch))
        entry["by_band"] = {
            b: (float(np.sqrt(v[0] / v[1])) if v[1] > 0 else float("nan"))
            for b, v in band_acc[ch].items()}
        out["channels"][ch] = entry
    return out


def stats_to_json(sc: dict, n_boot: int, seed: int,
                  keep_cluster_stats: bool = False) -> dict:
    j = {"n_targets": sc["n_targets"], "n_months": sc["n_months"],
         "n_wmos": sc["n_wmos"], "lead": sc["lead"], "channels": {}}
    for ch, e in sc["channels"].items():
        d = {"by_band": e["by_band"], "ci": {}}
        for unit, st in e["clusters"].items():
            if st is None:
                d["ci"][unit] = None; continue
            d["rmse"] = st.rmse()
            d["ci"][unit] = ci_rmse(
                st, n_boot=n_boot, seed=seed,
                kind="bca" if st.n_clusters < 20 else "percentile").to_dict()
            if keep_cluster_stats:
                d.setdefault("cluster_stats", {})[unit] = {
                    "ids": [str(x) for x in st.cluster_ids],
                    "se": [float(x) for x in st.se],
                    "n": [float(x) for x in st.n]}
        j["channels"][ch] = d
    return j


def macro_score(sc: dict) -> float:
    v = [e["clusters"]["source_month"].rmse()
         if e["clusters"]["source_month"] is not None else np.nan
         for e in sc["channels"].values()]
    return float(np.nanmean(v)) if v else float("nan")


# ------------------------------------------------------------------ training
def train_row(variant: str, seed: int) -> FusionRow:
    """Train one variant; select the checkpoint on the validation split.

    Model selection never touches `--eval-split`, so the reported numbers are
    out of sample in both float identity and year.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    row = FusionRow(variant, seed).to(dev)
    n = sum(p.numel() for p in row.net.parameters())
    opt = torch.optim.AdamW(row.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (min(1.0, (i + 1) / max(args.warmup, 1))
                        * (0.5 * (1 + np.cos(np.pi * min(
                            1.0, i / max(args.steps, 1))))) ** 0.5))
    rng = np.random.default_rng(seed)
    vmonths = elig["validation"]
    best, best_state, hist = np.inf, None, []
    row.train()
    for step in range(args.steps):
        m = int(rng.choice(elig["train"]))
        lead = int(rng.choice(leads))
        pr, tr = training_rows(c, m, lead, args.n_profiles, rng)
        if pr.size == 0 or tr.size == 0:
            continue
        s = build_argo_sample(c, norm, m, cfg=cfg_train, rng=rng, lead=lead,
                              profile_rows=pr, target_rows=tr)
        if s is None or s["target"].shape[0] == 0:
            continue
        s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        s["_prof_rows"] = pr
        pred = row(s)
        msk = s["target_mask"]
        if not msk.any():
            continue
        loss = (((pred - s["target"]) ** 2) * msk).sum() / msk.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(row.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            v = macro_score(score_model(row, vmonths, 0, seed))
            hist.append((step + 1, float(loss.item()), v))
            if np.isfinite(v) and v < best:
                best = v
                best_state = {k: t.detach().clone()
                              for k, t in row.state_dict().items()}
            print(f"    {variant:14s} step {step+1:5d}  loss {loss.item():.4f}"
                  f"  val {v:.4f}{'  *' if v == best else ''}", flush=True)
    if best_state is not None:
        row.load_state_dict(best_state)
    row.eval()
    print(f"  {variant:14s} params={n:,}  best val {best:.4f}", flush=True)
    return row, n, hist


# ---------------------------------------------------------------------- rows
trained: dict = {}
model_meta: dict = {}
for key, variant in ROWS.items():
    r, n, hist = train_row(variant, args.seed)
    trained[key] = r
    model_meta[key] = {"variant": variant, "params": n, "val_history": hist}

trained["train_climatology"] = TrainClimatology()
trained["objective_interpolation"] = ObjectiveInterpolation(OISettings()).to(dev)

pc = {m["params"] for m in model_meta.values()}
if len(pc) != 1:
    warnings_.append(
        f"the three learned rows are NOT parameter-matched: {sorted(pc)}; a "
        f"difference between them is no longer attributable to the mass rule "
        f"alone")
results["model"] = {"family": "fusion.D4RTFusion", "streams": ["profiles"],
                    "rows": model_meta, "parameter_matched": len(pc) == 1}

for name, factory in (("en4", GriddedReference.en4),
                      ("ecco", GriddedReference.ecco)):
    try:
        rdir = os.path.join(ROOT, "data", "reference")
        ref = used = None
        for cand in (args.region + args.reference_suffix, args.region):
            try:
                ref = factory(rdir, cand); used = cand; break
            except Exception:
                continue
        if ref is None:
            raise FileNotFoundError(f"no {name} files for {args.region}")
        if used != args.region + args.reference_suffix:
            warnings_.append(
                f"{name}: the '{args.reference_suffix}' cut is absent, so the "
                f"shallower '{used}' files were used; queries below its "
                f"deepest level are dropped, see query_coverage")
    except Exception as e:
        warnings_.append(f"{name} unavailable: {type(e).__name__}: {e}")
        continue
    covered = [m for lead in leads for m in eligible(args.eval_split, lead)
               if ref.covers(int(m))]
    if not covered:
        # Scoring it anyway would fall back to NaN everywhere and report an
        # empty row as if it were a measurement.
        warnings_.append(
            f"{name} does not cover any {args.eval_split} month (product range "
            f"{ref.time[0]}..{ref.time[-1]}); row omitted")
        print(f"  {name}: no coverage of {args.eval_split} — row omitted",
              flush=True)
        continue
    gr = GriddedRow(ref, norm); gr.ref_levels = LEVELS
    trained[name] = gr
    results.setdefault("reference_coverage", {})[name] = {
        "months_covered": len(covered), "files_used": used,
        "max_depth_m": float(ref.depth.max()),
        "product_range": [str(ref.time[0]), str(ref.time[-1])]}

# ------------------------------------------------------------------- scoring
per_row, cl_stats = {}, {}
for name, model in trained.items():
    per_lead = {}
    for lead in leads:
        sc = score_model(model, eligible(args.eval_split, lead), lead,
                         args.seed)
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

# the headline contrast, paired over held-out floats within this seed
d_key, u_key = "dfs_expertlocal_cbottle", "uniform_expertlocal_cbottle"
if d_key in cl_stats and u_key in cl_stats:
    diff = {}
    for ch in CHANNELS:
        a = cl_stats[d_key]["channels"][ch]["clusters"]["wmo"]
        b = cl_stats[u_key]["channels"][ch]["clusters"]["wmo"]
        if a is not None and b is not None:
            diff[ch] = ci_difference(a, b, n_boot=args.n_boot,
                                     seed=args.seed).to_dict()
    results["dfs_minus_uniform"] = diff

# `n_profiles` is the key `45_argo_real_data.py` already writes, so a table
# can read the density off the artifact instead of inferring it from a
# directory suffix -- which is how an arm gets mislabelled.
counts = {"n_profiles": args.n_profiles,
          "split_protocol": args.split_protocol,
          "eval_split": args.eval_split,
          "eval_months": int(eligible(args.eval_split, 0).size),
          "train_months": int(elig["train"].size),
          "validation_months": int(elig["validation"].size),
          "n_levels": int(LEVELS.size),
          "max_level_m": float(LEVELS.max()),
          "leads": leads, "steps": args.steps}

art = P.ResultArtifact(
    package="P0", track="A", region=args.region, results=results,
    seeds=[args.seed], command=" ".join(sys.argv), counts=counts,
    warnings=warnings_,
    notes=("Perceiver-IO family: fusion.D4RTFusion (DFS-Attention over a "
           "latent bottleneck, D4RT causal query decoder), profiles-only. "
           "Input: real Argo cohort floats. Target: real measurements from "
           "WMO-disjoint held-out floats. No held-out float appears in any "
           "training month. The three learned rows are the same network at the "
           "same parameter count and differ only in the observation-mass rule."
           )).finalize(ROOT)
path = os.path.join(OUT, f"artifact_P0_seed{args.seed}.json")
sha = art.write(path)
print(f"\nartifact: {path}\n  sha256 {sha}\n  protocol {art.protocol_hash_[:16]}"
      f"\ntotal {time.time()-t0:.0f}s")
