"""Query-anchored fusion: a satellite first guess, corrected by in-situ innovations.

The model of ``ocean_tokenizer.anchored``, trained and scored on the cells the
audit's runs are scored on (``62_sanity_train.py``): the same cohort, target,
train-year normalisation, months, input profiles and held-out profiles. A zero
prediction must reproduce the cell count and climatology error stored in a
finished run's summary before any number is written.

  first guess   a pointwise network at the query. ``--first-guess``:
                  none      zero anomaly (profiles are the only information)
                  position  position + calendar month (a learned mean field)
                  sat1deg   + the 1-degree monthly SST / SLA / SSS at the query
                  satday    + the daily 0.25-degree SLA on the query's own day
                For the training years it is cross-fitted by year, so the
                innovations the analysis is trained on are out of sample, as
                they are at test time.
  analysis      ``first guess + sum_i w_i * (profile_i - first guess at profile_i)``
                over the ``--k`` nearest input profiles. ``--weights``:
                  none      first guess only
                  softmax   normalised kernel average against a background key
                  dfs       the same with the DFS evidence prior, beta * log tau
                  kriging   the weights of the system the evidence comes from

``--region synthetic`` is the CESM2 cohort (profiles only, no day, no satellite):
``--first-guess none``. ``--region global`` is real Argo; the satellite values
at each profile come from ``experiments/data/49_satellite_at_profiles.py``.

  .venv/bin/python experiments/real_data/67_anchored_fusion.py --region synthetic \
      --first-guess none --weights kriging --tag anc_kriging
  .venv/bin/python experiments/real_data/67_anchored_fusion.py --region global \
      --first-guess satday --weights kriging --tag anc_satday_kriging
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import xarray as xr

from ocean_tokenizer import anchored as A
from ocean_tokenizer import dfs as DFS
from ocean_tokenizer.argo_obs import ArgoNorm
from ocean_tokenizer.audit_tools import cohort_path, load_cohort
from ocean_tokenizer.point_baselines import BANDS, CH, Scores, band_of_levels

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="global", choices=["global", "synthetic"])
ap.add_argument("--cohort", default=None,
                help="synthetic only: another cohort file under data/synthetic_argo/ "
                     "(name without .nc); default the audit's cesm2_uniform")
ap.add_argument("--anomaly", default=None, choices=["cell", "exact"],
                help="climatology at the cell centre or at the profile "
                     "(default: exact for synthetic, cell for global, as the reference runs)")
ap.add_argument("--first-guess", default="satday",
                choices=["none", "position", "sat1deg", "satday"])
ap.add_argument("--weights", default="kriging", choices=["none", "softmax", "dfs", "kriging"])
ap.add_argument("--k", type=int, default=32, help="nearest input profiles per query")
ap.add_argument("--steps", type=int, default=1500, help="optimiser steps of the analysis")
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--queries", type=int, default=512, help="target profiles per step")
ap.add_argument("--val-every", type=int, default=250)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--init-km", type=float, default=250.0)
ap.add_argument("--init-gamma", type=float, default=None,
                help="initial error ratio of the kriging rule (default 0.03 synthetic, 1.0 real)")
ap.add_argument("--no-time", action="store_true", help="real data: ignore the day separation")
ap.add_argument("--no-state", action="store_true",
                help="real data: ignore the sea-level separation between profiles")
ap.add_argument("--fg-epochs", type=int, default=24)
ap.add_argument("--fg-width", type=int, default=512)
ap.add_argument("--tag", default=None)
ap.add_argument("--out-root", default=None)
ap.add_argument("--freeze", action="store_true",
                help="score the initial scales without training (harness check against OI)")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REAL = args.region == "global"
if not REAL and args.first_guess != "none":
    raise SystemExit("the synthetic cohort has no satellite fields: use --first-guess none")
ANOM = args.anomaly or ("cell" if REAL else "exact")
SPLITS = ({"train": (2016, 2020), "validation": (2021, 2021), "development": (2022, 2023)}
          if REAL else
          {"train": (2000, 2003), "validation": (2004, 2004), "development": (2005, 2005)})
REFERENCE = (("setconv_surface" if ANOM == "cell" else "real_exact_r500_g1") if REAL
             else "syn_r500_g1")
TAG = args.tag or f"anc_{args.first_guess}_{args.weights}"
OUT = args.out_root or os.path.join(ROOT, "outputs", "audit", args.region, TAG)
QC = 10.0 if REAL else float("inf")      # gross-error check on INPUT values and the loss
EVAL_SEED, EVAL_CELLS = 20260918, 8000   # 62_sanity_train.py's validation draw
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
torch.manual_seed(args.seed); np.random.seed(args.seed)

# ------------------------------------------------------------------ data
c, _ = load_cohort(ROOT, args.region, SPLITS, anomaly=ANOM, **(
    {"cohort": args.cohort} if args.cohort else {}))
norm = ArgoNorm.fit(c, "train")
LEV = c.levels; L = LEV.size; NF = 2 * L
Y = np.concatenate([norm.z(ch, getattr(c, ch)) for ch in CH], axis=1)      # (P, 2L) z
P = Y.shape[0]
MONTHS = None
SAT = None
if REAL:
    SAT = np.load(os.path.join(ROOT, "data", "argo_cohort", "global_satellite.npz"))
    assert np.array_equal(SAT["month_index"], c.month_index), "satellite file is not this cohort"
    MONTHS = set(int(m) for m in np.unique(SAT["month_index"][np.isfinite(SAT["SLA_1deg"])]))
    d = xr.open_dataset(cohort_path(ROOT, "global"))
    order = np.argsort(np.asarray(d["month_index"].values, int), kind="stable")
    day = (np.asarray(d["time"].values)[order].astype("datetime64[D]")
           - np.datetime64("2000-01-01")).astype(float)
    d.close()


def eval_month(m, max_cells=0):
    src = c.month(m, float_split="cohort_float")
    tgt = c.month(m, float_split="heldout_float")
    R = tgt.size
    prof = np.repeat(np.arange(R), L); lev = np.tile(np.arange(L), R)
    if max_cells and R * L > max_cells:
        pick = np.random.default_rng([EVAL_SEED, int(m)]).choice(R * L, max_cells, replace=False)
        prof, lev = prof[pick], lev[pick]
    return dict(month=int(m), src=src, tgt=tgt, prof=prof, lev=lev,
                target={ch: Y[tgt][prof, k * L + lev] for k, ch in enumerate(CH)})


def months_of(split):
    return [int(m) for m in c.months_in(split)
            if (MONTHS is None or int(m) in MONTHS)
            and c.month(int(m), float_split="cohort_float").size
            and c.month(int(m), float_split="heldout_float").size]


def scored(evs, preds):
    """``preds``: one (R, 2L) array per month, at the month's held-out profiles."""
    s = Scores(LEV, norm.std)
    for ev, p in zip(evs, preds):
        for k, ch in enumerate(CH):
            s.add(ch, p[ev["prof"], k * L + ev["lev"]], ev["target"][ch], ev["lev"])
    return s.result()


val = [eval_month(m, EVAL_CELLS) for m in months_of("validation")]
test = [eval_month(m) for m in months_of("development")]
ref = json.load(open(os.path.join(ROOT, "outputs", "audit", args.region, REFERENCE,
                                  "summary_seed1234.json")))["scores"]
ident = {}
for split, evs in (("validation", val), ("development", test)):
    z = scored(evs, [np.zeros((ev["tgt"].size, NF)) for ev in evs])
    for ch in CH:
        a, b = z[ch], ref[split][ch]
        if a["n"] != b["n"] or abs(a["climatology_z"] / b["climatology_z"] - 1.0) > 1e-5:
            raise SystemExit(f"identity check failed on {split} {ch}: {a['n']} cells, "
                             f"climatology {a['climatology_z']} vs stored {b['n']}, "
                             f"{b['climatology_z']} ({REFERENCE})")
        ident[f"{split}/{ch}"] = {"n": a["n"], "climatology_z": a["climatology_z"]}
print(f"{TAG}: region={args.region} anomaly={ANOM} seed={args.seed} first_guess="
      f"{args.first_guess} weights={args.weights} k={args.k} device={dev}\n"
      f"  {len(val)} validation and {len(test)} test months; a zero prediction matches "
      f"{REFERENCE} ({time.time() - t0:.0f}s)", flush=True)

T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
LAT, LON = T(c.lat), T(c.lon % 360.0)
Yt = T(Y)
YIN = torch.where(Yt.abs() > QC, torch.full_like(Yt, float("nan")), Yt)    # inputs after QC
DAY = STATE = None
USE_TIME = REAL and not args.no_time
USE_STATE = REAL and not args.no_state and args.first_guess == "satday"
if REAL:
    DAY = T(day)
    sla = SAT["SLA_day_c"].astype(float) - SAT["SLA_clim_1deg"].astype(float)
    STATE = T(np.nan_to_num(sla / 0.09))           # 0 = no information on the separation
    STATE_OK = T(np.isfinite(sla), torch.bool)

# ------------------------------------------------------------------ first guess
FG = torch.zeros(P, NF, device=dev)
fg_info = {"kind": args.first_guess}
fg_checkpoint = None
if args.first_guess != "none":
    la, lo = np.deg2rad(c.lat), np.deg2rad(c.lon % 360.0)
    mon = c.month_index % 12 + 1
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    cols = [c.lat / 90.0, xyz]
    for k in range(6):
        cols += [np.sin(2 ** k * np.pi * xyz), np.cos(2 ** k * np.pi * xyz)]
    for h in (1, 2):
        cols += [np.sin(2 * np.pi * h * mon / 12.0), np.cos(2 * np.pi * h * mon / 12.0)]

    def col(name, scale):
        v = SAT[name].astype(float)
        return [np.nan_to_num(v / scale), np.isfinite(v).astype(float)]

    if args.first_guess in ("sat1deg", "satday"):
        cols += col("SLA_1deg_anom", 0.065) + col("SST_1deg_anom", 0.65) \
            + col("SSS_1deg_anom", 0.19) \
            + [np.nan_to_num((SAT["SST_1deg"].astype(float) - 14.0) / 11.0)]
    if args.first_guess == "satday":
        dc, de, dw, dn, ds_ = (SAT[f"SLA_day_{k}"].astype(float) for k in "cewns")
        ok = np.isfinite(dc) & np.isfinite(SAT["SLA_clim_1deg"])
        cols += [np.where(ok, (dc - SAT["SLA_clim_1deg"]) / 0.09, 0.0),
                 np.where(ok & np.isfinite(SAT["SLA_1deg"]), (dc - SAT["SLA_1deg"]) / 0.05, 0.0),
                 np.nan_to_num((de - dw) / 0.05), np.nan_to_num((dn - ds_) / 0.05),
                 ok.astype(float)]
    X = np.column_stack(cols)
    in_sat = np.isin(c.month_index, sorted(MONTHS))
    tr_rows = np.flatnonzero((c.year_split == "train") & (c.float_split == "cohort_float") & in_sat)
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Xt = T((X - mu) / sd)
    Mt = torch.isfinite(Yt) & (Yt.abs() <= QC)
    Y0 = torch.nan_to_num(Yt)

    def fit(rows, seed, select=True):
        """One first-guess network on ``rows``; with ``select`` the epoch is the
        best on the validation year's held-out profiles."""
        torch.manual_seed(seed)
        net = A.FirstGuess(X.shape[1], NF, width=args.fg_width).to(dev)
        bs = 4096
        steps = args.fg_epochs * (rows.size // bs)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        idx = T(rows, torch.long)
        best = (float("inf"), None, args.fg_epochs)
        for ep in range(args.fg_epochs):
            perm = idx[torch.randperm(idx.numel(), device=dev)]
            for i in range(0, perm.numel() - bs + 1, bs):
                b = perm[i:i + bs]
                loss = (((net(Xt[b]) - Y0[b]) ** 2) * Mt[b]).sum() / Mt[b].sum()
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            if select and (ep % 2 == 1 or ep == args.fg_epochs - 1):
                net.eval()
                with torch.no_grad():
                    sc = scored(val, [net(Xt[T(ev["tgt"], torch.long)]).double().cpu().numpy()
                                      for ev in val])
                net.train()
                if sc["macro_z"] < best[0]:
                    best = (sc["macro_z"], {k: v.detach().clone()
                                            for k, v in net.state_dict().items()}, ep + 1)
        if select:
            net.load_state_dict(best[1])
        return net.eval(), best[2]

    @torch.no_grad()
    def apply(net, rows):
        idx = T(rows, torch.long)
        for i in range(0, idx.numel(), 65536):
            FG[idx[i:i + 65536]] = net(Xt[idx[i:i + 65536]])

    final, ep = fit(tr_rows, args.seed)
    apply(final, np.flatnonzero(in_sat & np.isin(c.year_split, ["validation", "development"])))
    if args.weights != "none":
        # cross-fit the training years: each year's first guess comes from a
        # network that never saw that year, trained for the selected epochs
        args_epochs, args.fg_epochs = args.fg_epochs, ep
        for y in sorted(set(c.year[tr_rows].tolist())):
            fold, _ = fit(tr_rows[c.year[tr_rows] != y], args.seed + y, select=False)
            apply(fold, np.flatnonzero(in_sat & (c.year == y)))
        args.fg_epochs = args_epochs
    fg_info.update(inputs=int(X.shape[1]), train_profiles=int(tr_rows.size), best_epoch=int(ep),
                   params=sum(p.numel() for p in final.parameters()))
    fg_checkpoint = {
        "format_version": 1,
        "kind": args.first_guess,
        "state_dict": {k: v.detach().cpu() for k, v in final.state_dict().items()},
        "n_in": int(X.shape[1]), "n_fields": int(NF), "width": int(args.fg_width),
        "depth": 3, "best_epoch": int(ep),
        "feature_mean": torch.as_tensor(mu), "feature_std": torch.as_tensor(sd),
        "target_mean": {ch: torch.as_tensor(norm.mean[ch]) for ch in CH},
        "target_std": {ch: torch.as_tensor(norm.std[ch]) for ch in CH},
        "levels": torch.as_tensor(LEV), "channels": list(CH),
        "feature_recipe": "67_anchored_fusion.py:first guess",
        "feature_fingerprint": hashlib.sha256(memoryview(np.ascontiguousarray(X)).cast("B")).hexdigest(),
    }
    print(f"  first guess trained on {tr_rows.size:,} profiles, {X.shape[1]} inputs, "
          f"epoch {ep} ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ evidence (dfs rule)
TOK_EDGES = (0.0, 50.0, 200.0, 500.0, 1e9)         # the profile encoder's depth bands
BAND_ID = np.searchsorted(TOK_EDGES, LEV, side="right") - 1
BAND_MID = [25.0, 125.0, 350.0, (500.0 + float(LEV.max())) / 2.0]
with torch.no_grad():
    lh, _ = DFS.regime_scales(torch.tensor(BAND_MID))
BAND_ELL = [math.hypot(float(v), DFS.PROTOCOL_SCALE.dx_km) for v in lh]
BAND_NOISE = [0.01 / max(int((BAND_ID == b).sum()), 1) for b in range(4)]
FIELD_BAND = T(np.tile(BAND_ID, 2), torch.long)
ELL_DAYS = math.hypot(DFS.TEMPORAL_SCALE_D, DFS.PROTOCOL_SCALE.dt_d)


def log_mass(rows):
    """(N, 2L) log tau of the input profiles, by the depth band of each field."""
    t = DAY[rows] if USE_TIME else None
    tau = torch.stack([A.profile_leverage(LAT[rows], LON[rows], BAND_ELL[b], BAND_NOISE[b],
                                          k=32, t=t, ell_days=ELL_DAYS if USE_TIME else None)
                       for b in range(4)], dim=1)                    # (N, 4)
    return torch.log(tau.clamp(min=1e-6)).float()[:, FIELD_BAND]


# ------------------------------------------------------------------ analysis
model = None
if args.weights != "none":
    g0 = args.init_gamma if args.init_gamma is not None else (1.0 if REAL else 0.03)
    model = A.InnovationAnalysis(NF, mode=args.weights, k=args.k, init_km=args.init_km,
                                 init_gamma=g0, use_time=USE_TIME, use_state=USE_STATE).to(dev)


def analyse(tgt, src):
    """(R, 2L) prediction at profiles ``tgt`` from input profiles ``src`` (tensors)."""
    if model is None:
        return FG[tgt]
    kw = {}
    if USE_TIME:
        kw.update(q_t=DAY[tgt], o_t=DAY[src])
    if USE_STATE:
        # a missing sea level on either side says nothing about the separation
        kw.update(q_s=torch.where(STATE_OK[tgt], STATE[tgt], torch.zeros_like(STATE[tgt])),
                  o_s=STATE[src])
    if args.weights == "dfs":
        kw.update(log_mass=log_mass(src))
    return FG[tgt] + model(LAT[tgt], LON[tgt], LAT[src], LON[src], YIN[src] - FG[src], **kw)


@torch.no_grad()
def predict(evs, queries=None):
    out = []
    for ev in evs:
        src = T(ev["src"], torch.long)
        tgt = T(ev["tgt"] if queries is None else queries(ev), torch.long)
        out.append(analyse(tgt, src).double().cpu().numpy())
    return out


def describe():
    """Learned scales, averaged over the levels of each scoring band."""
    if model is None:
        return {}
    band = band_of_levels(LEV)
    out = {}
    for name, p in model.named_parameters():
        v = p.detach().cpu().numpy()
        v = np.exp(v) if name.startswith("log_") else v
        out[name.replace("log_", "")] = {
            ch: {b: float(v[k * L:(k + 1) * L][band == b].mean())
                 for b, _, _ in BANDS if (band == b).any()}
            for k, ch in enumerate(CH)}
    return out


hist = []
best = {"macro_z": float("inf"), "step": 0, "state": None}
if model is not None and not args.freeze:
    train_months = months_of("train")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(args.seed)
    run = []
    for step in range(1, args.steps + 1):
        m = int(rng.choice(train_months))
        pool = c.month(m, float_split="cohort_float")
        floats = np.unique(c.wmo[pool])
        held = rng.choice(floats, max(1, int(round(0.3 * floats.size))), replace=False)
        is_t = np.isin(c.wmo[pool], held)
        src, tgt = pool[~is_t], pool[is_t]
        if tgt.size > args.queries:
            tgt = rng.choice(tgt, args.queries, replace=False)
        src, tgt = T(src, torch.long), T(tgt, torch.long)
        pred = analyse(tgt, src)
        mask = torch.isfinite(Yt[tgt]) & (Yt[tgt].abs() <= QC)
        loss = (((pred - torch.nan_to_num(Yt[tgt])) ** 2) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        run.append(float(loss))
        if step % args.val_every == 0 or step == args.steps:
            sc = scored(val, predict(val))
            hist.append({"step": step, "train_loss": float(np.mean(run[-args.val_every:])),
                         "validation/macro_z": sc["macro_z"],
                         **{f"validation/{ch}_J": sc[ch]["J"] for ch in CH}})
            star = ""
            if sc["macro_z"] < best["macro_z"]:
                best = {"macro_z": sc["macro_z"], "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                star = " *"
            print(f"  step {step:5d} loss {hist[-1]['train_loss']:.4f}  validation J "
                  f"TEMP {sc['TEMP']['J']:.4f} SALT {sc['SALT']['J']:.4f}{star} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best["state"])

scores = {"validation": scored(val, predict(val)), "development": scored(test, predict(test))}
if model is not None:
    # copy test on unseen months: the queries ARE input profiles (the answer is
    # in the context); a profile is its own nearest neighbour here
    def some_inputs(ev):
        return np.sort(np.random.default_rng([1, ev["month"]]).choice(
            ev["src"], min(1520, ev["src"].size), replace=False))

    cp = []
    for ev in test:
        q = some_inputs(ev)
        R = q.size
        cp.append(dict(ev, tgt=q, prof=np.repeat(np.arange(R), L), lev=np.tile(np.arange(L), R)))
        cp[-1]["target"] = {ch: Y[q][cp[-1]["prof"], k * L + cp[-1]["lev"]]
                            for k, ch in enumerate(CH)}
    scores["copy_unseen"] = scored(cp, predict(cp))

summary = {"tag": TAG, "region": args.region, "seed": args.seed, "anomaly": ANOM,
           "splits": {k: list(v) for k, v in SPLITS.items()},
           "model": "anchored", "first_guess": fg_info, "weights": args.weights, "k": args.k,
           "use_time": bool(USE_TIME), "use_state": bool(USE_STATE),
           "params": int(sum(p.numel() for p in model.parameters())) if model else 0,
           "steps": args.steps, "lr": args.lr, "best_step": best["step"], "frozen": args.freeze,
           "input_qc_z": None if math.isinf(QC) else QC,
           "identity_check": {"reference": REFERENCE, **ident},
           "learned": describe(), "history": hist, "scores": scores,
           "runtime_s": time.time() - t0}
os.makedirs(OUT, exist_ok=True)
summary["cohort"] = args.cohort or ("global_global" if REAL else "cesm2_uniform")
summary["task"] = "same-month reconstruction; no causal observation cutoff"
summary["evidence_status"] = "development; not independent confirmation"
# Keep the satellite-only comparator on precisely the same queries as the
# fused prediction. Older summaries only saved the small interpolation head.
fg_info["scores"] = {
    "validation": scored(val, [FG[T(ev["tgt"], torch.long)].double().cpu().numpy() for ev in val]),
    "development": scored(test, [FG[T(ev["tgt"], torch.long)].double().cpu().numpy() for ev in test]),
}
fingerprint = hashlib.sha256()
for a in (c.month_index, c.lat, c.lon, c.wmo, LEV, Y):
    a = np.ascontiguousarray(a)
    fingerprint.update(str((a.shape, a.dtype.str)).encode())
    fingerprint.update(memoryview(a).cast("B"))
summary["cohort_fingerprint"] = fingerprint.hexdigest()
summary["source_sha256"] = {
    os.path.basename(path): hashlib.sha256(open(path, "rb").read()).hexdigest()
    for path in (__file__, A.__file__)
}
summary["artifacts"] = {}
if fg_checkpoint is not None:
    fg_name = f"first_guess_seed{args.seed}.pt"
    fg_checkpoint["cohort_fingerprint"] = summary["cohort_fingerprint"]
    fg_checkpoint["source_sha256"] = summary["source_sha256"]
    summary["feature_fingerprint"] = fg_checkpoint["feature_fingerprint"]
    torch.save(fg_checkpoint, os.path.join(OUT, fg_name))
    # Training-year values are out-of-fold; validation/development values are
    # from the final selected network. Save both for matched refiner ablations.
    cache_name = f"first_guess_predictions_seed{args.seed}.pt"
    torch.save({"format_version": 1, "predictions": FG.detach().cpu(),
                "cohort_fingerprint": summary["cohort_fingerprint"],
                "training_predictions": "cross-fit by year" if model is not None else "unused",
                "selected_epoch": int(ep)}, os.path.join(OUT, cache_name))
    summary["artifacts"].update(first_guess=fg_name, first_guess_predictions=cache_name)
if model is not None:
    summary["artifacts"]["analysis"] = f"model_seed{args.seed}.pt"
with open(os.path.join(OUT, f"summary_seed{args.seed}.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
if model is not None:
    torch.save(model.state_dict(), os.path.join(OUT, f"model_seed{args.seed}.pt"))
print(f"\n{TAG} seed {args.seed}: " + "  ".join(
    f"{k} TEMP {v['TEMP']['rmse_physical']:.4f} degC J {v['TEMP']['J']:.4f} / "
    f"SALT {v['SALT']['rmse_physical']:.4f} PSU J {v['SALT']['J']:.4f}"
    for k, v in scores.items()) + f"\n  {time.time() - t0:.0f}s -> {OUT}")
