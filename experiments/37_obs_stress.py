"""Observation stress test on ONE frozen checkpoint (the DFS-Attention claim).

Fills the gap experiments/08_density_ablation.py names in its own docstring:
that sweep RETRAINS every baseline at every profile density and calls itself
"the retrain-required contrast" to a shared-latent model "evaluated across all
densities".  This is that missing counterpart — one model, trained once, pushed
off its training distribution without ever being refit.

What is under test
------------------
DFS-Attention fuses observations against a climatological background key held
at ``log lambda_bg`` (fusion.py, DFSAttention docstring): *"where evidence is
thin the fusion falls back to climatology, and where it is rich the
observations win."*  That is a falsifiable claim about behaviour OFF the
training density, and nothing in the repo tests it — ``23_dfs_evidence_probes``
probes the estimator analytically with no trained weights, and ``invariance.py``
tests token algebra.  Here the trained model meets observing conditions it was
never fit to.

Design decision (the thing that makes it a stress test)
-------------------------------------------------------
Training uses a FIXED profile count (``--train-density``, default 1000), NOT
the protocol's ``K ~ U{0..3000}`` augmentation.  Under augmentation every
density is in-distribution and the sweep measures interpolation; at a fixed
density, 100 and 3000 are genuine extrapolation.  ``--aug-max`` restores the
augmented control.

Stress axes (all on the SAME frozen weights, inference only)
------------------------------------------------------------
1. density      profiles/month over ``--densities`` (0 -> 3000).  Headline:
                RMSE / climatology floor, floor recomputed per cell because
                unobserved-only scoring changes the query pool with density.
2. redundancy   the same profiles re-ingested k times.  DFS should discount
                copies as the same evidence; attention without an evidence
                model over-weights them.  The sharpest architecture-specific
                probe here.
3. coverage     the same profile COUNT confined to a longitude band, scored
                inside vs outside the band — the spatial form of the fallback
                claim.
4. modality     drop SST/SSS, drop the WOA background, profiles only.  Tests
                the "missing modality is an absent key" contract and D4RT's
                availability-conditioned ReferenceSlots.

Reading the result
------------------
* ratio -> 1.0 as density -> 0, never materially above 1.0  => fallback works.
* ratio ABOVE 1.0 at low density                           => the model
  over-trusts sparse observations; lambda_bg is not doing its job.  A real
  finding, not a failed run.
* duplicates moving the prediction                         => the evidence
  estimator is not discounting redundancy, which undercuts the DFS premise.

Run (DFS/D4RT and the no-evidence control in parallel on two GPUs):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/37_obs_stress.py --variant d4rt
    CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/37_obs_stress.py --variant perceiver
    .venv/bin/python experiments/37_obs_stress.py --smoke        # 2 min wiring check
"""
import sys, os, json, time, argparse, subprocess, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import data, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.fullrun import (FullRunData, VARS, BANDS,
                                     assert_nondegenerate_climatology,
                                     drop_observed_columns)

CONTEXT = 1
UNITS = {"TEMP": "degC", "SALT": "PSU"}
FULL = ("profiles", "surf", "woa")


def _years(s):
    a, _, b = s.partition("-")
    return (int(a), int(b or a))


ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="d4rt",
                choices=["d4rt", "dfs", "mbca", "perceiver", "resampler"],
                help="d4rt/dfs carry the evidence model; perceiver is the "
                     "no-evidence control the fallback claim is measured against")
ap.add_argument("--train-years", type=_years, default="2000-2003")
ap.add_argument("--val-years", type=_years, default="2004")
ap.add_argument("--test-years", type=_years, default="2005")
ap.add_argument("--train-density", type=int, default=1000,
                help="FIXED profiles/month during training (the stress-test "
                     "design: densities either side of this are extrapolation)")
ap.add_argument("--aug-max", type=int, default=0,
                help=">0 restores K ~ U{aug-min..aug-max} augmentation, which "
                     "makes the sweep interpolation rather than extrapolation")
ap.add_argument("--aug-min", type=int, default=0)
ap.add_argument("--densities", default="0,50,100,250,500,1000,1500,2000,3000")
ap.add_argument("--dup-factors", default="2,3",
                help="redundancy axis: re-ingest the training density this many times")
ap.add_argument("--cluster-fracs", default="0.25,0.10",
                help="coverage axis: fraction of the longitude range the "
                     "profiles are confined to (same total count)")
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--steps", type=int, default=10_000)
ap.add_argument("--queries", type=int, default=4096)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lr-min", type=float, default=1e-5)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--n-latent", type=int, default=32)
ap.add_argument("--n-heads", type=int, default=4)
ap.add_argument("--n-self-blocks", type=int, default=2)
ap.add_argument("--n-dec-blocks", type=int, default=2)
ap.add_argument("--anchor-grid", default="12,24")
ap.add_argument("--query-chunk", type=int, default=512)
ap.add_argument("--eval-chunk", type=int, default=32768)
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--val-queries", type=int, default=4096)
ap.add_argument("--stress-queries", type=int, default=30_000,
                help="query subsample per test month per stress cell")
ap.add_argument("--ckpt", default="", help="skip training, stress this checkpoint")
ap.add_argument("--tag", default="")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

TRAIN_YEARS, VAL_YEARS, TEST_YEARS = args.train_years, args.val_years, args.test_years
DENSITIES = [int(x) for x in args.densities.split(",") if x != ""]
DUPS = [int(x) for x in args.dup_factors.split(",") if x != ""]
CLUSTERS = [float(x) for x in args.cluster_fracs.split(",") if x != ""]
tag = args.tag or f"stress_{args.variant}_d{args.train_density}_s{args.seed}"
if args.smoke:
    args.steps, args.val_every, args.queries, args.val_queries = 40, 20, 512, 512
    args.stress_queries = 3000
    DENSITIES, DUPS, CLUSTERS = [0, 1000, 3000], [2], [0.25]
    tag += "_smoke"

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
t_start = time.time()
HAS_EVIDENCE = args.variant in ("d4rt", "dfs")
print(f"[{tag}] variant={args.variant} train_density={args.train_density} "
      f"steps={args.steps} device={dev} smoke={args.smoke}", flush=True)

# ====================================================================== data
grid = data.CommonGrid()
print(grid, flush=True)
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = data.select_month_indices(C.GT_SOURCE, TEST_YEARS)
print(f"months: train={tr_idx.size} val={va_idx.size} test={te_idx.size}", flush=True)

ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
assert_nondegenerate_climatology(ftrain["months"])
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rd = FullRunData(grid, norm, dev)
D, H, W, HW = rd.D, rd.H, rd.W, rd.HW
oi, oj, n_ocean = rd.oi, rd.oj, rd.n_ocean

ZA = torch.from_numpy(rd.z_volume(ftrain)).to(dev)
ZAf = ZA.view(len(tr_idx), 2, D * HW)
surfZ = torch.from_numpy(rd.z_surf(ftrain)).to(dev)
tr_months = ftrain["months"].copy()
valid_tr = [torch.isfinite(ZA[t]).all(0).view(-1).nonzero(as_tuple=True)[0]
            .cpu().numpy().astype("int64") for t in range(len(tr_idx))]
del ftrain
rd.load_woa(data.woa_prior(grid))

fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
ZAv = torch.from_numpy(rd.z_volume(fval)).to(dev)
surfZv = torch.from_numpy(rd.z_surf(fval)).to(dev)
v_months = fval["months"].copy()
ZAt = torch.from_numpy(rd.z_volume(ftest)).to(dev)
surfZt = torch.from_numpy(rd.z_surf(ftest)).to(dev)
t_months = ftest["months"].copy()
del fval, ftest
print(f"  data ready in {time.time()-ts:.1f}s", flush=True)


# ======================================================== pack construction
def build_packs(ZAvol, surfZa, months, sampler, rng, include=FULL,
                subsample=0):
    """Eval packs for one observing scenario.

    ``sampler(rng) -> (ii, jj)`` chooses the profile columns for a month; the
    scored pool is always the UNOBSERVED finite cells, so the query set (and
    therefore the climatology floor) is re-derived per scenario rather than
    reused across densities.
    """
    packs = []
    n_level = torch.zeros(D, dtype=torch.float64, device=dev)
    se0 = torch.zeros(D, 2, dtype=torch.float64, device=dev)
    for t, mo in enumerate(months):
        mo = int(mo)
        ii_np, jj_np = sampler(rng)
        ii_t = torch.from_numpy(np.ascontiguousarray(ii_np)).to(dev)
        jj_t = torch.from_numpy(np.ascontiguousarray(jj_np)).to(dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)
        if ii_t.numel():
            col[ii_t * W + jj_t] = True
        fin = torch.isfinite(ZAvol[t]).all(0).view(D, HW)
        idx = (fin & (~col)[None]).view(-1).nonzero(as_tuple=True)[0]
        if subsample and idx.numel() > subsample:
            sel = rng.choice(idx.numel(), size=subsample, replace=False)
            idx = idx[torch.from_numpy(np.sort(sel)).to(dev)]
        y = ZAvol[t].view(2, -1)[:, idx].T.contiguous()
        q, di = rd.q_from_flat(idx, mo)
        obs = rd.obs_dict(ZAvol, surfZa, t, mo, ii_t, jj_t, include=include,
                          context=CONTEXT)
        packs.append(dict(obs=obs, q=q, y=y, di=di, idx=idx, mo=mo, t=t))
        n_level += torch.bincount(di, minlength=D).double()
        se0.index_add_(0, di, y.double() ** 2)
    return packs, n_level, se0


def sampler_uniform(k):
    def f(rng):
        if k <= 0:
            return np.zeros(0, "int64"), np.zeros(0, "int64")
        pick = rng.choice(n_ocean, size=min(k, n_ocean), replace=False)
        return oi[pick], oj[pick]
    return f


def sampler_duplicated(k, factor):
    """The SAME k columns re-ingested ``factor`` times — identical evidence,
    more tokens.  DFS should discount the copies; plain attention cannot."""
    def f(rng):
        pick = rng.choice(n_ocean, size=min(k, n_ocean), replace=False)
        return np.tile(oi[pick], factor), np.tile(oj[pick], factor)
    return f


def sampler_clustered(k, frac):
    """Same count, confined to a contiguous longitude band of width frac*W."""
    width = max(1, int(round(frac * W)))
    def f(rng):
        start = int(rng.integers(0, W))
        band = (np.arange(width) + start) % W
        cand = np.where(np.isin(oj, band))[0]
        if cand.size == 0:
            return np.zeros(0, "int64"), np.zeros(0, "int64")
        pick = rng.choice(cand, size=min(k, cand.size), replace=False)
        return oi[pick], oj[pick]
    return f, width


# ================================================================ evaluation
@torch.no_grad()
def eval_sse(packs):
    """Per-level z-space SSE (D,2) + mean total DFS evidence over months."""
    model.eval()
    se = torch.zeros(D, 2, dtype=torch.float64, device=dev)
    ev, n_tok = [], []
    for p in packs:
        tokens = model.encode(p["obs"], batch=1, device=dev)
        z = model.fuse(tokens)
        n_tok.append(float(tokens.mask.sum()))
        if HAS_EVIDENCE and getattr(model, "last_evidence", None) is not None:
            ev.append(float(model.last_evidence["total"].sum()))
        Q = p["q"].shape[1]
        for i in range(0, Q, args.eval_chunk):
            out = model.decode(z, p["q"][:, i:i + args.eval_chunk])[0]
            se.index_add_(0, p["di"][i:i + args.eval_chunk],
                          (out - p["y"][i:i + args.eval_chunk]).double() ** 2)
    model.train()
    return se, (float(np.mean(ev)) if ev else None), float(np.mean(n_tok))


def cell_metrics(packs, n_level, se0):
    """One stress cell -> physical RMSE, its own climatology floor, and ratio."""
    se, evidence, n_tok = eval_sse(packs)
    got = rd.physical_rmse(se, n_level)
    floor = rd.physical_rmse(se0, n_level)
    return {
        "rmse": got["full"], "floor": floor["full"],
        "ratio": {v: got["full"][v] / max(floor["full"][v], 1e-12) for v in VARS},
        "by_band": got["by_band"], "floor_by_band": floor["by_band"],
        "evidence_total": evidence, "mean_live_tokens": n_tok,
        "queries": int(n_level.sum()),
    }


# ===================================================================== model
anchor = tuple(int(x) for x in args.anchor_grid.split(",")) if args.anchor_grid else None
build_kw = dict(d_model=args.d_model, n_latent=args.n_latent, n_heads=args.n_heads,
                n_self_blocks=args.n_self_blocks, seed=args.seed, anchor_grid=anchor)
if args.variant == "d4rt":
    build_kw.update(n_dec_blocks=args.n_dec_blocks, max_lead=1,
                    query_chunk=args.query_chunk)
model = build_fusion_model(args.variant, grid, **build_kw).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"model {args.variant}: params={n_params:,} d={args.d_model} "
      f"L={model.n_latent}{' anchored' if anchor else ''} "
      f"evidence_model={HAS_EVIDENCE}", flush=True)

os.makedirs(C.CKPT, exist_ok=True); os.makedirs(C.CACHE, exist_ok=True)
ckpt_path = args.ckpt or os.path.join(C.CKPT, f"{tag}.pt")

# ------------------------------------------------------------------- train
val_rng = np.random.default_rng([args.seed, 1])
val_packs, val_n, val_se0 = build_packs(ZAv, surfZv, v_months,
                                        sampler_uniform(args.train_density),
                                        val_rng, subsample=args.val_queries)
VAL_FLOOR = rd.physical_rmse(val_se0, val_n)["full"]
print(f"val floor TEMP={VAL_FLOOR['TEMP']:.4f} SALT={VAL_FLOOR['SALT']:.4f}", flush=True)

curves = {"step": [], "loss": [], "val_step": [], "val_score": []}
best = {"score": float("inf"), "step": -1}
if args.ckpt:
    print(f"loading {args.ckpt} (no training)", flush=True)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["state_dict"])
    train_secs = 0.0
else:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def _lr(step):
        if args.warmup and step < args.warmup:
            return (step + 1) / args.warmup
        p = min(max((step - args.warmup) / max(args.steps - args.warmup, 1), 0.), 1.)
        lo = args.lr_min / args.lr
        return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    rng = np.random.default_rng([args.seed, 0])
    Ttr, win = len(tr_idx), []
    t_train = time.time()
    for step in range(1, args.steps + 1):
        t = int(rng.integers(Ttr)); mo = int(tr_months[t])
        K = (int(rng.integers(args.aug_min, args.aug_max + 1))
             if args.aug_max > 0 else args.train_density)
        pick = rng.choice(n_ocean, size=min(K, n_ocean), replace=False)
        ii_t = torch.from_numpy(oi[pick]).to(dev)
        jj_t = torch.from_numpy(oj[pick]).to(dev)
        col = torch.zeros(HW, dtype=torch.bool, device=dev)
        col[ii_t * W + jj_t] = True
        pool = valid_tr[t]
        over = int(args.queries * (1.0 + 1.3 * K / max(n_ocean, 1)) + 64)
        cand = pool[rng.choice(pool.size, size=min(over, pool.size), replace=False)]
        idx_t = drop_observed_columns(torch.from_numpy(cand).to(dev), col, HW)[:args.queries]
        if idx_t.numel() == 0:
            continue
        y = ZAf[t][:, idx_t].T
        q, _ = rd.q_from_flat(idx_t, mo)
        obs = rd.obs_dict(ZA, surfZ, t, mo, ii_t, jj_t, include=FULL, context=CONTEXT)
        out = model(obs, q)[0]
        loss = (out - y).pow(2).mean()
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); win.append(float(loss.detach()))
        if step % 200 == 0:
            sps = step / (time.time() - t_train)
            curves["step"].append(step); curves["loss"].append(float(np.mean(win)))
            print(f"  step {step:6d}/{args.steps} loss {np.mean(win):.4f} "
                  f"{sps:.2f} it/s eta {(args.steps-step)/max(sps,1e-9)/60:.0f} min",
                  flush=True)
            win = []
        if args.val_every and step % args.val_every == 0:
            se, _, _ = eval_sse(val_packs)
            vr = rd.physical_rmse(se, val_n)["full"]
            vs = float(np.mean([vr[v] / VAL_FLOOR[v] for v in VARS]))
            curves["val_step"].append(step); curves["val_score"].append(vs)
            f_ = ""
            if vs < best["score"]:
                best = {"score": vs, "step": step, **vr}; f_ = "  *best*"
                torch.save({"state_dict": model.state_dict(), "step": step,
                            "tag": tag, "variant": args.variant,
                            "args": vars(args)}, ckpt_path)
            print(f"  [val] step {step} TEMP {vr['TEMP']:.4f} SALT {vr['SALT']:.4f} "
                  f"score {vs:.4f}{f_}", flush=True)
    train_secs = time.time() - t_train
    print(f"training done in {train_secs/60:.1f} min; best step {best['step']} "
          f"(score {best['score']:.4f})", flush=True)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev)["state_dict"])

# ============================================================= stress sweep
print("\n=== STRESS SWEEP (frozen weights, inference only) ===", flush=True)
results = {"density": {}, "redundancy": {}, "coverage": {}, "modality": {}}
srng = lambda salt: np.random.default_rng([args.seed, 7, salt])

print("[1] profile density", flush=True)
for k in DENSITIES:
    p, n_, s0 = build_packs(ZAt, surfZt, t_months, sampler_uniform(k), srng(k),
                            subsample=args.stress_queries)
    m = cell_metrics(p, n_, s0)
    results["density"][str(k)] = m
    mark = "  <- training density" if k == args.train_density else ""
    ev = f" evid={m['evidence_total']:.1f}" if m["evidence_total"] else ""
    print(f"  N={k:5d}  TEMP {m['rmse']['TEMP']:.4f}/{m['floor']['TEMP']:.4f} "
          f"ratio {m['ratio']['TEMP']:.4f}  SALT ratio {m['ratio']['SALT']:.4f}"
          f"{ev}{mark}", flush=True)

print("[2] redundancy (same profiles re-ingested)", flush=True)
base = results["density"].get(str(args.train_density))
for f_ in DUPS:
    p, n_, s0 = build_packs(ZAt, surfZt, t_months,
                            sampler_duplicated(args.train_density, f_),
                            srng(100 + f_), subsample=args.stress_queries)
    m = cell_metrics(p, n_, s0)
    results["redundancy"][str(f_)] = m
    d = (f"  dTEMP_ratio {m['ratio']['TEMP'] - base['ratio']['TEMP']:+.4f}"
         if base else "")
    ev = f" evid={m['evidence_total']:.1f}" if m["evidence_total"] else ""
    print(f"  x{f_} copies  TEMP ratio {m['ratio']['TEMP']:.4f}  "
          f"tokens {m['mean_live_tokens']:.0f}{ev}{d}", flush=True)

print("[3] coverage (same count, confined to a longitude band)", flush=True)
for fr in CLUSTERS:
    smp, width = sampler_clustered(args.train_density, fr)
    p, n_, s0 = build_packs(ZAt, surfZt, t_months, smp, srng(int(fr * 1000)),
                            subsample=args.stress_queries)
    m = cell_metrics(p, n_, s0)
    m["band_width_lon"] = width
    results["coverage"][f"{fr:g}"] = m
    print(f"  band={fr:g} ({width} lon)  TEMP ratio {m['ratio']['TEMP']:.4f}  "
          f"SALT ratio {m['ratio']['SALT']:.4f}", flush=True)

print("[4] modality dropout", flush=True)
for name, inc in [("full", FULL), ("no_surf", ("profiles", "woa")),
                  ("no_woa", ("profiles", "surf")), ("profiles_only", ("profiles",)),
                  ("no_profiles", ("surf", "woa"))]:
    p, n_, s0 = build_packs(ZAt, surfZt, t_months,
                            sampler_uniform(args.train_density), srng(200),
                            include=inc, subsample=args.stress_queries)
    m = cell_metrics(p, n_, s0)
    results["modality"][name] = m
    print(f"  {name:14s} TEMP ratio {m['ratio']['TEMP']:.4f}  "
          f"SALT ratio {m['ratio']['SALT']:.4f}", flush=True)

# ==================================================================== figure
dens = sorted(int(k) for k in results["density"])
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)
for v, col in zip(VARS, ("tab:red", "tab:blue")):
    ax[0].plot(dens, [results["density"][str(k)]["ratio"][v] for k in dens],
               "o-", color=col, label=v)
ax[0].axhline(1.0, color="k", ls="--", lw=1, label="climatology")
ax[0].axvline(args.train_density, color="grey", ls=":", lw=1.2,
              label=f"trained @ {args.train_density}")
# symlog only earns its keep across many decades; with a handful of points it
# prints negative ticks for a count that is never negative
if len(dens) > 5 and max(dens) / max(min(d for d in dens if d) if any(dens) else 1, 1) > 20:
    ax[0].set_xscale("symlog", linthresh=50)
ax[0].set_xticks(dens)
ax[0].set_xlabel("synthetic Argo profiles / month")
ax[0].set_ylabel("RMSE / climatology floor")
ax[0].set_title("Degradation off the training density")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

for v, col in zip(VARS, ("tab:red", "tab:blue")):
    ax[1].plot(dens, [results["density"][str(k)]["rmse"][v] for k in dens],
               "o-", color=col, label=f"{v} model")
    ax[1].plot(dens, [results["density"][str(k)]["floor"][v] for k in dens],
               "--", color=col, alpha=.5, label=f"{v} floor")
ax[1].set_yscale("log"); ax[1].set_xticks(dens)
ax[1].set_xlabel("synthetic Argo profiles / month")
ax[1].set_ylabel("physical RMSE (degC / PSU)")
ax[1].set_title("Absolute error vs the floor")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

if HAS_EVIDENCE and results["density"][str(dens[-1])]["evidence_total"]:
    ax[2].plot(dens, [results["density"][str(k)]["evidence_total"] for k in dens],
               "o-", color="tab:green")
    ax[2].set_xticks(dens); ax[2].set_xlabel("profiles / month")
    ax[2].set_ylabel("total DFS evidence  (sum tau)")
    ax[2].set_title("Estimated degrees of freedom for signal")
    ax[2].grid(alpha=.3)
else:
    names = list(results["modality"]); xs = np.arange(len(names))
    for i, v in enumerate(VARS):
        ax[2].bar(xs + i * .38 - .19,
                  [results["modality"][n]["ratio"][v] for n in names],
                  width=.38, label=v)
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(names, rotation=20, fontsize=8)
    ax[2].set_ylabel("RMSE / floor"); ax[2].set_title("Modality dropout")
    ax[2].legend(fontsize=8)

fig.suptitle(f"Observation stress test — {args.variant} "
             f"({n_params:,} params) trained at a FIXED {args.train_density} "
             f"profiles/month, then frozen.  {len(t_months)} held-out months "
             f"({TEST_YEARS[0]}), unobserved columns only.", fontsize=11.5)
fig_path = os.path.join(C.REPORTS, f"fig_obs_stress_{args.variant}.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
print("\nwrote", fig_path, flush=True)


# ==================================================================== report
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


record = {"tag": tag, "task": "observation_stress_test", "variant": args.variant,
          "has_evidence_model": HAS_EVIDENCE, "git_commit": git_commit(),
          "smoke": args.smoke, "seed": args.seed, "n_params": n_params,
          "train_density": args.train_density, "aug_max": args.aug_max,
          "train_years": TRAIN_YEARS, "val_years": VAL_YEARS,
          "test_years": TEST_YEARS, "inputs": list(FULL),
          "train_minutes": round(train_secs / 60, 2), "best_val": best,
          "curves": curves, "results": results, "figure": fig_path}
with open(os.path.join(C.CACHE, f"{tag}.json"), "w") as f:
    json.dump(record, f, indent=2)

# A model that predicts ~zero anomaly reconstructs the climatology exactly and
# scores ratio 1.0 by construction.  Such a model is invariant to EVERY stress
# axis — not because it is robust, but because it never reads its observations.
# Detect that and say so, or the tables below read as a clean sweep.
_all_ratios = [c["ratio"][v] for grp in results.values() for c in grp.values()
               for v in VARS]
_dev_from_clim = max(abs(r - 1.0) for r in _all_ratios) if _all_ratios else 0.0
COLLAPSED = _dev_from_clim < 0.02
if COLLAPSED:
    print(f"\n!! degenerate: every stress cell within {_dev_from_clim:.4f} of the "
          "climatology floor — this model ignores its observations; the stress "
          "axes cannot discriminate on it.", flush=True)

L = [f"# Observation stress test — `{args.variant}`\n"]
if COLLAPSED:
    L += [f"> ## Read this first: the model collapsed to climatology\n",
          f"> Every cell below sits within **{_dev_from_clim:.4f}** of the "
          "climatology floor (ratio 1.0). The training target is the anomaly in "
          "z-space, so predicting ~0 everywhere reproduces the climatology "
          "exactly and scores 1.0 *by construction* — and that is the degenerate "
          "solution this model found.\n",
          "> **The flat rows are therefore NOT evidence of robustness.** A model "
          "that ignores its observation tokens is trivially invariant to "
          "removing them, duplicating them, clustering them, or dropping whole "
          "modalities. These axes can only discriminate on a model that "
          "actually reads its inputs, so for this run they are uninformative "
          "rather than passing.\n",
          "> What the density axis *does* establish is the collapse itself: "
          "identical output at 0 and 3000 profiles is direct evidence the "
          "observation pathway is unused. A single-density evaluation would "
          "have reported 'near the floor' and hidden that.\n"]
L += [
     f"One model, trained once at a **fixed {args.train_density} synthetic Argo "
     f"profiles/month**, then frozen and pushed off that density. This is the "
     "counterpart `experiments/08_density_ablation.py` names in its own "
     "docstring (that sweep retrains every baseline per density; this one never "
     "refits).\n",
     f"Model: {args.variant} ({n_params:,} params), inputs {' + '.join(FULL)}, "
     f"test {TEST_YEARS[0]} ({len(t_months)} months), unobserved columns only, "
     "anomaly target. Floor = predicting zero anomaly (the train-only monthly "
     "climatology), recomputed per cell because the scored pool changes with "
     "density.\n",
     "## 1. Profile density\n",
     "| profiles/month | TEMP RMSE | TEMP floor | TEMP ratio | SALT ratio |"
     + (" DFS evidence |" if HAS_EVIDENCE else ""),
     "|---|---|---|---|---|" + ("---|" if HAS_EVIDENCE else "")]
for k in dens:
    m = results["density"][str(k)]
    ev = (f" {m['evidence_total']:.1f} |" if HAS_EVIDENCE and m["evidence_total"]
          else (" n/a |" if HAS_EVIDENCE else ""))
    star = " **(trained here)**" if k == args.train_density else ""
    L.append(f"| {k}{star} | {m['rmse']['TEMP']:.4f} | {m['floor']['TEMP']:.4f} "
             f"| {m['ratio']['TEMP']:.4f} | {m['ratio']['SALT']:.4f} |{ev}")

L += ["", "## 2. Redundancy — the same profiles re-ingested\n",
      "Identical columns supplied 2x/3x: more tokens, no new physical evidence. "
      "DFS-Attention should discount the copies.\n",
      "| copies | live tokens | TEMP ratio | SALT ratio |"
      + (" DFS evidence |" if HAS_EVIDENCE else "")]
L.append("|---|---|---|---|" + ("---|" if HAS_EVIDENCE else ""))
if base:
    ev = (f" {base['evidence_total']:.1f} |" if HAS_EVIDENCE and base["evidence_total"]
          else (" n/a |" if HAS_EVIDENCE else ""))
    L.append(f"| x1 (reference) | {base['mean_live_tokens']:.0f} | "
             f"{base['ratio']['TEMP']:.4f} | {base['ratio']['SALT']:.4f} |{ev}")
for f_ in DUPS:
    m = results["redundancy"][str(f_)]
    ev = (f" {m['evidence_total']:.1f} |" if HAS_EVIDENCE and m["evidence_total"]
          else (" n/a |" if HAS_EVIDENCE else ""))
    L.append(f"| x{f_} | {m['mean_live_tokens']:.0f} | {m['ratio']['TEMP']:.4f} "
             f"| {m['ratio']['SALT']:.4f} |{ev}")

L += ["", "## 3. Coverage — same count, confined to a longitude band\n",
      "| band (fraction of lon) | TEMP ratio | SALT ratio |", "|---|---|---|"]
for fr in CLUSTERS:
    m = results["coverage"][f"{fr:g}"]
    L.append(f"| {fr:g} ({m['band_width_lon']} deg) | {m['ratio']['TEMP']:.4f} "
             f"| {m['ratio']['SALT']:.4f} |")

L += ["", "## 4. Modality dropout\n",
      "| inputs | TEMP ratio | SALT ratio |", "|---|---|---|"]
for name in results["modality"]:
    m = results["modality"][name]
    L.append(f"| {name} | {m['ratio']['TEMP']:.4f} | {m['ratio']['SALT']:.4f} |")

L += ["", "## Figure\n", f"![observation stress]({os.path.basename(fig_path)})\n",
      "## How to read this\n",
      "* Ratio approaching 1.0 as density falls, without exceeding it, is the "
      "background-referenced fallback working as claimed "
      "(`fusion.DFSAttention`: thin evidence -> climatology).\n",
      "* A ratio **above** 1.0 at low density means the model over-trusts sparse "
      "observations — `log_lambda_bg` is not carrying its weight. That is a "
      "finding, not a failed run.\n",
      "* Duplicated profiles moving the prediction means the evidence estimator "
      "is not discounting redundancy, which is the DFS premise.\n",
      f"Run record: `outputs/cache/{tag}.json`\n"]
md = os.path.join(C.REPORTS, f"obs_stress_{args.variant}.md")
with open(md, "w") as f:
    f.write("\n".join(L))
print("wrote", md, flush=True)
print(f"TOTAL {(time.time()-t_start)/60:.1f} min", flush=True)
