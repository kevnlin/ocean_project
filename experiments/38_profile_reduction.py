"""Mentor stress test — eval-only Argo-profile reduction on FROZEN models.

Spec (mentor, verbatim intent): take the current setup and gradually reduce the
number of *input* Argo profiles — 100%, 75%, 50%, 25%, 10% — during EVALUATION
ONLY, no retraining, so the curve measures how robust the existing model is.
Keep the evaluation targets and the profile subsets the same for every method,
then plot RMSE against the number of available profiles.  1,000 profiles is
additionally marked as the fixed data-budget point for the coming real-data
phase.

How comparability is enforced (the two clauses that matter):

* **Same subsets** — one random permutation of ocean columns per test month;
  the k-profile subset is its first k entries.  Subsets are therefore NESTED
  (10% of a month's profiles is a strict subset of its 25%, etc.) and identical
  for every method.
* **Same targets** — the scored query pool is built ONCE per month, from cells
  unobserved relative to the FULL (100%) profile set.  Because every subset is
  contained in the full set, those cells stay unobserved at every reduction, so
  every point of every curve is scored on identical (location, depth, month)
  targets and the climatology floor is a single horizontal line.  This is
  deliberately different from experiments/37_obs_stress.py, which re-derives
  the pool (and floor) per density because its densities are not nested.

Relation to 37: that script retrains at a fixed density and measures
extrapolation of a NEW model (the training-regime axis the mentor defers to
"if we have time"); this one measures robustness of the EXISTING models.  Its
finding — the fixed-1000 model collapses without profiles because graceful
fallback has to be learned from density augmentation — predicts the fixed-1000
curve here should degrade faster than the augmentation-trained one.

Methods (all frozen checkpoints, inference only; missing files are skipped):
  d4rt_aug     current best  — D4RT, aug-trained K~U{0..3000}, prof+surf+woa
  d4rt_ssh     + pseudo-SSH modality (the ablation arm)
  d4rt_fixed   D4RT trained at FIXED 1000 (from 37; 4-modality ckpt, padded)
  perceiver    no-evidence control (collapsed to climatology; flat by design)

Checkpoint compatibility: ckpts trained before the ssh_grid registration carry
4-modality ``modality_emb`` / ``ReferenceSlots.avail_proj`` tensors; they are
zero-padded to the current 5-modality shapes on load (the ssh row is never
exercised — these models are fed no ssh tokens).

Run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/38_profile_reduction.py
    .venv/bin/python experiments/38_profile_reduction.py --smoke
"""
import sys, os, json, time, argparse, subprocess
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
from ocean_tokenizer.fullrun import FullRunData, VARS, assert_nondegenerate_climatology
from ocean_tokenizer.ssh import SSHAnom, load_ssh_cache, ssh_for_indices

UNITS = {"TEMP": "degC", "SALT": "PSU"}
FULL_INC = ("profiles", "surf", "woa")


def _years(s):
    a, _, b = s.partition("-")
    return (int(a), int(b or a))


ap = argparse.ArgumentParser()
ap.add_argument("--train-years", type=_years, default="2000-2003",
                help="months the checkpoints' climatology/normaliser were fit on")
ap.add_argument("--test-years", type=_years, default="2005")
ap.add_argument("--n-full", type=int, default=1500,
                help="the 100%% profile count (protocol_v1's 1500)")
ap.add_argument("--fractions", default="1.0,0.75,0.5,0.25,0.10")
ap.add_argument("--extra-counts", default="1000",
                help="absolute counts added to the sweep (mentor: the fixed "
                     "1000-profile data-budget point)")
ap.add_argument("--queries-per-month", type=int, default=30_000,
                help="fixed target subsample per test month (drawn once, "
                     "shared by every method and every reduction)")
ap.add_argument("--eval-chunk", type=int, default=32768)
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--tag", default="profile_reduction_s%d" % C.SEED)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

TRAIN_YEARS, TEST_YEARS = args.train_years, args.test_years
FRACTIONS = sorted({float(x) for x in args.fractions.split(",")}, reverse=True)
EXTRA = [int(x) for x in args.extra_counts.split(",") if x.strip()]
tag = args.tag + ("_smoke" if args.smoke else "")
if args.smoke:
    FRACTIONS = [1.0, 0.25]
    EXTRA = [1000]
    args.queries_per_month = 3000

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
t_start = time.time()

# counts actually evaluated: fraction points + the fixed-budget markers
COUNTS = sorted({int(np.ceil(f * args.n_full)) for f in FRACTIONS}
                | set(EXTRA), reverse=True)
print(f"[{tag}] n_full={args.n_full} counts={COUNTS} device={dev}", flush=True)

# ================================================================= data/norm
grid = data.CommonGrid()
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
te_idx = data.select_month_indices(C.GT_SOURCE, TEST_YEARS)
print(f"months: train={tr_idx.size} (stats only)  test={te_idx.size}", flush=True)

ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
assert_nondegenerate_climatology(ftrain["months"])
norm = AnomNorm(Climatology(ftrain, surf_train), ftrain, surf_train)
tr_months = ftrain["months"].copy()
del ftrain

rd = FullRunData(grid, norm, dev)
D, H, W, HW = rd.D, rd.H, rd.W, rd.HW
oi, oj, n_ocean = rd.oi, rd.oj, rd.n_ocean
rd.load_woa(data.woa_prior(grid))

ftest = data.load_gt_fields(te_idx, grid)
ZAt = torch.from_numpy(rd.z_volume(ftest)).to(dev)
surfZt = torch.from_numpy(rd.z_surf(ftest)).to(dev)
te_months = ftest["months"].copy()
del ftest

sshZ_te = None
ssh_path = os.path.join(C.CACHE, "ssh_dyn.npz")
if os.path.exists(ssh_path):
    ssh_all, ssh_ti = load_ssh_cache(ssh_path)
    sshnorm = SSHAnom(ssh_for_indices(ssh_all, ssh_ti, tr_idx), tr_months)
    sshZ_te = torch.from_numpy(
        rd.z_ssh(ssh_for_indices(ssh_all, ssh_ti, te_idx), te_months,
                 sshnorm)).to(dev)
print(f"  data ready in {time.time()-ts:.1f}s", flush=True)

# ====================== fixed monthly draws: permutation + frozen target pool
rng = np.random.default_rng([args.seed, 38])
months_setup = []
n_level = torch.zeros(D, dtype=torch.float64, device=dev)
se0 = torch.zeros(D, 2, dtype=torch.float64, device=dev)
for t, mo in enumerate(te_months):
    mo = int(mo)
    perm = rng.permutation(n_ocean)[:min(args.n_full, n_ocean)]
    ii_full = torch.from_numpy(oi[perm]).to(dev)     # nested subsets = prefixes
    jj_full = torch.from_numpy(oj[perm]).to(dev)
    col_full = torch.zeros(HW, dtype=torch.bool, device=dev)
    col_full[ii_full * W + jj_full] = True
    fin = torch.isfinite(ZAt[t]).all(0).view(D, HW)
    pool = (fin & (~col_full)[None]).view(-1).nonzero(as_tuple=True)[0]
    if pool.numel() > args.queries_per_month:
        sel = rng.choice(pool.numel(), size=args.queries_per_month, replace=False)
        pool = pool[torch.from_numpy(np.sort(sel)).to(dev)]
    y = ZAt[t].view(2, -1)[:, pool].T.contiguous()
    q, di = rd.q_from_flat(pool, mo)
    months_setup.append(dict(t=t, mo=mo, ii=ii_full, jj=jj_full, q=q, y=y, di=di))
    n_level += torch.bincount(di, minlength=D).double()
    se0.index_add_(0, di, y.double() ** 2)

FLOOR = rd.physical_rmse(se0, n_level)["full"]
print(f"fixed pool: {int(n_level.sum()):,} targets over {len(months_setup)} "
      f"months | clim floor TEMP={FLOOR['TEMP']:.4f} SALT={FLOOR['SALT']:.4f}",
      flush=True)

# =================================================================== methods
D4RT_KW = dict(d_model=64, n_latent=32, n_heads=4, n_self_blocks=2,
               n_dec_blocks=2, max_lead=1, query_chunk=512,
               anchor_grid=(12, 24), seed=args.seed)
PERC_KW = dict(d_model=64, n_latent=32, n_heads=4, n_self_blocks=2,
               anchor_grid=(12, 24), seed=args.seed)
METHODS = [
    ("d4rt_aug",   "D4RT (aug-trained)",
     "d4rt_recon_nossh_p1000_s1234.pt", "d4rt", FULL_INC, False, D4RT_KW),
    ("d4rt_ssh",   "D4RT + pseudo-SSH",
     "d4rt_recon_ssh_p1000_s1234.pt", "d4rt", FULL_INC + ("ssh",), True, D4RT_KW),
    ("d4rt_fixed", "D4RT (fixed-1000-trained)",
     "stress_d4rt_d1000_s1234.pt", "d4rt", FULL_INC, False, D4RT_KW),
    ("perceiver",  "Perceiver control",
     "stress_perceiver_d1000_s1234.pt", "perceiver", FULL_INC, False, PERC_KW),
]


def load_padded(model, path):
    """Load a pre-``ssh_grid`` checkpoint into the current 5-modality model.

    Any tensor whose shape differs only by the grown modality axis (4 -> 5:
    ``modality_emb``, ``ReferenceSlots.avail_proj``, ``EvidenceResampler``
    per-modality slots, ...) is embedded into the model's freshly initialised
    tensor, which keeps its init in the new region.  Safe because these models
    are fed no ssh tokens, so the new region is never exercised.
    """
    sd = torch.load(path, map_location=dev)["state_dict"]
    tgt = model.state_dict()
    for k, v in list(sd.items()):
        if k in tgt and v.shape != tgt[k].shape:
            assert v.dim() == tgt[k].dim() and all(
                a <= b for a, b in zip(v.shape, tgt[k].shape)), \
                (k, tuple(v.shape), tuple(tgt[k].shape))
            merged = tgt[k].clone()
            merged[tuple(slice(0, s) for s in v.shape)] = v
            sd[k] = merged
            print(f"    padded {k} {tuple(v.shape)} -> {tuple(merged.shape)}",
                  flush=True)
    model.load_state_dict(sd)


@torch.no_grad()
def eval_curve(model, include):
    """-> {count: physical_rmse dict}; fixed targets, nested profile prefixes."""
    model.eval()
    out = {}
    for k in COUNTS:
        se = torch.zeros(D, 2, dtype=torch.float64, device=dev)
        for m in months_setup:
            obs = rd.obs_dict(ZAt, surfZt, m["t"], m["mo"],
                              m["ii"][:k], m["jj"][:k], include=include,
                              sshZ=sshZ_te)
            pred = model(obs, m["q"])[0]
            for i in range(0, m["q"].shape[1], args.eval_chunk):
                sl = slice(i, i + args.eval_chunk)
                se.index_add_(0, m["di"][sl],
                              (pred[sl] - m["y"][sl]).double() ** 2)
        out[k] = rd.physical_rmse(se, n_level)["full"]
        print(f"    N={k:5d}  TEMP {out[k]['TEMP']:.4f} "
              f"({out[k]['TEMP']/FLOOR['TEMP']:.4f}x floor)  "
              f"SALT {out[k]['SALT']:.4f} "
              f"({out[k]['SALT']/FLOOR['SALT']:.4f}x floor)", flush=True)
    return out


results, labels = {}, {}
for key, label, fn, variant, include, with_ssh, kw in METHODS:
    path = os.path.join(C.CKPT, fn)
    if not os.path.exists(path):
        print(f"[{key}] SKIP — {path} missing", flush=True)
        continue
    if with_ssh and sshZ_te is None:
        print(f"[{key}] SKIP — ssh_dyn.npz missing", flush=True)
        continue
    print(f"[{key}] {label}  ({fn})", flush=True)
    model = build_fusion_model(variant, grid, with_ssh=with_ssh, **kw).to(dev)
    load_padded(model, path)
    results[key] = eval_curve(model, include)
    labels[key] = label
    del model
    torch.cuda.empty_cache()

if not results:
    raise SystemExit("no checkpoints found — nothing to plot")

# ==================================================================== figure
xs = sorted(COUNTS)
frac_counts = sorted({int(np.ceil(f * args.n_full)) for f in FRACTIONS})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
colors = plt.cm.tab10.colors
for a, v in zip(ax, VARS):
    for i, key in enumerate(results):
        a.plot(xs, [results[key][k][v] for k in xs], "o-",
               color=colors[i], label=labels[key])
    a.axhline(FLOOR[v], color="k", ls="--", lw=1.2, label="climatology floor")
    for e in EXTRA:
        a.axvline(e, color="grey", ls=":", lw=1.2)
        a.annotate(f"fixed budget ({e})", xy=(e, a.get_ylim()[1]),
                   xytext=(4, -12), textcoords="offset points",
                   fontsize=8, color="grey", rotation=90, va="top")
    a.set_xticks(frac_counts)
    a.set_xticklabels([f"{c}\n{100*c/args.n_full:.0f}%" for c in frac_counts],
                      fontsize=8)
    a.set_xlabel("input Argo profiles / month (evaluation only, no retraining)")
    a.set_ylabel(f"{v} RMSE ({UNITS[v]})")
    a.set_title(v)
    a.grid(alpha=.3)
ax[0].legend(fontsize=8)
fig.suptitle(f"Robustness to reduced Argo input — frozen models, identical "
             f"targets and nested profile subsets per month "
             f"({len(months_setup)} test months {TEST_YEARS[0]}, "
             f"{int(n_level.sum()):,} fixed targets)", fontsize=12)
fig_path = os.path.join(C.REPORTS, "fig_profile_reduction.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
print("wrote", fig_path, flush=True)


# ==================================================================== report
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


record = {"tag": tag, "task": "eval_only_profile_reduction",
          "git_commit": git_commit(), "smoke": args.smoke, "seed": args.seed,
          "n_full": args.n_full, "fractions": FRACTIONS, "counts": COUNTS,
          "extra_counts": EXTRA, "queries_per_month": args.queries_per_month,
          "targets_total": int(n_level.sum()), "floor": FLOOR,
          "test_years": TEST_YEARS, "methods": labels,
          "results": {k: {str(n): r for n, r in v.items()}
                      for k, v in results.items()},
          "figure": fig_path}
with open(os.path.join(C.CACHE, f"{tag}.json"), "w") as f:
    json.dump(record, f, indent=2)

L = ["# Eval-only Argo-profile reduction (mentor stress test)\n",
     "Frozen models, no retraining: the input profile count is reduced at "
     f"evaluation time — {', '.join(f'{100*f:.0f}%' for f in FRACTIONS)} of "
     f"{args.n_full}, plus the fixed data-budget point(s) "
     f"{EXTRA} — while the **evaluation targets and the profile subsets are "
     "identical for every method** (nested prefix subsets of one permutation "
     "per month; one fixed target pool drawn against the 100% set, so the "
     "climatology floor is a single line).\n",
     f"Test: {TEST_YEARS[0]} ({len(months_setup)} months), "
     f"{int(n_level.sum()):,} targets, unobserved columns only, physical "
     "full-column RMSE.\n",
     f"Floor: TEMP {FLOOR['TEMP']:.4f} degC · SALT {FLOOR['SALT']:.4f} PSU\n"]
for v in VARS:
    L += [f"## {v} ({UNITS[v]})\n",
          "| profiles | " + " | ".join(labels[k] for k in results) + " |",
          "|---" * (len(results) + 1) + "|"]
    for n in sorted(COUNTS, reverse=True):
        pct = f" ({100*n/args.n_full:.0f}%)" if n in frac_counts else " (budget)"
        L.append(f"| {n}{pct} | " + " | ".join(
            f"{results[k][n][v]:.4f}" for k in results) + " |")
    L.append("")
L += ["## Figure\n", f"![profile reduction]({os.path.basename(fig_path)})\n",
      "## Notes\n",
      "* Eval-only reduction measures robustness of the existing models; the "
      "training-regime companion (retrain at a fixed density, extrapolate) is "
      "`experiments/37_obs_stress.py` / `reports/obs_stress_d4rt.md`.\n",
      "* The Perceiver control collapsed to climatology during training "
      "(`reports/obs_stress_perceiver.md`), so its flat curve is the "
      "ignore-the-observations reference, not robustness.\n",
      f"Run record: `outputs/cache/{tag}.json`\n"]
with open(os.path.join(C.REPORTS, "profile_reduction.md"), "w") as f:
    f.write("\n".join(L))
print("wrote", os.path.join(C.REPORTS, "profile_reduction.md"), flush=True)
print(f"TOTAL {(time.time()-t_start)/60:.1f} min", flush=True)
