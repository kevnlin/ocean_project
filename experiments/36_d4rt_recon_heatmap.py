"""2-D reconstruction of T/S with the current architecture + spatial error heatmap.

The ask: reconstruct temperature and salinity from sparse Argo-like profiles
using the project's current model — DFS-Attention fusion over a Perceiver-IO
latent with the D4RT causal space-time query decoder (``fusion.D4RTFusion``) —
and map WHERE the reconstruction error lives, with more error drawn hotter
(red / orange) and less error darker.

Reconstruction only: every query is at lead 0 (``t_tgt == t_src``), so this is
the contemporaneous-analysis task, not forecasting.  The D4RT lead machinery is
present but pinned at 0 (``LeadEmbedding`` has ``padding_idx=0``, so lead 0
contributes exactly nothing and the forecasting rows never receive gradient).

Data / protocol
---------------
CESM2-LE 1x1deg (``data/cesm2_le_full_standard.zarr``) is the ground truth and
WOA23 decav91C0 monthly 1deg (``data/woa23_standard.zarr``) is the
climatological background — both built by ``experiments/standardize.py``.  The
model therefore sees the full protocol_v1 input set: sparse profiles + SST/SSS
+ the WOA background that DFS-Attention measures its evidence against.
``--no-woa`` drops the background modality (the background branch is then
simply skipped) if you want the ablation.

The month window is whatever has been standardized locally (``--train-years``
/ ``--val-years`` / ``--test-years``), which is shorter than protocol_v1's
276/36/12 split, so these numbers are not directly comparable to the
protocol_v1 tables in reports/.

Split (train-only climatology, no test leakage):
    train 2000-2003 (48 mo) | val 2004 (12 mo) | test 2005 (12 mo)
Targets are train-only monthly anomalies in z-space; the reported map is
physical RMSE (degC / PSU).  Scoring is UNOBSERVED-only: the whole column of
every supplied profile is excluded, so the model cannot score by echoing its
own input back out.

Outputs
-------
    reports/fig_d4rt_recon_heatmap.png   the figure
    reports/d4rt_recon_heatmap.md        the written-up numbers
    outputs/cache/<tag>.json             run record + per-band RMSE
    outputs/ckpt/<tag>.pt                best-validation weights

Run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/36_d4rt_recon_heatmap.py --smoke
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/36_d4rt_recon_heatmap.py --steps 20000
"""
import sys, os, json, time, argparse, subprocess, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from ocean_tokenizer import data, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm
from ocean_tokenizer.fusion import build_fusion_model
from ocean_tokenizer.ssh import SSHAnom, load_ssh_cache, ssh_for_indices
from ocean_tokenizer.fullrun import (FullRunData, VARS, BANDS,
                                     assert_nondegenerate_climatology,
                                     drop_observed_columns)

CONTEXT = 1                         # surface months in the input window
UNITS = {"TEMP": "degC", "SALT": "PSU"}
# map panels: oceanographic layers + the pooled column, matching the layout of
# experiments/11_layered_heatmap.py so the two figures read the same way
LAYERS = [("0-100 m", 0.0, 100.0), ("100-300 m", 100.0, 300.0),
          ("300-985 m", 300.0, 1e9)]


def _years(s):
    """'2000-2003' or '2005' -> (lo, hi) inclusive."""
    a, _, b = s.partition("-")
    return (int(a), int(b or a))


ap = argparse.ArgumentParser()
ap.add_argument("--train-years", type=_years, default="2000-2003")
ap.add_argument("--val-years", type=_years, default="2004")
ap.add_argument("--test-years", type=_years, default="2005")
ap.add_argument("--no-woa", action="store_true",
                help="drop the WOA23 climatological background modality "
                     "(DFS-Attention then has no background to reference)")
ap.add_argument("--ssh", action="store_true", default=True,
                help="include the pseudo-SSH (steric height) modality; needs "
                     "outputs/cache/ssh_dyn.npz from experiments/28_make_ssh.py")
ap.add_argument("--no-ssh", dest="ssh", action="store_false")
ap.add_argument("--variant", default="d4rt", choices=["d4rt", "dfs"])
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--steps", type=int, default=20_000)
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
ap.add_argument("--anchor-grid", default="12,24",
                help="'NLAT,NLON' geographically anchored latents "
                     "(overrides --n-latent); '' to disable")
ap.add_argument("--query-chunk", type=int, default=1024,
                help="D4RT decoder query chunk: memory only, never couples queries")
ap.add_argument("--aug-min", type=int, default=200)
ap.add_argument("--aug-max", type=int, default=3000)
ap.add_argument("--n-profiles-eval", type=int, default=1000,
                help="synthetic Argo profiles per val/test month")
ap.add_argument("--val-every", type=int, default=1000)
ap.add_argument("--val-queries", type=int, default=4096)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--min-steps", type=int, default=5000)
ap.add_argument("--log-every", type=int, default=200)
ap.add_argument("--map-subsample", type=int, default=0,
                help="0 = score the FULL unobserved pool for the map; >0 caps "
                     "queries per test month (faster, noisier map)")
ap.add_argument("--eval-chunk", type=int, default=65536)
ap.add_argument("--tag", default="")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = (args.train_years, args.val_years,
                                      args.test_years)
INCLUDE = ("profiles", "surf") if args.no_woa else ("profiles", "surf", "woa")
if args.ssh:
    INCLUDE = INCLUDE + ("ssh",)

tag = args.tag or f"d4rt_recon_s{args.seed}"
if args.smoke:
    args.steps, args.log_every, args.val_every = 60, 20, 30
    args.queries, args.val_queries = 512, 512
    args.n_profiles_eval, args.aug_max = 300, 600
    args.min_steps, args.patience = 0, 99
    args.map_subsample = 20_000
    tag += "_smoke"

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
t_start = time.time()
print(f"[{tag}] variant={args.variant} steps={args.steps} device={dev} "
      f"smoke={args.smoke}", flush=True)

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

ZA = torch.from_numpy(rd.z_volume(ftrain)).to(dev)              # (T,2,D,H,W)
ZAf = ZA.view(len(tr_idx), 2, D * HW)
surfZ = torch.from_numpy(rd.z_surf(ftrain)).to(dev)
tr_months = ftrain["months"].copy()
valid_tr = [torch.isfinite(ZA[t]).all(0).view(-1).nonzero(as_tuple=True)[0]
            .cpu().numpy().astype("int64") for t in range(len(tr_idx))]
if "woa" in INCLUDE:
    # the climatological background DFS-Attention measures evidence against
    rd.load_woa(data.woa_prior(grid))

# ---- pseudo-SSH: train-only climatology, then z-score every split with it ----
sshZ_tr = sshZ_va = sshZ_te = None
if "ssh" in INCLUDE:
    ssh_path = os.path.join(C.CACHE, "ssh_dyn.npz")
    if not os.path.exists(ssh_path):
        raise SystemExit(f"{ssh_path} missing — build it first:\n"
                         f"  .venv/bin/python experiments/28_make_ssh.py "
                         f"--years {TRAIN_YEARS[0]},{TEST_YEARS[1]}")
    ssh_all, ssh_ti = load_ssh_cache(ssh_path)
    sshnorm = SSHAnom(ssh_for_indices(ssh_all, ssh_ti, tr_idx), tr_months)
    to_dev = lambda a: torch.from_numpy(np.nan_to_num(a, nan=np.nan)).to(dev)
    sshZ_tr = to_dev(rd.z_ssh(ssh_for_indices(ssh_all, ssh_ti, tr_idx),
                              tr_months, sshnorm))
    rd.sshZ = sshZ_tr                       # used by the training loop
    print(f"  ssh: {ssh_all.shape[0]} cached months, "
          f"{100*np.isfinite(ssh_for_indices(ssh_all, ssh_ti, tr_idx)).mean():.1f}% "
          f"finite, anom std {sshnorm.std*100:.2f} cm", flush=True)
del ftrain

fval = data.load_gt_fields(va_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
if "ssh" in INCLUDE:
    sshZ_va = to_dev(rd.z_ssh(ssh_for_indices(ssh_all, ssh_ti, va_idx),
                              fval["months"], sshnorm))
    sshZ_te = to_dev(rd.z_ssh(ssh_for_indices(ssh_all, ssh_ti, te_idx),
                              ftest["months"], sshnorm))
val_packs, val_n, val_se0 = rd.make_packs(fval, np.random.default_rng([args.seed, 1]),
                                          args.n_profiles_eval, include=INCLUDE,
                                          subsample=args.val_queries, sshZ=sshZ_va)
test_packs, test_n, test_se0 = rd.make_packs(ftest, np.random.default_rng([args.seed, 2]),
                                             args.n_profiles_eval, include=INCLUDE,
                                             subsample=args.map_subsample, sshZ=sshZ_te)
del fval, ftest
VAL_FLOOR = rd.physical_rmse(val_se0, val_n)["full"]
TEST_FLOOR = rd.physical_rmse(test_se0, test_n)["full"]
print(f"  data ready in {time.time()-ts:.1f}s | clim floor "
      f"val {VAL_FLOOR['TEMP']:.4f}/{VAL_FLOOR['SALT']:.4f}  "
      f"test {TEST_FLOOR['TEMP']:.4f}/{TEST_FLOOR['SALT']:.4f}", flush=True)

# ===================================================================== model
anchor = (tuple(int(x) for x in args.anchor_grid.split(","))
          if args.anchor_grid else None)
build_kw = dict(d_model=args.d_model, n_latent=args.n_latent,
                n_heads=args.n_heads, n_self_blocks=args.n_self_blocks,
                seed=args.seed, anchor_grid=anchor,
                with_ssh=("ssh" in INCLUDE))
if args.variant == "d4rt":
    build_kw.update(n_dec_blocks=args.n_dec_blocks, max_lead=1,
                    query_chunk=args.query_chunk)
model = build_fusion_model(args.variant, grid, **build_kw).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)


def _lr_lambda(step):
    if args.warmup and step < args.warmup:
        return (step + 1) / args.warmup
    p = min(max((step - args.warmup) / max(args.steps - args.warmup, 1), 0.0), 1.0)
    lo = args.lr_min / args.lr
    return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * p))


sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
print(f"model {args.variant}: params={n_params:,} d={args.d_model} "
      f"L={model.n_latent}{' anchored' if anchor else ''}", flush=True)

os.makedirs(C.CKPT, exist_ok=True)
os.makedirs(C.CACHE, exist_ok=True)
ckpt_path = os.path.join(C.CKPT, f"{tag}.pt")


# ================================================================ evaluation
@torch.no_grad()
def eval_level_sse(packs):
    """Per-level z-space SSE (D,2) — the scalar metric, matches fullrun."""
    model.eval()
    se = torch.zeros(D, 2, dtype=torch.float64, device=dev)
    for p in packs:
        z = model.fuse(model.encode(p["obs"], batch=1, device=dev))
        Q = p["q"].shape[1]
        for i in range(0, Q, args.eval_chunk):
            out = model.decode(z, p["q"][:, i:i + args.eval_chunk])[0]
            se.index_add_(0, p["di"][i:i + args.eval_chunk],
                          (out - p["y"][i:i + args.eval_chunk]).double() ** 2)
    model.train()
    return se


# which map panel each analysis level belongs to (last panel = pooled column)
_layer_of_level = np.full(D, len(LAYERS), dtype=np.int64)
for _li, (_nm, _lo, _hi) in enumerate(LAYERS):
    _sel = (grid.depth > _lo) & (grid.depth <= _hi)
    if _lo <= grid.depth.min():
        _sel |= np.isclose(grid.depth, grid.depth.min())
    _layer_of_level[_sel] = _li
assert (_layer_of_level < len(LAYERS)).all(), "a depth level fell outside LAYERS"
LAYER_OF_LEVEL = torch.from_numpy(_layer_of_level).to(dev)
NPANEL = len(LAYERS) + 1                      # + pooled full column


@torch.no_grad()
def eval_cell_maps(packs):
    """Per-(lat,lon, layer) PHYSICAL squared error, pooled over depth & months.

    The z-space residual at level d is converted with that level's anomaly std
    before accumulation, so a cell's number is a physical RMSE in degC / PSU —
    the same quantity as the scalar headline, resolved in space and by
    oceanographic layer instead of pooled.  Every query contributes twice: once
    to its own layer and once to the pooled full-column panel.

    Returns (se_model, se_floor, n) shaped (HW, NPANEL, 2) / same / (HW, NPANEL).
    """
    model.eval()
    astd = torch.stack([rd.astd_t[v] for v in VARS], dim=1).double()   # (D,2)
    flat = lambda c: torch.zeros(HW * NPANEL, c, dtype=torch.float64, device=dev) \
        if c else torch.zeros(HW * NPANEL, dtype=torch.float64, device=dev)
    se_m, se_f, n = flat(2), flat(2), flat(0)
    for p in packs:
        z = model.fuse(model.encode(p["obs"], batch=1, device=dev))
        cell = (p["idx"] % HW)
        Q = p["q"].shape[1]
        for i in range(0, Q, args.eval_chunk):
            sl = slice(i, i + args.eval_chunk)
            out = model.decode(z, p["q"][:, sl])[0]
            di = p["di"][sl]
            w = astd[di]                                            # (q,2)
            e_m = ((out - p["y"][sl]).double() * w) ** 2
            # climatology floor = predicting zero anomaly, i.e. residual == y
            e_f = (p["y"][sl].double() * w) ** 2
            ones = torch.ones(di.numel(), dtype=torch.float64, device=dev)
            base = cell[sl] * NPANEL
            for tgt in (base + LAYER_OF_LEVEL[di], base + len(LAYERS)):
                se_m.index_add_(0, tgt, e_m)
                se_f.index_add_(0, tgt, e_f)
                n.index_add_(0, tgt, ones)
    model.train()
    return (se_m.view(HW, NPANEL, 2), se_f.view(HW, NPANEL, 2),
            n.view(HW, NPANEL))


def val_score():
    r = rd.physical_rmse(eval_level_sse(val_packs), val_n)["full"]
    return float(np.mean([r[v] / VAL_FLOOR[v] for v in VARS])), r


# ================================================================== training
rng = np.random.default_rng([args.seed, 0])
Ttr = len(tr_idx)
curves = {"step": [], "loss": [], "lr": [], "val_step": [], "val_score": [],
          "val_TEMP": [], "val_SALT": []}
best = {"score": float("inf"), "step": -1}
since_best, win = 0, []
t_train = time.time()

for step in range(1, args.steps + 1):
    t = int(rng.integers(Ttr))
    mo = int(tr_months[t])
    K = int(rng.integers(args.aug_min, args.aug_max + 1))
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

    y = ZAf[t][:, idx_t].T                                          # (Q,2)
    q, _ = rd.q_from_flat(idx_t, mo)
    obs = rd.obs_dict(ZA, surfZ, t, mo, ii_t, jj_t, include=INCLUDE,
                      context=CONTEXT)
    out = model(obs, q)[0]
    loss = (out - y).pow(2).mean()
    if not torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        continue
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    win.append(float(loss.detach()))

    if step % args.log_every == 0:
        sps = step / (time.time() - t_train)
        curves["step"].append(step); curves["loss"].append(float(np.mean(win)))
        curves["lr"].append(sched.get_last_lr()[0])
        print(f"  step {step:6d}/{args.steps}  loss {np.mean(win):.4f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {sps:.2f} it/s  "
              f"eta {(args.steps-step)/max(sps,1e-9)/60:.1f} min", flush=True)
        win = []

    if args.val_every and step % args.val_every == 0:
        vs, vr = val_score()
        curves["val_step"].append(step); curves["val_score"].append(vs)
        curves["val_TEMP"].append(vr["TEMP"]); curves["val_SALT"].append(vr["SALT"])
        flag = ""
        if vs < best["score"]:
            best = {"score": vs, "step": step, "TEMP": vr["TEMP"], "SALT": vr["SALT"]}
            since_best = 0; flag = "  *best*"
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "tag": tag, "variant": args.variant,
                        "args": vars(args)}, ckpt_path)
        else:
            since_best += 1
        print(f"  [val] step {step}  TEMP {vr['TEMP']:.4f} SALT {vr['SALT']:.4f}  "
              f"score {vs:.4f} (1.0 = climatology){flag}", flush=True)
        if since_best >= args.patience and step >= args.min_steps:
            print(f"early stop at step {step}", flush=True)
            break

train_secs = time.time() - t_train
print(f"training done in {train_secs/60:.1f} min; best step {best['step']} "
      f"(val score {best['score']:.4f})", flush=True)
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=dev)["state_dict"])

# ============================================================ test + mapping
print("scoring test months (full unobserved pool) ...", flush=True)
ts = time.time()
test_metrics = rd.physical_rmse(eval_level_sse(test_packs), test_n)
se_m, se_f, n_cell = eval_cell_maps(test_packs)
print(f"  mapped in {time.time()-ts:.1f}s", flush=True)

PANELS = [nm for nm, _, _ in LAYERS] + ["full column"]
n_np = n_cell.cpu().numpy().reshape(H, W, NPANEL)
have = n_np > 0                                        # (H,W,NPANEL)
# rmse_map[v][p] / floor_map[v][p] are (H,W) maps for panel p
rmse_map = {v: [] for v in VARS}
floor_map = {v: [] for v in VARS}
skill_map = {v: [] for v in VARS}
for k, v in enumerate(VARS):
    sm = se_m[:, :, k].cpu().numpy().reshape(H, W, NPANEL)
    sf = se_f[:, :, k].cpu().numpy().reshape(H, W, NPANEL)
    for p in range(NPANEL):
        ok = have[..., p]
        rm = np.full((H, W), np.nan); fm = np.full((H, W), np.nan)
        rm[ok] = np.sqrt(sm[..., p][ok] / n_np[..., p][ok])
        fm[ok] = np.sqrt(sf[..., p][ok] / n_np[..., p][ok])
        rmse_map[v].append(rm); floor_map[v].append(fm)
        with np.errstate(invalid="ignore", divide="ignore"):
            skill_map[v].append(1.0 - rm / np.where(fm > 0, fm, np.nan))

for v in VARS:
    print(f"  {v}: full-column RMSE {test_metrics['full'][v]:.4f} "
          f"{UNITS[v]} vs floor {TEST_FLOOR[v]:.4f} "
          f"(skill {1 - test_metrics['full'][v]/TEST_FLOOR[v]:+.1%})", flush=True)
    for p, nm in enumerate(PANELS):
        print(f"      {nm:12s} median RMSE {np.nanmedian(rmse_map[v][p]):.4f}  "
              f"floor {np.nanmedian(floor_map[v][p]):.4f}", flush=True)

# ==================================================================== figure
extent = [float(grid.lon.min()), float(grid.lon.max()),
          float(grid.lat.min()), float(grid.lat.max())]
# 'hot': black -> red -> orange -> yellow, i.e. darker = better reconstruction,
# redder/brighter = more error.  Land / never-scored cells drawn plain grey.
cmap = plt.cm.hot.copy(); cmap.set_bad("0.75")

fig, axes = plt.subplots(len(VARS), NPANEL, figsize=(4.1 * NPANEL, 6.6),
                         constrained_layout=True)
for r, v in enumerate(VARS):
    # one colour scale per variable across all panels -> layers are comparable
    finite = np.concatenate([m[np.isfinite(m)] for m in rmse_map[v]])
    vmax = float(np.nanpercentile(finite, 99))
    for c in range(NPANEL):
        ax = axes[r, c]
        im = ax.imshow(np.ma.masked_invalid(rmse_map[v][c]), origin="lower",
                       extent=extent, aspect="auto", cmap=cmap,
                       norm=Normalize(0, vmax))
        if r == 0:
            ax.set_title(PANELS[c], fontsize=11)
        if c == 0:
            ax.set_ylabel(f"{v} ({UNITS[v]})\nlatitude", fontsize=10)
        ax.set_xlabel("longitude" if r == len(VARS) - 1 else "")
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes[r, :], shrink=0.68, location="right", pad=0.012,
                 label=f"{v} RMSE ({UNITS[v]})")

fig.suptitle(
    f"2-D T/S reconstruction error by depth layer — DFS-Attention + D4RT query "
    f"decoder ({n_params:,} params)\n{len(test_packs)} held-out months "
    f"({TEST_YEARS[0]}), {args.n_profiles_eval} profiles/month, "
    f"unobserved columns only, anomaly target.  Inputs: {' + '.join(INCLUDE)}."
    "\ndark = accurate reconstruction, red / orange / white = more error; "
    "grey = land or unscored",
    fontsize=12)
fig_path = os.path.join(C.REPORTS, "fig_d4rt_recon_heatmap.png")
fig.savefig(fig_path, dpi=140, bbox_inches="tight")
print("wrote", fig_path, flush=True)

# ---- companion: is the model actually beating the climatology it references?
fig2, ax2 = plt.subplots(len(VARS), 3, figsize=(15.5, 6.6), constrained_layout=True)
skl = plt.cm.RdBu.copy(); skl.set_bad("0.75")     # red = worse than climatology
for r, v in enumerate(VARS):
    both = np.concatenate([m[np.isfinite(m)]
                           for m in (rmse_map[v][-1], floor_map[v][-1])])
    vmax = float(np.nanpercentile(both, 99))
    for c, (fld, ttl) in enumerate([
            (rmse_map[v][-1], "D4RT reconstruction"),
            (floor_map[v][-1], "climatology floor")]):
        im = ax2[r, c].imshow(np.ma.masked_invalid(fld), origin="lower",
                              extent=extent, aspect="auto", cmap=cmap,
                              norm=Normalize(0, vmax))
        if r == 0:
            ax2[r, c].set_title(ttl, fontsize=11)
        fig2.colorbar(im, ax=ax2[r, c], shrink=0.8,
                      label=f"RMSE ({UNITS[v]})")
    s = skill_map[v][-1]
    lim = float(np.nanpercentile(np.abs(s[np.isfinite(s)]), 98)) or 1.0
    im = ax2[r, 2].imshow(np.ma.masked_invalid(s), origin="lower", extent=extent,
                          aspect="auto", cmap=skl,
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
    if r == 0:
        ax2[r, 2].set_title("skill  (1 - RMSE / floor)", fontsize=11)
    fig2.colorbar(im, ax=ax2[r, 2], shrink=0.8,
                  label="blue beats climatology, red worse")
    ax2[r, 0].set_ylabel(f"{v} ({UNITS[v]})\nlatitude", fontsize=10)
for a in ax2[-1, :]:
    a.set_xlabel("longitude")
for a in ax2.ravel():
    a.tick_params(labelsize=8)
fig2.suptitle("Full-column reconstruction vs the climatology floor "
              f"({TEST_YEARS[0]} held-out months, unobserved columns only)",
              fontsize=12.5)
fig2_path = os.path.join(C.REPORTS, "fig_d4rt_recon_skill.png")
fig2.savefig(fig2_path, dpi=140, bbox_inches="tight")
print("wrote", fig2_path, flush=True)


# ==================================================================== report
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


record = {
    "tag": tag, "task": "2d_reconstruction_error_heatmap", "variant": args.variant,
    "git_commit": git_commit(), "smoke": args.smoke, "seed": args.seed,
    "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
    "woa_available": "woa" in INCLUDE, "inputs": list(INCLUDE), "lead": 0,
    "woa_source": "WOA23 decav91C0 monthly 1deg (NCEI), t_an/s_an",
    "protocol_note": "protocol_v1 inputs on a shorter month window than "
                     "protocol_v1's 276/36/12 split; not directly comparable "
                     "to the reports/ tables",
    "train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "test_years": TEST_YEARS,
    "train_months": int(tr_idx.size), "val_months": int(va_idx.size),
    "test_months": int(te_idx.size), "n_params": n_params,
    "steps_run": int(curves["step"][-1]) if curves["step"] else 0,
    "best_val": best, "train_minutes": round(train_secs / 60, 2),
    "test_floor": TEST_FLOOR, "test_full": test_metrics["full"],
    "test_by_band": test_metrics["by_band"], "test_by_depth": test_metrics["by_depth"],
    "skill_full": {v: 1 - test_metrics["full"][v] / TEST_FLOOR[v] for v in VARS},
    "map_panels": PANELS,
    "map_median_rmse": {v: {nm: float(np.nanmedian(rmse_map[v][p]))
                            for p, nm in enumerate(PANELS)} for v in VARS},
    "map_median_floor": {v: {nm: float(np.nanmedian(floor_map[v][p]))
                             for p, nm in enumerate(PANELS)} for v in VARS},
    "cells_mapped": int(have[..., -1].sum()), "curves": curves,
    "figure": fig_path, "figure_skill": fig2_path,
}
json_path = os.path.join(C.CACHE, f"{tag}.json")
with open(json_path, "w") as f:
    json.dump(record, f, indent=2)

L = ["# 2-D T/S Reconstruction + Spatial Error Heatmap\n",
     "Model: **DFS-Attention fusion over a Perceiver-IO latent + D4RT causal "
     "space-time query decoder** (`fusion.D4RTFusion`), lead 0 "
     f"(reconstruction only), {n_params:,} parameters.\n",
     f"Data: CESM2-LE 1x1deg — train {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} "
     f"({tr_idx.size} mo) / val {VAL_YEARS[0]} ({va_idx.size} mo) / "
     f"test {TEST_YEARS[0]} ({te_idx.size} mo), "
     f"{args.n_profiles_eval} synthetic Argo profiles per month.\n",
     f"Inputs: **{' + '.join(INCLUDE)}** "
     + ("(WOA23 decav91C0 monthly 1&deg;, the background DFS-Attention "
        "references its evidence against).\n" if "woa" in INCLUDE
        else "— WOA background ablated via `--no-woa`.\n"),
     "> **Caveat — short month window.** This run uses "
     f"{tr_idx.size}/{va_idx.size}/{te_idx.size} train/val/test months, not "
     "protocol_v1's 276/36/12, so the numbers are not directly comparable "
     "with the tables elsewhere in `reports/`.\n",
     "Scoring excludes the full column of every supplied profile "
     "(unobserved-only), so the model cannot score by echoing its input.\n",
     "## Headline (test, full column)\n",
     "| var | D4RT RMSE | climatology floor | skill |",
     "|-----|-----------|-------------------|-------|"]
for v in VARS:
    sk = 1 - test_metrics["full"][v] / TEST_FLOOR[v]
    L.append(f"| {v} | {test_metrics['full'][v]:.4f} {UNITS[v]} | "
             f"{TEST_FLOOR[v]:.4f} {UNITS[v]} | {sk:+.1%} |")
L += ["", "## By depth band\n",
      "| var | band | D4RT RMSE | floor | skill |",
      "|-----|------|-----------|-------|-------|"]
band_floor = rd.physical_rmse(test_se0, test_n)["by_band"]
for v in VARS:
    for name, _, _ in BANDS:
        m = test_metrics["by_band"][v][name]
        fl = band_floor[v][name]
        L.append(f"| {v} | {name} | {m:.4f} | {fl:.4f} | {1 - m/fl:+.1%} |")
L += ["", "## Error heatmap by depth layer\n",
      f"![reconstruction error heatmap]({os.path.basename(fig_path)})\n",
      "Rows are TEMP / SALT, columns are oceanographic layers plus the pooled "
      "column. **Darker = better reconstruction; brighter red / orange / "
      "yellow = more error.** Land and never-scored cells are grey. Each row "
      "shares one colour scale across its panels, so layers are directly "
      "comparable.\n",
      "| var | panel | median RMSE | median floor |",
      "|-----|-------|-------------|--------------|"]
for v in VARS:
    for p, nm in enumerate(PANELS):
        L.append(f"| {v} | {nm} | {np.nanmedian(rmse_map[v][p]):.4f} "
                 f"{UNITS[v]} | {np.nanmedian(floor_map[v][p]):.4f} {UNITS[v]} |")
L += ["", "## Against the climatology floor\n",
      f"![skill map]({os.path.basename(fig2_path)})\n",
      "Full-column reconstruction, the climatology floor on the same colour "
      "scale, and the skill ratio. Blue beats climatology, red is worse than "
      "it — the honest read of whether the model is adding anything over "
      "simply predicting the seasonal mean.\n",
      f"Per-cell values pool the levels in each layer x {te_idx.size} test "
      f"months ({int(have[..., -1].sum()):,} ocean cells scored), converted to "
      "physical units with each level's anomaly std, so a cell's number is a "
      "physical RMSE.\n",
      f"Run record: `outputs/cache/{tag}.json`\n"]
md_path = os.path.join(C.REPORTS, "d4rt_recon_heatmap.md")
with open(md_path, "w") as f:
    f.write("\n".join(L))
print("wrote", md_path, flush=True)
print(f"TOTAL {(time.time()-t_start)/60:.1f} min", flush=True)
