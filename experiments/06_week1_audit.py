"""Week-1 audit: the corrected baseline matrix.

Fixes the three evaluation defects that made the earlier sweep invalid:

  1. ANOMALY TARGET      — models regress the anomaly against a *train-only*
                           monthly CESM2 climatology, not the absolute field, so
                           they can no longer lean on the memorisable mean.
  2. UNOBSERVED-ONLY     — the headline metric excludes observed profile columns
                           (no scoring on cells fed the noise-free truth).  We
                           print the all-ocean number alongside it so the
                           leakage gap is explicit.
  3. TRAIN-ONLY FLOOR    — the RMSE floor is the model's own held-out
                           climatology (~0.65 degC), reported next to the
                           bias-dominated WOA prior it replaces.

Also adds the JOINT-DEPTH U-Net (whole water column as channels) as the strong
baseline the shared-latent method must beat.

Run:
    python experiments/06_week1_audit.py [--smoke]
"""
import sys, os, json, time, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from ocean_tokenizer import data, baselines as B, metrics, config as C
from ocean_tokenizer.anomaly import Climatology, AnomNorm

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

if args.smoke:
    C.N_TRAIN_MONTHS, C.N_TEST_MONTHS = 6, 3
    C.N_PROFILES = 600
    C.MLP_EPOCHS, C.UNET_EPOCHS = 4, 3
    C.UNET_JOINT_EPOCHS = 8
    C.MLP_POINTS_PER_MONTH = 30_000

t0 = time.time()
rng = np.random.default_rng(C.SEED)
device = C.DEVICE if torch.cuda.is_available() else "cpu"
print(f"device={device} smoke={args.smoke}")

grid = data.CommonGrid()
print(grid)

# ---- pick train/test months (same protocol / seed as the legacy sweep) ----
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))
print(f"train months={tr_idx.size} test months={te_idx.size}")

print("loading GT fields ..."); ts = time.time()
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
print(f"  loaded in {time.time()-ts:.1f}s")

# ---- train-only climatology + anomaly-space normaliser ----
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

# ---- assemble per-month samples (now carry an unobserved-column mask) ----
def make_samples(fields):
    return [B.prepare_month(fields, fields, woa, grid, t, rng, C.N_PROFILES)
            for t in range(len(fields["months"]))]

print("assembling samples ..."); ts = time.time()
train_samples = make_samples(ftrain)
test_samples = make_samples(ftest)
print(f"  assembled in {time.time()-ts:.1f}s")

TRUE = {v: np.stack([s["gt"][v] for s in test_samples], 0) for v in B.VARS}  # (N,D,H,W)
TEST_UNOBS = np.stack([s["unobs_mask"] for s in test_samples], 0)            # (N,H,W)
obs_frac = 1.0 - TEST_UNOBS.sum() / (grid.ocean.sum() * len(test_samples))
print(f"test observed-column fraction excluded from headline metric: {obs_frac:.3%}")

def stack_pred(preds):
    return {v: np.stack([p[v] for p in preds], 0) for v in B.VARS}

results = []
depth_tables = {}

def record(method, cfg, preds):
    P = stack_pred(preds)
    ev_all = metrics.evaluate(P, TRUE, grid.depth)                 # all ocean (leaky)
    ev_un = metrics.evaluate_masked(P, TRUE, TEST_UNOBS, grid.depth)  # unobserved-only
    row = {"method": method, "config": cfg,
           "TEMP_all": ev_all["overall"]["TEMP"], "SALT_all": ev_all["overall"]["SALT"],
           "TEMP_unobs": ev_un["overall"]["TEMP"], "SALT_unobs": ev_un["overall"]["SALT"]}
    results.append(row)
    depth_tables[(method, cfg)] = ev_un["by_depth"]
    print(f"  [{method:12s} | {cfg:18s}] "
          f"unobs TEMP={row['TEMP_unobs']:.4f} SALT={row['SALT_unobs']:.4f}  "
          f"(all-ocean TEMP={row['TEMP_all']:.4f})")

CONFIGS = {
    "profiles_only":     ("profiles",),
    "woa_only":          ("woa",),
    "profiles_woa":      ("profiles", "woa"),
    "profiles_woa_surf": ("profiles", "woa", "surf"),
}

# ============================ references / floors ============================
print("\n== reference floors ==")
record("woa_prior", "woa_only", [B.predict_climatology(s) for s in test_samples])
record("clim_floor", "train_clim", [B.predict_clim_floor(s, clim, grid) for s in test_samples])

print("\n== nearest ==")
record("nearest", "profiles_only", [B.predict_nearest(s, use_woa=False) for s in test_samples])
record("nearest", "profiles_woa", [B.predict_nearest(s, use_woa=True) for s in test_samples])

print("\n== mlp (anomaly target) ==")
for name, cfg in CONFIGS.items():
    preds = B.train_predict_mlp(train_samples, test_samples, grid, norm, cfg, rng, device)
    record("mlp", name, preds)

print("\n== unet depthwise (anomaly target, unobserved-only loss) ==")
for name, cfg in CONFIGS.items():
    preds = B.train_predict_unet(train_samples, test_samples, grid, norm, cfg, device,
                                 unobs_loss=True)
    record("unet_depthwise", name, preds)

print("\n== unet joint-depth (anomaly target, unobserved-only loss) ==")
for name, cfg in CONFIGS.items():
    preds = B.train_predict_unet_joint(train_samples, test_samples, grid, norm, cfg, device,
                                       unobs_loss=True)
    record("unet_joint", name, preds)

# ============================ reproducible config ============================
def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"

knobs = {k: getattr(C, k) for k in dir(C)
         if k.isupper() and isinstance(getattr(C, k), (int, float, str, list, tuple))}
run_cfg = {
    "git_commit": git_commit(),
    "smoke": args.smoke,
    "seed": C.SEED,
    "device": device,
    "train_months": tr_idx.tolist(),
    "test_months": te_idx.tolist(),
    "obs_column_fraction_excluded": float(obs_frac),
    "target": "train-only monthly CESM2 anomaly",
    "headline_metric": "unobserved-only RMSE (profile columns excluded)",
    "knobs": knobs,
}

os.makedirs(C.CACHE, exist_ok=True)
np.savez(os.path.join(C.CACHE, "week1_depth_tables.npz"),
         depths=grid.depth,
         **{f"{m}__{c}__{v}": dt[v]
            for (m, c), dt in depth_tables.items() for v in B.VARS})
with open(os.path.join(C.CACHE, "week1_results.json"), "w") as f:
    json.dump({"results": results, "run_config": run_cfg}, f, indent=2)

# ============================ markdown report ============================
floor_T = next(r["TEMP_unobs"] for r in results if r["method"] == "clim_floor")
floor_S = next(r["SALT_unobs"] for r in results if r["method"] == "clim_floor")

lines = []
lines.append("# Week-1 Audit — Corrected Baseline Matrix\n")
lines.append(f"- **Target:** train-only monthly CESM2 anomaly (was: absolute z-scored field)")
lines.append(f"- **Headline metric:** unobserved-only RMSE — {obs_frac:.2%} of ocean columns "
             f"(observed profiles) excluded from scoring")
lines.append(f"- **Floor:** train-only CESM2 climatology = **{floor_T:.3f} degC / {floor_S:.3f} PSU** "
             f"(the real floor; WOA prior is bias-dominated and reported separately)")
lines.append(f"- **Split:** {tr_idx.size} train / {te_idx.size} test months, seed {C.SEED}, "
             f"{C.N_PROFILES} profiles/month")
lines.append(f"- **Commit:** `{run_cfg['git_commit'][:12]}`  ·  smoke={args.smoke}\n")
lines.append("The two right-most columns are the honest numbers. The two 'all-ocean' columns "
             "include observed columns and are inflated by leakage — the gap between them and the "
             "unobserved columns is the leakage the earlier sweep was scoring on.\n")
lines.append("| method | config | TEMP unobs | SALT unobs | TEMP all-ocean | SALT all-ocean |")
lines.append("|---|---|---|---|---|---|")
def fmt(x): return f"{x:.4f}" if np.isfinite(x) else "—"
for r in results:
    lines.append(f"| {r['method']} | {r['config']} | **{fmt(r['TEMP_unobs'])}** | "
                 f"**{fmt(r['SALT_unobs'])}** | {fmt(r['TEMP_all'])} | {fmt(r['SALT_all'])} |")
lines.append("")
lines.append("### Skill vs train-only climatology floor (unobserved-only)\n")
lines.append("Skill = 1 − RMSE / RMSE_floor (positive = beats its own climatology).\n")
lines.append("| method | config | TEMP skill | SALT skill |")
lines.append("|---|---|---|---|")
for r in results:
    if r["method"] in ("woa_prior", "clim_floor"):
        continue
    sT = 1 - r["TEMP_unobs"] / floor_T if floor_T else np.nan
    sS = 1 - r["SALT_unobs"] / floor_S if floor_S else np.nan
    lines.append(f"| {r['method']} | {r['config']} | {sT:+.3f} | {sS:+.3f} |")
lines.append("")
report_path = os.path.join(C.REPORTS, "week1_audit.md")
with open(report_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nDONE in {time.time()-t0:.1f}s")
print(f"  -> reports/week1_audit.md")
print(f"  -> outputs/cache/week1_results.json")
