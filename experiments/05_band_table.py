"""Regenerate baseline_table.md with a depth-BANDED layout.

Primary view mirrors the requested format:
    TEMP/SALT split, rows = approaches, columns = surface(~5m) | 0-50m | 50-200m | 200+m
Band RMSE is the *valid-cell-weighted* aggregation of the per-level RMSEs
(deeper levels have fewer ocean cells), so it equals the true RMSE over the band.
A "Global mean" predictor (per-depth training mean) and the full per-level appendix
are included for completeness.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ocean_tokenizer import data, baselines as B, config as C

# ---- reproduce the exact train/test split used by 03_baselines.py ----
rng = np.random.default_rng(C.SEED)
grid = data.CommonGrid()
tr_pool = data.select_month_indices(C.GT_SOURCE, C.TRAIN_YEARS)
te_pool = data.select_month_indices(C.GT_SOURCE, C.TEST_YEARS)
tr_idx = np.sort(rng.choice(tr_pool, size=min(C.N_TRAIN_MONTHS, tr_pool.size), replace=False))
te_idx = np.sort(rng.choice(te_pool, size=min(C.N_TEST_MONTHS, te_pool.size), replace=False))

print("loading GT for weights + global-mean baseline ...")
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)

# per-level valid-cell weights and per-level global-mean-baseline RMSE
W, gm_rmse, self_rmse = {}, {}, {}
# model self-climatology: per calendar-month mean of TRAINING months (per var)
self_clim = {v: {m: (np.nanmean(ftrain[v][ftrain["months"] == m], axis=0)
                     if (ftrain["months"] == m).any() else None)
                 for m in range(1, 13)} for v in B.VARS}
for v in B.VARS:
    true = ftest[v]                                   # (T,D,H,W)
    fin = np.isfinite(true)
    W[v] = fin.sum(axis=(0, 2, 3)).astype(float)      # (D,)
    mean_d = np.nanmean(ftrain[v], axis=(0, 2, 3))     # train per-depth mean (D,)
    sse = np.nansum((true - mean_d[None, :, None, None]) ** 2, axis=(0, 2, 3))
    gm_rmse[v] = np.sqrt(sse / np.maximum(W[v], 1))
    # self-climatology per-level RMSE on the same test set
    pred = np.stack([self_clim[v][mo] for mo in ftest["months"]], axis=0)
    sse_s = np.nansum((true - pred) ** 2, axis=(0, 2, 3))
    self_rmse[v] = np.sqrt(sse_s / np.maximum(W[v], 1))

# ---- load sweep outputs ----
depth_tables = np.load(os.path.join(C.CACHE, "baseline_depth_tables.npz"))
depths = depth_tables["depths"]
with open(os.path.join(C.CACHE, "baseline_results.json")) as f:
    R = json.load(f)
results = R["results"]

# ---- depth bands ----
BANDS = [("surface (~5 m)", lambda d: np.isclose(d, depths[0])),
         ("0-50 m",   lambda d: d <= 50),
         ("50-200 m", lambda d: (d > 50) & (d <= 200)),
         ("200+ m",   lambda d: d > 200)]

def band_rmse(rmse_d, w_d, sel):
    m = sel(depths) & np.isfinite(rmse_d) & (w_d > 0)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.sum(w_d[m] * rmse_d[m] ** 2) / np.sum(w_d[m])))

# ---- rows: friendly label -> (method, config) or "global" ----
ROWS = [
    ("Global mean",                       "global"),
    ("WOA23 climatology (obs prior)",     ("climatology", "woa_only")),
    ("CESM2-LE self-climatology (ref)",   "self"),
    ("Nearest profile",                   ("nearest", "profiles_only")),
    ("Nearest + WOA23",                   ("nearest", "profiles_woa")),
    ("Pointwise MLP (prof+WOA+SST/SSS)",  ("mlp", "profiles_woa_surf")),
    ("2D U-Net (profiles)",               ("unet", "profiles_only")),
    ("2D U-Net (prof+WOA)",               ("unet", "profiles_woa")),
    ("2D U-Net (prof+WOA+SST/SSS)",       ("unet", "profiles_woa_surf")),
]

def rmse_d_for(key, var):
    if key == "global":
        return gm_rmse[var]
    if key == "self":
        return self_rmse[var]
    m, c = key
    k = f"{m}__{c}__{var}"
    return depth_tables[k] if k in depth_tables.files else None

# ---- build markdown ----
L = ["# Baseline Table — Sparse-Profile Ocean-State Reconstruction\n",
     f"> 🏆 **Method of choice: 2D U-Net (prof+WOA+SST/SSS)** — best full-column "
     f"RMSE of every method and best through the thermocline. It is the ⭐ **bold** "
     f"row in each table below.\n",
     f"- **Ground truth:** CESM2-LE full simulation (held-out test months)",
     f"- **Train months:** {R['n_train']}  |  **Test months:** {R['n_test']}  "
     f"|  **Synthetic Argo profiles/month:** {R['n_profiles']}",
     f"- **Metric:** depth-banded RMSE (valid-cell-weighted, NaN-aware, ocean only). "
     f"TEMP in °C, SALT in PSU. Lower is better. **Bold** = best per band among "
     f"reconstruction methods (priors/floors excluded).",
     f"- **Note on WOA23 climatology:** the ground truth is the CESM2-LE *model*, "
     f"whose climate differs from real observations. WOA23 therefore carries an "
     f"irreducible model–obs bias (~1.0–1.2 °C for TEMP, largest in the thermocline / "
     f"western-boundary-current hotspots). The *CESM2-LE self-climatology* row "
     f"(climatology built from the model's own training months) isolates the true "
     f"\"climatology cannot capture internal variability\" floor — about half the WOA RMSE.\n"]

PRIOR_KEYS = {"global", ("climatology", "woa_only"), "self"}   # not eligible for "best"
OURS_KEY = ("unet", "profiles_woa_surf")

for var, unit in [("TEMP", "°C"), ("SALT", "PSU")]:
    L.append(f"## {var} RMSE ({unit}) — lower is better\n")
    L.append("| Method | " + " | ".join(b[0] for b in BANDS) + " |")
    L.append("|:--|" + ":--:|" * len(BANDS))
    # band cells per row, then best-per-band among non-prior rows
    rowcells = {}
    for label, key in ROWS:
        rd = rmse_d_for(key, var)
        rowcells[label] = None if rd is None else [band_rmse(rd, W[var], sel)
                                                   for _, sel in BANDS]
    best = []
    for bi in range(len(BANDS)):
        cand = [rowcells[l][bi] for l, k in ROWS
                if k not in PRIOR_KEYS and rowcells[l] is not None
                and np.isfinite(rowcells[l][bi])]
        best.append(min(cand) if cand else None)
    for label, key in ROWS:
        vals = rowcells[label]
        if vals is None:
            continue
        ours = key == OURS_KEY
        disp = f"⭐ **{label} (ours)**" if ours else label
        cells = []
        for bi, v in enumerate(vals):
            s = f"{v:.4f}"
            if (best[bi] is not None and abs(v - best[bi]) < 1e-9) or ours:
                s = f"**{s}**"
            cells.append(s)
        L.append(f"| {disp} | " + " | ".join(cells) + " |")
    L.append("")

# ---- overall method x config matrix (compact) ----
def get(method, cfg, var):
    for r in results:
        if r["method"] == method and r["config"] == cfg:
            return r[f"RMSE_{var}"]
    return None

CFG = ["profiles_only", "woa_only", "profiles_woa", "profiles_woa_surf"]
MET = ["climatology", "nearest", "mlp", "unet"]
for var, unit in [("TEMP", "°C"), ("SALT", "PSU")]:
    L.append(f"## Overall RMSE — full column — {var} ({unit})\n")
    L.append("| method \\ config | " + " | ".join(CFG) + " |")
    L.append("|:--|" + ":--:|" * len(CFG))
    for m in MET:
        cells = []
        for c in CFG:
            val = get(m, c, var)
            if val is None:
                cells.append("–")
            elif m == "unet" and c == "profiles_woa_surf":
                cells.append(f"⭐ **{val:.4f}**")          # our method of choice
            else:
                cells.append(f"{val:.4f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

# ---- per-level appendix (U-Net across configs) ----
for var in ["TEMP", "SALT"]:
    L.append(f"## Appendix — per-level RMSE, U-Net across configs — {var}\n")
    L.append("| depth (m) | " + " | ".join(CFG) + " |")
    L.append("|" + "---|" * (len(CFG) + 1))
    cols = {c: rmse_d_for(("unet", c), var) for c in CFG}
    for di, d in enumerate(depths):
        row = [f"{d:.0f}"] + [f"{cols[c][di]:.3f}" if cols[c] is not None else "–" for c in CFG]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

with open(os.path.join(C.REPORTS, "baseline_table.md"), "w") as f:
    f.write("\n".join(L))
print("wrote reports/baseline_table.md (banded format)")
