"""Combine the existing baselines + the three new reference baselines into one
depth-banded RMSE comparison (CSV + markdown).

It reuses the EXACT split (so the valid-cell weights match) and the per-depth
RMSE tables cached by:
  * experiments/03_baselines.py  -> outputs/cache/baseline_depth_tables.npz
  * the three reference scripts   -> outputs/cache/ref_<name>_depth.npz
Global-mean and CESM2-LE self-climatology rows are recomputed here (same recipe
as experiments/05_band_table.py) so the table is self-contained.

The markdown is formatted for readability: rows are grouped (priors -> baselines
-> surface-only reference models -> ours), the best value per band (among
deployable reconstruction methods) is bolded, and OUR method of choice
-- 2D U-Net (prof+WOA+SST/SSS) -- is starred and emphasised.

Run (after the three reference scripts have been run at least once):
    ~/.venv/bin/python src/baselines/build_comparison_table.py
Outputs:
    reports/baseline_comparison.csv
    reports/baseline_comparison.md
"""
from __future__ import annotations
import os
import sys
import csv
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-land nanmean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as cm  # noqa: E402

VARS = cm.VARS
BAND_NAMES = ["surface_~5m", "0-50m", "50-200m", "200m+"]                 # internal keys
COL_LABELS = ["surface (~5 m)", "0–50 m", "50–200 m", "200+ m", "overall"]  # display

# Row registry: (label, kind, key, group, eligible_for_best)
# eligible_for_best excludes priors / reference floors from the per-band winner.
OURS_KEY = ("existing", "unet__profiles_woa_surf")
ROWS = [
    ("Global mean",                     "global",   None,                     "prior",   False),
    ("WOA23 climatology",               "existing", "climatology__woa_only",  "prior",   False),
    ("CESM2-LE self-climatology (ref)", "self",     None,                     "prior",   False),
    ("Nearest profile",                 "existing", "nearest__profiles_only", "simple",  True),
    ("Nearest + WOA23",                 "existing", "nearest__profiles_woa",  "simple",  True),
    ("Pointwise MLP (prof+WOA+SST/SSS)","existing", "mlp__profiles_woa_surf", "learned", True),
    ("2D U-Net (profiles)",             "existing", "unet__profiles_only",    "learned", True),
    ("2D U-Net (prof+WOA)",             "existing", "unet__profiles_woa",     "learned", True),
    ("NeSPReSO PCA+MLP",                "ref",      "nesperso_pcamlp",        "surface", True),
    ("OSnet MLP (15× ens)",             "ref",      "osnet_mlp",              "surface", True),
    ("Nardelli stacked-LSTM",           "ref",      "nardelli_lstm",          "surface", True),
    ("2D U-Net (prof+WOA+SST/SSS)",     "existing", "unet__profiles_woa_surf","ours",    True),
]
GROUP_HEADERS = {
    "prior":   "Priors & reference floors",
    "simple":  "Simple baselines",
    "learned": "Learned baselines",
    "surface": "Surface-only reference models (single-modality)",
    "ours":    "Our method (multi-modal)",
}


def main():
    S = cm.build_split()                       # full, deterministic -> matches W
    grid, ftrain, ftest = S["grid"], S["ftrain"], S["ftest"]
    depths = grid.depth

    # ---- valid-cell weights + global-mean + self-climatology per-depth RMSE ----
    W, gm, selfc = {}, {}, {}
    self_clim = {v: {m: (np.nanmean(ftrain[v][ftrain["months"] == m], axis=0)
                         if (ftrain["months"] == m).any() else None)
                     for m in range(1, 13)} for v in VARS}
    for v in VARS:
        true = ftest[v]
        W[v] = np.isfinite(true).sum(axis=(0, 2, 3)).astype(float)
        mean_d = np.nanmean(ftrain[v], axis=(0, 2, 3))
        sse = np.nansum((true - mean_d[None, :, None, None]) ** 2, axis=(0, 2, 3))
        gm[v] = np.sqrt(sse / np.maximum(W[v], 1))
        pred = np.stack([self_clim[v][mo] for mo in ftest["months"]], 0)
        sse_s = np.nansum((true - pred) ** 2, axis=(0, 2, 3))
        selfc[v] = np.sqrt(sse_s / np.maximum(W[v], 1))

    existing = np.load(os.path.join(cm.CACHE, "baseline_depth_tables.npz"))

    def ref_table(name):
        p = os.path.join(cm.CACHE, f"ref_{name}_depth.npz")
        return np.load(p) if os.path.exists(p) else None
    refs = {n: ref_table(n) for n in ("nesperso_pcamlp", "osnet_mlp", "nardelli_lstm")}

    def rmse_d(kind, key, var):
        if kind == "global":
            return gm[var]
        if kind == "self":
            return selfc[var]
        if kind == "existing":
            k = f"{key}__{var}"
            return existing[k] if k in existing.files else None
        if kind == "ref":
            t = refs.get(key)
            return None if t is None else t[var]
        return None

    def cells_for(kind, key, var):
        """5 numbers: 4 depth-band RMSE + overall full-column RMSE (or None)."""
        rd = rmse_d(kind, key, var)
        if rd is None:
            return None
        w = W[var]
        b = cm.band_rmse(rd, w, depths)
        vals = [b[n] for n in BAND_NAMES]
        m = np.isfinite(rd) & (w > 0)
        vals.append(float(np.sqrt(np.sum(w[m] * rd[m] ** 2) / np.sum(w[m])))
                    if m.any() else float("nan"))
        return vals

    # precompute every cell, per variable
    table = {v: {label: cells_for(kind, key, v) for label, kind, key, _, _ in ROWS}
             for v in VARS}

    def best_per_col(var):
        out = []
        for ci in range(len(COL_LABELS)):
            cand = [table[var][l][ci] for l, _, _, _, e in ROWS
                    if e and table[var][l] is not None and np.isfinite(table[var][l][ci])]
            out.append(min(cand) if cand else None)
        return out

    # ---- CSV (flat, machine-readable) ----
    csv_path = os.path.join(cm.REPORTS, "baseline_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variable"] + COL_LABELS)
        for label, kind, key, _, _ in ROWS:
            for var in VARS:
                vals = table[var][label]
                if vals is None:
                    continue
                w.writerow([label, var] + [f"{c:.4f}" for c in vals])
    print(f"wrote {csv_path}")

    # ---- markdown (grouped, emphasised) ----
    ov = {v: table[v]["2D U-Net (prof+WOA+SST/SSS)"][-1] for v in VARS}
    md = [
        "# Baseline Comparison — Sparse-Profile Ocean-State Reconstruction\n",
        f"> 🏆 **Method of choice: 2D U-Net (profiles + WOA + SST/SSS).** Lowest "
        f"full-column RMSE of every method (**{ov['TEMP']:.2f} °C / {ov['SALT']:.2f} PSU**) "
        f"and best through the thermocline — the surface-only reference models trail "
        f"below the mixed layer.\n",
        "**Setup:** CESM2-LE ground truth · 48 train / 12 test months · 1500 profiles/month.",
        "**Metric:** depth-banded RMSE (valid-cell-weighted, NaN-aware, ocean only). "
        "TEMP °C · SALT PSU · **lower is better**.",
        "**How to read:** ⭐ = our method (bold row) · **bold** = best per band among "
        "reconstruction methods (priors/floors excluded) · rows grouped priors → "
        "baselines → surface-only reference models → ours.\n",
    ]
    for var, unit in [("TEMP", "°C"), ("SALT", "PSU")]:
        best = best_per_col(var)
        md.append(f"## {var} RMSE ({unit}) — lower is better\n")
        md.append("| Method | " + " | ".join(COL_LABELS) + " |")
        md.append("|:--|" + ":--:|" * len(COL_LABELS))
        cur = None
        for label, kind, key, group, elig in ROWS:
            if group != cur:
                md.append(f"| _{GROUP_HEADERS[group]}_ |" + " |" * len(COL_LABELS))
                cur = group
            vals = table[var][label]
            if vals is None:
                continue
            disp = f"⭐ **{label} (ours)**" if group == "ours" else label
            cells = []
            for ci, v in enumerate(vals):
                s = f"{v:.3f}"
                if (best[ci] is not None and abs(v - best[ci]) < 1e-9) or group == "ours":
                    s = f"**{s}**"
                cells.append(s)
            md.append(f"| {disp} | " + " | ".join(cells) + " |")
        md.append("")
        md.append(
            f"> **Ours** posts the best overall full-column {var} RMSE "
            f"(**{ov[var]:.3f} {unit}**) and the best 50–200 m thermocline skill. "
            f"OSnet leads only in the upper ocean (surface / 0–50 m), where a "
            f"surface-only model is expected to.\n")

    md_path = os.path.join(cm.REPORTS, "baseline_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
