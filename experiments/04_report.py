"""Assemble the baseline table + depth-resolved tables into markdown reports."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ocean_tokenizer import config as C

with open(os.path.join(C.CACHE, "baseline_results.json")) as f:
    R = json.load(f)
dt = np.load(os.path.join(C.CACHE, "baseline_depth_tables.npz"))
depths = R["depths"]
results = R["results"]

CONFIG_ORDER = ["profiles_only", "woa_only", "profiles_woa", "profiles_woa_surf"]
METHOD_ORDER = ["climatology", "nearest", "mlp", "unet"]

def get(method, cfg, var):
    for r in results:
        if r["method"] == method and r["config"] == cfg:
            return r[f"RMSE_{var}"]
    return None

L = []
L.append("# Baseline Table — Sparse-Profile Ocean-State Reconstruction\n")
L.append(f"- **Ground truth:** CESM2-LE full simulation (held-out test months)")
L.append(f"- **Train months:** {R['n_train']}  |  **Test months:** {R['n_test']}  "
         f"|  **Synthetic Argo profiles/month:** {R['n_profiles']}")
L.append(f"- **Target depths:** {len(depths)} levels, {depths[0]:.0f}–{depths[-1]:.0f} m")
L.append(f"- **Metric:** RMSE (NaN-aware, ocean only). TEMP in °C, SALT in PSU.\n")

# ---- overall table: TEMP ----
for var, unit in [("TEMP", "°C"), ("SALT", "PSU")]:
    L.append(f"## Overall RMSE — {var} ({unit})\n")
    L.append("| method \\ config | profiles_only | woa_only | profiles_woa | profiles_woa_surf |")
    L.append("|---|---|---|---|---|")
    for m in METHOD_ORDER:
        cells = []
        for cfg in CONFIG_ORDER:
            v = get(m, cfg, var)
            cells.append(f"{v:.4f}" if v is not None else "–")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

# ---- best per config ----
L.append("## Best method per configuration (TEMP / SALT)\n")
L.append("| config | best TEMP | best SALT |")
L.append("|---|---|---|")
for cfg in CONFIG_ORDER:
    bt = min(((get(m, cfg, "TEMP"), m) for m in METHOD_ORDER
              if get(m, cfg, "TEMP") is not None), default=(None, "-"))
    bs = min(((get(m, cfg, "SALT"), m) for m in METHOD_ORDER
              if get(m, cfg, "SALT") is not None), default=(None, "-"))
    L.append(f"| {cfg} | {bt[0]:.4f} ({bt[1]}) | {bs[0]:.4f} ({bs[1]}) |")
L.append("")

# ---- depth-resolved table for the strongest method (unet) across configs ----
def depth_col(method, cfg, var):
    key = f"{method}__{cfg}__{var}"
    return dt[key] if key in dt.files else None

for var in ["TEMP", "SALT"]:
    L.append(f"## RMSE by depth — U-Net across configs — {var}\n")
    header = "| depth (m) | " + " | ".join(CONFIG_ORDER) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(CONFIG_ORDER) + 1))
    cols = {cfg: depth_col("unet", cfg, var) for cfg in CONFIG_ORDER}
    for di, d in enumerate(depths):
        row = [f"{d:.0f}"]
        for cfg in CONFIG_ORDER:
            c = cols[cfg]
            row.append(f"{c[di]:.3f}" if c is not None else "–")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

# ---- depth-resolved comparison of all methods at richest config ----
rich = "profiles_woa_surf"
for var in ["TEMP", "SALT"]:
    L.append(f"## RMSE by depth — all methods @ {rich} — {var}\n")
    methods = [m for m in METHOD_ORDER if depth_col(m, rich, var) is not None]
    L.append("| depth (m) | " + " | ".join(methods) + " |")
    L.append("|" + "---|" * (len(methods) + 1))
    cols = {m: depth_col(m, rich, var) for m in methods}
    for di, d in enumerate(depths):
        row = [f"{d:.0f}"] + [f"{cols[m][di]:.3f}" for m in methods]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

with open(os.path.join(C.REPORTS, "baseline_table.md"), "w") as f:
    f.write("\n".join(L))
print("wrote reports/baseline_table.md")
