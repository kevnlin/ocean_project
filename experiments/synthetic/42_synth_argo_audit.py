"""Data audit of the synthetic Argo task, before any training result is read.

Two plan items (2026-09-26) are data questions and are answered here from the
cohort alone:

* **cell-centre vs at-position anomaly.** The registered target subtracts the
  climatology at the centre of the 1-deg cell. At a profile ~50 km from that
  centre the difference ``clim(position) - clim(centre)`` is climatological
  gradient, not anomaly, yet it becomes part of the target. How big is it
  against the anomaly itself, by level?

* **salinity outliers / extreme z-scores.** Every value in this cohort is true
  (CESM2 interpolated, no sensor), so any tail here is physical, and anything a
  QC rule flags here is a false positive. Where are the extreme z-scores, how
  much of the loss do they carry, and does the per-level std need replacing?

Plus the split drift that sets the z unit of the held-out years.

  .venv/bin/python experiments/synthetic/42_synth_argo_audit.py
"""
from __future__ import annotations

import copy, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.argo_obs import ArgoNorm
from ocean_tokenizer.audit_tools import load_cohort, robust_qc, cohort_path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REP = os.path.join(ROOT, "reports", "synthetic")
OUTD = os.path.join(ROOT, "outputs", "audit", "synthetic")
os.makedirs(OUTD, exist_ok=True)
CH = ("TEMP", "SALT")
UNIT = {"TEMP": "°C", "SALT": "PSU"}
SPLITS = {"train": (2000, 2003), "validation": (2004, 2004),
          "development": (2005, 2005)}

#: (name, lat range, lon range in 0-360) — first match wins
BOXES = [("Persian Gulf", (23, 31), (47, 57)), ("Red Sea", (12, 30), (32, 44)),
         ("Black Sea", (40, 47), (27, 42)), ("Baltic", (53, 66), (9, 31)),
         ("Mediterranean", (30, 46), (354, 360)), ("Mediterranean", (30, 46), (0, 37)),
         ("Hudson Bay", (50, 66), (265, 295)), ("Arctic (>66N)", (66, 90), (0, 360)),
         ("Southern Ocean (<60S)", (-90, -60), (0, 360)),
         ("Bay of Bengal", (5, 23), (80, 95)), ("Amazon plume", (-5, 15), (295, 320)),
         ("Congo plume", (-10, 0), (5, 14)), ("Rio de la Plata", (-40, -33), (300, 310))]


def region(lat, lon):
    lon = lon % 360.0
    for name, (a, b), (c, d) in BOXES:
        if a <= lat < b and c <= lon < d:
            return name
    return "open ocean"


c_cell, _ = load_cohort(ROOT, "synthetic", SPLITS, anomaly="cell")
c_pos, _ = load_cohort(ROOT, "synthetic", SPLITS, anomaly="exact")
LEV = c_pos.levels
tr = c_pos.year_split == "train"
d = xr.open_dataset(cohort_path(ROOT, "synthetic"))
order = np.argsort(np.asarray(d["month_index"].values, int), kind="stable")
raw = {ch: np.asarray(d[ch].values, float)[order] for ch in CH}
raw_fill = {ch: np.asarray(d[f"{ch}_RAW"].values, float)[order] for ch in CH}
clim_pos = {ch: np.asarray(d[f"CLIM_POS_{ch}"].values, float)[order] for ch in CH}
out = {"n_profiles": int(c_pos.lat.size), "levels": LEV.tolist(),
       "fill_values_removed_grid": int(d.attrs["fill_values_removed"]),
       "fill_columns": int(d.attrs["fill_columns"])}
d.close()

# ------------------------------------------------------------ 1. the task card
per_month = np.bincount(c_pos.month_index)
q = c_pos.float_split == "heldout_float"
out["task"] = {"months": int(per_month.size), "per_month": int(per_month[0]),
               "queries_per_month": int(q.sum() / per_month.size),
               "finite_by_level": np.isfinite(c_pos.SALT).mean(0).round(4).tolist(),
               # mean spacing of the INPUT profiles if spread evenly over the ocean
               "input_spacing_km": float(np.sqrt(3.61e8 / ((~q).sum() / per_month.size)))}

# ------------------------------------------------- 2. cell-centre vs position
rep = {}
for ch in CH:
    spur = getattr(c_cell, ch) - getattr(c_pos, ch)       # = clim_pos - clim_cell
    rows = {}
    for name, m in (("train", tr), ("heldout_years", ~tr)):
        a = getattr(c_pos, ch)[m]
        s = spur[m]
        rows[name] = {
            "rms_anomaly": np.sqrt(np.nanmean(a ** 2, 0)).tolist(),
            "rms_spurious": np.sqrt(np.nanmean(s ** 2, 0)).tolist(),
            # share of the cell-centre target's mean square that is gradient
            "spurious_share": (np.nanmean(s ** 2, 0)
                               / np.nanmean(getattr(c_cell, ch)[m] ** 2, 0)).tolist()}
    a = np.abs(spur)
    k = np.nanargmax(np.where(np.isfinite(a), a, -1))
    i, l = np.unravel_index(k, a.shape)
    rows["max_abs"] = {"value": float(a[i, l]), "lat": float(c_pos.lat[i]),
                       "lon": float(c_pos.lon[i]), "depth": float(LEV[l])}
    # share of |spurious| above 25 % of the local anomaly std, all levels pooled
    sd = np.nanstd(getattr(c_pos, ch)[tr], 0)
    rows["frac_values_over_quarter_sd"] = float(np.nanmean(a > 0.25 * sd))
    top = np.argsort(np.nan_to_num(np.nanmax(a, 1), nan=-1))[::-1][:2000]
    reg = [region(c_pos.lat[t], c_pos.lon[t]) for t in top]
    u, n = np.unique(reg, return_counts=True)
    rows["top2000_profiles_by_region"] = {str(x): int(y) for x, y in
                                          sorted(zip(u, n), key=lambda z: -z[1])}
    rep[ch] = rows
out["cell_vs_position"] = rep

# ------------------------------------------------------ 3. z-score tail audit
norm = ArgoNorm.fit(c_pos, "train")
zaud = {}
for ch in CH:
    z = norm.z(ch, getattr(c_pos, ch))
    az = np.abs(z)
    fin = np.isfinite(az)
    v = az[fin]
    z2 = np.nansum(z ** 2)
    per_level = []
    for l in range(LEV.size):
        a = getattr(c_pos, ch)[tr, l]
        a = a[np.isfinite(a)]
        mad = 1.4826 * np.median(np.abs(a - np.median(a)))
        k4 = float(np.mean(((a - a.mean()) / a.std()) ** 4))
        per_level.append({"depth": float(LEV[l]), "std": float(a.std()),
                          "mad_scale": float(mad), "std_over_mad": float(a.std() / mad),
                          "kurtosis": k4,
                          "max_abs_z": float(np.nanmax(az[:, l]))})
    ext = np.argwhere(az > 8)
    reg = [region(c_pos.lat[i], c_pos.lon[i]) for i, _ in ext]
    u, n = np.unique(reg, return_counts=True)
    top = np.argsort(np.where(fin, az, -1).ravel())[::-1][:15]
    top_rows = []
    for k in top:
        i, l = np.unravel_index(k, az.shape)
        top_rows.append({"z": float(z[i, l]), "lat": float(c_pos.lat[i]),
                         "lon": float(c_pos.lon[i]), "depth": float(LEV[l]),
                         "month": int(c_pos.month_index[i]),
                         "value": float(raw[ch][i, l]),
                         "anomaly": float(getattr(c_pos, ch)[i, l]),
                         "region": region(c_pos.lat[i], c_pos.lon[i])})
    zaud[ch] = {
        "quantiles_abs_z": {str(p): float(np.quantile(v, p))
                            for p in (0.5, 0.9, 0.99, 0.999, 0.9999)},
        "max_abs_z": float(v.max()),
        "n_values": int(v.size),
        "count_over": {str(t): int((v > t).sum()) for t in (4, 5, 8, 20)},
        "share_of_sum_z2_over": {str(t): float(np.nansum(np.where(az > t, z ** 2, 0)) / z2)
                                 for t in (4, 5, 8)},
        "over8_by_region": {str(x): int(y) for x, y in
                            sorted(zip(u, n), key=lambda t: -t[1])},
        "per_level": per_level, "top": top_rows}
    # the regridding fill, had it been left in: what z would those values get?
    # (the climatology is undefined at a fill cell, so the reference is the same
    # profile's nearest wet level above — adjacent levels differ by far less
    # than the zero a fill corner mixes in)
    touched = np.isfinite(raw_fill[ch]) & ~np.isfinite(raw[ch])
    if touched.any():
        za, vals = [], []
        for i, l in np.argwhere(touched):
            above = np.where(np.isfinite(raw[ch][i, :l]))[0]
            if above.size:
                za.append((raw_fill[ch][i, l] - raw[ch][i, above[-1]]) / norm.std[ch][l])
                vals.append(raw_fill[ch][i, l])
        zaud[ch]["fill_if_kept"] = {"n": int(touched.sum()),
                                    "value_min": float(np.min(vals)),
                                    "value_max": float(np.max(vals)),
                                    "z_min": float(np.min(za)), "z_max": float(np.max(za))}
out["zscores"] = zaud

# what the real pipeline's 8-sigma QC would remove from this all-true cohort
cq = copy.deepcopy(c_pos)
qr = robust_qc(cq, k_sigma=8.0)
out["robust_qc_8sigma"] = {"values_flagged": qr.n_values_flagged,
                           "profile_channels_dropped": qr.n_profiles_dropped,
                           "std_change_max_rel": {
                               ch: float(np.nanmax(np.abs(np.array(qr.std_after[ch])
                                                          / np.array(qr.std_before[ch]) - 1)))
                               for ch in CH}}

# ------------------------------------------------------------ 4. split drift
drift = {}
for ch in CH:
    rows = {}
    for sp in SPLITS:
        m = c_pos.year_split == sp
        a = getattr(c_pos, ch)[m]
        rows[sp] = {"mean": np.nanmean(a, 0).tolist(), "std": np.nanstd(a, 0).tolist(),
                    "rms_z": float(np.sqrt(np.nanmean(norm.z(ch, a) ** 2)))}
    drift[ch] = rows
out["split_drift"] = drift

with open(os.path.join(OUTD, "data_audit.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("task", "robust_qc_8sigma")}, indent=1))
for ch in CH:
    print(ch, "cell-vs-pos rms spurious (train):",
          np.round(rep[ch]["train"]["rms_spurious"], 4).tolist())
    print(ch, "spurious share (train):",
          np.round(rep[ch]["train"]["spurious_share"], 4).tolist())
    print(ch, "z quantiles", zaud[ch]["quantiles_abs_z"], "max", zaud[ch]["max_abs_z"],
          "count_over", zaud[ch]["count_over"], "share", zaud[ch]["share_of_sum_z2_over"])
    print(ch, ">8 by region", zaud[ch]["over8_by_region"])
    print(ch, "drift rms_z", {k: round(v["rms_z"], 3) for k, v in drift[ch].items()})
    print(ch, "fill_if_kept", zaud[ch].get("fill_if_kept"))
