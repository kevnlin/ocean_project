"""Real-Argo data checks for the pre-shutdown plan (experiments A and D).

A (data half): how large is the cell-centre target artifact on REAL Argo?
    anomcell - anomx = C(profile position) - C(1-deg cell centre), WOA23,
    measured against the at-position anomaly itself, by depth band, on the
    training years.

D: are the extreme z-scores data errors or ocean signal? For the largest |z|
    in each channel: the observation, position, depth, climatology, anomaly,
    normalization mean/std and z, the float's data mode, and simple checks for
    fill values, a spike against the neighbouring levels, and a whole-profile
    offset. Then what the 8-sigma QC (`--ablation qc`) removes.

  .venv/bin/python experiments/real_data/65_target_qc_audit.py
  -> outputs/audit/global/target_qc_audit.json, reports/real_data/target_qc_audit.md
"""
from __future__ import annotations

import copy, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer.argo_obs import ArgoNorm
from ocean_tokenizer.audit_tools import load_cohort, robust_qc, cohort_path, band_of_lat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLITS = {"train": (2016, 2020), "validation": (2021, 2021), "development": (2022, 2023)}
CH = ("TEMP", "SALT")
UNIT = {"TEMP": "°C", "SALT": "PSU"}
BANDS = (("0-100 m", 0, 100), ("100-300 m", 100, 300), ("300-700 m", 300, 700),
         ("700-985 m", 700, 1000))
LATB = ("0-20°", "20-45°", "45-90°")

cell, _ = load_cohort(ROOT, "global", SPLITS, anomaly="cell")
pos, _ = load_cohort(ROOT, "global", SPLITS, anomaly="exact")
raw_ds = xr.open_dataset(cohort_path(ROOT, "global"))
order = np.argsort(np.asarray(raw_ds["month_index"].values, int), kind="stable")
raw = {ch: np.asarray(raw_ds[ch].values, float)[order] for ch in CH}
mode = np.asarray(raw_ds["data_mode"].values).astype(str)[order]
raw_ds.close()
LEV = pos.levels
tr = pos.year_split == "train"
out = {"splits": SPLITS, "n_profiles": int(pos.lat.size), "n_train": int(tr.sum())}


def bands(per_level):
    a = np.asarray(per_level, float)
    return [float(np.nanmean(a[(LEV > lo) & (LEV <= hi)] if lo else a[LEV <= hi]))
            for _, lo, hi in BANDS]


# ------------------------------------------------------------- A: C(p) - C(c)
A = {}
lb = band_of_lat(pos.lat)
for ch in CH:
    d = getattr(cell, ch) - getattr(pos, ch)          # = C(p) - C(c)
    a = getattr(pos, ch)
    ms_d, ms_a = np.nanmean(d[tr] ** 2, 0), np.nanmean(a[tr] ** 2, 0)
    ms_c = np.nanmean(getattr(cell, ch)[tr] ** 2, 0)
    ad = np.abs(d[tr])
    A[ch] = {"rms_artifact": np.sqrt(ms_d).tolist(), "rms_anomaly": np.sqrt(ms_a).tolist(),
             "share_of_cell_target": (ms_d / ms_c).tolist(),
             "p99_abs": np.nanquantile(ad, 0.99, axis=0).tolist(),
             "max_abs": float(np.nanmax(ad)),
             "frac_over_quarter_sd": float(np.nanmean(ad > 0.25 * np.nanstd(a[tr], 0))),
             "rms_by_lat_band": {LATB[b]: float(np.sqrt(np.nanmean(d[tr & (lb == b)] ** 2)))
                                 for b in range(3)}}
out["A_artifact"] = A

# ------------------------------------------------------------- D: extremes
norm = ArgoNorm.fit(pos, "train")
D = {}
for ch in CH:
    x, an = raw[ch], getattr(pos, ch)
    z = norm.z(ch, an)
    az = np.where(np.isfinite(z), np.abs(z), -1.0)
    v = az[az >= 0]
    top = np.argsort(az.ravel())[::-1][:25]
    rows, kinds = [], {}
    for k in top:
        i, l = np.unravel_index(k, az.shape)
        col = x[i]
        nb = [col[j] for j in (l - 1, l + 1) if 0 <= j < LEV.size and np.isfinite(col[j])]
        spike = (abs(col[l] - np.mean(nb)) > 5 * norm.std[ch][l]) if nb else False
        zp = np.abs(z[i][np.isfinite(z[i])])
        whole = bool(zp.size and np.median(zp) > 5)
        fill = bool(col[l] in (0.0, 99999.0, 99999.99) or col[l] < (-3 if ch == "TEMP" else 2)
                    or col[l] > (40 if ch == "TEMP" else 42))
        # impossible for this place: an anomaly no water mass there can carry
        # (15 degC, 5 PSU), or near-freezing water where the climatology is tropical
        clim = col[l] - an[i, l]
        wrong = bool(abs(an[i, l]) > (15.0 if ch == "TEMP" else 5.0)
                     or (ch == "TEMP" and col[l] < 2.0 and clim > 20.0))
        kind = ("fill/impossible value" if fill else
                "impossible for location" if wrong else
                "whole-profile offset" if whole else
                "single-level spike" if spike else "plausible ocean value")
        kinds[kind] = kinds.get(kind, 0) + 1
        rows.append({"z": float(z[i, l]), "value": float(col[l]),
                     "climatology": float(col[l] - an[i, l]), "anomaly": float(an[i, l]),
                     "norm_mean": float(norm.mean[ch][l]), "norm_std": float(norm.std[ch][l]),
                     "lat": float(pos.lat[i]), "lon": float(pos.lon[i]), "depth": float(LEV[l]),
                     "year": int(pos.year[i]), "month": int(pos.month_index[i] % 12 + 1),
                     "wmo": str(pos.wmo[i]), "data_mode": mode[i],
                     "median_abs_z_profile": float(np.median(zp)) if zp.size else None,
                     "kind": kind})
    D[ch] = {"quantiles_abs_z": {str(p): float(np.quantile(v, p)) for p in (0.5, 0.99, 0.999, 0.9999)},
             "max_abs_z": float(v.max()), "count_over": {str(t): int((v > t).sum()) for t in (8, 20, 100)},
             "share_sum_z2_over8": float(np.sum(np.where(az > 8, az ** 2, 0)) / np.sum(v ** 2)),
             "top25_kinds": kinds, "top": rows}

# what the 8-sigma QC removes (on a copy; same call as --ablation qc)
cq = copy.deepcopy(pos)
rep = robust_qc(cq, k_sigma=8.0, data_mode=mode)
qc = {"values_flagged": rep.n_values_flagged, "profile_channels_dropped": rep.n_profiles_dropped,
      "flagged_profiles_by_data_mode": rep.by_data_mode,
      "profiles_by_data_mode": {m: int((mode == m).sum()) for m in np.unique(mode)},
      "std_change_max_rel": {ch: float(np.nanmax(np.abs(np.array(rep.std_after[ch])
                                                        / np.array(rep.std_before[ch]) - 1)))
                             for ch in CH}}
for ch in CH:
    gone = np.isfinite(getattr(pos, ch)) & ~np.isfinite(getattr(cq, ch))
    rows = np.where(gone.any(1))[0]
    qc[f"{ch}_flagged_by_lat_band"] = {LATB[b]: int((lb[rows] == b).sum()) for b in range(3)}
    qc[f"{ch}_flagged_abs_z_median"] = float(np.nanmedian(np.abs(norm.z(ch, getattr(pos, ch)))[gone]))
D["qc_8sigma"] = qc
out["D_extremes"] = D

os.makedirs(os.path.join(ROOT, "outputs", "audit", "global"), exist_ok=True)
with open(os.path.join(ROOT, "outputs", "audit", "global", "target_qc_audit.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)

# ------------------------------------------------------------- report
def t(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows]) + "\n"


md = ["# Real-Argo target and QC audit\n",
      "<!-- generated by experiments/real_data/65_target_qc_audit.py -->\n",
      f"Global cohort, {out['n_profiles']:,} profiles; statistics on the training "
      f"years 2016-2020 ({out['n_train']:,} profiles) unless noted.\n",
      "## A. Size of the cell-centre artifact, C(position) − C(cell centre)\n",
      "WOA23 monthly climatology bilinearly interpolated to each profile, minus "
      "WOA23 at the centre of the profile's 1° cell. Band means of RMS artifact / "
      "RMS at-position anomaly, and in brackets the share of the cell-centre "
      "target's mean square that is artifact:\n",
      t(["", *[b for b, _, _ in BANDS]],
        [[f"{ch} ({UNIT[ch]})"] + [f"{a:.3f} / {b:.3f} ({100 * s:.1f} %)" for a, b, s in zip(
            bands(A[ch]["rms_artifact"]), bands(A[ch]["rms_anomaly"]),
            bands(A[ch]["share_of_cell_target"]))] for ch in CH]),
      t(["", "max |artifact|", "values where it exceeds ¼ of the level's anomaly std",
         "RMS by |lat| 0-20° / 20-45° / 45-90°"],
        [[ch, f"{A[ch]['max_abs']:.3f} {UNIT[ch]}", f"{100 * A[ch]['frac_over_quarter_sd']:.1f} %",
          " / ".join(f"{v:.3f}" for v in A[ch]["rms_by_lat_band"].values())] for ch in CH]),
      "## D. Extreme z-scores\n",
      "z = (at-position anomaly − train mean) / train std, per level.\n",
      t(["", "median |z|", "99 %", "99.9 %", "99.99 %", "max |z|", "|z| > 8", "|z| > 20",
         "|z| > 100", "Σz² share from |z| > 8"],
        [[ch, *(f"{D[ch]['quantiles_abs_z'][q]:.2f}" for q in ("0.5", "0.99", "0.999", "0.9999")),
          f"{D[ch]['max_abs_z']:.0f}", *(f"{D[ch]['count_over'][k]:,}" for k in ("8", "20", "100")),
          f"{100 * D[ch]['share_sum_z2_over8']:.0f} %"] for ch in CH]),
      "The 25 largest values per channel, classified: " + "; ".join(
          f"{ch}: " + ", ".join(f"{k} {n}" for k, n in D[ch]["top25_kinds"].items()) for ch in CH) + ".\n",
      "The ten largest per channel:\n",
      t(["", "z", "value", "WOA23 clim", "anomaly", "mean / std", "position", "depth", "date",
         "float (mode)", "profile median |z|", "kind"],
        [[ch, f"{r['z']:+.0f}", f"{r['value']:.3f}", f"{r['climatology']:.3f}", f"{r['anomaly']:+.3f}",
          f"{r['norm_mean']:+.3f} / {r['norm_std']:.3f}", f"{r['lat']:.2f}°, {r['lon']:.2f}°E",
          f"{r['depth']:.0f} m", f"{r['year']}-{r['month']:02d}", f"{r['wmo']} ({r['data_mode']})",
          f"{r['median_abs_z_profile']:.1f}", r["kind"]] for ch in CH for r in D[ch]["top"][:10]]),
      "## D. What the 8σ QC removes\n",
      t(["", "values flagged", "profile-channels dropped", "median |z| of flagged values",
         "flagged profiles by |lat| 0-20° / 20-45° / 45-90°"],
        [[ch, f"{qc['values_flagged'][ch]:,}", f"{qc['profile_channels_dropped'][ch]:,}",
          f"{qc[f'{ch}_flagged_abs_z_median']:.1f}",
          " / ".join(f"{v:,}" for v in qc[f"{ch}_flagged_by_lat_band"].values())] for ch in CH]),
      "Flagged profiles by Argo data mode (R real-time, A adjusted, D delayed), "
      "against all profiles: " + "; ".join(
          f"{ch}: " + ", ".join(f"{m} {n:,}" for m, n in qc["flagged_profiles_by_data_mode"].get(ch, {}).items())
          for ch in CH) + "; all profiles: " + ", ".join(
          f"{m} {n:,}" for m, n in qc["profiles_by_data_mode"].items()) + ".\n"]
with open(os.path.join(ROOT, "reports", "real_data", "target_qc_audit.md"), "w") as f:
    f.write("\n".join(md))
print("\n".join(md))
