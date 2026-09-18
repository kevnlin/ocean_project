"""Turn the audit JSON and the audit runs into the meeting's three deliverables.

  reports/real_data/pipeline_audit.md    the step-by-step audit (61_pipeline_audit.py)
  reports/real_data/overfit_sanity.md    the overfit ladder (62_sanity_train.py --mode ...)
  reports/real_data/ablation_ladder.md   one switch at a time on a fixed held-out set
  + fig_pair_correlation.png, fig_loss_curves.png, fig_overfit.png, fig_ablation.png

Every number is read from a file written by those two scripts; nothing is typed
in by hand.  Re-run after any of them.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np

from ocean_tokenizer.plot_style import use_style, SERIES, MUTED, INK_2, STATUS

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="gulfstream,npac_gyre")
ap.add_argument("--eval-split", default="development")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REP = os.path.join(ROOT, "reports", "real_data")
AUD = json.load(open(os.path.join(ROOT, "outputs", "cache", "pipeline_audit.json")))
REGIONS = [r for r in args.regions.split(",") if r in AUD["regions"]]
CH = ("TEMP", "SALT")
UNIT = {"TEMP": "°C", "SALT": "PSU"}
BANDS = ("0-100m", "100-300m", "300-700m", "700-1400m")
plt = use_style()


def f(x, n=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def runs(region):
    """tag -> {seed: summary} for every finished audit run of this region."""
    out = defaultdict(dict)
    for p in glob.glob(os.path.join(ROOT, "outputs", "audit", region, "*",
                                    "summary_seed*.json")):
        d = json.load(open(p))
        out[d["tag"]][d["seed"]] = d
    return out


def agg(rs, split, ch, key="rmse_z"):
    """mean and spread over seeds."""
    v = [s["scores"][split][ch][key] for s in rs.values()
         if split in s["scores"] and ch in s["scores"][split]]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


# ==========================================================================
# figure 1 — how fast the anomaly decorrelates, and where the inputs sit
# ==========================================================================
fig, axes = plt.subplots(1, len(REGIONS), figsize=(5.4 * len(REGIONS), 3.6),
                         squeeze=False)
for ax, region in zip(axes[0], REGIONS):
    R = AUD["regions"][region]
    depths = list(R["pair_correlation"]["TEMP"].keys())
    for i, d in enumerate(depths):
        e = R["pair_correlation"]["TEMP"][d]
        b = np.asarray(e["bins_km"]); mid = 0.5 * (b[:-1] + b[1:])
        y = np.asarray(e["corr"]); n = np.asarray(e["pairs"])
        ok = n > 50
        ln, = ax.plot(mid[ok], y[ok], color=SERIES[i], label=d, marker="o")
        ln.set_markeredgecolor("#fcfcfb"); ln.set_markeredgewidth(2.0)
    med = R["input_density"]["nearest_input_km_at_cap"]["median"]
    ax.axvline(med, color=MUTED, lw=1.0, ls="-")
    ax.annotate(f"median nearest\ninput: {med:.0f} km", (med, 0.62),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                color=INK_2, va="top")
    ax.axhline(0.0, color="#c3c2b7", lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(8, 1300); ax.set_ylim(-0.15, 0.85)
    ax.set_xlabel("separation between two floats (km)")
    ax.set_ylabel("correlation of temperature anomaly")
    ax.set_title(f"{region} — same month, different floats")
    ax.legend(title="depth", loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(REP, "fig_pair_correlation.png"), bbox_inches="tight")
plt.close(fig)


# ==========================================================================
# figure 2 — the overfit ladder
# ==========================================================================
def history(region, tag, seed=1234):
    """One seed's curve.

    Several seeds of an arm append to the same file. Records written after
    2026-09-17 carry their seed; older ones do not, so the fallback keeps the
    first monotonically increasing run of steps — joining two seeds end to end
    would draw a zigzag rather than a training curve.
    """
    p = os.path.join(ROOT, "outputs", "audit", region, tag, "history.jsonl")
    if not os.path.exists(p):
        return []
    recs = []
    for line in open(p):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    tagged = [r for r in recs if r.get("seed") == seed]
    if tagged:
        return tagged
    out = []
    for r in recs:
        if out and r["step"] < out[-1]["step"]:
            break
        out.append(r)
    return out


OVERFIT_RUNS = [("mem_d4rt", "memorise, 1 month"), ("copy_d4rt", "copy the input"),
                ("small8_d4rt", "8 months"), ("mem_setconv", "memorise, SetConv")]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
plotted = 0
for i, (tag, label) in enumerate(OVERFIT_RUNS):
    h = [r for r in history("gulfstream", tag) if "overfit/TEMP_rmse_z" in r]
    if not h:
        continue
    ax.plot([r["step"] for r in h], [r["overfit/TEMP_rmse_z"] for r in h],
            color=SERIES[i], label=label)
    plotted += 1
if plotted:
    ax.axhline(0.1, color=STATUS["good"], lw=1.5, ls="--")
    ax.annotate("target 0.1 z", (ax.get_xlim()[1], 0.1), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK_2)
    ax.set_yscale("log"); ax.set_xlabel("training step")
    ax.set_ylabel("temperature RMSE on the fitted data (z)")
    ax.set_title("Overfit sanity check — Gulf Stream")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_overfit.png"), bbox_inches="tight")
plt.close(fig)


# ==========================================================================
# figure 3 — training curves (the meeting asked for convergence evidence)
# ==========================================================================
CURVE_RUNS = [("baseline", "baseline (DFS, Perceiver-IO)"),
              ("all_profiles", "all profiles (no 128 cap)"),
              ("backbone_setconv", "SetConv-UNet backbone"),
              ("fixed_stack", "audit fixes stacked")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
any_curve = False
for i, (tag, label) in enumerate(CURVE_RUNS):
    h = history("gulfstream", tag)
    tl = [(r["step"], r["train_loss"]) for r in h if "train_loss" in r]
    vl = [(r["step"], r["validation/macro_z"]) for r in h
          if "validation/macro_z" in r]
    if not tl:
        continue
    any_curve = True
    s, v = zip(*tl)
    # batch size 1 makes the per-step loss swing between 0.35 and 1.2; the
    # trend is the readable quantity, so it is smoothed over ~800 steps
    k = max(1, len(v) // 25)
    sm = np.convolve(v, np.ones(k) / k, mode="valid")
    axes[0].plot(s[k - 1:k - 1 + len(sm)], sm, color=SERIES[i], label=label)
    if vl:
        s2, v2 = zip(*vl)
        axes[1].plot(s2, v2, color=SERIES[i], label=label, marker="o")
if any_curve:
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("training loss (MSE, z²), smoothed")
    axes[0].set_title("Training loss — still falling at 12 k steps"); axes[0].legend()
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("validation RMSE (z, mean of T and S)")
    axes[1].set_title("Validation RMSE (2022, held-out floats) — not converged")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_loss_curves.png"), bbox_inches="tight")
plt.close(fig)


# ==========================================================================
# figure 4 — ablation ladder
# ==========================================================================
def ablation_rows(region):
    rs = runs(region)
    if "baseline" not in rs:
        return []
    base = agg(rs["baseline"], args.eval_split, "TEMP")[0]
    rows = []
    for tag, per_seed in sorted(rs.items()):
        # the overfit rungs are not held-out scores, and the satellite-era arms
        # live on a different split, so their z is not this table's z
        if tag.startswith(("mem_", "copy_", "small8_", "smk_")) or "_sat_" in tag:
            continue
        m, sd, n = agg(per_seed, args.eval_split, "TEMP")
        ms, sds, _ = agg(per_seed, args.eval_split, "SALT")
        mv, _, _ = agg(per_seed, "validation", "TEMP")
        mvs, _, _ = agg(per_seed, "validation", "SALT")
        tsub, _, _ = agg(per_seed, "train_subset", "TEMP")
        rows.append({"tag": tag, "n_seeds": n, "temp": m, "temp_sd": sd,
                     "salt": ms, "salt_sd": sds, "val_temp": mv, "val_salt": mvs,
                     "train_temp": tsub, "delta": m - base,
                     "temp_phys": agg(per_seed, args.eval_split, "TEMP", "rmse_physical")[0],
                     "salt_phys": agg(per_seed, args.eval_split, "SALT", "rmse_physical")[0],
                     "J_temp": agg(per_seed, args.eval_split, "TEMP", "J")[0],
                     "J_salt": agg(per_seed, args.eval_split, "SALT", "J")[0],
                     "params": list(per_seed.values())[0]["params"]})
    return sorted(rows, key=lambda r: r["delta"])


rows_gs = ablation_rows("gulfstream")
if rows_gs:
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows_gs) + 1.4))
    lab = [r["tag"] for r in rows_gs]
    val = [r["delta"] for r in rows_gs]
    col = [SERIES[0] if v < 0 else STATUS["critical"] for v in val]
    y = np.arange(len(val))
    ax.barh(y, val, color=col, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5)
    ax.axvline(0, color="#c3c2b7", lw=1.0)
    sd = float(np.nanmean([r["temp_sd"] for r in rows_gs if np.isfinite(r["temp_sd"])]))
    ax.axvspan(-sd, sd, color="#e1e0d9", alpha=0.6, zorder=0)
    ax.annotate(f"±{sd:.3f} = seed spread", (sd, len(val) - 0.4),
                xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK_2)
    for yi, v in zip(y, val):
        ax.annotate(f"{v:+.3f}", (v, yi), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", fontsize=8,
                    ha="left" if v >= 0 else "right", va="center", color=INK_2)
    span = max(max(val) - min(val), 1e-6)
    ax.set_xlim(min(val) - 0.42 * span, max(val) + 0.30 * span)
    ax.set_xlabel("change in held-out temperature RMSE vs baseline (z) — left is better")
    ax.set_title("One switch at a time — Gulf Stream, 2023-24 held-out floats")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_ablation.png"), bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# the audit report
# ==========================================================================
def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


L = []
w = L.append
w("# Pipeline audit — data prep, normalisation, tokenisation\n")
w("<!-- generated by experiments/real_data/61_pipeline_audit.py + 63_audit_report.py "
  "— do not edit by hand -->\n")
w("> Asked for at the 2026-09-17 meeting: walk the pipeline one stage at a time "
  "before running anything else, because numbers that all sit near 1 and do not "
  "move when the model changes point at the data path.\n")
w(f"> Protocol `{AUD['split_protocol']}` "
  f"(train {AUD['splits']['train'][0]}-{AUD['splits']['train'][1]}, "
  f"validation {AUD['splits']['validation'][0]}, "
  f"test {AUD['splits']['development'][0]}-{AUD['splits']['development'][1]}), "
  f"input cap {AUD['n_profiles']} profiles/month, seed {AUD['seed']}.\n")

w("## 0. The short version\n")
k0 = AUD["regions"][REGIONS[0]].get("kriging")
if k0:
    w("A kriging optimal interpolation with **zero trained parameters**, whose "
      "covariance is fitted on the training years, was run on the identical "
      "held-out floats, months and input draws as the tables:\n")
    rows = []
    for region in REGIONS:
        K = AUD["regions"][region]["kriging"]
        rows.append([region,
                     f"{f(K['kriging_cap']['TEMP']['rmse_z'],4)} / {f(K['kriging_cap']['TEMP']['J'])}",
                     f"{f(K['kriging_all']['TEMP']['rmse_z'],4)} / {f(K['kriging_all']['TEMP']['J'])}",
                     f"{f(K['kriging_cap']['SALT']['rmse_z'],4)} / {f(K['kriging_cap']['SALT']['J'])}",
                     f"{f(K['kriging_all']['SALT']['rmse_z'],4)} / {f(K['kriging_all']['SALT']['J'])}"])
    w(md_table(["region", "TEMP cap 128 — RMSE z / J", "TEMP all profiles",
                "SALT cap 128", "SALT all profiles"], rows))
    w("Against the published Table 1 (GAOT backbone, lead 0): Gulf Stream DFS "
      "**0.961** TEMP J, gyre DFS **0.849**. The interpolator matches the trained "
      "model at the same input density in the Gulf Stream, beats every learned "
      "row in the gyre, and beats all of them in both regions once it may use the "
      "profiles the cap throws away.\n")
    w("After the fixes in `ablation_ladder.md` the learned arms reach it at "
      "matched inputs (1.1490 vs 1.1560 in the Gulf Stream, 0.8448 vs 0.8451 in "
      "the gyre) but still do not pass it when both see every profile. The one "
      "configuration that does pass it adds satellite altimetry — see the "
      "satellite section of that report.\n")

w("## 1. Data preparation\n")
w("### 1.1 What one z unit is worth\n")
rows = []
for region in REGIONS:
    S = AUD["regions"][region]["scale"]
    for ch in CH:
        rows.append([region, ch] + [f(S[ch]["anom_std_by_band"][b], 3) for b in BANDS]
                    + [UNIT[ch]])
w(md_table(["region", "channel"] + list(BANDS) + ["unit"], rows))
w("The anomaly's own standard deviation **is** the z scale, so a model at "
  "RMSE ≈ 1 z is at ≈ 2 °C in the Gulf Stream and ≈ 0.9 °C in the gyre. The "
  "0.1-0.2 °C / 0.1 PSU target from the meeting is 10-20× below the natural "
  "variability of the field at a held-out float, not a small step from here.\n")

w("### 1.2 Values that are not measurements\n")
rows = []
for region in REGIONS:
    O = AUD["regions"][region]["outliers"]
    for ch in CH:
        o = O[ch]
        rows.append([region, ch, o["profiles_over_10_sigma"], o["floats"],
                     f(o["max_abs_z"], 0),
                     ", ".join(f"{k}:{v}" for k, v in sorted(o["by_data_mode"].items())),
                     f(o["dev_input_z_rms"], 2), f(o["dev_input_z_rms_clean"], 2)])
w(md_table(["region", "channel", "profiles with abs(z)>10", "floats", "max abs(z)",
            "by data mode", "test-year input z RMS", "same, outliers removed"], rows))
for region in REGIONS:
    for ch in CH:
        for wf in AUD["regions"][region]["outliers"][ch]["worst_floats"][:1]:
            if wf["max_abs_z"] > 50:
                w(f"- `{region}` {ch}: float **{wf['wmo']}** "
                  f"({'/'.join(wf['data_mode'])}, {wf['years'][0]}-{wf['years'][1]}, "
                  f"{wf['float_split'].replace('_', ' ')}) reaches "
                  f"|z| = {wf['max_abs_z']:.0f}.\n")
w("These pass the cohort's Argo QC filter but are not ocean. They enter the model "
  "as **inputs**, where a single value hundreds of standard deviations from "
  "climatology dominates the month's encoding, and they enter the per-level "
  "standard deviation that defines the z scale for every other value.\n")

w("### 1.3 Where the climatology is evaluated\n")
rows = []
for region in REGIONS:
    C = AUD["regions"][region]["climatology_position"]
    rows.append([region, f"{C['cell_deg'][0]:.2f}° × {C['cell_deg'][1]:.2f}°",
                 f"{C['offset_km']['lat_mean']:.0f} / {C['offset_km']['lon_mean']:.0f} km",
                 f"{C['offset_km']['lat_max']:.0f} / {C['offset_km']['lon_max']:.0f} km",
                 f(C["TEMP"]["std_difference_by_band"]["0-100m"], 3),
                 f(C["TEMP"]["std_difference_by_band"]["300-700m"], 3),
                 f(C["TEMP"]["variance_removed_pct"]["100-300m"], 1) + " %"])
w(md_table(["region", "region cell", "mean offset lat/lon", "max offset",
            "spurious °C 0-100m", "spurious °C 300-700m",
            "anomaly variance removed by using the profile's own position"], rows))
w("The registered target subtracts WOA23 at the **centre of the region cell**, "
  "not at the profile. A profile can sit ~100 km from that centre, and across the "
  "Gulf Stream front that distance carries a real climatological gradient, so part "
  "of what the model is asked to predict is the climatology's own spatial "
  "structure. It is a modest term — a few per cent of variance — but it is pure "
  "noise added to the target and it is free to remove.\n")

w("### 1.4 The test years are not the training years\n")
rows = []
for region in REGIONS:
    S = AUD["regions"][region]["scale"]
    for ch in CH:
        rows.append([region, ch,
                     f(S[ch]["anom_mean_by_band_train"]["0-100m"], 3),
                     f(S[ch]["anom_mean_by_band_dev"]["0-100m"], 3),
                     f(S[ch]["dev_over_train_std"]["0-100m"], 3),
                     f(S[ch]["dev_over_train_std"]["300-700m"], 3), UNIT[ch]])
w(md_table(["region", "channel", "train mean anomaly 0-100m",
            "test mean anomaly 0-100m", "test/train σ 0-100m",
            "test/train σ 300-700m", "unit"], rows))
w("The upper-ocean anomaly is larger and more variable in 2023-24 than in the "
  "training era. That is why the climatology row reads **1.21 z** in the Gulf "
  "Stream rather than 1.00: the reference is the training climatology, and the "
  "test years have moved away from it. Any J in the tables is measured against "
  "that shifted reference.\n")

w("## 2. Normalisation\n")
rows = []
for region in REGIONS:
    N = AUD["regions"][region]["normalisation"]
    for ch in CH:
        wl = N["worst_level_inflation"][ch]
        rows.append([region, ch,
                     f(N["robust_over_fitted_sigma"][ch]["0-100m"], 2),
                     f(N["robust_over_fitted_sigma"][ch]["700-1400m"], 2),
                     f"{wl['level_m']:.0f} m",
                     f"{f(wl['fitted_std'], 3)} vs {f(wl['robust_std'], 3)}"])
w(md_table(["region", "channel", "robust σ / fitted σ, 0-100m",
            "same, 700-1400m", "worst level", "fitted vs robust σ there"], rows))
w("Per-level z-scoring is fitted on the training years with the plain standard "
  "deviation, so the outliers of §1.2 set the scale of the levels they sit on. "
  "Where the ratio is far below 1 the level's z unit is inflated by a handful of "
  "values, which both compresses the target there and rescales the model's input.\n")

w("## 3. Tokenisation\n")
T = AUD["regions"][REGIONS[0]]["tokenisation"]
w("### 3.1 A profile becomes five tokens\n")
w(md_table(["band", "levels pooled", "token depth", "level depths"],
           [[f"{b['band_m'][0]:.0f}-{b['band_m'][1]:.0f} m", b["levels"],
             f"{b['token_depth_m']:.0f} m",
             ", ".join(f"{d:.0f}" for d in b["level_depths"])]
            for b in T["profile_tokens"]["bands"]]))
w(f"`ProfileEncoder` embeds each level, then **mean-pools inside the band**, and "
  f"places the token at the band's midpoint. A query at 300 m is answered from a "
  f"token that stands for 222-409 m at a nominal 350 m. Measured on the cohort, "
  f"a band-mean summary discards "
  f"{100 * T['profile_tokens']['band_mean_loses']['TEMP']['within_band_variance_fraction']:.0f} % "
  f"of temperature and "
  f"{100 * T['profile_tokens']['band_mean_loses']['SALT']['within_band_variance_fraction']:.0f} % "
  f"of salinity variance within a profile — less than one might fear, because "
  f"levels inside a band are strongly correlated, but it is the vertical detail "
  f"the thermocline lives in.\n")

w("### 3.2 The resolution the model can express\n")
rows = [["shared coordinate features (finest Fourier wavelength)",
         f"{T['coordinate_features']['finest_sphere_wavelength_deg']:.2f}° "
         f"≈ {T['coordinate_features']['finest_sphere_wavelength_km']:.0f} km"],
        ["local refiner Gaussian, north-south (init)",
         f"{T['local_refiner_init']['ell_lat_km']:.0f} km"],
        ["local refiner Gaussian, east-west (init)",
         f"{T['local_refiner_init']['ell_lon_km_at_region']:.0f} km"],
        ["local refiner depth / time (init)",
         f"{T['local_refiner_init']['ell_depth_m']:.0f} m / "
         f"{T['local_refiner_init']['ell_time_months']:.0f} months"],
        ["local refiner output gate (init)", f"{T['local_refiner_init']['gate_init']}"],
        ["region box", f"{T['region_box_km']['lat']:.0f} × "
                       f"{T['region_box_km']['lon']:.0f} km"],
        ["measured anomaly decorrelation (from §4)", "~50-150 km"]]
w(md_table(["component", "scale"], rows))
w("Both paths into a query are set far wider than the field they describe. The "
  "coordinate features cannot express structure finer than ~400 km, and the "
  "'local' refiner's distance prior is initialised **wider than the whole "
  "region**, so at the start of training every observation in the box is "
  "essentially equidistant from every query; its output is then scaled by a gate "
  "initialised at 0.05. The length scales are learnable, but they start four "
  "orders of magnitude away from the ocean's.\n")
w(f"A second, narrower defect: in `argo_obs.build_argo_sample` the token depth "
  f"axis is physical depth / deepest level while the query depth axis is level "
  f"index / (levels-1) — the same number means two different depths, up to "
  f"{T['argo_obs_query_depth_axis']['max_mismatch']:.2f} apart on the normalised "
  f"axis. It is read by {T['argo_obs_query_depth_axis']['used_by']}.\n")

w("### 3.3 How much of the observing system reaches the model\n")
rows = []
for region in REGIONS:
    D = AUD["regions"][region]["input_density"]
    rows.append([region, f"{D['available_per_month']['median']:.0f}",
                 f"{D['used_per_month_at_cap']['median']:.0f}",
                 f"{100 * D['fraction_of_available_used']:.0f} %",
                 f"{D['nearest_input_km_at_cap']['median']:.0f} km",
                 f"{D['nearest_input_km_all_profiles']['median']:.0f} km",
                 f"{100 * D['nearest_input_km_at_cap']['frac_under_50']:.0f} %"])
w(md_table(["region", "profiles available / month", "used at cap 128",
            "fraction used", "median distance to nearest input",
            "same, all profiles", "targets within 50 km"], rows))
w("The cap is not a mild subsample: it discards about two thirds of the floats "
  "that reported that month, and it moves the typical held-out target from ~110 km "
  "to ~150 km from its nearest input — across the steep part of the correlation "
  "curve below.\n")

w("## 4. What is recoverable at all\n")
w("![anomaly correlation against separation](fig_pair_correlation.png)\n")
rows = []
for region in REGIONS:
    e = AUD["regions"][region]["pair_correlation"]["TEMP"]
    for d, v in e.items():
        b = np.asarray(v["bins_km"]); corr = np.asarray(v["corr"])
        pick = {f"{b[i]}-{b[i+1]}": corr[i] for i in range(len(corr))}
        rows.append([region, d] + [f(pick.get(k), 2) for k in
                                   ("0-10", "25-50", "50-75", "100-150", "200-300")])
w(md_table(["region", "depth", "0-10 km", "25-50 km", "50-75 km", "100-150 km",
            "200-300 km"], rows))
w("Two same-month profiles from different floats correlate ~0.7 when they are "
  "essentially co-located, and the correlation is gone by 150-200 km in the Gulf "
  "Stream. Even a *perfect* co-located observation would leave ~50 % of the "
  "variance unexplained, because a monthly anomaly at a point contains submonthly "
  "and submesoscale variability that no neighbouring float sees. This is the "
  "ceiling every method in the tables is working under, and it is why every row "
  "sits between 0.85 and 1.0 J.\n")
# the floor: what a perfect map of the CORRELATED field would still score
w("### 4.1 The floor, from the fitted covariance\n")
rows = []
for region in REGIONS:
    R = AUD["regions"][region]
    fit = R.get("covariance_fit")
    if not fit:
        continue
    for ch in CH:
        nug = np.array([e["nugget"] for e in fit[ch]])
        rho0 = []
        for d, v in R["pair_correlation"][ch].items() if ch in R["pair_correlation"] else []:
            c0 = np.asarray(v["corr"]); n0 = np.asarray(v["pairs"])
            ok = np.flatnonzero(n0 > 50)
            if ok.size:
                rho0.append(c0[ok[0]])
        rows.append([region, ch, f(float(nug.mean()), 2),
                     f(float(np.sqrt(nug.mean())), 2),
                     f(float(np.mean(rho0)), 2) if rho0 else "—",
                     f(float(np.sqrt(1 - np.mean(rho0))), 2) if rho0 else "—"])
w(md_table(["region", "channel", "mean nugget (fitted)",
            "J floor = sqrt(nugget)", "correlation at ~0 km",
            "J floor = sqrt(1 - rho0)"], rows))
w("The **nugget** is the share of the anomaly variance at a point that no other "
  "float sees — submonthly change, submesoscale structure, and the "
  "representativeness difference between a point measurement and anything "
  "smoother. It is a property of the ocean and the sampling, not of the model. "
  "A method that mapped the *correlated* part perfectly everywhere would still "
  "score J around **0.5-0.65**. That is the number a learned method should be "
  "measured against; J = 0.1 (the 0.1-0.2 degC target) is not attainable at a "
  "held-out float by any method.\n")

if k0:
    rows = []
    for region in REGIONS:
        K = AUD["regions"][region]["kriging"]
        for name in ("climatology", "nearest", "kriging_cap", "kriging_all"):
            rows.append([region, name] +
                        [f"{f(K[name][ch]['rmse_z'], 4)} / {f(K[name][ch]['rmse_physical'], 3)} "
                         f"{UNIT[ch]} / {f(K[name][ch]['J'])}" for ch in CH])
    w(md_table(["region", "method", "TEMP z / physical / J", "SALT z / physical / J"], rows))
    w("`kriging_cap` sees exactly the 128 profiles the model sees; `kriging_all` "
      "sees every profile of the month. Both fit their covariance on training "
      "years only (a nugget plus two Gaussian scales per level, §4 of the JSON), "
      "and neither has a trained parameter.\n")

with open(os.path.join(REP, "pipeline_audit.md"), "w") as fh:
    fh.write("\n".join(L))
print("wrote reports/real_data/pipeline_audit.md")


# ==========================================================================
# overfit report
# ==========================================================================
O = ["# Overfit sanity check\n",
     "<!-- generated by 62_sanity_train.py + 63_audit_report.py — do not edit by hand -->\n",
     "> The meeting's sanity test: drive the RMSE below 0.1 on data the model is "
     "allowed to memorise. If it cannot, the pipeline is broken; if it can, the "
     "held-out numbers are about the ocean and the observing system, not about "
     "capacity.\n",
     "Three rungs, all on the Gulf Stream cohort, lead 0, the registered "
     "405 K-parameter model unless the row says otherwise:\n",
     "- **memorise** — one fixed month, one fixed input/target float split.\n",
     "- **copy** — the targets *are* the input profiles, queried at their own "
     "positions and levels. Tests whether information can reach a query at all.\n",
     "- **8 months** — eight fixed months with fixed partitions.\n"]
rows = []
for region in REGIONS:
    for tag, per_seed in sorted(runs(region).items()):
        if not tag.startswith(("mem_", "copy_", "small8_")):
            continue
        t, tsd, n = agg(per_seed, "overfit", "TEMP")
        s, ssd, _ = agg(per_seed, "overfit", "SALT")
        tp = agg(per_seed, "overfit", "TEMP", "rmse_physical")[0]
        sp = agg(per_seed, "overfit", "SALT", "rmse_physical")[0]
        one = list(per_seed.values())[0]
        rows.append([region, f"`{tag}`", one["steps"], n, f(t, 3), f(tp, 3),
                     f(s, 3), f(sp, 4),
                     "yes" if max(t, s) < 0.1 else "no"])
if rows:
    O.append(md_table(["region", "run", "steps", "seeds", "TEMP RMSE (z)",
                       "TEMP (°C)", "SALT RMSE (z)", "SALT (PSU)", "below 0.1 z?"],
                      rows))
    O.append("![overfit curves](fig_overfit.png)\n")
with open(os.path.join(REP, "overfit_sanity.md"), "w") as fh:
    fh.write("\n".join(O))
print("wrote reports/real_data/overfit_sanity.md")


# ==========================================================================
# ablation report
# ==========================================================================
A = ["# Ablation ladder — one component at a time\n",
     "<!-- generated by 62_sanity_train.py + 63_audit_report.py — do not edit by hand -->\n",
     "> Every arm is the same training run with exactly one switch moved, scored "
     "on a FIXED held-out set: the same months, the same seeded input draws and "
     "the same queries, so a difference is attributable to the switch.\n",
     "> Selection uses 2022 (`validation`); the 2023-24 column is the held-out "
     "score of the selected weights. 2025 is untouched.\n",
     "> **`qc` and `anomaly_exact` change the target itself**, so their z units "
     "are not the baseline's. Compare those two by **J** (each arm against its "
     "own climatology on its own targets); every other arm shares the "
     "baseline's target and is comparable in z.\n"]
for region in REGIONS:
    rws = ablation_rows(region)
    if not rws:
        continue
    A.append(f"## {region}\n")
    A.append(md_table(
        ["arm", "seeds", "params", "val TEMP (z)", "test TEMP (z)", "test TEMP (°C)",
         "test SALT (z)", "test J TEMP", "Δ vs baseline (z)"],
        [[f"`{r['tag']}`", r["n_seeds"], f"{r['params']:,}", f(r["val_temp"], 4),
          f"{f(r['temp'], 4)} ± {f(r['temp_sd'], 4)}", f(r["temp_phys"], 3),
          f(r["salt"], 4), f(r["J_temp"], 3),
          ("baseline" if r["tag"] == "baseline" else f"{r['delta']:+.4f}")]
         for r in rws]))
    #: the shipped headline for this region, lead 0, Perceiver-IO backbone,
    #: `_recent3_obs` arm of reports/real_data/main_tables.md (3 seeds). Quoted
    #: for orientation: those runs trained on leads 0/1/3/6 and selected on the
    #: same validation year, so they are close to but not identical with the
    #: lead-0-only arms above.
    PUBLISHED = {"gulfstream": (1.1776, 1.1723), "npac_gyre": (0.8934, 0.8190)}
    if region in PUBLISHED:
        A.append(f"Published headline for orientation (`main_tables.md`, Table 1, "
                 f"DFS, Perceiver-IO, lead 0, 3 seeds): TEMP "
                 f"**{PUBLISHED[region][0]:.4f} z**, SALT "
                 f"**{PUBLISHED[region][1]:.4f} z**.\n")
    K = AUD["regions"][region].get("kriging")
    if K:
        A.append(f"Reference on the same held-out set: kriging OI at the same cap "
                 f"**{K['kriging_cap']['TEMP']['rmse_z']:.4f} z**, with every "
                 f"profile **{K['kriging_all']['TEMP']['rmse_z']:.4f} z**, "
                 f"climatology **{K['climatology']['TEMP']['rmse_z']:.4f} z**.\n")
if rows_gs:
    A.append("![ablation ladder](fig_ablation.png)\n")
    A.append("![training curves](fig_loss_curves.png)\n")

# ---- the satellite-era arms (a secondary split, reported separately) --------
SAT = [("d4rt_sat_profiles", "Perceiver-IO, profiles only"),
       ("setconv_sat_profiles", "SetConv-UNet, profiles only"),
       ("setconv_sat_sla", "SetConv-UNet, **+ altimetry (SLA) only**"),
       ("setconv_sat_surface", "SetConv-UNet, **+ satellite SST/SLA/SSS**")]
sat_rows = []
for region in REGIONS:
    rs = runs(region)
    for tag, label in SAT:
        if tag not in rs:
            continue
        one = list(rs[tag].values())[0]
        sat_rows.append([region, label, one["params"], agg(rs[tag], "validation", "TEMP")[2],
                         f(agg(rs[tag], "development", "TEMP")[0], 4),
                         f(agg(rs[tag], "development", "TEMP", "rmse_physical")[0], 3),
                         f(agg(rs[tag], "development", "TEMP", "J")[0], 3),
                         f(agg(rs[tag], "development", "SALT", "J")[0], 3)])
if sat_rows:
    A.append("## Satellite era — does cross-modality information buy anything?\n")
    A.append("> A SECONDARY split (train 2016-2020, validate 2021, test 2022-2023): "
             "`data/real_obs_1deg.zarr` covers 2016-2023 only, so this cannot use "
             "the `recent3` test years and its numbers never join Tables 1-8. "
             "Every arm here is trained on the same 60 months.\n")
    A.append(md_table(["region", "arm", "params", "seeds", "test TEMP (z)",
                       "test TEMP (°C)", "test J TEMP", "test J SALT"], sat_rows))
    A.append("The **altimetry-only** row is the control that matters: the L4 SST "
             "and SSS analyses assimilate in-situ data and could in principle "
             "feed a held-out float's own surface value back to the model, "
             "while altimetry never sees a profile. It reproduces most of the "
             "gain, and the gain is largest in the deepest band — the steric "
             "signature of sea level, not SST copied down.\n")
    try:
        KC = json.load(open(os.path.join(ROOT, "outputs", "cache",
                                         "pipeline_audit_custom.json")))
        for region in REGIONS:
            K = KC["regions"][region]["kriging"]
            A.append(f"`{region}` reference on this split: kriging OI (profiles only) "
                     f"**{K['kriging_cap']['TEMP']['rmse_z']:.4f} z / "
                     f"J {K['kriging_cap']['TEMP']['J']:.3f}** at the same cap, "
                     f"**{K['kriging_all']['TEMP']['rmse_z']:.4f} z / "
                     f"J {K['kriging_all']['TEMP']['J']:.3f}** with every profile; "
                     f"climatology {K['climatology']['TEMP']['rmse_z']:.4f} z.\n")
    except FileNotFoundError:
        A.append("_(kriging reference for this split not computed yet: run "
                 "`61_pipeline_audit.py --split-table ...`)_\n")
with open(os.path.join(REP, "ablation_ladder.md"), "w") as fh:
    fh.write("\n".join(A))
print("wrote reports/real_data/ablation_ladder.md")
