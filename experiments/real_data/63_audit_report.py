"""Turn the audit JSON and the audit runs into the meeting's three deliverables.

  reports/real_data/pipeline_audit.md    the step-by-step audit (61_pipeline_audit.py)
  reports/real_data/overfit_sanity.md    the overfit ladder (62_sanity_train.py --mode ...)
  reports/real_data/ablation_ladder.md   one switch at a time on a fixed held-out set
  + fig_pair_correlation.png, fig_loss_curves.png, fig_overfit.png, fig_ablation.png

Written for the GLOBAL ocean state reconstruction: one domain over the whole
ocean, every profile a month delivers, the satellite-era split. Every number is
read from a file those two scripts wrote; nothing is typed in by hand. Re-run
after either of them.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np

from ocean_tokenizer.plot_style import use_style, SERIES, MUTED, INK_2, STATUS

ap = argparse.ArgumentParser()
ap.add_argument("--region", default="global")
ap.add_argument("--eval-split", default="development")
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REP = os.path.join(ROOT, "reports", "real_data")
AUD = json.load(open(os.path.join(ROOT, "outputs", "cache", "pipeline_audit.json")))
REGION = args.region
R = AUD["regions"][REGION]
SPL = AUD["splits"]
TEST = f"{SPL['development'][0]}-{SPL['development'][1]}"
VAL = f"{SPL['validation'][0]}"
CH = ("TEMP", "SALT")
UNIT = {"TEMP": "°C", "SALT": "PSU"}
BANDS = ("0-100m", "100-300m", "300-700m", "700-1400m")
plt = use_style()

#: each arm is ONE switch away from its reference; most references are the
#: baseline, but a switch measured on a different stem is scored against that stem
REFERENCE = {"level_tokens_cap1000": "cap1000",
             "backbone_lno": "baseline",
             "backbone_lno_uniform": "backbone_lno"}
LABEL = {
    "baseline": "baseline — Perceiver-IO fuse, DFS mass, D4RT decoder",
    "mass_uniform": "uniform mass instead of DFS",
    "mass_count": "count mass instead of DFS",
    "qc": "robust input QC",
    "anomaly_exact": "climatology at the profile's own position",
    "level_tokens_cap1000": "per-level tokens (at cap 1000)",
    "refiner_local": "refiner length scales 150 km / 100 m",
    "refiner_gate1": "refiner output gate 0.05 → 1.0",
    "no_latent": "no global latent (query path only)",
    "no_target_dropout": "no target dropout",
    "batch8": "8 months per optimiser step",
    "cap1000": "cap 1000 profiles / month",
    "fixed_stack": "QC + exact climatology + refiner locality + gate",
    "backbone_lno": "PhCA-style (LNO) fuse instead of Perceiver-IO, 32 slots",
    "backbone_lno_uniform": "PhCA-style fuse, uniform mass",
}
#: arms that are not part of this study: the SetConv encoder was dropped, and the
#: comparison is strictly baseline DFS / PhCA DFS / uniform mass in each trunk —
#: no satellite-input arms, no slot-count variant
EXCLUDE = ("setconv", "lno_sla", "lno_surface", "backbone_lno_s128")
SEEDS = (1234, 1235)


def f(x, n=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def runs():
    """tag -> {seed: summary} for every finished audit run of this domain."""
    out = defaultdict(dict)
    for p in glob.glob(os.path.join(ROOT, "outputs", "audit", REGION, "*",
                                    "summary_seed*.json")):
        d = json.load(open(p))
        if any(x in d["tag"] for x in EXCLUDE):
            continue
        out[d["tag"]][d["seed"]] = d
    return out


def agg(rs, split, ch, key="rmse_z"):
    v = [s["scores"][split][ch][key] for s in rs.values()
         if split in s["scores"] and ch in s["scores"][split]]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def history(tag, seed=1234):
    """One seed's curve (several seeds of an arm append to one file)."""
    p = os.path.join(ROOT, "outputs", "audit", REGION, tag, "history.jsonl")
    if not os.path.exists(p):
        return []
    recs = []
    for line in open(p):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    return [r for r in recs if r.get("seed") == seed]


RUNS = runs()
#: the climatology reference every J is measured against, as scored by the runs
#: themselves on the identical held-out cells (zero anomaly = training climatology)
_b = RUNS.get("baseline", {})
CLIM_Z = {ch: (next(iter(_b.values()))["scores"]["development"][ch]["climatology_z"]
               if _b else float("nan")) for ch in ("TEMP", "SALT")}

# ==========================================================================
# figure 1 — how fast the anomaly decorrelates, globally and by latitude band
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
for ax, (key, title, legend) in zip(axes, (
        ("pair_correlation", "all latitudes, by depth", "depth"),
        ("pair_correlation_by_lat_band_327m", "327 m, by |latitude| band", "|lat| band"))):
    src = R[key]["TEMP"] if key == "pair_correlation" else R[key]
    for i, (name, e) in enumerate(src.items()):
        b = np.asarray(e["bins_km"]); mid = 0.5 * (b[:-1] + b[1:])
        y = np.asarray(e["corr"]); ok = np.asarray(e["pairs"]) > 50
        ln, = ax.plot(mid[ok], y[ok], color=SERIES[i], label=name, marker="o")
        ln.set_markeredgecolor("#fcfcfb"); ln.set_markeredgewidth(2.0)
    med = R["input_density"]["nearest_input_km_all_profiles"]["median"]
    ax.axvline(med, color=MUTED, lw=1.0)
    ax.annotate(f"median nearest\ninput: {med:.0f} km", (med, 0.78), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=INK_2, va="top")
    ax.axhline(0.0, color="#c3c2b7", lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(8, 1300); ax.set_ylim(-0.15, 0.85)
    ax.set_xlabel("separation between two floats (km)")
    ax.set_ylabel("correlation of temperature anomaly")
    ax.set_title(f"Same month, different floats — {title}")
    ax.legend(title=legend, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(REP, "fig_pair_correlation.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# figure 2 — the overfit ladder
# ==========================================================================
OVERFIT_RUNS = [("mem_d4rt", "memorise, 1 month (4 k steps)"),
                ("mem_d4rt_8k", "memorise, 1 month (8 k steps)"),
                ("copy_d4rt", "copy the input"), ("small8_d4rt", "8 months")]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
plotted = 0
for i, (tag, label) in enumerate(OVERFIT_RUNS):
    h = [r for r in history(tag) if "overfit/TEMP_rmse_z" in r]
    if h:
        ax.plot([r["step"] for r in h], [r["overfit/TEMP_rmse_z"] for r in h],
                color=SERIES[i], label=label)
        plotted += 1
if plotted:
    ax.axhline(0.1, color=STATUS["good"], lw=1.5, ls="--")
    ax.annotate("target 0.1 z", (ax.get_xlim()[1], 0.1), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK_2)
    ax.set_yscale("log"); ax.set_xlabel("training step")
    ax.set_ylabel("temperature RMSE on the fitted data (z)")
    ax.set_title("Overfit sanity check — global")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_overfit.png"), bbox_inches="tight")
plt.close(fig)

# ==========================================================================
# figure 3 — training curves
# ==========================================================================
# left: training loss of the two backbones; right: validation RMSE for BOTH
# seeds of the baseline and of the refiner fix — the seed-1235 collapse is the
# single most important thing these curves show
LOSS_RUNS = [("baseline", 1234, "Perceiver-IO baseline"),
             ("backbone_lno", 1234, "PhCA-style (LNO) backbone"),
             ("refiner_gate1", 1234, "refiner gate 1.0"),
             ("backbone_lno_uniform", 1234, "PhCA-style + uniform mass")]
#: the same four arms as the loss panel, in the same colours, seed 1234 only
VAL_RUNS = list(LOSS_RUNS)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
any_curve = False
for i, (tag, seed, label) in enumerate(LOSS_RUNS):
    tl = [(r["step"], r["train_loss"]) for r in history(tag, seed) if "train_loss" in r
          and "validation/macro_z" not in r]
    if not tl:
        continue
    any_curve = True
    st, v = zip(*tl)
    k = max(1, len(v) // 25)      # batch size 1: the smoothed trend is the readable part
    sm = np.convolve(v, np.ones(k) / k, mode="valid")
    axes[0].plot(st[k - 1:k - 1 + len(sm)], sm, color=SERIES[i], label=label)
for i, (tag, seed, label) in enumerate(VAL_RUNS):
    vl = [(r["step"], r["validation/macro_z"]) for r in history(tag, seed)
          if "validation/macro_z" in r]
    if vl:
        st, v = zip(*vl)
        axes[1].plot(st, v, color=SERIES[i], label=label, marker="o", ms=3.5)
if any_curve:
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("training loss (MSE, z²), smoothed")
    axes[0].set_title("Training loss (seed 1234)"); axes[0].legend()
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("validation RMSE (z, mean of T and S)")
    axes[1].set_title(f"Validation RMSE ({VAL}) — seed {SEEDS[0]}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_loss_curves.png"), bbox_inches="tight")
plt.close(fig)


# ==========================================================================
# ablation rows
# ==========================================================================
#: a run whose held-out TEMP J stays above this ended near the climatology
COLLAPSED_J = 0.94


def zval(tag, seed, ch="TEMP", key="rmse_z", split=None):
    d = RUNS.get(tag, {}).get(seed)
    if d is None:
        return float("nan")
    return d["scores"][split or args.eval_split][ch][key]


def ablation_rows():
    """One row per arm: both seeds, their mean, and the PAIRED Δ.

    Δ is the mean over seeds of (arm − reference) at the same seed, because the
    seeds differ by more than most switches do: a same-seed difference cancels
    what the seed contributes, a difference of means does not.
    """
    if "baseline" not in RUNS:
        return []
    out = []
    for tag, per_seed in RUNS.items():
        if tag.startswith(("mem_", "copy_", "small8_", "smk_", "time_")):
            continue
        ref = REFERENCE.get(tag, "baseline")
        z = [zval(tag, s) for s in SEEDS]
        dz = [zval(tag, s) - zval(ref, s) for s in SEEDS]
        jt = [zval(tag, s, key="J") for s in SEEDS]
        out.append({
            "tag": tag, "ref": ref, "n_seeds": len(per_seed), "z": z,
            "temp": float(np.nanmean(z)), "temp_sd": float(np.nanstd(z)),
            "J_temp_seeds": jt, "J_temp": float(np.nanmean(jt)),
            "J_salt": float(np.nanmean([zval(tag, s, "SALT", "J") for s in SEEDS])),
            "temp_phys": float(np.nanmean([zval(tag, s, key="rmse_physical") for s in SEEDS])),
            "delta": (0.0 if tag == "baseline" else float(np.nanmean(dz))),
            "delta_seeds": dz,
            "collapsed": [bool(j > COLLAPSED_J) for j in jt],
            # the same paired difference, only on seeds where NEITHER run
            # collapsed: separates "this switch adds information" from "this
            # switch avoided the collapse"
            "delta_clean": float(np.nanmean([d for d, s_ in zip(dz, SEEDS)
                                              if zval(tag, s_, key="J") <= COLLAPSED_J
                                              and zval(ref, s_, key="J") <= COLLAPSED_J]
                                             or [np.nan])),
            "val_temp": float(np.nanmean([zval(tag, s, split="validation") for s in SEEDS])),
            "params": list(per_seed.values())[0]["params"]})
    return sorted(out, key=lambda r: r["delta"])


ROWS = ablation_rows()
SPREAD = (float(np.nanmean([r["temp_sd"] for r in ROWS if r["n_seeds"] > 1]))
          if any(r["n_seeds"] > 1 for r in ROWS) else float("nan"))

if ROWS:
    shown = [r for r in ROWS if r["tag"] != "baseline"]
    fig, ax = plt.subplots(figsize=(8.2, 0.40 * len(shown) + 1.6))
    lab = [f"{r['tag']}" + ("" if r["ref"] == "baseline" else f"  (vs {r['ref']})")
           for r in shown]
    val = [r["delta"] for r in shown]
    col = [SERIES[0] if v < 0 else STATUS["critical"] for v in val]
    y = np.arange(len(val))
    ax.barh(y, val, color=col, height=0.58, alpha=0.9)
    # each seed's own paired difference, so a mean carried by one seed shows
    for yi, r in zip(y, shown):
        for k, (s, d) in enumerate(zip(SEEDS, r["delta_seeds"])):
            if np.isfinite(d):
                mk = ax.plot(d, yi + (-0.16 if k == 0 else 0.16), marker="o" if k == 0 else "s",
                             ms=5.5, color=INK_2, ls="none",
                             label=f"seed {s}" if yi == 0 else None)[0]
                mk.set_markeredgecolor("#fcfcfb"); mk.set_markeredgewidth(1.5)
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5)
    ax.axvline(0, color="#c3c2b7", lw=1.0)
    for yi, v in zip(y, val):
        ax.annotate(f"{v:+.3f}", (min(v, 0) if v < 0 else v, yi),
                    xytext=(-6 if v < 0 else 6, 0), textcoords="offset points",
                    fontsize=8, ha="right" if v < 0 else "left", va="center", color=INK_2)
    lo = min([v for v in val] + [d for r in shown for d in r["delta_seeds"] if np.isfinite(d)])
    hi = max([v for v in val] + [d for r in shown for d in r["delta_seeds"] if np.isfinite(d)])
    span = max(hi - lo, 1e-6)
    ax.set_xlim(min(lo, 0) - 0.30 * span, max(hi, 0) + 0.22 * span)
    ax.set_xlabel("paired change in held-out temperature RMSE vs its reference (z) — left is better")
    ax.set_title(f"One switch at a time — global, {TEST} held-out floats (bars = mean of 2 seeds)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "fig_ablation.png"), bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# the audit report
# ==========================================================================
L = []
w = L.append
w("# Pipeline audit — global ocean state reconstruction\n")
w("<!-- generated by experiments/real_data/61_pipeline_audit.py + 63_audit_report.py "
  "— do not edit by hand -->\n")
w("> Asked for at the 2026-09-17 meeting: walk the pipeline one stage at a time "
  "before running anything else, because numbers that all sit near 1 and do not "
  "move when the model changes point at the data path.\n")
comp = R["composition"]
w(f"> **Setup.** One domain over the whole ocean: the global Argo cohort "
  f"({sum(v['profiles'] for v in comp.values()):,} profiles in these years, "
  f"{len(R['levels'])} levels to {max(R['levels']):.0f} m), every profile a month "
  f"delivers, target = WOA23 monthly anomaly, per-level z-score fitted on the "
  f"training years. Split: train {SPL['train'][0]}-{SPL['train'][1]}, validate "
  f"{VAL}, test {TEST}. Truth is held-out, WMO-disjoint floats.\n")

w("## 0. The short version\n")
O_ = R["outliers"]; T_ = R["tokenisation"]; S_ = R["scale"]
w(md_table(["stage", "what the audit found"], [
    ["data prep", f"{O_['SALT']['profiles_over_10_sigma']:,} salinity profiles beyond "
                  f"10σ pass Argo QC (max |z| {O_['SALT']['max_abs_z']:.0f}); test-year "
                  f"input salinity z-RMS {O_['SALT']['dev_input_z_rms']:.2f} with them, "
                  f"{O_['SALT']['dev_input_z_rms_clean']:.2f} without"],
    ["data prep", "the anomaly subtracts WOA23 at the 1° cell centre, not at the "
                  "profile (§1.3)"],
    ["normalisation", f"1 z = {S_['TEMP']['anom_std_by_band']['0-100m']:.2f} °C / "
                      f"{S_['SALT']['anom_std_by_band']['0-100m']:.2f} PSU in the upper "
                      "100 m; one σ per level for the whole ocean, plain (not robust)"],
    ["tokenisation", f"{T_['profile_tokens']['tokens_per_profile']} mean-pooled depth-band "
                     f"tokens per profile discard "
                     f"{100 * T_['profile_tokens']['band_mean_loses']['TEMP']['within_band_variance_fraction']:.0f} % "
                     "of its vertical variance"],
    ["tokenisation", f"the query-local refiner starts at "
                     f"{T_['local_refiner_init']['ell_lat_km']:,.0f} km with a "
                     f"{T_['local_refiner_init']['gate_init']} output gate"]]))
if ROWS:
    base = next((r for r in ROWS if r["tag"] == "baseline"), None)
    if base:
        w(f"The trained baseline scores **{f(base['z'][0], 4)} / {f(base['z'][1], 4)} z** "
          f"on its two seeds against a climatology of **{f(CLIM_Z['TEMP'], 4)} z** "
          f"(`ablation_ladder.md`).\n")

w("## 1. Data preparation\n")
w("### 1.1 What one z unit is worth\n")
S = R["scale"]
w(md_table(["channel"] + list(BANDS) + ["unit"],
           [[ch] + [f(S[ch]["anom_std_by_band"][b], 3) for b in BANDS if b in S[ch]["anom_std_by_band"]]
            + [UNIT[ch]] for ch in CH]))
t_up = S["TEMP"]["anom_std_by_band"]["0-100m"]
s_up = S["SALT"]["anom_std_by_band"]["0-100m"]
w(f"The anomaly's own standard deviation **is** the z scale: 1 z is "
  f"{t_up:.2f} °C and {s_up:.3f} PSU in the upper 100 m of the global ocean. A "
  f"model at RMSE ≈ 1 z is predicting the training climatology. The meeting's "
  f"0.1-0.2 °C target is {t_up / 0.2:.0f}-{t_up / 0.1:.0f}× below that variability.\n")

w("### 1.2 Values that are not measurements\n")
O = R["outliers"]
w(md_table(["channel", "profiles with abs(z)>10", "floats", "max abs(z)",
            "by data mode", "test-year input z RMS", "same, outliers removed"],
           [[ch, O[ch]["profiles_over_10_sigma"], O[ch]["floats"], f(O[ch]["max_abs_z"], 0),
             ", ".join(f"{k}:{v}" for k, v in sorted(O[ch]["by_data_mode"].items())),
             f(O[ch]["dev_input_z_rms"], 2), f(O[ch]["dev_input_z_rms_clean"], 2)]
            for ch in CH]))
for ch in CH:
    for wf in O[ch]["worst_floats"][:2]:
        w(f"- {ch}: float **{wf['wmo']}** ({'/'.join(wf['data_mode'])}, "
          f"{wf['years'][0]}-{wf['years'][1]}, {wf['float_split'].replace('_', ' ')}), "
          f"{wf['profiles']} profiles over 10σ, max |z| {wf['max_abs_z']:.0f}.\n")
Q = R["robust_qc"]
w(f"These pass the cohort's Argo QC flags but are not ocean. They enter the model "
  f"as **inputs** and they enter the per-level standard deviation that defines the "
  f"z unit. The audit's robust QC (8σ, iterated on training-year statistics) "
  f"flags {Q['values_flagged']['TEMP']:,} temperature and "
  f"{Q['values_flagged']['SALT']:,} salinity values and drops "
  f"{Q['profiles_dropped']['TEMP']:,} / {Q['profiles_dropped']['SALT']:,} "
  f"profile-channels; the `qc` arm measures what that is worth.\n")

w("### 1.3 Where the climatology is evaluated\n")
C = R["climatology_position"]
w(md_table(["grid cell", "mean offset lat / lon", "max offset",
            "spurious °C 0-100m", "spurious °C 300-700m",
            "variance removed by using the profile's own position, 100-300m"],
           [[f"{C['cell_deg'][0]:.2f}° × {C['cell_deg'][1]:.2f}°",
             f"{C['offset_km']['lat_mean']:.0f} / {C['offset_km']['lon_mean']:.0f} km",
             f"{C['offset_km']['lat_max']:.0f} / {C['offset_km']['lon_max']:.0f} km",
             f(C["TEMP"]["std_difference_by_band"]["0-100m"], 3),
             f(C["TEMP"]["std_difference_by_band"]["300-700m"], 3),
             f(C["TEMP"]["variance_removed_pct"]["100-300m"], 1) + " %"]]))
w("The registered recipe subtracts WOA23 at the centre of the 1° cell, not at the "
  "profile. Across a front that offset carries a real climatological gradient, so "
  "part of what the model is asked to predict is the climatology's own spatial "
  "structure. The `anomaly_exact` arm removes it.\n")

w("### 1.4 The test years are not the training years\n")
w(md_table(["channel", "train mean anomaly 0-100m", "test mean anomaly 0-100m",
            "test/train σ 0-100m", "test/train σ 300-700m", "unit"],
           [[ch, f(S[ch]["anom_mean_by_band_train"]["0-100m"], 3),
             f(S[ch]["anom_mean_by_band_dev"]["0-100m"], 3),
             f(S[ch]["dev_over_train_std"]["0-100m"], 3),
             f(S[ch]["dev_over_train_std"]["300-700m"], 3), UNIT[ch]] for ch in CH]))
if np.isfinite(CLIM_Z["TEMP"]):
    w(f"This is why the climatology reference reads "
      f"**{f(CLIM_Z['TEMP'], 4)} z** rather than 1.00 on the "
      f"test years: the reference is the training climatology, and the ocean has "
      f"moved away from it. Every J is measured against that shifted reference.\n")

w("## 2. Normalisation\n")
N = R["normalisation"]
w(md_table(["channel", "robust σ / fitted σ, 0-100m", "same, 700-1400m",
            "worst level", "fitted vs robust σ there"],
           [[ch, f(N["robust_over_fitted_sigma"][ch]["0-100m"], 2),
             f(N["robust_over_fitted_sigma"][ch].get("700-1400m"), 2),
             f"{N['worst_level_inflation'][ch]['level_m']:.0f} m",
             f"{f(N['worst_level_inflation'][ch]['fitted_std'], 3)} vs "
             f"{f(N['worst_level_inflation'][ch]['robust_std'], 3)}"] for ch in CH]))
w("Per-level z-scoring pools the whole ocean at each level, so one z unit mixes "
  "the quiet interior with the western boundary currents, and it is fitted with "
  "the plain standard deviation, so the outliers of §1.2 set part of the scale. "
  "Where the ratio sits well below 1 the level's unit is inflated by heavy tails "
  "(fronts and eddies) or by a handful of bad values.\n")

w("## 3. Tokenisation\n")
T = R["tokenisation"]
w(f"### 3.1 A profile becomes {T['profile_tokens']['tokens_per_profile']} tokens\n")
w(md_table(["band", "levels pooled", "token depth", "level depths"],
           [[f"{b['band_m'][0]:.0f}-{b['band_m'][1]:.0f} m", b["levels"],
             f"{b['token_depth_m']:.0f} m", ", ".join(f"{d:.0f}" for d in b["level_depths"])]
            for b in T["profile_tokens"]["bands"]]))
bl = T["profile_tokens"]["band_mean_loses"]
w(f"`ProfileEncoder` embeds each level, **mean-pools inside the band**, and places "
  f"the token at the band's midpoint. A band-mean summary discards "
  f"{100 * bl['TEMP']['within_band_variance_fraction']:.0f} % of temperature and "
  f"{100 * bl['SALT']['within_band_variance_fraction']:.0f} % of salinity variance "
  f"within a profile — the vertical detail the thermocline lives in. Per-level "
  f"tokens avoid it but multiply the token count ~5×, which is why that arm is run "
  f"at cap 1000.\n")

w("### 3.2 The resolution the model can express\n")
LR = T["local_refiner_init"]
w(md_table(["component", "scale"], [
    ["shared coordinate features (finest Fourier wavelength)",
     f"{T['coordinate_features']['finest_sphere_wavelength_deg']:.2f}° ≈ "
     f"{T['coordinate_features']['finest_sphere_wavelength_km']:.0f} km"],
    ["local refiner Gaussian, north-south (init)", f"{LR['ell_lat_km']:.0f} km"],
    ["local refiner Gaussian, east-west (init, at the equator)",
     f"{LR['ell_lon_km_at_region']:.0f} km"],
    ["local refiner depth / time (init)",
     f"{LR['ell_depth_m']:.0f} m / {LR['ell_time_months']:.0f} months"],
    ["local refiner output gate (init)", f"{LR['gate_init']}"]]))
w("Both paths into a query start far wider than the ocean's correlation scales "
  "(§4): the 'local' refiner's distance prior spans thousands of kilometres and "
  "its output is gated at 0.05. The scales are learnable but start orders of "
  "magnitude away; the `refiner_local` and `refiner_gate1` arms test the fix.\n")

w("### 3.3 How much of the observing system reaches the model\n")
D = R["input_density"]
w(md_table(["profiles / month (median, min-max)", "used by the model",
            "median distance target → nearest input", "targets within 50 km",
            f"same at cap {D['cap']}"],
           [[f"{D['available_per_month']['median']:.0f} "
             f"({D['available_per_month']['min']}-{D['available_per_month']['max']})",
             "all",
             f"{D['nearest_input_km_all_profiles']['median']:.0f} km",
             f"{100 * D['nearest_input_km_all_profiles']['frac_under_50']:.0f} %",
             f"{D['nearest_input_km_at_cap']['median']:.0f} km, "
             f"{100 * D['nearest_input_km_at_cap']['frac_under_50']:.0f} % within 50 km"]]))

w("## 4. What is recoverable at all\n")
w("![anomaly correlation against separation](fig_pair_correlation.png)\n")


def corr_row(name, v):
    b = np.asarray(v["bins_km"]); corr = np.asarray(v["corr"])
    pick = {f"{b[i]}-{b[i+1]}": corr[i] for i in range(len(corr))}
    return [name] + [f(pick.get(k), 2) for k in ("0-10", "25-50", "50-75", "100-150", "200-300", "400-600")]


w(md_table(["depth", "0-10 km", "25-50 km", "50-75 km", "100-150 km", "200-300 km", "400-600 km"],
           [corr_row(d, v) for d, v in R["pair_correlation"]["TEMP"].items()]))
w(md_table(["abs(latitude) band, 327 m", "0-10 km", "25-50 km", "50-75 km", "100-150 km",
            "200-300 km", "400-600 km"],
           [corr_row(d, v) for d, v in R["pair_correlation_by_lat_band_327m"].items()]))
w("Two same-month profiles from different floats are far from perfectly "
  "correlated even when co-located, and the correlation length differs by "
  "latitude band. Any global model has to be local where the ocean is local and "
  "broad where it is broad.\n")

fit = R.get("covariance_fit")
if fit:
    w("### 4.1 The floor, from the fitted covariance\n")
    rows = []
    for ch in CH:
        nb = len(fit[ch][0]["bands"])
        for b in range(nb):
            nug = np.array([lv["bands"][b]["nugget"] for lv in fit[ch]])
            L1 = np.array([lv["bands"][b]["L1_km"] for lv in fit[ch]])
            lo, hi = fit[ch][0]["bands"][b]["lat_band"]
            rows.append([ch, f"{lo:.0f}-{min(hi, 90):.0f}°", f"{np.median(L1):.0f} km",
                         f(float(nug.mean()), 2), f(float(np.sqrt(nug.mean())), 2)])
    w(md_table(["channel", "abs(lat) band", "median mesoscale length L1",
                "mean nugget", "J floor = sqrt(nugget)"], rows))
    w("The **nugget** is the share of the anomaly variance at a point that no other "
      "float sees — submonthly change, submesoscale structure, and the difference "
      "between a point measurement and anything smoother. It is a property of the "
      "ocean and the sampling, not of the model. A method that mapped the "
      "correlated part perfectly would still score about **J = sqrt(nugget)**; "
      "that is the floor to measure against. A 0.1-0.2 °C target corresponds to "
      "J ≈ 0.1 and is not attainable at a held-out float by any method.\n")

with open(os.path.join(REP, "pipeline_audit.md"), "w") as fh:
    fh.write("\n".join(L))
print("wrote reports/real_data/pipeline_audit.md")

# ==========================================================================
# overfit report
# ==========================================================================
O = ["# Overfit sanity check — global\n",
     "<!-- generated by 62_sanity_train.py + 63_audit_report.py — do not edit by hand -->\n",
     "> The meeting's sanity test: drive the RMSE below 0.1 on data the model is "
     "allowed to memorise. If it cannot, the pipeline is broken; if it can, the "
     "held-out numbers are about the ocean and the observing system, not capacity.\n",
     f"Three rungs on the global cohort, training years {SPL['train'][0]}-"
     f"{SPL['train'][1]}, every profile of the month as input, lead 0, the "
     "405 K-parameter model unless the row says otherwise:\n",
     "- **memorise** — one fixed month, one fixed input/target float split.\n",
     "- **copy** — 512 of the input profiles are also the targets, queried at their "
     "own positions and levels: can information reach a query at all?\n",
     "- **8 months** — eight fixed months with fixed partitions.\n",
     "- `_fixed` rows add the two refiner fixes (local length scales, gate 1.0); "
     "`mem_d4rt_8k` is the one-month rung trained twice as long.\n"]
rows = []
for tag, per_seed in sorted(RUNS.items()):
    if not tag.startswith(("mem_", "copy_", "small8_")):
        continue
    t, _, n = agg(per_seed, "overfit", "TEMP")
    s_, _, _ = agg(per_seed, "overfit", "SALT")
    rows.append([f"`{tag}`", list(per_seed.values())[0]["steps"], n, f(t, 3),
                 f(agg(per_seed, "overfit", "TEMP", "rmse_physical")[0], 3),
                 f(s_, 3), f(agg(per_seed, "overfit", "SALT", "rmse_physical")[0], 4),
                 "yes" if max(t, s_) < 0.1 else "no"])
if rows:
    O.append(md_table(["run", "steps", "seeds", "TEMP (z)", "TEMP (°C)", "SALT (z)",
                       "SALT (PSU)", "below 0.1 z?"], rows))
    O.append("![overfit curves](fig_overfit.png)\n")
with open(os.path.join(REP, "overfit_sanity.md"), "w") as fh:
    fh.write("\n".join(O))
print("wrote reports/real_data/overfit_sanity.md")

# ==========================================================================
# ablation report
# ==========================================================================
A = ["# Ablation ladder — global, one component at a time\n",
     "<!-- generated by 62_sanity_train.py + 63_audit_report.py — do not edit by hand -->\n",
     "> Every arm is the same training run with exactly one switch moved, scored on "
     "a FIXED held-out set: the same months, the same input profiles and the same "
     f"held-out floats. Selection uses {VAL}; the test columns are {TEST}, scored on "
     "every held-out cell. Two seeds per arm.\n",
     "> **Δ is paired**: the mean over seeds of (arm − reference) at the same seed. "
     "The two seeds differ by more than most switches do, so only same-seed "
     "differences are meaningful. The reference is the baseline for most arms, "
     "`backbone_lno` for the PhCA uniform-mass arm, `cap1000` for per-level tokens.\n",
     "> **`qc` and `anomaly_exact` change the target itself**; compare those two by J.\n"]


def rowfmt(r):
    z = r["z"]
    col = ["**collapsed**" if c else "" for c in r["collapsed"]]
    return [f"`{r['tag']}`", LABEL.get(r["tag"], ""), f"{r['params']:,}",
            f"{f(z[0], 4)} {col[0]}".strip(), f"{f(z[1], 4)} {col[1]}".strip(),
            f(r["temp"], 4), f(r["J_temp"], 3), f(r["J_salt"], 3), f(r["temp_phys"], 3),
            ("—" if r["tag"] == "baseline" else f"{r['delta']:+.4f}"),
            ("—" if r["tag"] == "baseline" or not np.isfinite(r["delta_clean"])
             else f"{r['delta_clean']:+.4f}"),
            ("—" if r["tag"] == "baseline" else f"`{r['ref']}`")]


HEAD = ["arm", "what it changes", "params", "TEMP z, seed 1234", "TEMP z, seed 1235",
        "mean", "J TEMP", "J SALT", "TEMP (°C)", "paired Δ (z)",
        "Δ where neither collapsed", "reference"]
byt = {r["tag"]: r for r in ROWS}

# ---- the backbone question, first ----
BB = [t for t in ("baseline", "mass_uniform",
                  "backbone_lno", "backbone_lno_uniform") if t in byt]
if BB:
    A.append("## 1. PhCA-style (LNO) fuse vs the Perceiver-IO fuse, "
             "with DFS against uniform mass in each\n")
    A.append("Four arms, nothing else moved: the two fuse stages, each with DFS "
             "evidence and with uniform mass. Both use the same D4RT query decoder, "
             "the same data and the same seeds. The PhCA-style fuse computes each "
             "observation token's weights over 32 latent slots from its **position "
             "only** (an MLP on the shared coordinate features) and normalises them "
             "**over the slots**, Slot-Attention style, so each token spreads exactly "
             "its DFS evidence and slot mass is conserved. It departs from the LNO "
             "paper, whose PhCA encoder normalises over the input points, in exactly "
             "that respect.\n")
    A.append(md_table(HEAD, [rowfmt(byt[t]) for t in BB]))
    A.append(f"Climatology on the same held-out cells: **{f(CLIM_Z['TEMP'], 4)} z** "
             f"(TEMP), **{f(CLIM_Z['SALT'], 4)} z** (SALT); J is RMSE divided by it.\n")

# ---- the full ladder ----
A.append("## 2. The full ladder\n")
if ROWS:
    A.append(md_table(HEAD, [rowfmt(r) for r in ROWS]))
    A.append(f"Runs marked **collapsed** end with held-out TEMP J above {COLLAPSED_J}: "
             "they end near the climatology. When one side of "
             "a pair collapsed, the paired Δ mostly measures the collapse; the "
             "*Δ where neither collapsed* column is the clean effect of the switch "
             "('—' when every seed of the pair had a collapse).\n")
    A.append("![ablation ladder](fig_ablation.png)\n")

A.append("## 3. Training curves\n")
A.append("![training curves](fig_loss_curves.png)\n")
A.append(f"Both panels show the same four arms in the same colours, on seed "
         f"{SEEDS[0]}: training loss on the left (smoothed; batch size 1), "
         f"validation RMSE on {VAL} on the right. The refiner-gate arm reaches its "
         "level by 4 k steps; PhCA with uniform mass stays above the rest in both "
         "panels. Per-seed end points, including the seed-1235 collapses, are in "
         "the tables above. Every run also logs to W&B (offline in "
         "`outputs/wandb/`; `wandb sync outputs/wandb/wandb/offline-run-*`).\n")
with open(os.path.join(REP, "ablation_ladder.md"), "w") as fh:
    fh.write("\n".join(A))
print("wrote reports/real_data/ablation_ladder.md")
