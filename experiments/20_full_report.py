"""Task 8c (Week 4) — aggregate the full-scale runs into the Week-4 report.

Collects outputs/cache/full_<variant>_s<seed>.json (trainer results),
outputs/cache/full_eval_full_<variant>_s<seed>.json (probes + flexibility),
and the certified week-3 baselines (audit_depthwise_e40 / audit_joint_e400_cos),
then writes:

    reports/full_training_report.md
    reports/fig_full_val_curves.png
    reports/fig_full_rmse_depth.png
    reports/fig_full_count_sweep.png

Robust to partial availability: aggregates whatever exists (so it can be run
mid-queue), and states what is missing.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_tokenizer import config as C

VARIANTS = ("mbca", "perceiver", "resampler")
VNAMES = {"mbca": "MBCA (method)", "perceiver": "Standard Perceiver",
          "resampler": "Fixed-budget resampler"}
SEEDS = (1234, 1235, 1236)
VARS = ("TEMP", "SALT")
REPORTS = C.REPORTS


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


PREFIX = os.environ.get("REPORT_PREFIX", "fullA")   # final-config runs
runs = {}                 # (variant, seed) -> trainer json (status done only)
for v in VARIANTS:
    for s in SEEDS:
        d = load(os.path.join(C.CACHE, f"{PREFIX}_{v}_s{s}.json"))
        if d and d.get("status") == "done":
            runs[(v, s)] = d
evals = {}
for v in VARIANTS:
    for s in SEEDS:
        d = load(os.path.join(C.CACHE, f"full_eval_{PREFIX}_{v}_s{s}.json"))
        if d:
            evals[(v, s)] = d
# small-config seed-1234 trio (the first full-scale attempt) — ablation rows
small_runs = {v: load(os.path.join(C.CACHE, f"full_{v}_s1234.json"))
              for v in VARIANTS}
small_runs = {v: d for v, d in small_runs.items()
              if d and d.get("status") == "done"}
aud_dw = load(os.path.join(C.CACHE, "audit_depthwise_e40.json"))
aud_j = load(os.path.join(C.CACHE, "audit_joint_e400_cos.json"))

any_run = next(iter(runs.values()), None)
if any_run is None:
    sys.exit("no completed full_* runs yet — nothing to report")
FLOOR = any_run["test_floor"]

lines = []
A = lines.append
A("# Week-4 Report — First Full-Scale Training of the Shared-Latent Variants")
A("")
A("*Task 8 of the Week-4 plan (reports/full_training_plan.md): three fusion "
  "variants, identical everything except the fusion rule, trained under "
  "protocol_v1 with profile-count augmentation U{0..3000}; validation-selected "
  "checkpoints; pinned test months scored once per run.  Metric: "
  "**unobserved-only anomaly RMSE** (degC / PSU), pooled over the 12 pinned "
  "test months.*")
A("")
missing = [f"full_{v}_s{s}" for v in VARIANTS for s in SEEDS
           if (v, s) not in runs]
if missing:
    A(f"> **Partial report** — runs not yet complete: {', '.join(missing)}.")
    A("")

# ---------------- training-setup note ----------------
A("## Setup deviations from the week-3 plan (declared)")
A("")
A("protocol_v1 (split, metric, inputs, eval) is unchanged.  The week-3 POC "
  "froze the architecture on an overfit gate only; at full scale that config "
  "could not learn at all, and the following changes were required — every "
  "one applied identically to all three variants, so the comparison remains "
  "fusion-rule-isolating:")
A("")
A("1. **Fourier coordinate features** (`coord_features` fourier_v2, shared by "
  "all encoders and the query decoder): with the original 7 smooth features "
  "the training loss stayed pinned at 1.0 even when queries were sampled AT "
  "observed profile columns (the copy diagnostic, "
  "`18_full_train.py --probe-observed`) — attention had no spatially "
  "selective basis to read tokens with.")
A("2. **Geographically anchored latents** (`--anchor-grid 18,30` -> 540 "
  "latent tokens, one per 10x12-deg cell, initialised with the shared "
  "coordinate featurisation of the cell centre): best validation peak of "
  "every configuration searched; the unanchored 128-latent original is kept "
  "as an ablation row below.")
A("3. **Training-side augmentation/regularisation**: 2k-step LR warmup; 25% "
  "observed-column query bootstrap (MAE-style; evaluation remains strictly "
  "unobserved-only — protocol_v1 prohibits scoring, not training, on "
  "observed columns); grid-token dropout 0.3; input noise 0.05 sigma; "
  "AdamW weight decay 0.01.")
A("")
A("**Training-dynamics finding (the honest headline of this run).** Every "
  "configuration searched — 1M and 4.7M params, 128/512/540-anchored "
  "latents, grid-drop up to 0.5, no-bootstrap, lr 1e-4, input noise up to "
  "0.5 sigma — peaks at a validation score of ~0.93-0.95 within ~5k steps "
  "and then degrades while the training loss keeps falling.  With 276 "
  "training fields, exact-column observations uniquely fingerprint each "
  "month, so reconstruct-the-field training collapses into month-identity "
  "recall for a globally attending token model; masking/noise levels that "
  "would destroy the fingerprint also destroy the signal.  Checkpoints are "
  "therefore validation-selected near the peak (protocol-clean), and closing "
  "the generalisation gap to the convolutional baselines — e.g. more "
  "training members/years, field-space decoding, or local-attention "
  "decoders — is the top week-5 priority.  Search-run curves: "
  "`outputs/cache/{scale_,reg_,anchor_}*.json` (their test numbers are "
  "quarantined diagnostics, never used for selection or reported).")
A("")

# ---------------- headline table ----------------
A("## 1. Headline: test unobserved-only anomaly RMSE (mean ± std over seeds)")
A("")
A("| model | TEMP (degC) | SALT (PSU) | TEMP 0-100m | TEMP 100-300m | TEMP 300-max |")
A("|---|---|---|---|---|---|")
A(f"| Climatology floor (train-only) | {FLOOR['TEMP']:.4f} | {FLOOR['SALT']:.4f} "
  f"| — | — | — |")
if aud_dw:
    t = aud_dw["test"]
    A(f"| Depthwise U-Net (certified, seed 1234) | {t['TEMP']:.4f} | "
      f"{t['SALT']:.4f} | {t['by_band']['TEMP']['0-100m']:.4f} | "
      f"{t['by_band']['TEMP']['100-300m']:.4f} | "
      f"{t['by_band']['TEMP']['300-max']:.4f} |")
if aud_j:
    t = aud_j["test"]
    A(f"| Joint-depth U-Net (certified, seed 1234) | {t['TEMP']:.4f} | "
      f"{t['SALT']:.4f} | {t['by_band']['TEMP']['0-100m']:.4f} | "
      f"{t['by_band']['TEMP']['100-300m']:.4f} | "
      f"{t['by_band']['TEMP']['300-max']:.4f} |")


def agg(variant, field):
    vals = [runs[(variant, s)]["test"][field] for s in SEEDS
            if (variant, s) in runs]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)


def agg_band(variant, var, band):
    vals = [runs[(variant, s)]["test"]["by_band"][var][band] for s in SEEDS
            if (variant, s) in runs]
    return np.mean(vals) if vals else np.nan


for v in VARIANTS:
    mT, sT, n = agg(v, "TEMP")
    mS, sS, _ = agg(v, "SALT")
    if n == 0:
        continue
    A(f"| {VNAMES[v]} ({n} seed{'s' if n > 1 else ''}) | "
      f"{mT:.4f} ± {sT:.4f} | {mS:.4f} ± {sS:.4f} | "
      f"{agg_band(v, 'TEMP', '0-100m'):.4f} | "
      f"{agg_band(v, 'TEMP', '100-300m'):.4f} | "
      f"{agg_band(v, 'TEMP', '300-max'):.4f} |")
for v in VARIANTS:
    if v in small_runs:
        t = small_runs[v]["test"]
        A(f"| {VNAMES[v]} — unanchored 128-latent ablation (seed 1234) | "
          f"{t['TEMP']:.4f} | {t['SALT']:.4f} | "
          f"{t['by_band']['TEMP']['0-100m']:.4f} | "
          f"{t['by_band']['TEMP']['100-300m']:.4f} | "
          f"{t['by_band']['TEMP']['300-max']:.4f} |")
A("")
A("Skill vs floor = 1 − RMSE/floor.  The certified U-Net numbers are the "
  "week-3 audit checkpoints (seed 1234, fixed 1500 profiles, no count "
  "augmentation); the shared-latent rows carry count augmentation and are "
  "additionally capable of the section-3 sweeps with the same checkpoint.")
A("")

# ---------------- sensitivity probes ----------------
if evals:
    A("## 2. Sensitivity at the selected checkpoint (Task-6 protocol, real data)")
    A("")
    A("Relative output change under information-preserving token manipulations "
      "(validation months; lower = more invariant):")
    A("")
    A("| probe | " + " | ".join(VNAMES[v] for v in VARIANTS) + " |")
    A("|---|" + "---|" * len(VARIANTS))
    for probe in ("duplicate_half", "patch_refine_2x", "profile_resample_2x"):
        row = [f"| {probe} "]
        for v in VARIANTS:
            vals = [evals[(v, s)]["probes"][probe]["rel_output_change_mean"]
                    for s in SEEDS if (v, s) in evals]
            row.append(f"| {np.mean(vals):.5f} " if vals else "| — ")
        A("".join(row) + "|")
    A("")
    A("| probe | variant | TEMP RMSE base -> probed |")
    A("|---|---|---|")
    for probe in ("duplicate_half", "patch_refine_2x", "profile_resample_2x"):
        for v in VARIANTS:
            vals = [(evals[(v, s)]["probes"][probe]["base_rmse_TEMP"],
                     evals[(v, s)]["probes"][probe]["probe_rmse_TEMP"])
                    for s in SEEDS if (v, s) in evals]
            if vals:
                b = np.mean([x[0] for x in vals]); p = np.mean([x[1] for x in vals])
                A(f"| {probe} | {VNAMES[v]} | {b:.4f} -> {p:.4f} |")
    A("")

    # ---------------- flexibility ----------------
    A("## 3. Flexibility with ONE checkpoint (no retraining)")
    A("")
    A("### 3a. Profile-count sweep (test months; week-2 density axis)")
    A("")
    dens = None
    for v in VARIANTS:
        e = next((evals[(v, s)] for s in SEEDS if (v, s) in evals), None)
        if e:
            dens = [r["density"] for r in e["count_sweep"]]
            break
    if dens:
        A("| density | floor TEMP | " +
          " | ".join(f"{VNAMES[v]} TEMP" for v in VARIANTS) + " | " +
          " | ".join(f"{VNAMES[v]} SALT" for v in VARIANTS) + " |")
        A("|---|---|" + "---|" * (2 * len(VARIANTS)))
        for i, d in enumerate(dens):
            row = [f"| {d} "]
            fl = [evals[(v, s)]["count_sweep"][i]["floor_TEMP"]
                  for v in VARIANTS for s in SEEDS if (v, s) in evals]
            row.append(f"| {np.mean(fl):.4f} ")
            for var in ("TEMP", "SALT"):
                for v in VARIANTS:
                    vals = [evals[(v, s)]["count_sweep"][i][var]
                            for s in SEEDS if (v, s) in evals]
                    row.append(f"| {np.mean(vals):.4f} " if vals else "| — ")
            A("".join(row) + "|")
        A("")
        A("The week-2 ablation retrained the depthwise U-Net *per density* "
          "(reports/week2_density_ablation.md); every shared-latent number "
          "above comes from a single checkpoint per seed.")
        A("")
    A("### 3b. Missing-modality matrix (test months, headline masks fixed)")
    A("")
    A("| inputs | " + " | ".join(f"{VNAMES[v]} TEMP" for v in VARIANTS) +
      " | " + " | ".join(f"{VNAMES[v]} SALT" for v in VARIANTS) + " |")
    A("|---|" + "---|" * (2 * len(VARIANTS)))
    for row_name in ("full", "drop_surf", "drop_woa", "drop_profiles"):
        row = [f"| {row_name} "]
        for var in ("TEMP", "SALT"):
            for v in VARIANTS:
                vals = [evals[(v, s)]["modality_matrix"][row_name][var]
                        for s in SEEDS if (v, s) in evals]
                row.append(f"| {np.mean(vals):.4f} " if vals else "| — ")
        A("".join(row) + "|")
    A("")
    A("No baseline in the repo can produce 3a/3b without retraining one model "
      "per row.")
    A("")

# ---------------- gate ----------------
A("## 4. Submission gate #2 (from the week-3 plan)")
A("")
mbca_T, _, n_m = agg("mbca", "TEMP")
perc_T, _, n_p = agg("perceiver", "TEMP")
if n_m and n_p:
    verdict = "PASS" if mbca_T <= perc_T * 1.005 else "AT RISK"
    A(f"F-C (MBCA) vs F-A (Perceiver) accuracy: {mbca_T:.4f} vs {perc_T:.4f} "
      f"degC -> **{verdict}** on the accuracy axis; see section 2 for the "
      f"invariance axis (the insurance policy).")
else:
    A("Gate not yet evaluable — waiting for both MBCA and Perceiver runs.")
A("")
A("## 5. Provenance")
A("")
for (v, s), d in sorted(runs.items()):
    A(f"- `{d['tag']}`: best step {d['best']['step']} "
      f"(val score {d['best']['score']:.4f}), {d['gpu_hours']:.2f} GPU-h, "
      f"commit `{d['git_commit'][:8]}`, obs_query_frac "
      f"{d.get('obs_query_frac')}, coord {d.get('coord_features')}")
A("")

with open(os.path.join(REPORTS, "full_training_report.md"), "w") as f:
    f.write("\n".join(lines))
print(f"wrote {os.path.join(REPORTS, 'full_training_report.md')}")

# ---------------- figures ----------------
# val curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for v in VARIANTS:
    for s in SEEDS:
        if (v, s) not in runs:
            continue
        c = runs[(v, s)]["curves"]
        lab = VNAMES[v] if s == SEEDS[0] else None
        col = {"mbca": "C0", "perceiver": "C1", "resampler": "C2"}[v]
        axes[0].plot(c["step"], c["val_TEMP"], col, alpha=0.7, label=lab)
        axes[1].plot(c["step"], c["val_SALT"], col, alpha=0.7, label=lab)
axes[0].axhline(any_run["val_floor"]["TEMP"], ls="--", c="k", lw=0.8,
                label="clim floor")
axes[1].axhline(any_run["val_floor"]["SALT"], ls="--", c="k", lw=0.8)
if aud_dw:
    axes[0].axhline(aud_dw["best"]["val_TEMP"], ls=":", c="0.4", lw=0.8,
                    label="depthwise U-Net (val)")
    axes[1].axhline(aud_dw["best"]["val_SALT"], ls=":", c="0.4", lw=0.8)
axes[0].set_ylabel("val unobs anomaly RMSE TEMP (degC)")
axes[1].set_ylabel("val unobs anomaly RMSE SALT (PSU)")
for ax in axes:
    ax.set_xlabel("step"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "fig_full_val_curves.png"), dpi=140)
print("wrote fig_full_val_curves.png")

# rmse by depth
import ocean_tokenizer.data as data_mod
depth = None
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for v in VARIANTS:
    prof = [runs[(v, s)]["test"]["by_depth"] for s in SEEDS if (v, s) in runs]
    if not prof:
        continue
    col = {"mbca": "C0", "perceiver": "C1", "resampler": "C2"}[v]
    D = len(prof[0]["TEMP"])
    depth = depth if depth is not None else np.arange(D)
    for k, var in enumerate(VARS):
        m = np.mean([p[var] for p in prof], 0)
        axes[k].plot(m, np.arange(D), col, label=VNAMES[v])
for k, var in enumerate(VARS):
    axes[k].invert_yaxis()
    axes[k].set_xlabel(f"{var} RMSE"); axes[k].set_ylabel("depth level idx")
    axes[k].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(REPORTS, "fig_full_rmse_depth.png"), dpi=140)
print("wrote fig_full_rmse_depth.png")

# count sweep
if evals:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v in VARIANTS:
        rows = [evals[(v, s)]["count_sweep"] for s in SEEDS if (v, s) in evals]
        if not rows:
            continue
        col = {"mbca": "C0", "perceiver": "C1", "resampler": "C2"}[v]
        dens = [r["density"] for r in rows[0]]
        for k, var in enumerate(VARS):
            m = np.mean([[r[var] for r in rr] for rr in rows], 0)
            axes[k].plot(dens, m, col + "-o", ms=3, label=VNAMES[v])
    fl = [r["floor_TEMP"] for r in next(iter(evals.values()))["count_sweep"]]
    axes[0].plot(dens, fl, "k--", lw=0.8, label="clim floor")
    for k, var in enumerate(VARS):
        axes[k].set_xlabel("profiles / month (one checkpoint)")
        axes[k].set_ylabel(f"test {var} RMSE")
        axes[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "fig_full_count_sweep.png"), dpi=140)
    print("wrote fig_full_count_sweep.png")
