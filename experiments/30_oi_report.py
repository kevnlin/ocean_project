"""Phase-1 / Task 1.5 — turn the OI caches into the two M1 reports + figures.

Reads
    outputs/cache/oi_tuning_val.json     (required)
    outputs/cache/oi_tuning_train.json   (optional stability check)
    outputs/cache/oi_vs_unet_seed<seed>.json + _maps.npz
Writes
    reports/oi_tuning.md
    reports/oi_baseline.md
    reports/fig_oi_rmse_bars.png
    reports/fig_oi_vs_unet_error_map.png

Run:  python experiments/30_oi_report.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ocean_tokenizer import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--suffix", default="",
                help="read <cache>_<suffix>.json and write reports/<name>_<suffix>.md; "
                     "use '_smoke' to exercise this script without overwriting the "
                     "real reports")
args = ap.parse_args()
SFX = args.suffix

VARS = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
BANDS = ["0-100m", "100-300m", "300-max"]
LABEL = {"clim_floor": "Climatology floor (train-only)",
         "woa_prior": "WOA23 prior",
         "nearest": "Nearest-profile fill",
         "oi": "**Optimal interpolation**",
         "unet_depthwise_profiles_only": "Depthwise U-Net (profiles_only)",
         "unet_depthwise_pws": "Depthwise U-Net (profiles_woa_surf, certified)"}
ORDER = ["woa_prior", "clim_floor", "nearest", "oi",
         "unet_depthwise_profiles_only", "unet_depthwise_pws"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


tun = load(os.path.join(C.CACHE, f"oi_tuning_val{SFX}.json"))
tun_tr = load(os.path.join(C.CACHE, f"oi_tuning_train{SFX}.json"))
cmp_path = os.path.join(C.CACHE, f"oi_vs_unet_seed{args.seed}{SFX}.json")
cmp = load(cmp_path)
if tun is None and cmp is None:
    raise SystemExit("no OI caches found — run experiments/run_oi_queue.sh first")

# =========================================================================
# reports/oi_tuning.md
# =========================================================================
if tun is not None:
    L = sorted({r["L_km"] for r in tun["stage_a"]["TEMP"]})
    G = sorted({r["gamma"] for r in tun["stage_a"]["TEMP"]})
    out = ["# OI hyperparameter tuning (protocol_v1)", "",
           f"*Selection split: **{tun['selection_split']}** months "
           f"({len(tun['months'])} snapshots: {', '.join(m[:7] for m in tun['months'])}). "
           f"Seed {tun['seed']}, {tun['n_profiles']} profiles/month. "
           f"Commit `{tun['git_commit'][:8]}`. {tun['cpu_hours']:.2f} CPU-hours.*", "",
           "Hyperparameter selection **is** model selection, so under protocol_v1 it "
           "uses the validation months (2008-2010). The intern plan said \"training "
           "months only\"; protocol_v1 is the stricter rule and the frozen protocol "
           "wins. The 12 pinned test months are never touched here.", "",
           "Metric: unobserved-only anomaly RMSE, squared errors pooled over every "
           "scored cell of every month before taking the root (identical pooling to "
           "`metrics.evaluate_masked` on a stacked array).", ""]

    for v in VARS:
        grid_a = {(r["L_km"], r["gamma"]): r["rmse"] for r in tun["stage_a"][v]}
        best = min(grid_a, key=grid_a.get)
        out += [f"## Stage A — {v} ({UNITS[v]}), k = {tun['k_stage_a']}", "",
                "| L_km \\ gamma | " + " | ".join(f"{g}" for g in G) + " |",
                "|---" * (len(G) + 1) + "|"]
        for l in L:
            cells = []
            for g in G:
                x = grid_a.get((l, g))
                cells.append("—" if x is None else
                             (f"**{x:.4f}**" if (l, g) == best else f"{x:.4f}"))
            out.append(f"| {l:.0f} | " + " | ".join(cells) + " |")
        out += ["", f"Optimum: **L = {best[0]:.0f} km, gamma = {best[1]}** "
                    f"→ {grid_a[best]:.4f} {UNITS[v]}.", ""]

    for v in VARS:
        rows = sorted(tun["stage_b"][v], key=lambda r: r["k"])
        if not rows:
            continue
        bk = min(rows, key=lambda r: r["rmse"])
        out += [f"## Stage B — {v}: neighbour count k at the stage-A optimum", "",
                "| k | RMSE |", "|---|---|"]
        for r in rows:
            mark = " **(chosen)**" if r["k"] == bk["k"] else ""
            out.append(f"| {r['k']} | {r['rmse']:.4f}{mark} |")
        out.append("")

    b = tun["best"]
    out += ["## Frozen hyperparameters", "",
            "| variable | L_km | gamma | k | selection RMSE |", "|---|---|---|---|---|"]
    for v in VARS:
        out.append(f"| {v} | {b[v]['L_km']:.0f} | {b[v]['gamma']} | {b[v]['k']} | "
                   f"{b[v]['rmse']:.4f} {UNITS[v]} |")
    out += ["", "These are the values `27_oi_vs_unet.py` reads from "
                "`outputs/cache/oi_tuning_val.json`. **Frozen** — changing them "
                "requires re-running this sweep and re-freezing.", ""]

    if tun_tr is not None:
        out += ["## Stability check — the same sweep on training months", "",
                "If the optimum moved between splits, the choice would be fitting "
                "noise rather than the covariance structure of the ocean.", "",
                "| variable | val optimum | train optimum | agree? |",
                "|---|---|---|---|"]
        for v in VARS:
            bv, bt = tun["best"][v], tun_tr["best"][v]
            same = (bv["L_km"], bv["gamma"]) == (bt["L_km"], bt["gamma"])
            out.append(f"| {v} | L={bv['L_km']:.0f}, gamma={bv['gamma']} | "
                       f"L={bt['L_km']:.0f}, gamma={bt['gamma']} | "
                       f"{'yes' if same else '**NO — investigate**'} |")
        out.append("")

    out += ["---", "",
            f"Rerun: `python experiments/26_oi_tuning.py --split "
            f"{tun['selection_split']}`", ""]
    open(os.path.join(C.REPORTS, f"oi_tuning{SFX}.md"), "w").write("\n".join(out))
    print(f"wrote reports/oi_tuning{SFX}.md")

# =========================================================================
# reports/oi_baseline.md  + figures
# =========================================================================
if cmp is not None:
    g = cmp["results"]["global"]
    na = cmp["results"]["north_atlantic"]
    present = [m for m in ORDER if m in g]
    hp = cmp["oi_hyperparams"]

    def pct(a, b):
        """improvement of b over a, in %"""
        return 100.0 * (a - b) / a

    n_test = len(cmp["test_months"])
    smoke_warn = ("\n> **SMOKE RUN — not a result.** Only "
                  f"{n_test} of the 12 pinned test months were scored.\n"
                  if cmp.get("smoke") else "")
    out = ["# Milestone M1 — Optimal Interpolation vs the learned reconstructors", "",
           f"*protocol_v1 · {n_test} pinned test month{'s' if n_test != 1 else ''} · "
           f"{cmp['n_profiles']} profiles/month · "
           f"seed {cmp['seed']} · commit `{cmp['git_commit'][:8]}` · "
           f"{cmp['cpu_hours']:.2f} CPU-hours.*", smoke_warn, "",
           "**The M1 question**: does the learned method match or beat optimal "
           "interpolation, the operational standard behind EN4, the Roemmich-Gilson "
           "Argo climatology and ISAS? If not, the method is not yet useful.", "",
           "## Setup", "",
           f"* OI hyperparameters (frozen on validation months, see "
           f"[oi_tuning.md](oi_tuning.md)): "
           + "; ".join(f"{v} L={hp[v]['L_km']:.0f} km, gamma={hp[v]['gamma']}, "
                       f"k={hp[v]['k']}" for v in VARS),
           f"* **Every method sees byte-identical observations.** "
           f"`27_oi_vs_unet.py` replays the certified run's RNG "
           f"({cmp['rng_draws_skipped']} discarded profile draws) so the profile "
           f"positions are exactly those `{cmp['certified_ckpt']}` was scored on.",
           ]
    rep = cmp["certified_reproduction"]
    worst = max(abs(rep[v]["delta"]) for v in VARS)
    ok = worst < 1e-4
    if cmp.get("smoke"):
        note = ("n/a in a smoke run — a subset of months cannot match a "
                "12-month cached number")
    else:
        note = "PASS" if ok else "**MISMATCH — investigate before trusting this table**"
    out += [f"* Verification: the certified checkpoint reproduces its cached test "
            f"RMSE to {worst:.1e} ({note}).", ""]

    out += ["## 1. Headline — unobserved-only anomaly RMSE", "",
            "| method | TEMP (degC) | SALT (PSU) | skill vs floor (TEMP) |",
            "|---|---|---|---|"]
    floor = g["clim_floor"]["full"]
    for m in present:
        f = g[m]["full"]
        out.append(f"| {LABEL[m]} | {f['TEMP']:.4f} | {f['SALT']:.4f} | "
                   f"{1 - f['TEMP']/floor['TEMP']:+.3f} |")
    out.append("")

    if "oi" in g and "unet_depthwise_pws" in g:
        oi_t, un_t = g["oi"]["full"]["TEMP"], g["unet_depthwise_pws"]["full"]["TEMP"]
        oi_s, un_s = g["oi"]["full"]["SALT"], g["unet_depthwise_pws"]["full"]["SALT"]
        verdict = ("**beats**" if un_t < oi_t else "**does NOT beat**")
        out += [f"### Verdict", "",
                f"The full system (depthwise U-Net, profiles+WOA+SST/SSS) {verdict} "
                f"optimal interpolation: **{pct(oi_t, un_t):+.1f} % on TEMP** "
                f"({oi_t:.4f} → {un_t:.4f} degC) and "
                f"**{pct(oi_s, un_s):+.1f} % on SALT** "
                f"({oi_s:.4f} → {un_s:.4f} PSU).", ""]
        if "unet_depthwise_profiles_only" in g:
            po = g["unet_depthwise_profiles_only"]["full"]
            out += [f"Like-for-like (both see profiles only): U-Net "
                    f"{po['TEMP']:.4f} vs OI {oi_t:.4f} degC "
                    f"({pct(oi_t, po['TEMP']):+.1f} %). This isolates *whose "
                    f"interpolator is better* from *whose inputs are richer*.", ""]
        else:
            out += ["> The `profiles_only` U-Net row (the like-for-like "
                    "information comparison) still needs a free GPU: "
                    "`CUDA_VISIBLE_DEVICES=N python experiments/27_oi_vs_unet.py "
                    "--train-profiles-only`. Until it lands, the comparison above "
                    "confounds *better interpolator* with *more inputs*.", ""]

    # ---- context rows from other protocol_v1 runs (different RNG position) ----
    ctx = load(os.path.join(C.CACHE, "baselines_protocol_v1.json"))
    if ctx and "oi" in g:
        oi_t = g["oi"]["full"]["TEMP"]
        mlp = ctx["summary"].get("mlp")
        near_ref = ctx["summary"].get("nearest")
        out += ["### Context: where OI sits among the existing protocol_v1 rows", "",
                "> **Caveat.** These rows come from "
                "`baselines_protocol_v1.json`, which drew its test-month profiles "
                "from a *different position* in the RNG stream (it skips 276 train "
                "draws; this script skips 276 + 36 to land on the certified U-Net's "
                "state). They are therefore the same protocol and the same "
                "distribution but **not the same profiles** — indicative, not "
                "identical-sample. The table in §1 is the identical-sample one. "
                f"For scale, the nearest-profile row differs by "
                f"{abs(g['nearest']['full']['TEMP'] - near_ref['TEMP_mean']):.4f} degC "
                f"between the two draws.", "",
                "| method | TEMP (degC) | SALT (PSU) |", "|---|---|---|"]
        if mlp:
            out.append(f"| Pointwise MLP (3 seeds) | {mlp['TEMP_mean']:.4f} ± "
                       f"{mlp['TEMP_std']:.4f} | {mlp['SALT_mean']:.4f} ± "
                       f"{mlp['SALT_std']:.4f} |")
        out.append(f"| **Optimal interpolation** (this run) | **{oi_t:.4f}** | "
                   f"**{g['oi']['full']['SALT']:.4f}** |")
        out.append("| Shared-latent fusion variants (3 seeds) | ~0.52 | ~0.12 |")
        out.append("")
        if mlp and oi_t < mlp["TEMP_mean"]:
            out += [f"**A properly tuned classical baseline beats the learned "
                    f"pointwise model**: OI {oi_t:.4f} vs MLP "
                    f"{mlp['TEMP_mean']:.4f} degC "
                    f"({pct(mlp['TEMP_mean'], oi_t):+.1f} %). Only the "
                    f"convolutional U-Nets clear it. This is worth carrying into "
                    f"the paper line: an OI row belongs in any table that claims a "
                    f"learned method is useful, and the shared-latent variants "
                    f"(~0.52, near the floor) are currently far below it.", ""]

    out += ["## 2. Depth bands (global)", "",
            "| method | " + " | ".join(f"TEMP {b}" for b in BANDS) + " |",
            "|---" * (len(BANDS) + 1) + "|"]
    for m in present:
        bb = g[m]["by_band"]["TEMP"]
        out.append(f"| {LABEL[m]} | " +
                   " | ".join(f"{bb[b]:.4f}" for b in BANDS) + " |")
    out.append("")
    if "oi" in g and "unet_depthwise_pws" in g:
        gains = {b: pct(g["oi"]["by_band"]["TEMP"][b],
                        g["unet_depthwise_pws"]["by_band"]["TEMP"][b]) for b in BANDS}
        top = max(gains, key=gains.get)
        out += ["Gain of the U-Net over OI, by band: "
                + " · ".join(f"**{b}** {gains[b]:+.1f} %" for b in BANDS) + ".", ""]
        if top == "0-100m":
            out += ["**The gain is largest at the surface, not in the thermocline "
                    "— and that is the expected result once stated carefully.** The "
                    "plan predicted the 100-300 m band on the reasoning that \"OI "
                    "cannot use SST/SSS\". The premise is right but the conclusion "
                    "does not follow: SST and SSS are *dense observations of the "
                    "0-100 m layer itself*, so the modality OI lacks constrains the "
                    "surface directly and the thermocline only indirectly, through "
                    "learned covariance. The ordering above is what that mechanism "
                    "predicts.", "",
                    "Note also that OI is the *only* method here whose 100-300 m "
                    "error exceeds its 0-100 m error "
                    f"({g['oi']['by_band']['TEMP']['100-300m']:.4f} > "
                    f"{g['oi']['by_band']['TEMP']['0-100m']:.4f}): with profiles "
                    "alone, the thermocline is genuinely the hardest layer. The "
                    "U-Net inverts that ordering by using the surface fields.", "",
                    "**Testable prediction this makes**: the `profiles_only` U-Net "
                    "(pending a GPU) should show a *much* smaller 0-100 m advantage "
                    "over OI than the full system does, because it loses exactly the "
                    "modality that produces this band's gain. If it does not, this "
                    "explanation is wrong and the advantage is coming from the "
                    "convolutional prior instead.", ""]
        else:
            out += [f"Largest gain: **{top}**. The plan predicted 100-300 m on the "
                    f"reasoning that OI cannot use SST/SSS — check whether the "
                    f"ordering here supports that mechanism.", ""]

    out += ["## 3. Regional check — North Atlantic box "
            f"({cmp['na_box']['lat'][0]}-{cmp['na_box']['lat'][1]} N, "
            f"{cmp['na_box']['lon'][0]}-{cmp['na_box']['lon'][1]} E, "
            f"{cmp['na_box']['cells']} cells)", "",
            "Gulf Stream + subtropical gyre — a high-eddy-energy region where "
            "interpolation is hardest.", "",
            "| method | TEMP (degC) | SALT (PSU) | TEMP vs global |",
            "|---|---|---|---|"]
    for m in present:
        f, fg = na[m]["full"], g[m]["full"]
        out.append(f"| {LABEL[m]} | {f['TEMP']:.4f} | {f['SALT']:.4f} | "
                   f"{f['TEMP']/fg['TEMP']:.2f}x |")
    out.append("")
    if "oi" in na and "unet_depthwise_pws" in na:
        rg = pct(na["oi"]["full"]["TEMP"], na["unet_depthwise_pws"]["full"]["TEMP"])
        gg = pct(g["oi"]["full"]["TEMP"], g["unet_depthwise_pws"]["full"]["TEMP"])
        agree = "consistent with" if (rg > 0) == (gg > 0) else "**opposite to**"
        out += [f"Regional margin over OI: {rg:+.1f} % vs {gg:+.1f} % globally — "
                f"{agree} the global conclusion.", ""]

    out += ["## 4. Figures", "",
            "![RMSE by method](fig_oi_rmse_bars.png)", "",
            "![OI vs U-Net error maps](fig_oi_vs_unet_error_map.png)", "",
            "## 5. References", "",
            "* Bretherton, Davis & Fandry (1976), *A technique for objective "
            "analysis and design of oceanographic experiments*, Deep-Sea Res. — "
            "the OI weight equation implemented in `oi.py`.",
            "* Roemmich & Gilson (2009), *The 2004-2008 mean and annual cycle of "
            "T/S from Argo*, Prog. Oceanogr. — the operational Argo OI climatology.",
            "* Good, Martin & Rayner (2013), *EN4*, JGR Oceans.",
            "* Gaillard et al. (2016), *ISAS*, J. Climate.", "",
            "---", "",
            f"Rerun: `python experiments/27_oi_vs_unet.py --verify-unet` then "
            f"`python experiments/30_oi_report.py`", ""]
    open(os.path.join(C.REPORTS, f"oi_baseline{SFX}.md"), "w").write("\n".join(out))
    print(f"wrote reports/oi_baseline{SFX}.md")

    # ---------------- figure 1: RMSE bars ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    show = [m for m in present if m != "woa_prior"]     # woa dwarfs the scale
    colors = {"clim_floor": "#888888", "nearest": "#c46210", "oi": "#d62728",
              "unet_depthwise_profiles_only": "#7aa6d6",
              "unet_depthwise_pws": "#2a78d6"}
    for ax, v in zip(axes, VARS):
        vals = [g[m]["full"][v] for m in show]
        ax.bar(range(len(show)), vals,
               color=[colors.get(m, "#444") for m in show])
        ax.axhline(g["clim_floor"]["full"][v], ls="--", c="k", lw=1,
                   label="climatology floor")
        for i, x in enumerate(vals):
            ax.text(i, x, f"{x:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(show)))
        ax.set_xticklabels([LABEL[m].replace("**", "").replace(" (", "\n(")
                            for m in show], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(f"unobserved-only RMSE ({UNITS[v]})")
        ax.set_title(f"{v} — global, 12 pinned test months")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, f"fig_oi_rmse_bars{SFX}.png"), dpi=140)
    plt.close(fig)
    print(f"wrote reports/fig_oi_rmse_bars{SFX}.png")

    # ---------------- figure 2: spatial error maps ----------------
    mp = os.path.join(C.CACHE, f"oi_vs_unet_seed{args.seed}{SFX}_maps.npz")
    if os.path.exists(mp):
        z = np.load(mp)
        pair = [m for m in ("oi", "unet_depthwise_pws") if f"{m}_TEMP" in z]
        if len(pair) == 2:
            a, b = z[f"{pair[0]}_TEMP"], z[f"{pair[1]}_TEMP"]
            vmax = float(np.nanpercentile(np.concatenate(
                [a[np.isfinite(a)], b[np.isfinite(b)]]), 99))
            fig, axes = plt.subplots(3, 1, figsize=(9.5, 11))
            for ax, arr, ttl in zip(
                    axes[:2], (a, b),
                    (f"OI — TEMP RMSE (L={hp['TEMP']['L_km']:.0f} km, "
                     f"gamma={hp['TEMP']['gamma']}, k={hp['TEMP']['k']})",
                     "Depthwise U-Net (profiles_woa_surf) — TEMP RMSE")):
                im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                               extent=[0, 360, -90, 90], aspect="auto")
                ax.set_title(ttl, fontsize=10)
                ax.set_ylabel("lat")
                fig.colorbar(im, ax=ax, label="degC", fraction=0.03)
            d = a - b
            lim = float(np.nanpercentile(np.abs(d[np.isfinite(d)]), 99))
            im = axes[2].imshow(d, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                                extent=[0, 360, -90, 90], aspect="auto")
            axes[2].set_title("OI minus U-Net (red = U-Net better)", fontsize=10)
            axes[2].set_xlabel("lon"); axes[2].set_ylabel("lat")
            fig.colorbar(im, ax=axes[2], label="degC", fraction=0.03)
            for ax in axes:      # North Atlantic box
                ax.plot([275, 335, 335, 275, 275], [20, 20, 55, 55, 20],
                        c="w", lw=1.2, ls="--")
            fig.tight_layout()
            fig.savefig(os.path.join(C.REPORTS, f"fig_oi_vs_unet_error_map{SFX}.png"),
                        dpi=140)
            plt.close(fig)
            print(f"wrote reports/fig_oi_vs_unet_error_map{SFX}.png")

print("done.")
