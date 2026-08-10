"""Reconstruction visualisation for the Monday deck — CURRENT U-Net results.

Not a new experiment: inference only, from checkpoints that already exist.
Nothing here is trained, and no number in the reports changes.

  Figure 1  ground truth | OI | U-Net        (shared diverging scale)
            d|error|     | OI |err| | U-Net |err|   (shared sequential scale)
  Figure 2  U-Net without SSH | with SSH | d|error|  (shared scales)

Both figures show the **100-300 m thermocline band** over the North Atlantic
box (20-55 N, 275-335 E) for one held-out test month, with the month's Argo
profile positions overlaid and the regional RMSE printed on every panel.

Which model is shown
--------------------
Figure 1's U-Net is `audit_depthwise_e40` -- the certified **depthwise 2-D
U-Net** of protocol_v1 (0.1580 degC global).  Figure 2's two arms are the
control / treatment of the SSH ablation.  **None of these is the shared-latent
DFS / MBCA / Perceiver model**, which sits near the climatology floor (~0.52)
and is not plotted here.  Every panel is labelled accordingly.

What is plotted
---------------
The **anomaly** (field minus the train-only monthly climatology), because that
is what protocol_v1 defines as the prediction target and what the RMSE measures.
Plotting absolute temperature would make all three panels look identical -- the
climatology dominates the field -- and would hide exactly the signal the models
are judged on.

The maps are the band **mean** over the 7 levels in 100-300 m; the quoted RMSE
is **pooled** over all levels and cells in the band (the same quantity as
`metrics.evaluate_layers`), so it is comparable to the report tables rather than
being the RMSE of the displayed mean.

Month selection
---------------
`--month auto` (default) ranks all 12 pinned test months by the U-Net's regional
advantage over OI and picks the **median** one -- deliberately representative
rather than the most flattering.  The full ranking is printed so the choice is
auditable, and `--month <i>` overrides it.

Run:
    python experiments/32_reconstruction_figure.py
"""
import argparse
import glob
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

from ocean_tokenizer import baselines as B, config as C, data
from ocean_tokenizer.anomaly import AnomNorm, Climatology
from ocean_tokenizer.oi import predict_oi
from ocean_tokenizer.ssh import SSHAnom, load_ssh_cache, ssh_for_indices
from ocean_tokenizer.unet import UNet2D

TRAIN_YEARS, VAL_YEARS = (1985, 2007), (2008, 2010)
TEST_TIME_INDICES = [1933, 1935, 1936, 1938, 1942, 1946, 1952, 1953,
                     1956, 1965, 1967, 1976]
CFG_PWS = ("profiles", "woa", "surf")
CFG_SSH = ("profiles", "woa", "surf", "ssh")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--var", default="TEMP", choices=("TEMP", "SALT"))
ap.add_argument("--band", default="100,300", help="depth band in metres")
ap.add_argument("--lat", default="20,55")
ap.add_argument("--lon", default="275,335")
ap.add_argument("--month", default="auto",
                help="'auto' = median U-Net advantage, or a 0-11 index")
ap.add_argument("--device", default="cpu",
                help="inference only, ~20 slices -- cpu is plenty and avoids "
                     "competing for a shared card")
ap.add_argument("--no-ssh", action="store_true", help="skip figure 2")
args = ap.parse_args()

V = args.var
UNIT = "degC" if V == "TEMP" else "PSU"
b_lo, b_hi = (float(x) for x in args.band.split(","))
lat_lo, lat_hi = (float(x) for x in args.lat.split(","))
lon_lo, lon_hi = (float(x) for x in args.lon.split(","))

grid = data.CommonGrid()
band = np.where((grid.depth > b_lo) & (grid.depth <= b_hi))[0]
ii = np.where((grid.lat >= lat_lo) & (grid.lat <= lat_hi))[0]
jj = np.where((grid.lon >= lon_lo) & (grid.lon <= lon_hi))[0]
EXTENT = [grid.lon[jj[0]] - .5, grid.lon[jj[-1]] + .5,
          grid.lat[ii[0]] - .5, grid.lat[ii[-1]] + .5]
print(f"band {b_lo:.0f}-{b_hi:.0f} m -> levels {band} "
      f"({grid.depth[band].min():.0f}-{grid.depth[band].max():.0f} m)", flush=True)
print(f"region lat {lat_lo}-{lat_hi} lon {lon_lo}-{lon_hi} "
      f"-> {int(grid.ocean[np.ix_(ii, jj)].sum())} ocean cells", flush=True)


def crop(a):
    """(...,H,W) -> the display window."""
    return a[..., ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]


# ---------------- fields, climatology, samples (RNG replay) ----------------
tr_idx = data.select_month_indices(C.GT_SOURCE, TRAIN_YEARS)
va_idx = data.select_month_indices(C.GT_SOURCE, VAL_YEARS)
te_idx = np.asarray(TEST_TIME_INDICES)
print(f"loading fields ({tr_idx.size} train) ...", flush=True)
ftrain = data.load_gt_fields(tr_idx, grid)
ftest = data.load_gt_fields(te_idx, grid)
woa = data.woa_prior(grid)
surf_train = {v: ftrain[v] for v in C.VARS_SURF if v in ftrain}
clim = Climatology(ftrain, surf_train)
norm = AnomNorm(clim, ftrain, surf_train)

rng = np.random.default_rng(args.seed)
n_ocean = int(grid.ocean.sum())
for _ in range(tr_idx.size + va_idx.size):          # land on the certified state
    rng.choice(n_ocean, size=min(C.N_PROFILES, n_ocean), replace=False)
te = [B.prepare_month(ftest, ftest, woa, grid, t, rng, C.N_PROFILES)
      for t in range(len(ftest["months"]))]

# pseudo-SSH, train-only statistics (only needed for figure 2)
ssh_ok = os.path.exists(os.path.join(C.CACHE, "ssh_dyn.npz")) and not args.no_ssh
if ssh_ok:
    ssh_all, ssh_idx = load_ssh_cache(os.path.join(C.CACHE, "ssh_dyn.npz"))
    sshnorm = SSHAnom(ssh_for_indices(ssh_all, ssh_idx, tr_idx), ftrain["months"])
    ssh_te = ssh_for_indices(ssh_all, ssh_idx, te_idx)
    for k, s in enumerate(te):
        s["ssh_z"] = sshnorm.z(ssh_te[k], s["month"])


# ---------------- models ----------------
def load_unet(tag, cfg):
    p = os.path.join(C.CKPT, f"{tag}.pt")
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    c_in = B._unet_channels(te[0], grid, norm, cfg).shape[1]
    m = UNet2D(c_in, len(B.VARS), base=C.UNET_BASE).to(args.device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def unet_predict(model, s, cfg):
    X = torch.from_numpy(B._unet_channels(s, grid, norm, cfg)).to(args.device)
    out = model(X).cpu().numpy()                      # (D, 2, H, W)
    k = B.VARS.index(V)
    arr = norm.unz3d(V, out[:, k], s["month"])
    return np.where(grid.ocean[None], arr, np.nan).astype("float32")


hp = json.load(open(os.path.join(C.CACHE, "oi_tuning_val.json")))["best"]
OI_HP = {v: {kk: hp[v][kk] for kk in ("L_km", "gamma", "k")} for v in B.VARS}

certified = load_unet("audit_depthwise_e40", CFG_PWS)
assert certified is not None, "certified checkpoint missing"


# ---------------- metrics ----------------
def band_rmse(pred, s):
    """Pooled RMSE over the band, region and unobserved cells (report-comparable)."""
    t = s["gt"][V][band][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
    p = pred[band][:, ii[0]:ii[-1] + 1, jj[0]:jj[-1] + 1]
    m = crop(s["unobs_mask"])[None] & np.isfinite(t) & np.isfinite(p)
    d = (p - t)[m]
    return float(np.sqrt(np.mean(d * d)))


def band_anom_map(field3d, month):
    """Band-mean ANOMALY over the display window; land/observed -> NaN."""
    a = field3d[band] - clim.clim3d(V, month)[band]
    return np.nanmean(crop(a), axis=0)


# ---------------- month selection ----------------
print("\nper-month regional RMSE in the band (unobserved cells only):", flush=True)
print(f"  {'#':>2} {'month':>10} {'OI':>8} {'U-Net':>8} {'gain':>7}", flush=True)
rows = []
for k, s in enumerate(te):
    p_oi = predict_oi(s, norm, grid,
                      L_km={v: OI_HP[v]["L_km"] for v in B.VARS},
                      gamma={v: OI_HP[v]["gamma"] for v in B.VARS},
                      k={v: OI_HP[v]["k"] for v in B.VARS})[V]
    p_un = unet_predict(certified, s, CFG_PWS)
    r_oi, r_un = band_rmse(p_oi, s), band_rmse(p_un, s)
    rows.append({"i": k, "time": str(ftest["time"][k])[:7], "oi": r_oi,
                 "unet": r_un, "gain": 100 * (r_oi - r_un) / r_oi,
                 "p_oi": p_oi, "p_un": p_un})
    print(f"  {k:>2} {rows[-1]['time']:>10} {r_oi:8.4f} {r_un:8.4f} "
          f"{rows[-1]['gain']:6.1f}%", flush=True)

if args.month == "auto":
    order = sorted(rows, key=lambda r: r["gain"])
    pick = order[len(order) // 2]                     # median, not maximum
    print(f"\n-> median-gain month selected: #{pick['i']} {pick['time']} "
          f"({pick['gain']:.1f}%).  Range across months: "
          f"{order[0]['gain']:.1f}% .. {order[-1]['gain']:.1f}%", flush=True)
else:
    pick = rows[int(args.month)]
    print(f"\n-> month #{pick['i']} {pick['time']} selected explicitly", flush=True)

s = te[pick["i"]]
mo_label = pick["time"]

# ---------------- panels ----------------
gt_map = band_anom_map(s["gt"][V], s["month"])
oi_map = band_anom_map(pick["p_oi"], s["month"])
un_map = band_anom_map(pick["p_un"], s["month"])
obs_hole = ~crop(s["unobs_mask"]) & crop(grid.ocean)   # observed columns
for m in (gt_map, oi_map, un_map):
    m[obs_hole] = np.nan                               # excluded from scoring


def abs_err(pred3d):
    e = np.abs(pred3d[band] - s["gt"][V][band])
    e = np.nanmean(crop(e), axis=0)
    e[obs_hole] = np.nan
    return e


oi_err, un_err = abs_err(pick["p_oi"]), abs_err(pick["p_un"])

# profile positions inside the window
plat, plon = s["prof"]["lat"], s["prof"]["lon"]
inwin = ((plat >= EXTENT[2]) & (plat <= EXTENT[3])
         & (plon >= EXTENT[0]) & (plon <= EXTENT[1]))
plat, plon = plat[inwin], plon[inwin]
print(f"   {plat.size} Argo profiles in the window", flush=True)

LAND = np.where(crop(grid.ocean), np.nan, 1.0)
VMAX = float(np.nanpercentile(np.abs(gt_map), 99))
EMAX = float(np.nanpercentile(np.concatenate(
    [oi_err[np.isfinite(oi_err)], un_err[np.isfinite(un_err)]]), 99))


def panel(ax, arr, title, cmap, vmin, vmax, rmse=None, rlabel="RMSE"):
    ax.imshow(LAND, origin="lower", extent=EXTENT, cmap="Greys", vmin=0, vmax=1.6,
              interpolation="nearest", aspect="auto")
    im = ax.imshow(arr, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="nearest", aspect="auto")
    ax.scatter(plon, plat, s=7, facecolor="none", edgecolor="k", linewidths=.6,
               alpha=.85, zorder=5)
    ax.set_title(title, fontsize=9.5, pad=4)
    ax.set_xticks([280, 300, 320]); ax.set_yticks([25, 35, 45, 55])
    ax.tick_params(labelsize=7.5)
    if rmse is not None:
        ax.text(.025, .965, f"{rlabel} {rmse:.4f} {UNIT}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, fontweight="bold", color="w",
                path_effects=[pe.withStroke(linewidth=2.6, foreground="black")],
                zorder=6)
    return im


# =========================== FIGURE 1 ===========================
fig, ax = plt.subplots(2, 3, figsize=(15.2, 8.0))
fig.suptitle(
    f"Reconstruction of the {b_lo:.0f}–{b_hi:.0f} m thermocline, North Atlantic — "
    f"held-out test month {mo_label}\n"
    f"{V} anomaly (field − train-only climatology) · depthwise U-Net "
    f"`audit_depthwise_e40` — NOT the DFS/Perceiver shared-latent model",
    fontsize=12.5, y=.985)

im1 = panel(ax[0, 0], gt_map, "Ground truth (CESM2-LE)", "RdBu_r", -VMAX, VMAX)
panel(ax[0, 1], oi_map, "Optimal interpolation", "RdBu_r", -VMAX, VMAX,
      pick["oi"])
panel(ax[0, 2], un_map, "Depthwise U-Net (profiles+WOA+SST/SSS)", "RdBu_r",
      -VMAX, VMAX, pick["unet"])

diff = oi_err - un_err
DMAX = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99))
im3 = panel(ax[1, 0], diff, "|error| difference:  OI − U-Net\n(red = U-Net closer to truth)",
            "RdBu_r", -DMAX, DMAX)
im2 = panel(ax[1, 1], oi_err, "OI  |error|", "magma", 0, EMAX, pick["oi"])
panel(ax[1, 2], un_err, "U-Net  |error|", "magma", 0, EMAX, pick["unet"])

for a in ax[:, 0]:
    a.set_ylabel("latitude (°N)", fontsize=8.5)
for a in ax[1]:
    a.set_xlabel("longitude (°E)", fontsize=8.5)

fig.subplots_adjust(left=.045, right=.90, top=.865, bottom=.155,
                    hspace=.30, wspace=.16)
cb1 = fig.colorbar(im1, ax=ax[0], fraction=.020, pad=.012)
cb1.set_label(f"{V} anomaly ({UNIT})", fontsize=8.5); cb1.ax.tick_params(labelsize=7.5)
cb2 = fig.colorbar(im2, ax=ax[1], fraction=.020, pad=.012)
cb2.set_label(f"|error| ({UNIT})", fontsize=8.5); cb2.ax.tick_params(labelsize=7.5)

fig.text(.045, .085,
         "Circles = the month's Argo profile columns.  Those columns are excluded from "
         "scoring at every level (the white gaps), per protocol_v1.\n"
         "The top row shares one diverging colour scale; the two |error| panels share "
         "one sequential scale, so panels are directly comparable.\n"
         f"Maps are the band mean over {band.size} levels "
         f"({grid.depth[band].min():.0f}–{grid.depth[band].max():.0f} m); the quoted "
         "RMSE is pooled over all levels and unobserved cells in the box, matching the "
         "report tables.\n"
         f"Month chosen as the MEDIAN of the 12 held-out months by U-Net advantage "
         f"(range {min(r['gain'] for r in rows):.0f}–{max(r['gain'] for r in rows):.0f} %) "
         f"— representative, not the most favourable.",
         fontsize=8.0, color="#333", va="top", linespacing=1.55)
out1 = os.path.join(C.REPORTS, "fig_reconstruction_na_thermocline.png")
fig.savefig(out1, dpi=155)
plt.close(fig)
print(f"\nwrote {out1}", flush=True)

# =========================== FIGURE 2 (SSH) ===========================
if ssh_ok:
    ctrl = load_unet(f"ssh_control_pws_s{args.seed}", CFG_PWS)
    trt = load_unet(f"ssh_treat_pws_ssh_s{args.seed}", CFG_SSH)
    if ctrl is not None and trt is not None:
        p_c, p_t = unet_predict(ctrl, s, CFG_PWS), unet_predict(trt, s, CFG_SSH)
        r_c, r_t = band_rmse(p_c, s), band_rmse(p_t, s)
        e_c, e_t = abs_err(p_c), abs_err(p_t)
        EMAX2 = float(np.nanpercentile(np.concatenate(
            [e_c[np.isfinite(e_c)], e_t[np.isfinite(e_t)]]), 99))
        d2 = e_c - e_t
        DMAX2 = float(np.nanpercentile(np.abs(d2[np.isfinite(d2)]), 99))

        f2, a2 = plt.subplots(1, 3, figsize=(15.2, 4.3))
        f2.suptitle(
            f"Does the pseudo-SSH channel help in the thermocline?  "
            f"{b_lo:.0f}–{b_hi:.0f} m, North Atlantic, {mo_label} — "
            f"same depthwise U-Net, one extra input channel",
            fontsize=12.5, y=.98)
        ia = panel(a2[0], e_c, "U-Net  without  SSH  |error|", "magma", 0, EMAX2, r_c)
        panel(a2[1], e_t, "U-Net  with  SSH  |error|", "magma", 0, EMAX2, r_t)
        ib = panel(a2[2], d2,
                   "|error| difference:  without − with\n(red = SSH closer to truth)",
                   "RdBu_r", -DMAX2, DMAX2,
                   100 * (r_c - r_t) / r_c, rlabel="regional gain")
        a2[2].texts[-1].set_text(f"regional gain {100*(r_c-r_t)/r_c:+.1f} %")
        for a in a2:
            a.set_xlabel("longitude (°E)", fontsize=8.5)
        a2[0].set_ylabel("latitude (°N)", fontsize=8.5)
        f2.subplots_adjust(left=.045, right=.90, top=.795, bottom=.245, wspace=.16)
        cb = f2.colorbar(ia, ax=a2[:2], fraction=.020, pad=.012)
        cb.set_label(f"|error| ({UNIT})", fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
        cb2b = f2.colorbar(ib, ax=a2[2], fraction=.040, pad=.012)
        cb2b.set_label(f"Δ|error| ({UNIT})", fontsize=8.5); cb2b.ax.tick_params(labelsize=7.5)
        f2.text(.045, .135,
                "Both arms are the same depthwise U-Net trained identically; only the "
                "input config differs (10 vs 12 channels).\n"
                "The pseudo-SSH is derived from the same T/S being reconstructed, so "
                "this is an UPPER BOUND on what real altimetry would give.\n"
                f"Global 3-seed effect: 0.1572 → 0.1368 degC (+13.0 %), largest in this "
                f"very band (+15.1 ± 0.8 %) — see reports/ssh_ablation.md.",
                fontsize=8.0, color="#333", va="top", linespacing=1.55)
        out2 = os.path.join(C.REPORTS, "fig_reconstruction_na_ssh.png")
        f2.savefig(out2, dpi=155)
        plt.close(f2)
        print(f"wrote {out2}   (no-SSH {r_c:.4f} -> with-SSH {r_t:.4f} {UNIT}, "
              f"{100*(r_c-r_t)/r_c:+.1f}%)", flush=True)

        json.dump({"month": mo_label, "month_index": pick["i"], "seed": args.seed,
                   "band_m": [b_lo, b_hi], "region": {"lat": [lat_lo, lat_hi],
                                                      "lon": [lon_lo, lon_hi]},
                   "var": V, "n_profiles_in_window": int(plat.size),
                   "rmse": {"oi": pick["oi"], "unet_certified": pick["unet"],
                            "unet_no_ssh": r_c, "unet_with_ssh": r_t},
                   "per_month": [{k: r[k] for k in ("i", "time", "oi", "unet", "gain")}
                                 for r in rows]},
                  open(os.path.join(C.CACHE, "reconstruction_figure.json"), "w"),
                  indent=2)
print("done.", flush=True)
