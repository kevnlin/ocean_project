"""Step-by-step audit of the real-Argo pipeline: data prep, normalisation, tokenisation.

Asked for in the 2026-09-17 group meeting: before any further experiments, walk
the pipeline one stage at a time and report what each stage does to the numbers,
because "everything reads ~1 and nothing moves it" is a symptom that points at
the data path rather than at the model.

Four sections, each a question with a number attached:

  1 DATA PREP      what is in the cohort, what the anomaly target is worth in
                   physical units, which values are not measurements, and where
                   the climatology is evaluated
  2 NORMALISATION  what one z unit is in degC / PSU per depth band, what the
                   per-level scale is fitted on, and what the train->test shift
                   does to the climatology reference
  3 TOKENISATION   what a profile becomes before the model sees it, and at what
                   spatial and vertical resolution the coordinate path and the
                   local refiner can still tell two places apart
  4 FLOOR          how far the anomaly at a held-out float is predictable at
                   all: correlation against separation, and the share of the
                   variance no other float sees (the nugget of a covariance
                   fitted on the training years, per latitude band)

Default: the GLOBAL cohort (one domain over the whole ocean, 1.79 M profiles,
20 levels to 985 m), every profile a month delivers, satellite-era split (train
2016-2020, validate 2021, test 2022-23). The anomaly is WOA23 at the centre of
the profile's 1 deg cell, as the registered cohorts were built. Covariances
are fitted per absolute-latitude band, because the tropics and the mid-latitude
fronts do not share one. `--regions gulfstream,npac_gyre` reruns the committed
regional study.

Nothing here trains anything and nothing here writes to the registered
artifacts.

  .venv/bin/python experiments/real_data/61_pipeline_audit.py
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr

from ocean_tokenizer import protocol as P
from ocean_tokenizer.argo_obs import (ArgoCohort, ArgoNorm, _select_profiles,
                                      ArgoObsConfig)
from ocean_tokenizer.audit_tools import (BINS_KM, LAT_BANDS, band_of_lat,
                                         cohort_path, fit_cov, haversine_km,
                                         load_cohort, pair_correlation)
from ocean_tokenizer.token_api import default_depth_bands, _DEPTH_SCALE, _N_FREQ_SPHERE
from ocean_tokenizer.query_decoder import _L_INIT, _GATE_INIT
from ocean_tokenizer.objective_interpolation import OISettings

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="global")
ap.add_argument("--split-protocol", default="recent3", choices=sorted(P.SPLIT_PROTOCOLS))
ap.add_argument("--split-table",
                default='{"train":[2016,2020],"validation":[2021,2021],"development":[2022,2023]}',
                help="JSON override of the year splits (for a secondary era, "
                     "e.g. the satellite-covered 2016-2023 window)")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--n-profiles", type=int, default=1000,
                help="the capped comparison arm; the main reference uses every profile")
ap.add_argument("--corr-months", type=int, default=150)
ap.add_argument("--no-covariance", action="store_true",
                help="skip the (slow) per-level covariance fits behind the floor")
ap.add_argument("--out", default=None)
args = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REGIONS = [r for r in args.regions.split(",") if r]
SPLITS = P.SPLIT_PROTOCOLS[args.split_protocol]
if args.split_table:
    SPLITS = {k: tuple(v) for k, v in json.loads(args.split_table).items()}
    A_PROTOCOL = "custom"
BANDS = (("0-100m", 0.0, 100.0), ("100-300m", 100.0, 300.0),
         ("300-700m", 300.0, 700.0), ("700-1400m", 700.0, 1401.0))
CH = ("TEMP", "SALT")
UNITS = {"TEMP": "degC", "SALT": "PSU"}
OUT_MD = args.out or os.path.join(ROOT, "reports", "real_data", "pipeline_audit.md")
OUT_JSON = os.path.join(ROOT, "outputs", "cache", "pipeline_audit.json")
FIG = os.path.join(ROOT, "reports", "real_data")
t0 = time.time()
A: dict = {"split_protocol": args.split_protocol, "splits": SPLITS,
           "seed": args.seed, "n_profiles": args.n_profiles, "regions": {}}


def band_of(levels):
    out = []
    for d in levels:
        for name, lo, hi in BANDS:
            if (lo < d <= hi) or (d <= levels.min() and lo <= 0):
                out.append(name); break
        else:
            out.append(BANDS[-1][0])
    return np.array(out)


def band_rms(levels, per_level):
    """RMS of a per-level quantity inside each depth band."""
    b = band_of(levels)
    return {name: float(np.sqrt(np.nanmean(np.asarray(per_level)[b == name] ** 2)))
            for name, _, _ in BANDS}


def eval_targets(c, leads=(0,), split="development"):
    """The table's fixed-target month set, replicated exactly (58_argo_fusion_tables)."""
    return [int(T) for T in c.months_in(split)
            if c.month(T, float_split="heldout_float").size
            and all(c.month(T - L, float_split="cohort_float").size for L in leads)]


for region in REGIONS:
    print(f"\n===== {region}", flush=True)
    R: dict = {}
    c, _ = load_cohort(ROOT, region, SPLITS)
    raw = ArgoCohort.load(cohort_path(ROOT, region))
    raw.apply_splits(SPLITS)
    norm = ArgoNorm.fit(c, "train")
    LEV = c.levels
    ds = xr.open_dataset(cohort_path(ROOT, region))
    order = np.argsort(np.asarray(ds["month_index"].values, int), kind="stable")
    dmode = np.asarray(ds["data_mode"].values).astype(str)[order]

    # ---------------------------------------------------------------- 1 data
    comp = {}
    for sp in ("train", "validation", "development", "holdout"):
        m = c.year_split == sp
        comp[sp] = {
            "profiles": int(m.sum()),
            "cohort_profiles": int((m & (c.float_split == "cohort_float")).sum()),
            "heldout_profiles": int((m & (c.float_split == "heldout_float")).sum()),
            "floats": int(np.unique(c.wmo[m]).size),
            "months": int(np.unique(c.month_index[m]).size)}
    R["composition"] = comp
    R["levels"] = LEV.tolist()

    scale = {}
    for ch in CH:
        tr = c.year_split == "train"
        scale[ch] = {
            "anom_std_per_level": norm.std[ch].tolist(),
            "anom_std_by_band": band_rms(LEV, norm.std[ch]),
            "raw_std_by_band": band_rms(LEV, np.nanstd(getattr(raw, ch)[tr], 0)),
            "anom_mean_by_band_train": band_rms(LEV, np.nanmean(getattr(c, ch)[tr], 0)),
        }
        dv = c.year_split == "development"
        scale[ch]["anom_mean_by_band_dev"] = band_rms(
            LEV, np.nanmean(getattr(c, ch)[dv], 0))
        scale[ch]["dev_over_train_std"] = band_rms(
            LEV, np.nanstd(getattr(c, ch)[dv], 0) / norm.std[ch])
    R["scale"] = scale

    # outliers, as the registered cohort ships them
    out = {}
    for ch in CH:
        z = (getattr(c, ch) - norm.mean[ch]) / norm.std[ch]
        with np.errstate(invalid="ignore"):
            mx = np.nanmax(np.abs(z), axis=1)
        bad = np.nan_to_num(mx) > 10
        per = {}
        for sp in ("train", "validation", "development", "holdout"):
            for fs in ("cohort_float", "heldout_float"):
                n = int((bad & (c.year_split == sp) & (c.float_split == fs)).sum())
                if n:
                    per[f"{sp}/{fs}"] = n
        u, n = np.unique(dmode[bad], return_counts=True)
        worst = sorted(np.unique(c.wmo[bad]).tolist(),
                       key=lambda w: -int((bad & (c.wmo == w)).sum()))[:3]
        out[ch] = {"profiles_over_10_sigma": int(bad.sum()),
                   "floats": int(np.unique(c.wmo[bad]).size),
                   "max_abs_z": float(np.nan_to_num(mx).max()),
                   "by_split": per,
                   "by_data_mode": {str(k): int(v) for k, v in zip(u, n)},
                   "worst_floats": [
                       {"wmo": str(w), "profiles": int((bad & (c.wmo == w)).sum()),
                        "years": [int(c.year[bad & (c.wmo == w)].min()),
                                  int(c.year[bad & (c.wmo == w)].max())],
                        "float_split": str(np.unique(c.float_split[c.wmo == w])[0]),
                        "data_mode": sorted({str(x) for x in dmode[bad & (c.wmo == w)]}),
                        "max_abs_z": float(np.nan_to_num(np.nanmax(np.abs(z[c.wmo == w]))))}
                       for w in worst]}
        # what those values do to the model's INPUT stream in the test years
        dmask = (c.year_split == "development") & (c.float_split == "cohort_float")
        zz = z[dmask]
        keep = np.nan_to_num(np.nanmax(np.abs(zz), axis=1)) <= 10
        out[ch]["dev_input_z_rms"] = float(np.sqrt(np.nanmean(zz ** 2)))
        out[ch]["dev_input_z_rms_clean"] = float(np.sqrt(np.nanmean(zz[keep] ** 2)))
    R["outliers"] = out

    cq, rep = load_cohort(ROOT, region, SPLITS, qc=True)
    R["robust_qc"] = {"k_sigma": rep.k_sigma, "values_flagged": rep.n_values_flagged,
                      "profiles_dropped": rep.n_profiles_dropped,
                      "by_data_mode": rep.by_data_mode,
                      "std_change_by_band": {
                          ch: {"before": band_rms(LEV, rep.std_before[ch]),
                               "after": band_rms(LEV, rep.std_after[ch])} for ch in CH}}

    # climatology evaluated at the cell centre vs at the profile
    box = (dict(lat=(-90.0, 90.0), lon=(0.0, 360.0)) if region == "global"
           else P.REGIONS[region])
    NY, NX = c.grid
    la0, la1 = (float(x) for x in box["lat"]); lo0, lo1 = (float(x) for x in box["lon"])
    latc = la0 + (raw.grid_y + 0.5) * (la1 - la0) / NY
    lonc = lo0 + (raw.grid_x + 0.5) * (lo1 - lo0) / NX
    dlat_km = np.abs(raw.lat - latc) * 111.195
    dlon_km = np.abs((raw.lon % 360.0) - lonc) * 111.195 * np.cos(np.deg2rad(raw.lat))
    cx, _ = load_cohort(ROOT, region, SPLITS, anomaly="exact")
    tr = c.year_split == "train"
    clim = {"cell_deg": [float((la1 - la0) / NY), float((lo1 - lo0) / NX)],
            "offset_km": {"lat_mean": float(dlat_km.mean()), "lat_max": float(dlat_km.max()),
                          "lon_mean": float(dlon_km.mean()), "lon_max": float(dlon_km.max())}}
    for ch in CH:
        a_cell = getattr(c, ch)[tr]; a_exact = getattr(cx, ch)[tr]
        clim[ch] = {
            "std_cell_by_band": band_rms(LEV, np.nanstd(a_cell, 0)),
            "std_exact_by_band": band_rms(LEV, np.nanstd(a_exact, 0)),
            "std_difference_by_band": band_rms(LEV, np.nanstd(a_cell - a_exact, 0)),
            "variance_removed_pct": {
                b: float(100 * (1 - (band_rms(LEV, np.nanstd(a_exact, 0))[b] ** 2)
                                / max(band_rms(LEV, np.nanstd(a_cell, 0))[b] ** 2, 1e-12)))
                for b, _, _ in BANDS}}
    R["climatology_position"] = clim

    # ------------------------------------------------------- 2 normalisation
    nrm = {"fitted_on": "train years of this protocol, per level, per channel",
           "z_to_physical_by_band": {ch: scale[ch]["anom_std_by_band"] for ch in CH}}
    for ch in CH:
        sd = norm.std[ch]
        med = np.nanmedian(getattr(c, ch)[tr], 0)
        mad = 1.4826 * np.nanmedian(np.abs(getattr(c, ch)[tr] - med), 0)
        nrm.setdefault("robust_over_fitted_sigma", {})[ch] = band_rms(LEV, mad / sd)
        nrm.setdefault("worst_level_inflation", {})[ch] = {
            "level_m": float(LEV[int(np.nanargmin(mad / sd))]),
            "fitted_std": float(sd[int(np.nanargmin(mad / sd))]),
            "robust_std": float(mad[int(np.nanargmin(mad / sd))])}
    R["normalisation"] = nrm

    # ------------------------------------------------------- 3 tokenisation
    bands = default_depth_bands(float(LEV.max()))
    per_band = []
    for (lo, hi) in bands:
        sel = (LEV >= lo) & (LEV < hi)
        if (lo, hi) == bands[-1]:
            sel |= LEV >= hi
        per_band.append({"band_m": [lo, hi], "levels": int(sel.sum()),
                         "token_depth_m": float((lo + hi) / 2),
                         "level_depths": LEV[sel].tolist()})
    # how much of a profile's vertical structure a band-mean summary cannot carry
    loss = {}
    for ch in CH:
        z = (getattr(c, ch)[tr] - norm.mean[ch]) / norm.std[ch]
        num = den = 0.0
        for (lo, hi) in bands:
            sel = (LEV >= lo) & (LEV < hi)
            if not sel.any():
                continue
            zz = z[:, sel]
            mu = np.nanmean(zz, axis=1, keepdims=True)
            num += float(np.nansum((zz - mu) ** 2)); den += float(np.nansum(zz ** 2))
        loss[ch] = {"within_band_variance_fraction": num / max(den, 1e-12)}
    lam_deg = 2.0 / (2 ** (_N_FREQ_SPHERE - 1)) * 180.0 / np.pi
    R["tokenisation"] = {
        "profile_tokens": {"bands": per_band, "tokens_per_profile": len(bands),
                           "pooling": "masked mean of per-level embeddings",
                           "band_mean_loses": loss},
        "coordinate_features": {
            "finest_sphere_wavelength_deg": float(lam_deg),
            "finest_sphere_wavelength_km": float(lam_deg * 111.195),
            "depth_scale_m": _DEPTH_SCALE, "deepest_level_m": float(LEV.max())},
        "local_refiner_init": {
            "ell_lat_deg": _L_INIT[1] * 90.0, "ell_lat_km": _L_INIT[1] * 90.0 * 111.195,
            "ell_lon_deg_at_region": _L_INIT[0] * 180.0,
            "ell_lon_km_at_region": float(_L_INIT[0] * 180.0 * 111.195
                                          * np.cos(np.deg2rad((la0 + la1) / 2))),
            "ell_depth_m": _L_INIT[2] * 1000.0, "ell_time_months": _L_INIT[3],
            "gate_init": _GATE_INIT},
        "region_box_km": {"lat": float((la1 - la0) * 111.195),
                          "lon": float((lo1 - lo0) * 111.195
                                       * np.cos(np.deg2rad((la0 + la1) / 2)))},
        "argo_obs_query_depth_axis": {
            "token_z": "physical depth / max level",
            "query_z": "level index / (n_levels - 1)",
            "used_by": "objective_interpolation + GodasRowModel (45); the fusion "
                       "family passes physical metres and is unaffected",
            "max_mismatch": float(np.max(np.abs(
                np.arange(LEV.size) / (LEV.size - 1) - LEV / LEV.max())))},
        "causal_oi_settings": {
            "ell_xy_km_lat": OISettings().ell_xy * (la1 - la0) * 111.195,
            "ell_xy_km_lon": float(OISettings().ell_xy * (lo1 - lo0) * 111.195
                                   * np.cos(np.deg2rad((la0 + la1) / 2))),
            "rho0": OISettings().rho0}}

    # realised input density and how far a target sits from its nearest input.
    # The model sees EVERY profile; the capped draw is the comparison arm.
    from scipy.spatial import cKDTree

    def _xyz(rows):
        la, lo = np.deg2rad(c.lat[rows]), np.deg2rad(c.lon[rows])
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    def nearest_km(tg, src):
        d, _ = cKDTree(_xyz(src)).query(_xyz(tg), k=1)
        return 2 * 6371.0 * np.arcsin(np.clip(d / 2, 0, 1))

    dens, dmin_cap, dmin_all = [], [], []
    ET = eval_targets(c)
    for T in ET:
        tg = c.month(T, float_split="heldout_float")
        pool = c.month(T, float_split="cohort_float")
        cap = _select_profiles(c, pool, args.n_profiles,
                               np.random.default_rng([args.seed, T, 0]))
        dens.append((pool.size, cap.size))
        if tg.size and pool.size:
            dmin_cap.append(nearest_km(tg, cap)); dmin_all.append(nearest_km(tg, pool))
    dmin_cap = np.concatenate(dmin_cap); dmin_all = np.concatenate(dmin_all)
    pool_n = np.array([d[0] for d in dens]); cap_n = np.array([d[1] for d in dens])

    def dist_summary(x):
        return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
                "p75": float(np.percentile(x, 75)),
                "frac_under_50": float((x < 50).mean()),
                "frac_under_100": float((x < 100).mean())}

    R["input_density"] = {
        "eval_months": len(ET), "cap": args.n_profiles,
        "available_per_month": {"median": float(np.median(pool_n)),
                                "min": int(pool_n.min()), "max": int(pool_n.max())},
        "used_per_month": {"median": float(np.median(pool_n))},
        "cap_fraction_of_available": float(cap_n.sum() / pool_n.sum()),
        "nearest_input_km_all_profiles": dist_summary(dmin_all),
        "nearest_input_km_at_cap": dist_summary(dmin_cap)}

    # ------------------------------------------------------------ 4 floor
    tr_months = c.months_in("train")
    corr = {}
    for ch in CH:
        corr[ch] = {}
        for lv_m in (5.0, 105.0, 326.9, 707.6):
            lv = int(np.argmin(np.abs(LEV - lv_m)))
            rho, cnt = pair_correlation(c, norm, ch, lv, tr_months,
                                        max_months=args.corr_months, seed=args.seed)
            corr[ch][f"{LEV[lv]:.0f}m"] = {"bins_km": BINS_KM.tolist(),
                                           "corr": rho.tolist(), "pairs": cnt.tolist()}
    R["pair_correlation"] = corr
    # the same curve per absolute-latitude band at the thermocline: the global
    # covariance is not one covariance
    lat_band = band_of_lat(c.lat)
    by_band = {}
    lv327 = int(np.argmin(np.abs(LEV - 326.9)))
    for b, (lo, hi) in enumerate(LAT_BANDS):
        rho, cnt = pair_correlation(c, norm, "TEMP", lv327, tr_months,
                                    max_months=args.corr_months, seed=args.seed,
                                    rows=np.flatnonzero(lat_band == b))
        by_band[f"{lo:.0f}-{min(hi, 90):.0f}"] = {"bins_km": BINS_KM.tolist(),
                                                 "corr": rho.tolist(), "pairs": cnt.tolist()}
    R["pair_correlation_by_lat_band_327m"] = by_band

    if not args.no_covariance:
        print("  fitting covariances per latitude band ...", flush=True)
        cov = {ch: [] for ch in CH}
        for ch in CH:
            for lv in range(LEV.size):
                per = []
                for b in range(len(LAT_BANDS)):
                    rho, cnt = pair_correlation(
                        c, norm, ch, lv, tr_months, max_months=args.corr_months,
                        seed=args.seed, rows=np.flatnonzero(lat_band == b))
                    per.append(fit_cov(rho, cnt))
                cov[ch].append(per)
        # the nugget of each fit is the share of variance no neighbouring float
        # sees: the floor on J at a held-out float
        R["covariance_fit"] = {
            ch: [{"level_m": float(LEV[i]), "bands": [
                    {"lat_band": list(LAT_BANDS[b]), "a0": m.a0, "a1": m.a1,
                     "L1_km": m.L1, "a2": m.a2, "L2_km": m.L2, "nugget": m.nugget}
                    for b, m in enumerate(per)]}
                 for i, per in enumerate(cov[ch])] for ch in CH}

    A["regions"][region] = R

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(A, f, indent=1, default=float)
print(f"\nwrote {OUT_JSON}  ({time.time()-t0:.0f}s)")
