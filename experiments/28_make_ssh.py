"""Phase-4 / Task 4.1 — build the pseudo-SSH ("satellite altimeter") modality.

Route A of the plan: derive the **steric / baroclinic** component of sea-surface
height from the TEMP/SALT truth fields via TEOS-10 dynamic height, because the
standardized stores carry no SSH variable.  See
[src/ocean_tokenizer/ssh.py](../src/ocean_tokenizer/ssh.py) for the method and
its limitations (baroclinic only, referenced to 990 dbar, derived from the same
fields the model reconstructs).

Writes one (H, W) field per month to ``outputs/cache/ssh_dyn.npz`` together with
the zarr time indices, so any split can align to it by index lookup.

Diagnostics printed (and cached) so the report can state what the modality
actually contains:
  * coverage: fraction of ocean cells with a full column to the reference level
  * the anomaly standard deviation, which is the signal a model could use
  * correlation of the SSH anomaly with the SST anomaly and with the 100-300 m
    mean temperature anomaly -- SSH is supposed to be a *thermocline* proxy, so
    if it only tracks SST it adds nothing beyond the SST channel we already have

Run:
    python experiments/28_make_ssh.py                 # 1985-2014, ~15 min CPU
    python experiments/28_make_ssh.py --smoke         # 6 months
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np

from ocean_tokenizer import config as C, data
from ocean_tokenizer.ssh import P_REF_DBAR, steric_height_field

ap = argparse.ArgumentParser()
ap.add_argument("--years", default="1985,2014",
                help="inclusive year range covering every protocol_v1 split")
ap.add_argument("--chunk", type=int, default=12, help="months loaded at a time")
ap.add_argument("--out", default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

y0, y1 = (int(x) for x in args.years.split(","))
t0 = time.time()
grid = data.CommonGrid()
idx = data.select_month_indices(C.GT_SOURCE, (y0, y1))
if args.smoke:
    idx = idx[:6]
print(f"grid={grid}\nmonths={idx.size} ({y0}-{y1})  p_ref={P_REF_DBAR} dbar",
      flush=True)

H, W = grid.nlat, grid.nlon
ssh = np.full((idx.size, H, W), np.nan, dtype="float32")
months = np.zeros(idx.size, dtype=int)
times = []
sst_a_all, t100_300_a_all = [], []

# 100-300 m levels, for the thermocline-proxy diagnostic
band = np.where((grid.depth > 100.0) & (grid.depth <= 300.0))[0]
sst_store, t_band_store = [], []

for s in range(0, idx.size, args.chunk):
    e = min(s + args.chunk, idx.size)
    f = data.load_gt_fields(idx[s:e], grid)
    for i in range(e - s):
        ssh[s + i] = steric_height_field(f["TEMP"][i], f["SALT"][i], grid)
    months[s:e] = f["months"]
    times.extend(list(f["time"]))
    sst_store.append(f["SST"].copy())
    t_band_store.append(np.nanmean(f["TEMP"][:, band], axis=1))
    print(f"  months {s+1}-{e}/{idx.size}  ({time.time()-t0:.0f}s)", flush=True)

SST = np.concatenate(sst_store, 0)                     # (T,H,W)
TBAND = np.concatenate(t_band_store, 0)                # (T,H,W)

# ---- diagnostics: anomalies about the *whole-archive* monthly climatology ----
# (This archive is not a split; it is the raw modality.  Split-correct anomalies
#  are taken later by the ablation script, using train months only.)
def monthly_anom(x):
    """Deviation from this archive's own monthly mean.

    A calendar month sampled only once has a mean equal to itself, so its
    anomaly is identically zero and carries no information.  Those months are
    returned as NaN rather than as a spurious zero, which would silently
    deflate every statistic computed below.
    """
    out = np.full_like(x, np.nan)
    for m in range(1, 13):
        sel = months == m
        if sel.sum() >= 2:
            out[sel] = x[sel] - np.nanmean(x[sel], axis=0)
    return out


ssh_a, sst_a, tb_a = monthly_anom(ssh), monthly_anom(SST), monthly_anom(TBAND)
ocean_valid = np.isfinite(ssh).all(axis=0) & grid.ocean
coverage = float(ocean_valid.sum() / max(grid.ocean.sum(), 1))
# a calendar month needs >= 2 samples before an anomaly means anything
n_repeated = int(sum((months == m).sum() >= 2 for m in range(1, 13)))
anom_defined = n_repeated > 0


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))


diag = {
    "coverage_frac_of_ocean": coverage,
    "ssh_mean_m": float(np.nanmean(ssh)),
    "ssh_anom_std_m": float(np.nanstd(ssh_a)),
    "ssh_anom_std_cm": float(np.nanstd(ssh_a) * 100),
    "corr_ssh_anom_vs_sst_anom": corr(ssh_a, sst_a),
    "corr_ssh_anom_vs_temp100_300_anom": corr(ssh_a, tb_a),
}
print("\ndiagnostics:", flush=True)
for k, v in diag.items():
    print(f"  {k:38s} {v: .4f}", flush=True)

# Physical guards.  The absolute field is checked always; the anomaly statistics
# only once a calendar month is actually repeated (a 6-month smoke archive has
# one sample per calendar month, so its anomalies are zero by construction and
# say nothing about the field).
assert 0.1 < diag["ssh_mean_m"] < 3.0, (
    f"mean steric height {diag['ssh_mean_m']:.4g} m is not physical: check the "
    f"units, the reference pressure, or the depth sign convention")
if anom_defined:
    assert 0.001 < diag["ssh_anom_std_m"] < 0.5, (
        f"pseudo-SSH anomaly std {diag['ssh_anom_std_m']:.4g} m is not physical")
else:
    print("  (anomaly diagnostics skipped: no calendar month is repeated)",
          flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


meta = {"task": "make_ssh", "method": "TEOS-10 steric height (baroclinic only)",
        "p_ref_dbar": P_REF_DBAR, "years": [y0, y1], "smoke": args.smoke,
        "git_commit": git_commit(), "n_months": int(idx.size),
        "diagnostics": diag,
        "cpu_hours": round((time.time() - t0) / 3600, 3)}

out = args.out or os.path.join(C.CACHE,
                               "ssh_dyn_smoke.npz" if args.smoke else "ssh_dyn.npz")
np.savez_compressed(out, ssh=ssh, time_index=idx, months=months,
                    time=np.array(times), meta=json.dumps(meta))
json.dump(meta, open(os.path.splitext(out)[0] + "_meta.json", "w"), indent=2)
print(f"\nDONE in {(time.time()-t0)/60:.1f} min -> {out} "
      f"({os.path.getsize(out)/1e6:.0f} MB)", flush=True)
