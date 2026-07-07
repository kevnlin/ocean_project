"""Generate synthetic Argo-like profiles from CESM2-LE and summarise them."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ocean_tokenizer import data, argo, config as C

rng = np.random.default_rng(C.SEED)
grid = data.CommonGrid()

# one recent monthly snapshot for illustration
ti = data.select_month_indices(C.GT_SOURCE, (2014, 2014))
f = data.load_gt_fields(ti[:1], grid)
prof = argo.sample_profiles(f, 0, grid, C.N_PROFILES, rng)

os.makedirs(C.CACHE, exist_ok=True)
np.savez(os.path.join(C.CACHE, "synthetic_argo_example.npz"),
         ij=prof["ij"], lat=prof["lat"], lon=prof["lon"],
         month=prof["month"], TEMP=prof["TEMP"], SALT=prof["SALT"],
         depth=grid.depth)

# summary
P, D = prof["TEMP"].shape
lines = ["# Synthetic Argo Profiles (from CESM2-LE)\n",
         f"- source month: {f['time'][0]}",
         f"- profiles sampled: {P} random ocean columns",
         f"- depth levels per profile: {D} ({grid.depth.min():.0f}-{grid.depth.max():.0f} m)",
         f"- TEMP range: [{np.nanmin(prof['TEMP']):.2f}, {np.nanmax(prof['TEMP']):.2f}] degC",
         f"- SALT range: [{np.nanmin(prof['SALT']):.2f}, {np.nanmax(prof['SALT']):.2f}] PSU",
         f"- lat coverage: [{prof['lat'].min():.1f}, {prof['lat'].max():.1f}]",
         f"- lon coverage: [{prof['lon'].min():.1f}, {prof['lon'].max():.1f}]",
         "",
         "Profiles are full vertical columns of TEMP/SALT drawn at random ocean "
         "(lat,lon) locations — emulating sparse Argo float sampling of the dense "
         "simulated ocean state.  Saved to `outputs/cache/synthetic_argo_example.npz`.",
         ""]
with open(os.path.join(C.REPORTS, "synthetic_argo.md"), "w") as fp:
    fp.write("\n".join(lines))
print("\n".join(lines))
