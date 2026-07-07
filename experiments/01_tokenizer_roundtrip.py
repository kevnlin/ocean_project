"""Round-trip test for all four tokenizers: field -> tokens -> field, error ~ 0.

Without masking the reconstruction must be exact (max_abs == 0, NaN mask
preserved).  We test on a real CESM2-LE monthly field on the common grid, plus a
masked/subsampled point-query case to show graceful degradation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ocean_tokenizer import data, config as C
from ocean_tokenizer.tokenizers import (
    GridPatchTokenizer2D, VolumePatchTokenizer3D,
    VerticalProfileTokenizer, PointQueryTokenizer, roundtrip_error)

grid = data.CommonGrid()
ti = data.select_month_indices(C.GT_SOURCE, (2000, 2000))[:1]
f = data.load_gt_fields(ti, grid)

# build (C,D,H,W) volume with TEMP,SALT ; and (C,H,W) surface with SST,SSS
vol = np.stack([f["TEMP"][0], f["SALT"][0]], axis=0)              # (2,D,H,W)
surf = np.stack([f["SST"][0], f["SSS"][0]], axis=0)              # (2,H,W)
print("volume field:", vol.shape, "surface field:", surf.shape)

lines = ["# Tokenizer Round-Trip Test\n",
         f"Field: CESM2-LE {f['time'][0]} on common grid "
         f"(D={grid.ndepth}, H={grid.nlat}, W={grid.nlon}).\n",
         "Reconstruction error ignores NaN land but verifies the NaN mask matches.\n",
         "| tokenizer | field | tokens shape | token_dim | max_abs | rmse | nan_match | exact |",
         "|-----------|-------|--------------|-----------|---------|------|-----------|-------|"]

def row(name, fld, ts, rec):
    e = roundtrip_error(fld, rec)
    lines.append(f"| {name} | {tuple(fld.shape)} | {tuple(ts.tokens.shape)} | "
                 f"{ts.tokens.shape[1]} | {e['max_abs']:.2e} | {e['rmse']:.2e} | "
                 f"{e['nan_match']} | {'YES' if e['exact'] else 'no'} |")
    print(name, e)
    return e

ok = True

# (a) 2D grid patch on surface (2,H,W)
tok = GridPatchTokenizer2D(patch=(15, 16))
ts = tok.encode(surf); rec = tok.decode(ts); ok &= row("2D grid patch (surface)", surf, ts, rec)["exact"]

# (a) 2D grid patch on a single depth level treated as (C=D? ) -> use volume per-depth via channels=D
ts = tok.encode(vol[0]); rec = tok.decode(ts)  # TEMP (D,H,W) as (C=D,H,W)
ok &= row("2D grid patch (TEMP all depths as channels)", vol[0], ts, rec)["exact"]

# (b) 3D volume patch on (2,D,H,W)
tok3 = VolumePatchTokenizer3D(patch=(5, 30, 36))
ts = tok3.encode(vol); rec = tok3.decode(ts); ok &= row("3D volume patch", vol, ts, rec)["exact"]

# (c) vertical profile on (2,D,H,W)
tokp = VerticalProfileTokenizer()
ts = tokp.encode(vol); rec = tokp.decode(ts); ok &= row("vertical profile", vol, ts, rec)["exact"]

# (d) point query — full (exact) and subsampled (masked, not exact)
tokq = PointQueryTokenizer(coords=(grid.lat, grid.lon, grid.depth))
ts = tokq.encode(vol); rec = tokq.decode(ts); ok &= row("point query (full)", vol, ts, rec)["exact"]

rng = np.random.default_rng(0)
ntot = vol[0].size
sample_idx = rng.choice(ntot, size=ntot // 10, replace=False)
ts = tokq.encode(vol, sample_idx=sample_idx); rec = tokq.decode(ts)
e = roundtrip_error(vol, rec)
lines.append(f"| point query (10% masked) | {tuple(vol.shape)} | {tuple(ts.tokens.shape)} | "
             f"{ts.tokens.shape[1]} | {e['max_abs']:.2e} | {e['rmse']:.2e} | "
             f"{e['nan_match']} | (expected non-exact: only 10% kept) |")
print("point query masked", e)

lines.append("")
lines.append(f"**All un-masked tokenizers exact:** {'YES ✅' if ok else 'NO ❌'}")
with open(os.path.join(C.REPORTS, "tokenizer_roundtrip.md"), "w") as fp:
    fp.write("\n".join(lines))
print("\nAll exact (unmasked):", ok)
assert ok, "round-trip not exact!"
