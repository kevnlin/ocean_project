# Tokenizer Round-Trip Test

Field: CESM2-LE 2000-01-18 on common grid (D=20, H=180, W=360).

Reconstruction error ignores NaN land but verifies the NaN mask matches.

| tokenizer | field | tokens shape | token_dim | max_abs | rmse | nan_match | exact |
|-----------|-------|--------------|-----------|---------|------|-----------|-------|
| 2D grid patch (surface) | (2, 180, 360) | (276, 480) | 480 | 0.00e+00 | 0.00e+00 | True | YES |
| 2D grid patch (TEMP all depths as channels) | (20, 180, 360) | (276, 4800) | 4800 | 0.00e+00 | 0.00e+00 | True | YES |
| 3D volume patch | (2, 20, 180, 360) | (240, 10800) | 10800 | 0.00e+00 | 0.00e+00 | True | YES |
| vertical profile | (2, 20, 180, 360) | (64800, 40) | 40 | 0.00e+00 | 0.00e+00 | True | YES |
| point query (full) | (2, 20, 180, 360) | (1296000, 5) | 5 | 0.00e+00 | 0.00e+00 | True | YES |
| point query (10% masked) | (2, 20, 180, 360) | (129600, 5) | 5 | nan | nan | False | (expected non-exact: only 10% kept) |

**All un-masked tokenizers exact:** YES ✅