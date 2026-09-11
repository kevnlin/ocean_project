# Equatorial Pacific — 2-D reconstruction, and what more Argo buys

<!-- generated alongside experiments/52_argo_recon_map.py -->

> **One track.** Track A only; Track B has not run this package.
> Input is real QC'd Argo; targets are real measurements from **WMO-disjoint
> held-out floats** appearing in no training month.

## Why this region

The global CESM2 error maps put their brightest band along the equator — the
cold tongue and its sharp, shallow thermocline are the hardest water in the
reconstruction. This package asks the same question of real observations there.

Box: **lat −12.5…12.5, lon 180…231** (central equatorial Pacific), the same
25° × 51° span as the Gulf Stream and North Pacific gyre boxes, so the grid,
token budget and model are unchanged and a regional difference cannot be a
configuration difference.

Cohort: **105,249 profiles, 986 floats, 301 months**, of which **197 floats /
20,864 profiles** are held out. 36 eligible development months, 60 held-out
WMOs scored.

## The equator behaves differently, and the baselines say so first

| row | TEMP (z) | SALT (z) | skill vs climatology (TEMP) |
|---|---:|---:|---:|
| `dfs_expertlocal_cbottle` | 0.565 | 0.434 | 44 % |
| `uniform_expertlocal_cbottle` | 0.568 | 0.434 | 44 % |
| `objective_interpolation` | 0.940 | 0.899 | 7 % |
| `train_climatology` | 1.014 | 1.012 | — |
| `source_persistence` | **1.314** | **1.253** | **−30 %** |

**`source_persistence` is worse than doing nothing.** Taking the nearest float's
measurement as the answer is *actively misleading* here, which it is not in
either mid-latitude box (Gulf Stream 0.80 against a 1.01 floor). That is the
signature of tropical instability waves and the equatorial wave guide: the field
decorrelates faster than the float array samples it. Optimal interpolation also
nearly collapses — 7 % skill against 21 % in the Gulf Stream.

The learned rows hold up (44 % skill), so the difficulty is real but not fatal.

## Feeding it the profiles that actually exist

The model has used **24 profiles/month** throughout, subsampled from a median of
**~260** the array actually delivers — about 10 % of the real data. Both arms
below are identical except for that budget: same region, same kernel, same
seeds, same held-out floats.

| arm | input profiles/month | TEMP RMSE (z) [95 % CI, seed+WMO] | SALT RMSE (z) |
|---|---|---|---|
| `p24` | 24 | 0.5575 [0.5362, 0.5822] | 0.4458 [0.4182, 0.4709] |
| `pall` | **all available** | **0.5377 [0.5177, 0.5577]** | 0.4374 [0.4134, 0.4598] |

Paired over the same seeds and the same floats — the arms' errors are strongly
correlated through the ocean state they both reconstruct, so the paired test is
the right one and two overlapping marginal CIs badly understate the evidence:

| channel | `pall` − `p24` [95 % CI] | excludes zero | per-seed |
|---|---|---|---|
| **TEMP** | **−0.0198 [−0.0342, −0.0046]** | **yes** | −0.0288, −0.0262, −0.0045 |
| SALT | −0.0084 [−0.0301, +0.0091] | no | +0.0094, −0.0043, −0.0302 |

**Temperature improves robustly — ~3.6 % relative, negative in all three seeds.
Salinity does not improve detectably.**

Cost: 187 ms/step against 58 ms, and 0.47 GB of GPU memory. ~12 min for 4,000
steps instead of ~4. The token count rises from ~380 to ~3,600.

### Where the extra observations helped

`fig_argo_recon_eq_pacific_improvement.png` maps
`RMSE(p24) − RMSE(pall)`; red is where the extra profiles helped.

The gain concentrates in the **100–300 m band between about 2° N and 10° N**,
strongest west of ~200° E — the thermocline ridge under the North Equatorial
Countercurrent, which is also the brightest part of the error map. The deep
band (300–985 m) is nearly neutral: it was already well reconstructed, so extra
observations have nothing to add. A few cells south of the equator get worse.

That pattern is what one would hope for — the additional evidence pays off where
the field is most structured — but it is read off a noisy cell-level difference
and should be treated as suggestive, not established.

## Figures

* `fig_argo_recon_eq_pacific_p24.png` — 24-profile error maps
* `fig_argo_recon_eq_pacific_pall.png` — all-profile error maps (**shared colour
  scale with the above**, or the improvement hides in the colour bar)
* `fig_argo_recon_eq_pacific_improvement.png` — the difference
* `fig_argo_recon_eq_pacific_coverage.png` — held-out values per cell

Cells are drawn only where at least 8 held-out values fell (474 of 988). Blank
cells are **not** zero error and are not interpolated: no held-out float
reported there, and painting them would invent skill where there is no evidence.

## Caveats

1. **A kernel change rode along.** These runs use `--region-kernel`, which sizes
   the DFS support kernel by this box's physical span. 51° of longitude is
   5,671 km at the equator against 4,499 km at the Gulf Stream, so the inherited
   default made equatorial correlation lengths ~26 % too short. Both arms use it,
   so the `pall − p24` contrast is clean — but the equatorial numbers are not
   directly comparable with the mid-latitude tables, which do not.
2. **Salinity is unresolved.** A null with three seeds is weak evidence of
   absence.
3. Only `dfs` and `uniform` were run here, not the full registered ladder.
