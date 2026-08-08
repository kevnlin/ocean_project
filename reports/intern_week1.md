# Intern week 1 — OI baseline, SSH modality, architecture spec

*2026-08-08. Branch `intern/oi-multimodal/setup`. Execution of
[Plan_OI_MultiModal_RealData.md](../Plan_OI_MultiModal_RealData.md).*

**Headline**: the OI baseline (milestone M1) is implemented, tested, tuned and
scored; the SSH modality is built and characterised; the architecture spec is
written. Two GPU-bound experiments are written and smoke-verified but not run —
every card on the box was at 73–75 GB / 80 GB all day.

---

## 0. Read this first — the plan was written against a week-2 snapshot

The plan describes the repo as it stood in week 2. The tree is at **week 4 with
DFS-Attention merged**. Per the plan's own tie-break rule ("when the plan
conflicts with code reality: the code wins — note the correction in your weekly
report"), every conflict below was resolved in favour of the code.

| # | Plan states | Code reality | Consequence |
|---|---|---|---|
| 1 | Split 312 train / 12 test; floor **0.535 / 0.125** | **protocol_v1** frozen 2026-07-17: 276 train / 36 val / 12 **pinned** test; floor **0.5523 / 0.1305** | All new work is built under protocol_v1 so its numbers can enter the headline table |
| 2 | Best model = depthwise U-Net **0.1492 / 0.0312** | Certified under protocol_v1: **0.1580 / 0.0325** | The M1 bar is 0.1580, not 0.1492 |
| 3 | "Tune OI on **training** months only" | protocol_v1: hyperparameter selection **is** model selection → **validation** months | Tuned on validation (stricter); a training-month sweep runs as a stability check |
| 4 | Imitate `08_density_ablation.py` | That script is pre-protocol | Imitated `21_baselines_protocol_v1.py` instead |
| 5 | Task 4.3: *create* `src/ocean_tokenizer/fusion.py` | **Exists**, four fusion variants, trained at full scale in week 4 | Task 4.3 is obsolete — and its stated bar was **missed** (see #6) |
| 6 | Fusion acceptance: "pass = beat joint U-Net 0.1787" | Shared latent sits at **~0.52** vs joint U-Net 0.1948 (month-identity recall collapse) | Not "implement the fusion core" but "fix the training dynamics" — a different task |
| 7 | `coord_features` is 7-dim | **68-dim fourier_v2** since week 4 | Spec documents the real featurisation; do not revert |
| 8 | ProfileEncoder cuts 4 index segments; 23-level grid trips `D % n_segments`; fix = 5-6-6-6 | **Already fixed, differently**: levels are banded by *physical depth*, no divisibility constraint | The plan's "known issue" is closed |
| 9 | Create `experiments/13_`, `14_`, `17_`, `18_` | All four numbers taken | Renumbered to 26–31 |
| 10 | Data lives elsewhere; rsync 42 GB to `/DATA2/zihao` | **This box is the source** — `processed/` is here (38 GB) | Task 0.2 is a no-op; `ROOT` made overridable via `OCEAN_ROOT` so the intern's box works too |
| 11 | cBottle = three models (3D/Video/SR) | Abstract describes **two stages**; no video variant confirmed | Flagged as unverified in the reading note |

## 1. What ran, with commands

```bash
# Phase 1 (M1) — CPU, ~2 h end to end
bash experiments/run_oi_queue.sh          # tune on val -> score test -> stability check
python experiments/30_oi_report.py        # -> reports/oi_{tuning,baseline}.md + figures

# Phase 4.1 — CPU, 13 min
python experiments/28_make_ssh.py         # -> outputs/cache/ssh_dyn.npz (47 MB, gitignored)

# Phase 2 — CPU, seconds
python experiments/31_density_powerlaw.py # -> reports/fig_density_powerlaw.png

# whole suite
python -m pytest tests/ -q                # 155 passed (114 before, +41 new)
```

## 2. Milestone M1 — optimal interpolation

**The repo had no OI.** `predict_nearest` is a distance-gated nearest fill with
no covariance model, so it cannot handle observation redundancy — two profiles
5 km apart count twice. [`src/ocean_tokenizer/oi.py`](../src/ocean_tokenizer/oi.py)
implements the real thing (Bretherton et al. 1976), level-by-level in z-scored
anomaly space so it shares the anomaly target, the normalisation and the
scoring mask with the U-Net:

$$\hat z_g = C_{go}\,(C_{oo} + \gamma I)^{-1} d, \qquad C(r) = e^{-r^2/2L^2}$$

* **22 unit tests** ([tests/test_oi.py](../tests/test_oi.py)), all analytic
  rather than regression snapshots: single-obs shrinkage `d/(1+γ)`, duplicate-obs
  saturation `2d/(2+γ)` (the redundancy handling a weighted average lacks),
  relaxation to background far from data, and exactness of the k-NN localisation
  against a dense global solve at k = n.
* Tuned on **validation** months: **L = 500 km, γ = 0.1, k = 10** for both TEMP
  and SALT — an interior point of the 4×3 grid on a smooth, convex surface, so a
  genuine optimum rather than a boundary artifact. k = 10 slightly beats k = 20
  and k = 40 (0.2034 / 0.2049 / 0.2054), i.e. the k-NN localisation is fully
  converged; extra near-zero-correlation neighbours only add noise to the solve.
  See [oi_tuning.md](oi_tuning.md).
* **Stability check passes**: repeating the whole sweep on *training* months
  returns the identical optimum (L = 500 km, γ = 0.1, k = 10) for both
  variables. The choice reflects the covariance structure of the ocean, not
  noise in one split.

### Results (12 pinned test months, identical samples)

| method | TEMP (°C) | SALT (PSU) | NA box TEMP | NA box SALT |
|---|---|---|---|---|
| WOA23 prior | 1.5670 | 0.6342 | 2.1768 | 0.5809 |
| Climatology floor | 0.5520 | 0.1305 | 0.6459 | 0.1455 |
| Nearest-profile fill | 0.9670 | 0.3119 | 1.3265 | 0.3437 |
| **Optimal interpolation** | **0.2287** | **0.0528** | **0.3139** | **0.0625** |
| Depthwise U-Net (certified) | 0.1580 | 0.0325 | 0.1938 | 0.0335 |

**The answer to M1: yes.** The full system beats OI by **+30.9 % on TEMP** and
**+38.4 % on SALT**, and the margin is *larger* in the North Atlantic
(+38.3 %) than globally — the conclusion is regionally consistent, not a global
average hiding a regional failure.

Three findings worth carrying beyond this report:

1. **OI beats the pointwise MLP** (0.2287 vs 0.2983 °C, +23.4 %). A properly
   tuned classical baseline outperforms a learned pointwise model; only the
   convolutional U-Nets clear it. An OI row belongs in *any* table that claims a
   learned method is useful — including the paper line's, where the
   shared-latent variants currently sit at ~0.52, i.e. **well below OI**.
2. **The U-Net's gain over OI is largest at the surface** (0–100 m +36.7 %),
   not in the thermocline (+26.9 %) as the plan predicted. The plan's premise
   was right — OI cannot use SST/SSS — but the conclusion did not follow: SST
   and SSS *are* dense observations of the 0–100 m layer, so they constrain the
   surface directly and the thermocline only indirectly.
3. **OI is the only method whose 100–300 m error exceeds its 0–100 m error**
   (0.2622 > 0.2509). With profiles alone, the thermocline really is the hardest
   layer; the surface fields are what invert that ordering. This yields a
   falsifiable prediction for the pending `profiles_only` run — see
   [oi_baseline.md](oi_baseline.md) §2.

The error map ([fig_oi_vs_unet_error_map.png](fig_oi_vs_unet_error_map.png))
localises OI's error to the equatorial Pacific cold tongue, the Kuroshio, the
Gulf Stream and the Agulhas — the high-eddy-energy regions where a stationary
isotropic Gaussian covariance is least appropriate. The difference map is red
almost everywhere: the U-Net is better nearly globally, not on average.

**Every method sees byte-identical observations.** `27_oi_vs_unet.py` replays
the certified run's RNG (312 discarded profile draws — `prepare_month` touches
the generator exactly once per month, with a draw that depends only on the ocean
mask) and then loads `audit_depthwise_e40.pt` for inference. `--verify-unet`
asserts the checkpoint reproduces its cached test RMSE, which proves the replay
and the inference path both match. This is stronger than the plan asked for: it
compares OI and the U-Net on the same profiles, not merely on the same settings.

**Still open**: the `profiles_only` U-Net row — the like-for-like comparison
(OI sees profiles only, so a profiles-only U-Net isolates *whose interpolator is
better* from *whose inputs are richer*). It needs training, hence a GPU.

## 3. Phase 2 — the density extrapolation is not what the plan says

The plan quotes "RMSE ∝ N^−0.41 → 4000 profiles = 0.100 °C". That α is the
**last segment only**. Fitting the whole curve gives α = 0.269, and the local
slope rises monotonically (0.151 → 0.254 → 0.340 → 0.410), so the curve is not
a power law at all.

| model | N at 0.1 °C |
|---|---|
| global fit (conservative) | **6143** |
| tail slope (= the plan's, optimistic) | **3973** |

The honest answer is **~4000–6100 profiles/month**, and the ambiguity is exactly
why the 4000/6000 runs are worth the GPU time. Full reasoning, including why the
slope steepens (low-density saturation against the no-observation limit) in
[density_4000_6000.md](density_4000_6000.md).

I also removed a footgun: `08_density_ablation.py` wrote to a fixed cache path,
so an extension run would have silently overwritten committed week-2 results. It
now takes `--out-suffix` (default empty = unchanged), and
[merge_density_json.py](../experiments/merge_density_json.py) folds extensions
back in — idempotently, refusing to merge runs whose `run_config` essentials
differ, and writing `.bak_premerge` on first touch.

## 4. Phase 3 — the specification

[doc/architecture_spec.md](../doc/architecture_spec.md): data pipeline (with
dataflow diagram), **normalization with every formula and its provenance**,
tokenization per modality with token counts, the model/loss as implemented, the
final-vision architecture marked exists-vs-missing, and the four cBottle
takeaways. It ends with an appendix listing the corrections it makes to the
plan. Companion reading note: [reading_cbottle.md](reading_cbottle.md), grounded
in the fetched abstract rather than recollection.

The token-imbalance number worth remembering: at 1500 profiles/month the WOA
prior contributes **~10 800** tokens against **6 000** profile tokens, so under
plain softmax attention the prior can outvote the observations. That is what the
fusion-rule comparison exists to study.

## 5. Phase 4 — the SSH modality

There was no SSH anywhere in the pipeline. Built the steric-height (dynamic
height, TEOS-10) pseudo-SSH from the T/S fields:
[ssh.py](../src/ocean_tokenizer/ssh.py) + [28_make_ssh.py](../experiments/28_make_ssh.py),
360 monthly fields, 11 physical unit tests (warm column stands taller than cold,
fresh taller than salty, short columns undefined, train-only statistics).

The characterisation is the interesting part:

| | value |
|---|---:|
| coverage | 86.7 % of ocean cells reach 990 dbar |
| SSH anomaly std | 2.89 cm |
| corr with **SST** anomaly | 0.470 |
| corr with **100–300 m T** anomaly | **0.708** |

It correlates *more* with thermocline temperature than with SST, so it is not a
restatement of the SST channel the model already has — which is the mechanism
the pre-registered hypothesis depends on. Hypothesis, design and launch command:
[ssh_ablation.md](ssh_ablation.md), written **before** any model was trained.

The `ssh` cfg token is strictly additive; 8 tests (4 of them one per
pre-existing config) pin that every config without `ssh` is **bit-identical**
with the SSH code present, so `audit_depthwise_e40` keeps its c_in = 10 and
every historical number stays reproducible.

## 6. Blocked on GPU

All eight A100s were at 73–75 GB / 80 GB for the entire session. Both blocked
items are written, smoke-verified end-to-end, and one command away:

| item | command | note |
|---|---|---|
| SSH ablation (Phase 4.2) | `CUDA_VISIBLE_DEVICES=N python experiments/29_ssh_ablation.py` | add `--cpu-tensors` to fit ~4 GB instead of ~17 GB |
| Density 4000/6000 (Phase 2) | `CUDA_VISIBLE_DEVICES=N python experiments/08_density_ablation.py --seed 1234 --densities 4000,6000 --out-suffix _ext` | ~half a day per seed |
| `profiles_only` U-Net (Phase 1) | `CUDA_VISIBLE_DEVICES=N python experiments/27_oi_vs_unet.py --train-profiles-only` | completes the M1 like-for-like row |

## 7. Next week

1. Run the three GPU items above; 3 seeds for anything headline.
2. Answer H1/H2 in [ssh_ablation.md](ssh_ablation.md) **without editing the
   pre-registered §3**.
3. Phase 4.4 loss study — per-modality renormalisation (cBottle idea 1) is the
   cheapest concrete answer to the advisor's "look into the loss function".
4. Begin Phase 5 (M2) scoping: EN4 profiles + OISST for the North Atlantic. The
   protocol shift needs thinking through *before* any download — the anomaly
   reference becomes WOA23, the floor becomes "predict WOA23", and the
   train/test split must be by **WMO float ID**, never by profile.
5. `oi.py` needs no changes for real data (it consumes lat/lon/values only), but
   γ will have to be re-tuned — real profiles carry instrument and
   representativeness error the OSSE's noise-free columns do not.

## 8. Files

**New**: `src/ocean_tokenizer/{oi,ssh}.py` · `tests/test_{oi,ssh,unet_channels_ssh}.py`
· `experiments/{26_oi_tuning,27_oi_vs_unet,28_make_ssh,29_ssh_ablation,30_oi_report,31_density_powerlaw,merge_density_json}.py`
· `experiments/run_oi_queue.sh` · `doc/architecture_spec.md`
· `reports/{oi_tuning,oi_baseline,reading_cbottle,ssh_ablation,density_4000_6000,intern_week1}.md`

**Modified**: `config.py` (ROOT via `OCEAN_ROOT`) · `baselines.py` (additive
`ssh` cfg token) · `08_density_ablation.py` (`--out-suffix`) · `.gitignore`.

**Not touched** (paper line, per the plan): `src/dfs_attention/`, `ICLR2027_*`,
`PlanB_*`, `reports/new.md`.
