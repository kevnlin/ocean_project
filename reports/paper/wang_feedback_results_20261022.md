# Results of the post-meeting experiments (2026-10-22, 04:30)

Spec: `docs/superpowers/specs/2026-10-21-wang-feedback-experiments.md`. Every run 100,000 steps, three
seeds, checkpoint chosen on the selection years. Numbers below are SELECTION-YEAR numbers (CESM2
2004-2005, real 2021), not test numbers; no sealed test has been opened for any of them.

## CESM2 (selection 2004-2005; score = mean standardised RMSE over leads, lower is better)

| arm | what changes | score | ratio to c_flow100 | T lead 0 | T lead 1 | T lead 3 | T lead 6 | S lead 0 | S lead 6 | best steps |
|---|---|---|---|---|---|---|---|---|---|---|
| A at 70k (paper) | - | 0.8057 | 1.016 | 0.0598 | | | 0.5601 | | | 66-70k |
| c_flow100 | A at 100k steps (comparison basis) | 0.7928 ± 0.0035 | 1.000 | 0.0579 | 0.2792 | 0.4503 | 0.5556 | 0.0134 | 0.0899 | 90-100k |
| c_deep | Gaussian kernel in a learned feature space (fixed init, ka-dev8) | 0.8043 ± 0.0056 | 1.015 | 0.0587 | 0.2842 | 0.4592 | 0.5572 | 0.0131 | 0.0910 | 72-94k |
| c_product | flow kernel x feature Gaussian (fixed init, ka-dev8) | 0.7905 ± 0.0006 | 0.997 | 0.0560 | 0.2765 | 0.4486 | 0.5561 | 0.0132 | 0.0900 | 70-100k |
| c_multiscale | two flow kernels (second 3x longer) | 0.7920 ± 0.0054 | 0.999 | 0.0572 | 0.2766 | 0.4498 | 0.5558 | 0.0137 | 0.0901 | 82-100k |
| c_mv | T and S solved jointly (coregionalisation) | 0.7937 ± 0.0049 | 1.001 | 0.0579 | 0.2800 | 0.4547 | 0.5574 | 0.0134 | 0.0896 | 90-96k |
| c_density | 1,520-6,080 input profiles a step | 0.7983 ± 0.0030 | 1.007 | 0.0591 | 0.2789 | 0.4530 | 0.5588 | 0.0136 | 0.0896 | 78-100k |
| c_draws | three draws a month (cohort aug3) | 0.8000 ± 0.0045 | 1.009 | 0.0595 | 0.2794 | 0.4530 | 0.5572 | 0.0139 | 0.0902 | 74-100k |
| c_both | aug3 + random density | 0.7938 ± 0.0029 | 1.001 | 0.0580 | 0.2773 | 0.4484 | 0.5588 | 0.0133 | 0.0903 | 86-100k |
| **c_y154** | **154 training years (1850-2003)** | **0.7495 ± 0.0021** | **0.945** | 0.0571 | **0.2565** | **0.3987** | **0.5075** | 0.0135 | **0.0850** | 86-96k |

Reading:
* 100k steps improve A by 1.6 % on the selection score (the 70k A was nearly converged; the gain is
  mostly at lead 0: 0.0598 -> 0.0579).
* None of the three learned covariances, nor the multivariate solve, beats the flow kernel beyond
  the seed spread (sd 0.003-0.006). The product kernel is 0.3 % better overall and 3 % better at lead 0
  (0.0560 vs 0.0579), within one standard deviation of the score. The feature-space kernel is 1.5 %
  worse. Augmentation (random density, more draws) does not help at the full 40-year scale.
* The 154-year window is the only large effect: -5.5 % on the score, -8 % / -11.5 % / -8.7 % in
  temperature at leads 1/3/6, -5.5 % in salinity at lead 6; lead 0 only -1.4 %. More simulated years
  mainly buy forecast skill.

Protocol consequence: by the pre-registered rule (a variant enters the paper only if it beats the
comparison basis beyond the seed spread, and a changed main model needs the new sealed CESM2
2008-2010 period), only c_y154 (and c_flow100 as its basis) go to the sealed v2 test, which is being
built (`docs/superpowers/specs/2026-10-22-sealed-cesm2-test2.md`, not opened). The kernel,
multivariate and augmentation rows are reported as selection-year ablations.

## Real Argo (selection 2021)

| model | training years | score | T 0 | T 1 | T 3 | T 6 | S 0 | S 6 | best step |
|---|---|---|---|---|---|---|---|---|---|
| KA-Ocean (paper) | 2016-2020, 30k steps | 0.8801 | 0.8332 | 0.9737 | 1.0563 | 1.0863 | 0.1400 | 0.1760 | 11-16k |
| r_y10 | 2011-2020, 100k steps | 0.8697 | 0.8103 | 0.9601 | 1.0414 | 1.0768 | 0.1377 | 0.1757 | 26-41k |

Ten real years instead of five: -1.2 % score, -2.7 % at lead 0, -0.9 % at lead 6 (same normalisation,
same 2021 cells; checks in `ka-dev7/outputs/real_y10_20261021/checks.json`). A declared extra pass of
r_y10 on the opened 2024 test is being scored (`ka-dev7/outputs/real_y10_20261021/sealed_2024/`).

## Controlled GP benchmark with non-Gaussian truths (151; RMSE, 3 seeds; oracle = exact GP with the true covariance)

| truth | OI | softmax | KA stationary | KA flow | KA deep (fixed init) | KA product (fixed init) | KA multiscale | oracle |
|---|---|---|---|---|---|---|---|---|
| flow (paper's control) | 0.4897 | 0.5150 | 0.4892 | 0.3873 | 0.4061 | 0.3875 | 0.3873 | 0.3846 |
| warped (curved lobes) | 0.3950 | 0.4949 | 0.3913 | **0.3466** | 0.3586 | 0.3466 | 0.3466 | 0.3455 |
| exponential (cusp) | 0.7450 | 0.7544 | 0.7448 | 0.7434 | 0.7442 | 0.7434 | **0.7378** | 0.7370 |
| mixture (two scales) | 0.3032 | 0.4130 | 0.3020 | 0.3019 | 0.3020 | 0.3019 | **0.3001** | 0.3000 |

The control reproduces 125's numbers to all printed digits. On the curved-lobe truth the flow kernel
reaches the oracle (ratio 1.003) while tuned OI and the stationary kernel sit 13 % above it: at the
32-neighbour scale a curved correlation is locally a shear, which the per-point anisotropic matrices
represent. A Gaussian in a learned feature space (the "net instead of the Gaussian geometry") learns
features once its saddle initialisation is fixed, but ends between the stationary and the flow kernel.
The two-scale kernel closes the last 0.1-0.8 % on the cusp and two-scale truths. Softmax is the worst
read-out on every truth.

Note for the record: the first benchmark run found that with every feature at zero the feature
Gaussian's gradient is exactly zero (a saddle), so the first c_deep/c_product campaign runs were
discarded and rerun from ka-dev8 (commit 66cdd57, small random feature rows).

## Reference analyses at the held-out real floats (160/161; both products assimilate these floats)

| 2024 sealed test, RMSE | T 0 | T 1 | T 3 | T 6 | S 0 | S 6 |
|---|---|---|---|---|---|---|
| ARMOR3D product (analysis of the target month / persisted anomaly) | 0.681 | 1.002 | 1.210 | 1.286 | 0.1303 | 0.2072 |
| GLORYS12 product | 0.829 | 1.117 | 1.320 | 1.399 | 0.1355 | 0.2268 |
| KA-Ocean (paper) | 0.849 | 1.009 | 1.095 | 1.131 | 0.1399 | 0.1830 |
| Persisted OI (paper) | 0.997 | 1.079 | 1.153 | 1.198 | 0.1583 | 0.1864 |

Reference rows, not baselines. GLORYS12 (interim product after 2021, +0.18 degC bias) is within 2.4 %
of KA-Ocean at lead 0; both products' persisted anomalies are worse than the persisted OI from lead 1
and worse than climatology by lead 3. Full tables: `outputs/reference_products_20261021/reference_rows.md`.

## Still running (2026-10-22 04:30)

* GLORYS12 OSSE: 264 training/selection months downloaded and reduced; cohort, caches, OI, queue
  (GPUs 0/1/5/6) and the sealed 2019-2020 test cohort are being built.
* Sealed CESM2 2008-2010 v2 test for c_flow100 vs c_y154: being built (check only; opened on request).
* Kevin's k100 (OI-anchored transformer, 4 vs 40 years, 100k): at ~90k steps.
