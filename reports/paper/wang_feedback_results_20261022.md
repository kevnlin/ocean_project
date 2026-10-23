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

### Declared extra pass of r_y10 on the opened 2024 test (ka-dev7 165; scored once; cells, persisted OI and climatology bit-identical to the main pass)

| 2024, RMSE | T 0 | T 1 | T 3 | T 6 | S 0 | S 1 | S 3 | S 6 |
|---|---|---|---|---|---|---|---|---|
| KA-Ocean (paper, 2016-2020) | 0.849 | 1.009 | 1.095 | 1.131 | 0.1399 | 0.1633 | 0.1762 | 0.1830 |
| KA-Ocean 10 years (2011-2020), mean of 3 | 0.841 | 1.002 | 1.094 | 1.139 | 0.1401 | 0.1637 | 0.1770 | 0.1839 |
| sd over seeds | 0.003 | 0.010 | 0.010 | 0.012 | 0.0012 | 0.0020 | 0.0010 | 0.0018 |

Ten real years help reconstruction (lead 0 T -1 %, about 3 sd) and not the forecasts (lead 6 slightly
worse, within 1 sd; seed spread at leads 1-6 is 2-4x the paper model's). Real data stays
data-limited in a different way than CESM2: more years of the same floats do not add forecast skill.
Not a replacement for the paper's real model; usable as a data-efficiency remark.

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

## Kevin's k100: the OI-anchored transformer at 100k steps (finished 05:28)

| baseline | window | score | T 0 | T 6 | S 0 | S 6 | best steps |
|---|---|---|---|---|---|---|---|
| f_y40 at 70k (paper's baseline row) | 40 y | 0.8212 | 0.0757 | 0.5810 | | | 46-68k |
| k100_f_y40 at 100k | 40 y | 0.8319 | 0.0745 | 0.5738 | 0.0191 | 0.0948 | 62-78k |
| k100_f_y4 at 100k | 4 y | 0.9945 | 0.1031 | 0.7038 | 0.0215 | 0.1134 | 14-20k |

The baseline does not gain from 100k steps (its chosen checkpoints are at 62-78k; the k100 runs
validate on 2004 alone with 2005 held out, so the scores are not strictly the paper's protocol, but
the ordering is clear). The 4-year window overfits by 14-20k steps, 19 % worse. So the shared budget
question is settled: KA-Ocean gains 1.6 % from 100k, the baseline nothing; the paper's 70k baseline
row stands, and the 100k KA model only enters through the sealed v2 test together with c_y154.

## GLORYS12 OSSE (third simulator; selection years = GLORYS 2017-2018; 100k steps, 3 seeds; finished 10:15)

| model | score | T 0 | T 1 | T 3 | T 6 | S 0 | S 6 |
|---|---|---|---|---|---|---|---|
| KA-Ocean (g_ka) | 0.7634 ± 0.0014 | 0.325 | 0.555 | 0.694 | 0.744 | 0.0722 | 0.1460 |
| softmax read-out | 0.7686 ± 0.0013 | 0.331 | 0.558 | 0.696 | 0.749 | 0.0733 | 0.1466 |
| ARMOR3D-style OI | 0.7754 ± 0.0031 | 0.337 | 0.566 | 0.700 | 0.747 | 0.0739 | 0.1472 |
| Aardvark-style (none) | 0.7853 ± 0.0022 | 0.360 | 0.567 | 0.700 | 0.756 | 0.0786 | 0.1476 |
| persisted OI | | 0.539 | | | 0.833 | | |

Same ordering as CESM2 and iHESP: KA-Ocean best at every lead and variable; lead-0 temperature
3.5 % below ARMOR3D-style OI, 40 % below the persisted OI. The sealed GLORYS 2019-2020 test
(cohort glorys12_test) is built and its scorer checked; it is opened only on request.

### Sealed GLORYS12 test, 2019-2020 (cohort 2006-2007), scored ONCE 2026-10-22 18:01 (Zihao approved; 12 checkpoints; never re-score)

RMSE on every held-out cell (729,600 per lead), mean of 3 seeds:

| model | T 0 | T 1 | T 3 | T 6 | S 0 | S 1 | S 3 | S 6 |
|---|---|---|---|---|---|---|---|---|
| KA-Ocean | **0.3392** | **0.5772** | **0.7250** | **0.7903** | **0.0787** | **0.1179** | **0.1467** | **0.1589** |
| softmax read-out | 0.3460 | 0.5800 | 0.7269 | 0.7915 | 0.0799 | 0.1184 | 0.1477 | 0.1602 |
| ARMOR3D-style OI | 0.3517 | 0.5879 | 0.7335 | 0.7940 | 0.0812 | 0.1199 | 0.1484 | 0.1609 |
| Aardvark-style | 0.3802 | 0.5900 | 0.7401 | 0.8200 | 0.0855 | 0.1207 | 0.1489 | 0.1612 |
| persisted OI | 0.5577 | 0.6796 | 0.8139 | 0.8772 | 0.1186 | 0.1375 | 0.1649 | 0.1780 |
| climatology | 0.8094 | 0.8094 | 0.8094 | 0.8094 | 0.1808 | 0.1808 | 0.1808 | 0.1808 |

Pre-registered comparison (KA vs best end-to-end baseline, lead 0): T -3.6 %, S -3.1 %. KA-Ocean is
best in 8/8 RMSE and 8/8 calibrated-CRPS settings (CRPS T 0.160/0.272/0.344/0.375 vs ARMOR3D-style
0.166/0.278/0.349/0.378). Seed sd at lead 0: 0.001-0.004. Same ordering as CESM2 and iHESP.

### Sealed CESM2 2008-2010 v2 test, scored ONCE 2026-10-22 18:20 (Zihao approved; 9 checkpoints; never re-score)

RMSE on every held-out cell (729,600 per lead), mean of 3 seeds (seed sd at lead 0 / lead 6 T: 0.0003-0.0006 / 0.008-0.009):

| model | training | T 0 | T 1 | T 3 | T 6 | S 0 | S 1 | S 3 | S 6 |
|---|---|---|---|---|---|---|---|---|---|
| A, 70k (paper) | 40 y | 0.0581 | 0.2529 | 0.3918 | 0.4769 | 0.0139 | 0.0430 | 0.0674 | 0.0845 |
| A, 100k (c_flow100) | 40 y | 0.0564 | 0.2519 | 0.3942 | 0.4834 | 0.0135 | 0.0424 | 0.0669 | 0.0846 |
| **A, 100k, 154 years (c_y154)** | 154 y | **0.0556** | **0.2392** | **0.3594** | **0.4292** | **0.0135** | **0.0415** | **0.0637** | **0.0780** |
| persisted OI | | 0.0842 | 0.2922 | 0.4785 | 0.5773 | 0.0220 | 0.0504 | 0.0835 | 0.1049 |
| climatology | | 0.5366 | 0.5366 | 0.5366 | 0.5366 | 0.1179 | 0.1179 | 0.1179 | 0.1179 |

c_y154 vs the paper's A: T -4.3 / -5.4 / -8.3 / -10.0 % at leads 0/1/3/6, S -2.9 / -3.5 / -5.5 / -7.7 %.
c_flow100 vs A: -2.9 % at lead 0 but +1.4 % at lead 6 (100k steps alone buy reconstruction, not
forecasts). The selection-year ordering is confirmed on a period no run had read: the 154-year
window is the one change that improves forecasts. Only these three models are on this period; every
baseline of the paper's main table remains on 2006-2007 at 40 years.

## Still running (2026-10-22 18:25)

* nothing; every planned run and both sealed tests are done.


* Sealed CESM2 2008-2010 v2 test for c_flow100 vs c_y154: built and checked; opened on request.
