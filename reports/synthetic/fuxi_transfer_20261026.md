# KA-Ocean against a FuXi-Ocean-style model, and both in transfer to GLORYS12

KA-Ocean x1.375 (the size study's choice) and a FuXi-Ocean-style model (Huang et al., arXiv:2506.03210, re-implemented: its code and weights are unreleased), trained alike on `cesm2_vlong_uniform` 1904-2003: 200,000 steps of 1,024 queries, leads 0, 1, 3 and 6 months in one model, the checkpoint chosen on 2004. Spec: `docs/superpowers/specs/2026-10-26-fuxi-glorys-comparison-design.md`.

## Models

| model | parameters | trained on | seeds | hours a run | best steps |
|---|---|---|---|---|---|
| KA-Ocean | 17,730,974 | 100 CESM2 years, 200k steps | 3 | 6.6 ± 0.6 | 198000, 180000, 170000 |
| FuXi-Ocean-style | 17,508,596 | 100 CESM2 years, 200k steps | 3 | 4.8 ± 0.1 | 186000, 174000, 188000 |
| g_ka (KA-Ocean, GLORYS-trained) | 9,422,894 | 20 GLORYS years, 100k steps | 3 | | |

## Verdicts

* CESM2 2005 (held out): **KA better**; score KA-Ocean (17.7M) 0.7207, FuXi-Ocean-style (17.5M) 0.8634, larger seed sd 0.0161.
* GLORYS12 2019-2020 (transfer): **FuXi-style better**; score KA-Ocean (17.7M) 0.9648, FuXi-Ocean-style (17.5M) 0.9360, larger seed sd 0.0066.

Score: mean over leads of the per-variable standardised RMSE (per-depth z units of the scored cohort), mean over seeds; a difference counts only beyond the larger seed sd.

## CESM2 2005 (held out): temperature RMSE (°C)

| lead | persisted OI | climatology | KA-Ocean (17.7M) | FuXi-Ocean-style (17.5M) |
|---|---|---|---|---|
| 0 | 0.0860 | 0.6528 | 0.0526 ± 0.0011 | 0.1329 ± 0.0009 |
| 1 | 0.3101 | 0.6528 | 0.2473 ± 0.0023 | 0.3001 ± 0.0075 |
| 3 | 0.5235 | 0.6528 | 0.4028 ± 0.0035 | 0.4657 ± 0.0131 |
| 6 | 0.6794 | 0.6528 | 0.5214 ± 0.0091 | 0.5614 ± 0.0231 |

## CESM2 2005 (held out): salinity RMSE (PSU)

| lead | persisted OI | climatology | KA-Ocean (17.7M) | FuXi-Ocean-style (17.5M) |
|---|---|---|---|---|
| 0 | 0.0226 | 0.1119 | 0.0132 ± 0.0003 | 0.0251 ± 0.0000 |
| 1 | 0.0529 | 0.1119 | 0.0412 ± 0.0002 | 0.0464 ± 0.0007 |
| 3 | 0.0902 | 0.1119 | 0.0649 ± 0.0006 | 0.0702 ± 0.0015 |
| 6 | 0.1176 | 0.1119 | 0.0843 ± 0.0013 | 0.0868 ± 0.0024 |

## GLORYS12 2019-2020 (transfer): temperature RMSE (°C)

| lead | persisted OI | climatology | g_ka (KA-Ocean, GLORYS-trained) (9.4M) | KA-Ocean (17.7M) | FuXi-Ocean-style (17.5M) |
|---|---|---|---|---|---|
| 0 | 0.5577 | 0.8094 | 0.3392 ± 0.0021 | 0.5512 ± 0.0053 | 0.4764 ± 0.0047 |
| 1 | 0.6796 | 0.8094 | 0.5772 ± 0.0006 | 0.6688 ± 0.0027 | 0.6790 ± 0.0047 |
| 3 | 0.8139 | 0.8094 | 0.7250 ± 0.0027 | 0.8090 ± 0.0106 | 0.8240 ± 0.0134 |
| 6 | 0.8772 | 0.8094 | 0.7903 ± 0.0032 | 0.8715 ± 0.0174 | 0.8708 ± 0.0192 |

## GLORYS12 2019-2020 (transfer): salinity RMSE (PSU)

| lead | persisted OI | climatology | g_ka (KA-Ocean, GLORYS-trained) (9.4M) | KA-Ocean (17.7M) | FuXi-Ocean-style (17.5M) |
|---|---|---|---|---|---|
| 0 | 0.1186 | 0.1808 | 0.0787 ± 0.0003 | 0.1186 ± 0.0028 | 0.1010 ± 0.0001 |
| 1 | 0.1375 | 0.1808 | 0.1179 ± 0.0004 | 0.1320 ± 0.0005 | 0.1346 ± 0.0001 |
| 3 | 0.1649 | 0.1808 | 0.1467 ± 0.0006 | 0.1608 ± 0.0012 | 0.1639 ± 0.0006 |
| 6 | 0.1780 | 0.1808 | 0.1589 ± 0.0013 | 0.1742 ± 0.0023 | 0.1763 ± 0.0011 |

g_ka trained on 20 GLORYS years (the in-domain reference); KA-Ocean and FuXi-Ocean-style never saw GLORYS.

## Transfer cost: GLORYS RMSE as a ratio to g_ka's

| lead | KA-Ocean (17.7M) temperature | KA-Ocean (17.7M) salinity | FuXi-Ocean-style (17.5M) temperature | FuXi-Ocean-style (17.5M) salinity |
|---|---|---|---|---|
| 0 | 1.625 | 1.507 | 1.405 | 1.284 |
| 1 | 1.159 | 1.120 | 1.176 | 1.142 |
| 3 | 1.116 | 1.096 | 1.137 | 1.117 |
| 6 | 1.103 | 1.097 | 1.102 | 1.110 |

## Protocol and asymmetries

* Equal: data, inputs (one month of profiles and surface grids), steps, queries, objective, optimiser, selection on 2004, the read-out after the state. FuXi-Ocean-style has no kriging read-out (KA's own).
* Left out of FuXi-Ocean: Mixture-of-Time (one input month), autoregression (one pass per lead), the Charbonnier loss and its two training stages. Its own size is unpublished; it is matched to KA's.
* KA-Ocean x1.375 won a three-size selection on 2004 (gain 0.0015, 0.2 %); FuXi-Ocean-style has one size.
* One optimiser rule for both: AdamW's weight decay 0.01 also reaches FuXi-Ocean-style's position embedding, attention temperatures and position-bias MLP, which Swin-v2 usually exempts.
* GLORYS: a declared extra pass on the sealed 2019-2020 cells; GLORYS's own statistics; g_ka's rows are the 2026-10-22 main pass's, not re-scored. g_ka is smaller and trained half as long: a reference, not a competitor.
