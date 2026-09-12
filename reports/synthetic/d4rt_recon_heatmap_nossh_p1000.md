# 2-D T/S Reconstruction + Spatial Error Heatmap

Model: **DFS-Attention fusion over a Perceiver-IO latent + D4RT causal space-time query decoder** (`fusion.D4RTFusion`), lead 0 (reconstruction only), 425,863 parameters.

Data: CESM2-LE 1x1deg — train 2000-2003 (48 mo) / val 2004 (12 mo) / test 2005 (12 mo), 1000 synthetic Argo profiles per month.

Inputs: **profiles + surf + woa** (WOA23 decav91C0 monthly 1&deg;, the background DFS-Attention references its evidence against).

> **Caveat — short month window.** This run uses 48/12/12 train/val/test months, not protocol_v1's 276/36/12, so the numbers are not directly comparable with the tables elsewhere in `reports/`.

Scoring excludes the full column of every supplied profile (unobserved-only), so the model cannot score by echoing its input.

## Headline (test, full column)

| var | D4RT RMSE | climatology floor | skill |
|-----|-----------|-------------------|-------|
| TEMP | 0.3041 degC | 0.5898 degC | +48.4% |
| SALT | 0.0723 PSU | 0.1130 PSU | +36.0% |

## By depth band

| var | band | D4RT RMSE | floor | skill |
|-----|------|-----------|-------|-------|
| TEMP | 0-100m | 0.3473 | 0.7307 | +52.5% |
| TEMP | 100-300m | 0.3395 | 0.5901 | +42.5% |
| TEMP | 300-max | 0.0957 | 0.1385 | +30.9% |
| SALT | 0-100m | 0.1017 | 0.1594 | +36.2% |
| SALT | 100-300m | 0.0502 | 0.0778 | +35.5% |
| SALT | 300-max | 0.0145 | 0.0201 | +27.8% |

## Error heatmap by depth layer

![reconstruction error heatmap](fig_d4rt_recon_heatmap.png)

Rows are TEMP / SALT, columns are oceanographic layers plus the pooled column. **Darker = better reconstruction; brighter red / orange / yellow = more error.** Land and never-scored cells are grey. Each row shares one colour scale across its panels, so layers are directly comparable.

| var | panel | median RMSE | median floor |
|-----|-------|-------------|--------------|
| TEMP | 0-100 m | 0.2484 degC | 0.4831 degC |
| TEMP | 100-300 m | 0.2197 degC | 0.3140 degC |
| TEMP | 300-985 m | 0.0631 degC | 0.0825 degC |
| TEMP | full column | 0.2226 degC | 0.3844 degC |
| SALT | 0-100 m | 0.0592 PSU | 0.0949 PSU |
| SALT | 100-300 m | 0.0387 PSU | 0.0540 PSU |
| SALT | 300-985 m | 0.0095 PSU | 0.0128 PSU |
| SALT | full column | 0.0485 PSU | 0.0788 PSU |

## Against the climatology floor

![skill map](fig_d4rt_recon_skill.png)

Full-column reconstruction, the climatology floor on the same colour scale, and the skill ratio. Blue beats climatology, red is worse than it — the honest read of whether the model is adding anything over simply predicting the seasonal mean.

Per-cell values pool the levels in each layer x 12 test months (42,337 ocean cells scored), converted to physical units with each level's anomaly std, so a cell's number is a physical RMSE.

Run record: `outputs/cache/d4rt_recon_nossh_p1000_s1234.json`
