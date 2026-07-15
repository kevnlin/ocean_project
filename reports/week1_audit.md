# Week-1 Audit — Corrected Baseline Matrix

- **Target:** train-only monthly CESM2 anomaly (was: absolute z-scored field)
- **Headline metric:** unobserved-only RMSE — 3.54% of ocean columns (observed profiles) excluded from scoring
- **Floor:** train-only CESM2 climatology = **0.535 degC / 0.125 PSU** (the real floor; WOA prior is bias-dominated and reported separately)
- **Split:** 312 train / 12 test months, seed 1234, 1500 profiles/month
- **Commit:** `a4bcf76ffadf`  ·  smoke=False

The two right-most columns are the honest numbers. The two 'all-ocean' columns include observed columns and are inflated by leakage — the gap between them and the unobserved columns is the leakage the earlier sweep was scoring on.

| method | config | TEMP unobs | SALT unobs | TEMP all-ocean | SALT all-ocean |
|---|---|---|---|---|---|
| woa_prior | woa_only | **1.5672** | **0.6343** | 1.5672 | 0.6343 |
| clim_floor | train_clim | **0.5350** | **0.1249** | 0.5351 | 0.1249 |
| nearest | profiles_only | **1.0211** | **0.4433** | 1.0027 | 0.4353 |
| nearest | profiles_woa | **0.9593** | **0.3105** | 0.9421 | 0.3049 |
| mlp | profiles_only | **0.3914** | **0.0869** | 0.3849 | 0.0854 |
| mlp | woa_only | **0.5350** | **0.1249** | 0.5352 | 0.1249 |
| mlp | profiles_woa | **0.3888** | **0.0871** | 0.3824 | 0.0856 |
| mlp | profiles_woa_surf | **0.2797** | **0.0496** | 0.2756 | 0.0489 |
| unet_depthwise | profiles_only | **0.1723** | **0.0467** | 0.1694 | 0.0460 |
| unet_depthwise | woa_only | **0.5361** | **0.1251** | 0.5363 | 0.1250 |
| unet_depthwise | profiles_woa | **0.1704** | **0.0464** | 0.1676 | 0.0456 |
| unet_depthwise | profiles_woa_surf | **0.1474** | **0.0307** | 0.1450 | 0.0302 |
| unet_joint | profiles_only | **0.2090** | **0.0554** | 0.2064 | 0.0547 |
| unet_joint | woa_only | **0.5353** | **0.1249** | 0.5354 | 0.1249 |
| unet_joint | profiles_woa | **0.2064** | **0.0539** | 0.2042 | 0.0532 |
| unet_joint | profiles_woa_surf | **0.1792** | **0.0380** | 0.1775 | 0.0377 |

### Skill vs train-only climatology floor (unobserved-only)

Skill = 1 − RMSE / RMSE_floor (positive = beats its own climatology).

| method | config | TEMP skill | SALT skill |
|---|---|---|---|
| nearest | profiles_only | -0.909 | -2.550 |
| nearest | profiles_woa | -0.793 | -1.486 |
| mlp | profiles_only | +0.268 | +0.304 |
| mlp | woa_only | -0.000 | -0.000 |
| mlp | profiles_woa | +0.273 | +0.303 |
| mlp | profiles_woa_surf | +0.477 | +0.603 |
| unet_depthwise | profiles_only | +0.678 | +0.626 |
| unet_depthwise | woa_only | -0.002 | -0.001 |
| unet_depthwise | profiles_woa | +0.681 | +0.628 |
| unet_depthwise | profiles_woa_surf | +0.725 | +0.754 |
| unet_joint | profiles_only | +0.609 | +0.556 |
| unet_joint | woa_only | -0.001 | -0.000 |
| unet_joint | profiles_woa | +0.614 | +0.569 |
| unet_joint | profiles_woa_surf | +0.665 | +0.696 |
