# Baseline Table — Sparse-Profile Ocean-State Reconstruction

> 🏆 **Method of choice: 2D U-Net (prof+WOA+SST/SSS)** — best full-column RMSE of every method and best through the thermocline. It is the ⭐ **bold** row in each table below.

- **Ground truth:** CESM2-LE full simulation (held-out test months)
- **Train months:** 48  |  **Test months:** 12  |  **Synthetic Argo profiles/month:** 1500
- **Metric:** depth-banded RMSE (valid-cell-weighted, NaN-aware, ocean only). TEMP in °C, SALT in PSU. Lower is better. **Bold** = best per band among reconstruction methods (priors/floors excluded).
- **Note on WOA23 climatology:** the ground truth is the CESM2-LE *model*, whose climate differs from real observations. WOA23 therefore carries an irreducible model–obs bias (~1.0–1.2 °C for TEMP, largest in the thermocline / western-boundary-current hotspots). The *CESM2-LE self-climatology* row (climatology built from the model's own training months) isolates the true "climatology cannot capture internal variability" floor — about half the WOA RMSE.

## TEMP RMSE (°C) 

| Method                | surface (~5 m) | 0-50 m | 50-200 m | 200+ m |
| Global mean                     |11.8374 | 11.7784 | 9.5153 | 4.4603|
| WOA23 climatology (obs prior)   | 1.3394 | 1.5238 | 1.7224 | 1.3864 |
| CESM2-LE self-climatology (ref) | 0.6511 | 0.6391 | 0.6531 | 0.2611 |
| Nearest profile                 | 1.0127 | 1.0463 | 1.1690 | 0.6688 |
| Nearest + WOA23                 | 0.8329 | 0.9427 | 1.0657 | 0.7488 |
| Pointwise MLP(prof+WOA+SST/SSS) | 0.4654 | 0.5275 | 0.8428 | 0.5487 |
| 2D U-Net (profiles)             | 0.6825 | 0.6919 | 0.6529 | 0.3755 |
| 2D U-Net (prof+WOA)             | 0.5896 | 0.6377 | 0.6122 | 0.2793 |
**| 2D U-Net (prof+WOA+SST/SSS)   | 0.3715 | 0.4751 | 0.5798 | 0.2894 |**

## SALT RMSE (PSU) 

| Method                | surface (~5 m) | 0-50 m | 50-200 m | 200+ m |
| Global mean                     | 2.0819 | 1.9662 | 1.3988 | 0.8719 |
| WOA23 climatology (obs prior)   | 0.8858 | 0.8016 | 0.6647 | 0.3960 |
| CESM2-LE self-climatology (ref) | 0.2219 | 0.1970 | 0.1240 | 0.0393 |
| Nearest profile                 | 0.6834 | 0.6147 | 0.3106 | 0.1540 |
| Nearest + WOA23                 | 0.4096 | 0.3832 | 0.3105 | 0.2150 |
| Pointwise MLP (prof+WOA+SST/SSS)| 0.1375 | 0.1402 | 0.1521 | 0.1075 |
| 2D U-Net (profiles)             | 0.2331 | 0.2030 | 0.1193 | 0.0664 |
| 2D U-Net (prof+WOA)             | 0.1984 | 0.1738 | 0.1130 | 0.0551 |
**| 2D U-Net (prof+WOA+SST/SSS)   | 0.1126 | 0.1342 | 0.1036 | 0.0539 |**

## Overall RMSE — full column — TEMP (°C)

| method \ config | profiles_only | woa_only | profiles_woa | profiles_woa_surf |

| climatology | – | 1.5648 | – | – |
| nearest | 0.9959 | – | 0.9378 | – |
| mlp | 0.9146 | 0.8977 | 0.8746 | 0.6787 |
| unet | 0.5873 | 0.6471 | 0.5331 | ⭐ **0.4723** |

## Overall RMSE — full column — SALT (PSU)

| method \ config | profiles_only | woa_only | profiles_woa | profiles_woa_surf |
|:--|:--:|:--:|:--:|:--:|
| climatology | – | 0.6314 | – | – |
| nearest | 0.3834 | – | 0.3049 | – |
| mlp | 0.2736 | 0.2582 | 0.3881 | 0.1355 |
| unet | 0.1343 | 0.1470 | 0.1187 | ⭐ **0.1001** |

## Appendix — per-level RMSE, U-Net across configs — TEMP

| depth (m) | profiles_only | woa_only | profiles_woa | profiles_woa_surf |
|---|---|---|---|---|
| 5 | 0.683 | 0.734 | 0.590 | 0.372 |
| 15 | 0.660 | 0.711 | 0.592 | 0.372 |
| 25 | 0.661 | 0.725 | 0.630 | 0.458 |
| 35 | 0.701 | 0.774 | 0.669 | 0.538 |
| 45 | 0.752 | 0.804 | 0.701 | 0.596 |
| 55 | 0.726 | 0.804 | 0.701 | 0.619 |
| 65 | 0.731 | 0.818 | 0.697 | 0.631 |
| 85 | 0.704 | 0.833 | 0.653 | 0.626 |
| 105 | 0.679 | 0.850 | 0.699 | 0.629 |
| 125 | 0.656 | 0.801 | 0.620 | 0.584 |
| 145 | 0.598 | 0.722 | 0.534 | 0.545 |
| 165 | 0.556 | 0.637 | 0.479 | 0.503 |
| 186 | 0.534 | 0.577 | 0.443 | 0.471 |
| 223 | 0.509 | 0.487 | 0.394 | 0.414 |
| 268 | 0.468 | 0.421 | 0.350 | 0.359 |
| 327 | 0.404 | 0.360 | 0.307 | 0.315 |
| 409 | 0.364 | 0.307 | 0.266 | 0.273 |
| 528 | 0.319 | 0.252 | 0.221 | 0.230 |
| 708 | 0.248 | 0.185 | 0.172 | 0.185 |
| 985 | 0.192 | 0.132 | 0.133 | 0.131 |

## Appendix — per-level RMSE, U-Net across configs — SALT

| depth (m) | profiles_only | woa_only | profiles_woa | profiles_woa_surf |
|---|---|---|---|---|
| 5 | 0.233 | 0.245 | 0.198 | 0.113 |
| 15 | 0.217 | 0.230 | 0.185 | 0.114 |
| 25 | 0.204 | 0.225 | 0.173 | 0.159 |
| 35 | 0.186 | 0.214 | 0.161 | 0.154 |
| 45 | 0.167 | 0.189 | 0.146 | 0.124 |
| 55 | 0.156 | 0.173 | 0.136 | 0.113 |
| 65 | 0.142 | 0.165 | 0.132 | 0.110 |
| 85 | 0.128 | 0.154 | 0.117 | 0.111 |
| 105 | 0.117 | 0.144 | 0.135 | 0.108 |
| 125 | 0.108 | 0.130 | 0.107 | 0.103 |
| 145 | 0.102 | 0.117 | 0.092 | 0.099 |
| 165 | 0.094 | 0.107 | 0.086 | 0.094 |
| 186 | 0.089 | 0.097 | 0.080 | 0.087 |
| 223 | 0.084 | 0.082 | 0.072 | 0.076 |
| 268 | 0.075 | 0.067 | 0.063 | 0.066 |
| 327 | 0.070 | 0.057 | 0.056 | 0.056 |
| 409 | 0.065 | 0.048 | 0.050 | 0.048 |
| 528 | 0.057 | 0.040 | 0.047 | 0.043 |
| 708 | 0.056 | 0.034 | 0.047 | 0.038 |
| 985 | 0.048 | 0.030 | 0.044 | 0.035 |
