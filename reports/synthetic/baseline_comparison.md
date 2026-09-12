# Baseline Comparison — Sparse-Profile Ocean-State Reconstruction

- **Ground truth:** CESM2-LE full simulation (held-out test months)
- **Train months:** 48  |  **Test months:** 12  |  **Synthetic Argo profiles/month:** 1500
- **Metric:** depth-banded RMSE (valid-cell-weighted, NaN-aware, ocean only). TEMP in °C, SALT in PSU. Lower is better.
- **New reference baselines** (`src/baselines/`, marked *(new)*): NeSPReSO PCA+MLP, OSnet-style 15×-ensemble MLP, Buongiorno-Nardelli stacked-LSTM. All three drop the SSH/ADT predictor (unavailable in this synthetic setup), so each is weakest in the 50–200 m thermocline band.

## TEMP RMSE (°C)
| Method             |surface (~5 m) | 0-50 m | 50-200 m | 200+ m|
**| 2D U-Net             | 0.3715 | 0.4751 | 0.5798 | 0.2894 |**
| NeSPReSO PCA+MLP       | 0.7059 | 0.7365 | 0.8674 | 0.4896 |
| OSnet MLP 15x-ens.     | 0.1111 | 0.3549 | 0.9449 | 0.5700 |
| Nardelli stacked-LSTM  | 0.3062 | 0.5428 | 0.8696 | 0.4815 |

## SALT RMSE (PSU)
| Method               |surface (~5 m) | 0-50 m | 50-200 m | 200+ m|
**| 2D U-Net              | 0.1126 | 0.1342 | 0.1036 | 0.0539 |**
| NeSPReSO PCA+MLP        | 0.1531 | 0.1623 | 0.1656 | 0.1075 |
| OSnet MLP 15x-ens       | 0.0227 | 0.1190 | 0.1791 | 0.1185 |
| Nardelli stacked-LSTM   | 0.2440 | 0.2555 | 0.1793 | 0.0972 |
