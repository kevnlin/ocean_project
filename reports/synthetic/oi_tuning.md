# OI hyperparameter tuning (protocol_v1)

*Selection split: **val** months (6 snapshots: 2008-01, 2008-05, 2008-09, 2009-03, 2009-07, 2009-11). Seed 1234, 1500 profiles/month. Commit `ac90064d`. 1.02 CPU-hours.*

Hyperparameter selection **is** model selection, so under protocol_v1 it uses the validation months (2008-2010). The intern plan said "training months only"; protocol_v1 is the stricter rule and the frozen protocol wins. The 12 pinned test months are never touched here.

Metric: unobserved-only anomaly RMSE, squared errors pooled over every scored cell of every month before taking the root (identical pooling to `metrics.evaluate_masked` on a stacked array).

## Stage A — TEMP (degC), k = 20

| L_km \ gamma | 0.03 | 0.1 | 0.3 |
|---|---|---|---|
| 300 | 0.2337 | 0.2388 | 0.2546 |
| 500 | 0.2099 | **0.2049** | 0.2112 |
| 800 | 0.2190 | 0.2184 | 0.2254 |
| 1200 | 0.2299 | 0.2363 | 0.2483 |

Optimum: **L = 500 km, gamma = 0.1** → 0.2049 degC.

## Stage A — SALT (PSU), k = 20

| L_km \ gamma | 0.03 | 0.1 | 0.3 |
|---|---|---|---|
| 300 | 0.0559 | 0.0568 | 0.0597 |
| 500 | 0.0530 | **0.0522** | 0.0539 |
| 800 | 0.0555 | 0.0558 | 0.0579 |
| 1200 | 0.0586 | 0.0605 | 0.0635 |

Optimum: **L = 500 km, gamma = 0.1** → 0.0522 PSU.

## Stage B — TEMP: neighbour count k at the stage-A optimum

| k | RMSE |
|---|---|
| 10 | 0.2034 **(chosen)** |
| 20 | 0.2049 |
| 40 | 0.2054 |

## Stage B — SALT: neighbour count k at the stage-A optimum

| k | RMSE |
|---|---|
| 10 | 0.0521 **(chosen)** |
| 20 | 0.0522 |
| 40 | 0.0524 |

## Frozen hyperparameters

| variable | L_km | gamma | k | selection RMSE |
|---|---|---|---|---|
| TEMP | 500 | 0.1 | 10 | 0.2034 degC |
| SALT | 500 | 0.1 | 10 | 0.0521 PSU |

These are the values `27_oi_vs_unet.py` reads from `outputs/cache/oi_tuning_val.json`. **Frozen** — changing them requires re-running this sweep and re-freezing.

## Stability check — the same sweep on training months

If the optimum moved between splits, the choice would be fitting noise rather than the covariance structure of the ocean.

| variable | val optimum | train optimum | agree? |
|---|---|---|---|
| TEMP | L=500, gamma=0.1 | L=500, gamma=0.1 | yes |
| SALT | L=500, gamma=0.1 | L=500, gamma=0.1 | yes |

---

Rerun: `python experiments/26_oi_tuning.py --split val`
