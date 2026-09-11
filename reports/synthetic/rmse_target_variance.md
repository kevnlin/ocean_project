# RMSE and target variance — reply to the external request

*Request: "for the most successful model you have, can you give me RMSE and
target variance for SST and SSH respectively". Generated 2026-08-09 from
`experiments/33_variance_table.py`; full per-level numbers in
`outputs/cache/variance_table.json`.*

---

## First, a mismatch worth resolving before the numbers are used

Our system **does not predict SST or SSH**. Under
[protocol_v1](protocol_v1.md) it predicts **TEMP and SALT as 3-D anomaly
fields** on 20 levels (5–985 m). SST, SSS and the pseudo-SSH are **inputs** —
dense satellite-analogue fields the model is *given*.

So "RMSE for SST" and "RMSE for SSH" do not exist as reconstruction skills here.
Two things do, and one of them is probably what is wanted:

| what was likely meant | what we can give |
|---|---|
| skill at the surface | RMSE + target variance at **level 0 (5 m)** — the SST analogue |
| a normaliser to compare across groups | **target variance per level / band / full column**, below |

⚠️ **The surface number is not a fair "SST reconstruction skill".** The model is
*handed* SST as an input channel, so reproducing the 5 m field is close to a
copy: RMSE 0.049 °C against a target std of 0.626 °C, i.e. 99.4 % explained
variance. Quoting that as reconstruction skill would badly overstate the method.
**The honest numbers are sub-surface**, where the model is genuinely
interpolating from sparse profiles.

---

## Definitions

* **Target** = the anomaly, `a = field − train-only monthly climatology`. That is
  what protocol_v1 asks the model to predict and what the RMSE measures.
* **Target variance** `var₀ = mean(a²)` — the second moment **about zero**. This
  is the relevant one: our reported floor is "predict zero anomaly", so
  `sqrt(var₀)` *is* the climatology-floor RMSE exactly.
  (`varₘ`, the variance about the target's own mean, is in the JSON; it differs
  only by the squared spatial-mean anomaly.)
* **Explained variance** `EV = 1 − MSE / var₀`.
* **Scoring set**: 12 pinned held-out test months, **unobserved ocean cells
  only** (profile columns excluded at every level).
* **Models**: `U-Net+SSH` = depthwise U-Net with the pseudo-SSH channel (our best,
  3-seed 0.1368 ± 0.0002 °C). `U-Net cert.` = the protocol_v1 certified model.
  OI = tuned optimal interpolation, for reference. **None of these is the
  shared-latent DFS/Perceiver model**, which is not competitive (~0.52).

---

## TEMP (°C)

| level / band | depth (m) | target std | target var | RMSE U-Net+SSH | RMSE U-Net cert. | RMSE OI | EV (+SSH) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L00 — SST analogue** | 5.0 | 0.6263 | 0.39228 | 0.0494 | 0.0490 | 0.1876 | 0.994 |
| L04 | 45.0 | 0.6467 | 0.41827 | 0.1579 | 0.1746 | 0.2503 | 0.940 |
| L08 | 105.0 | 0.7185 | 0.51621 | 0.1980 | 0.2332 | 0.3492 | 0.924 |
| L12 | 186.3 | 0.5144 | 0.26465 | 0.1501 | 0.1738 | 0.2192 | 0.915 |
| L16 | 408.8 | 0.2537 | 0.06436 | 0.0847 | 0.0991 | 0.1130 | 0.889 |
| L19 | 984.7 | 0.0826 | 0.00682 | 0.0335 | 0.0376 | 0.0401 | 0.835 |
| **0–100 m** | | 0.6485 | 0.42058 | 0.1401 | 0.1589 | 0.2509 | 0.953 |
| **100–300 m** | | 0.5844 | 0.34153 | 0.1631 | 0.1916 | 0.2622 | 0.922 |
| **300–max** | | 0.2140 | 0.04578 | 0.0718 | 0.0837 | 0.0953 | 0.887 |
| **FULL COLUMN** | | **0.5520** | **0.30475** | **0.1366** | **0.1580** | **0.2287** | **0.939** |

## SALT (PSU)

| level / band | depth (m) | target std | target var | RMSE U-Net+SSH | RMSE U-Net cert. | RMSE OI | EV (+SSH) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L00 — SSS analogue** | 5.0 | 0.2063 | 0.04257 | 0.0163 | 0.0161 | 0.0922 | 0.994 |
| L08 | 105.0 | 0.1284 | 0.01648 | 0.0396 | 0.0399 | 0.0490 | 0.905 |
| L16 | 408.8 | 0.0318 | 0.00101 | 0.0130 | 0.0139 | 0.0163 | 0.833 |
| **0–100 m** | | 0.1804 | 0.03255 | 0.0380 | 0.0393 | 0.0726 | 0.956 |
| **100–300 m** | | 0.0974 | 0.00949 | 0.0324 | 0.0332 | 0.0396 | 0.889 |
| **300–max** | | 0.0279 | 0.00078 | 0.0112 | 0.0119 | 0.0139 | 0.838 |
| **FULL COLUMN** | | **0.1305** | **0.01703** | **0.0316** | **0.0325** | **0.0528** | **0.941** |

*All 20 levels are in `outputs/cache/variance_table.json`.*

## Fields that are INPUTS, not targets

Variance of their anomaly over the same scoring set, for reference only — we
never predict these, so there is no RMSE to report:

| field | anomaly std | anomaly var | role |
|---|---:|---:|---|
| **SSH** (pseudo, steric height) | **0.0369 m = 3.69 cm** | 0.00136 m² | input channel, derived from T/S |
| SST | 0.6263 °C | 0.39228 | dense input |
| SSS | 0.2063 PSU | 0.04257 | dense input |

Note the SSH here is a **pseudo-SSH**: the steric/baroclinic component computed
from our own T/S fields via TEOS-10 dynamic height, referenced to 990 dbar. It
has no barotropic term and is not comparable to an altimeter product in absolute
terms ([ssh_ablation.md](ssh_ablation.md) §1).

---

## Suggested one-paragraph reply

> Our model predicts 3-D TEMP/SALT **anomalies**, not SST or SSH — those are
> inputs on our side, so we have no reconstruction RMSE for them. What we can
> give is RMSE against target variance per level. Full column, best model
> (depthwise U-Net + SSH channel), held-out test months, unobserved cells only:
> **TEMP RMSE 0.1366 °C against a target std of 0.5520 °C (var 0.3048), EV 0.939;
> SALT RMSE 0.0316 PSU against target std 0.1305 (var 0.0170), EV 0.941.**
> At the surface (5 m) it is 0.049 °C against 0.626 °C — but please don't read
> that as SST skill: the model is *given* SST, so the surface level is close to a
> copy. The meaningful numbers are sub-surface, e.g. the 100–300 m thermocline:
> **RMSE 0.1631 °C against target std 0.5844, EV 0.922.**
> For reference our pseudo-SSH input has an anomaly std of 3.69 cm. Happy to send
> the per-level table or re-run against a different target definition.

## Reproduce

```bash
python experiments/33_variance_table.py      # ~5 min, CPU
```
