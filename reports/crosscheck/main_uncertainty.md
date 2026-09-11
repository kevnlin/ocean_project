# P6 — Post-hoc predictive uncertainty

<!-- generated alongside experiments/real_data/54_argo_uncertainty.py -->

> **Two tracks.** Track B independently recomputes the calibration metrics
> below with a Monte-Carlo CRPS and empirical quantile counts, sharing no
> formula with the closed-form Gaussian CRPS used here — see
> [`crosscheck_uncertainty.md`](crosscheck_uncertainty.md).
> Frozen real-Argo base checkpoints; scored on WMO-disjoint held-out floats.

The plan approves uncertainty as a **post-hoc module only** and states the
constraint plainly: *"the uncertainty module may not change the registered mean
prediction."* `ocean_tokenizer.uncertainty` makes that structural rather than
procedural — the base is `requires_grad_(False)` in eval, the head consumes
**detached** features, and `forward` returns the base model's own tensor for the
mean, so bit-identity holds by construction. This report verifies the structure
survives contact with real trained checkpoints and a real optimizer.

Head input: the frozen mean prediction, the query geometry, and the DFS
evidence total `sum(omega)` for the sample — evidence is the natural predictor
of confidence, since it is exactly the quantity that says how well-observed a
neighbourhood is. Trained by Gaussian NLL, which is proper.

## gulfstream

Guarantees across 3 seeds: base checkpoint unchanged, mean bit-identical, zero gradient into base — **ALL PASS**

| input density | 90% coverage | 95% coverage | spread-skill slope | CRPS (TEMP) | mean sigma |
|---|---:|---:|---:|---:|---:|
|  100% | 0.866 | 0.908 | 1.09 | 0.2865 | 0.458 |
|   75% | 0.891 | 0.929 | 1.09 | 0.2870 | 0.500 |
|   50% | 0.920 | 0.955 | 1.13 | 0.2898 | 0.564 |
|   25% | 0.957 | 0.977 | 1.26 | 0.2999 | 0.685 |
|   10% | 0.970 | 0.983 | 0.96 | 0.3178 | 0.808 |

## npac_gyre

Guarantees across 3 seeds: base checkpoint unchanged, mean bit-identical, zero gradient into base — **ALL PASS**

| input density | 90% coverage | 95% coverage | spread-skill slope | CRPS (TEMP) | mean sigma |
|---|---:|---:|---:|---:|---:|
|  100% | 0.931 | 0.959 | 0.86 | 0.1851 | 0.376 |
|   75% | 0.950 | 0.971 | 0.79 | 0.1876 | 0.420 |
|   50% | 0.969 | 0.984 | 0.63 | 0.1962 | 0.496 |
|   25% | 0.986 | 0.994 | 0.34 | 0.2166 | 0.630 |
|   10% | 0.993 | 0.997 | 0.25 | 0.2421 | 0.769 |

## What the sparsity sweep found

This is the column the plan asks for and it is not decoration. **The calibrator
is fitted at full observation density and does not transfer to sparse input.**
As profiles are removed, coverage climbs *above* nominal and the spread-skill
slope falls — the head keeps widening its intervals faster than the error
actually grows, so it becomes over-dispersed rather than over-confident.

The effect is mild in the Gulf Stream (slope stays near 1) and severe in the
North Pacific gyre, where the slope collapses to ~0.25 at 10 % density. A
practical consequence: an interval quoted by this head at one observing density
should not be believed at another without re-fitting, and the sparsity column
is what makes that visible.

Over-dispersion is the safer failure direction — the intervals are too wide, not
too narrow — but it is still miscalibration, and it is worth saying which way it
fails rather than reporting a single well-behaved number at full density.

## Caveats

1. Track B recomputes CRPS and coverage independently and agrees; it does not
   independently re-fit the calibrator, so the head itself is single-track.
2. Three seeds; the between-seed spread in slope (0.96–1.22 Gulf Stream,
   0.59–1.20 gyre) is comparable to the effect being measured at full density.
3. The head is small (a 7-feature input) by design — it is a calibrator, not a
   second model. A richer feature set is the obvious next step if the sparsity
   transfer matters.
