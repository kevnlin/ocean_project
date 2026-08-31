# Phase 0 registration — operating point calibration

**Registered 2026-08-31, before any Phase 0 number was read.**
Companion to `work_plan.md` Phase 0. This file exists so the selection rule for
`s*` is on record ahead of the measurements, per the standing rule *"register
the rule before reading the number"*.

Git commit at registration: recorded by `29_dfs_operating_point.py` in its run
JSON (`git_commit_at_registration`), so the ordering is verifiable rather than
asserted.

---

## What is being calibrated

`batched_dfs.dfs_omega` computes whitened ridge leverage

    psi~_i  = psi_i / sqrt(lambda_i)
    A       = Psi~^T Psi~ + I
    omega_i = psi~_i^T A^-1 psi~_i

with `psi_i` the support-integrated random-Fourier features and
`lambda_i = noise_density_i * |support_i|`.

Before Phase 0 the support measure was **dimensionless unity**
(`godas_model.observation_mass` passed `torch.ones(N, 1)` as the quadrature
weight), so `lambda_i` was set entirely by the pilot constants
`NOISE_DENSITY_POINT = 0.08` / `NOISE_DENSITY_PATCH = 0.35` and the resulting
operating point was a side effect of a normalization constant rather than a
modelling choice.

After Phase 0 the quadrature weight is the token's **physical support area in
km²** and the noise density is an **observation-error variance density**
(variance per km²), so

    s_token = |psi~_i|^2 = |support_i| / noise_density_i          (units cancel)

is a deliberate, interpretable quantity: support area divided by error-variance
density.

## Quantities reported (the operating point)

Per run, on validation source months only:

| symbol | definition | why |
|---|---|---|
| `s_token` | `|psi~_i|^2` per token, before the joint solve | the per-token operating point the reparameterization controls |
| `omega_i` | whitened ridge leverage after the joint solve | the evidence actually used |
| `s_measured` | `mean(omega) / (1 - mean(omega))` | the plan's headline `s`, comparable to the pre-Phase-0 value |
| `sum omega` | total set evidence | bounded by `F`; shows headroom against the feature ceiling |
| `omega` histogram | per-modality | required by the standing rules |
| `trace(A^-1)` | — | `sum omega = F - trace(A^-1)`; separates "capped by F" from "capped by lambda" |

## Selection rule for `s*` — registered in advance

`s` is swept by scaling the per-modality noise-error variance densities by a
common factor `c`, holding geometry fixed, over a grid registered in the script
(`SWEEP_C`).

**Primary rule.** `s*` is the smallest `c` whose validation accuracy `J` is
within 1 standard error of the best `J` on the sweep (a one-standard-error rule,
biased toward the more conservative / higher-noise end). Validation `J` is the
macro score already used by the GODAS line, reported separately for temperature
and salinity per the standing rules, on validation source months only, with
three seeds.

**Explicitly excluded from selection.** Duplicate-collapse strength, the
`k = 8` discount, and any redundancy metric. Those are the *predictions* of the
calibration and are read only after `s*` is fixed. Selecting on them would be
choosing the mechanism on the metric it is supposed to predict.

**Tie-break.** If several `c` are within one standard error, prefer the one
closest to the physically-motivated prior (`c = 1`, i.e. the declared
observation-error variance densities taken at face value).

**Fallback if validation training cannot be run.** The geometry-only half of the
sweep — `s_token`, `omega`, `sum omega`, duplicate collapse — is computable
without training, because `omega` depends on support geometry, noise density and
variable group alone. If accuracy `J` is unavailable, `s*` is **not** selected;
the geometry sweep is reported as a calibration curve and `s*` selection is
deferred, rather than substituting a geometry-only criterion for the registered
accuracy criterion.

## Exit criteria (from the work plan, restated)

1. `29_dfs_operating_point.py` reports `s = O(1)` and emits no degeneracy warning.
2. `s*` registered with the selection rule, timestamped before the numbers.
3. Conservative refinement (P2', mass preservation) still holds under the new
   units — asserted by test, not by inspection.
4. RFF convergence table on record at the raised feature count.

## Decision point (three legitimate outcomes, from the plan)

| outcome | reading | next |
|---|---|---|
| duplicate discount at `k = 8` rises substantially and validation `J` holds | calibration was the blocker | Phase 1 |
| discount rises but `J` regresses materially | a real accuracy/robustness trade-off | that curve *is* the result; reframe and proceed |
| discount does not respond to `s` | the audit's diagnosis is wrong | stop, re-derive, bring numbers back |

Which outcome obtains is reported verbatim, pass or fail.
