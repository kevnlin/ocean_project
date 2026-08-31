# Phase 0 — operating point calibration: gate result

Registered rule: [`doc/phase0_registration.md`](../doc/phase0_registration.md),
commit `6f300e3`, committed **before** any number below was read.
Instrument: `experiments/29_dfs_operating_point.py`.
Run record: `outputs/cache/dfs_operating_point.json`,
`outputs/cache/dfs_operating_point_dup.json`.

> **Verdict: the gate does NOT open.** Calibrating the operating point does not
> lift the estimator out of the near-independent-vote regime. The duplicate
> discount stays at ~0.15–0.19 across five orders of magnitude of the noise
> scale. This is decision-point outcome 3 in the work plan — *"the discount does
> not respond to `s`; stop, re-derive, bring numbers back before spending
> more"* — with the qualification that it does respond, weakly, and saturates
> far below the level that would indicate real collapse.

---

## 1. What Phase 0 changed

| item | before | after |
|---|---|---|
| support measure | dimensionless `1` (`torch.ones(N,1)`) | **physical footprint in km²** |
| noise | pilot constants 0.08 / 0.35, dimensionless | **error-variance density as a noise area (km²)** |
| kernel length scales | `(0.35, 0.35, 0.25, 2.0)` in normalised box units | **`(1544 km, 973 km, 237 m, 2 months)`** — numerically equivalent |
| random features | 32 | **256** (`N_FEATURES_LEGACY = 32` retained for the frozen record) |

The per-token operating point is now readable:

```
s_token = |psi~|² = support_area / noise_area
```

— support area divided by error-variance density. The declared noise areas set
`s_token = 1` at each stream's nominal footprint (profile point 7 854 km²,
4×4 patch 212 093 km²), which is a *declared normalisation*, not a measured
GODAS error table. That is the honest successor to constants whose operating
point was a side effect.

**The reparameterisation exposed the accidental anisotropy the audit predicted.**
The same `0.35` means **1544 km in x but 973 km in y** — a factor 1.59 — purely
because the box spans 50° of longitude and 25° of latitude. It was never a
modelling choice. (Note: the work plan quotes 773 × 481 km, almost exactly half
these; that looks like a half-width convention. The values here are computed
from the box extent with a `cos(mean latitude)` correction, and the *ratio* —
the part that matters — agrees.)

## 2. Operating point: before vs after

6 source months, 664 live tokens each (384 profile points, 140 surface patches,
140 SSH patches).

| | legacy (F=32, unit support) | Phase 0 (F=256, km² support) |
|---|---|---|
| `s_token` mean | 8.34 | 0.95 |
| mean `omega` | 0.0432 | 0.0657 |
| `sum omega` | 28.68 / 32 — **89.6% of the ceiling** | 43.62 / 256 — **17.0%** |
| **`s_measured`** | **0.045** | **0.070** |

**The two ceilings were confounded, and Phase 0 separated them.** The legacy
configuration was sitting at **89.6% of the feature ceiling**: 664 tokens were
competing for 32 features' worth of evidence, so `omega` was crushed by the
*basis*, not by the noise. Raising `p` to 256 removed that constraint (17%
headroom). But `s_measured` only moved 0.045 → 0.070. **It was never the
binding constraint.**

## 3. The s-family — and why the sweep cannot rescue it

Sweeping the noise scale `c` (geometry only; no training needed, per the plan):

| `c` | `s_token` | mean `omega` | `sum omega` (% of F) | `s_measured` |
|---|---|---|---|---|
| 0.001 | 950 | 0.181 | 120.4 (47.0%) | **0.221** |
| 0.01 | 95 | 0.145 | 96.5 (37.7%) | 0.170 |
| 0.1 | 9.5 | 0.107 | 70.7 (27.6%) | 0.119 |
| 1 (prior) | 0.95 | 0.066 | 43.6 (17.0%) | 0.070 |
| 10 | 0.095 | 0.028 | 18.9 (7.4%) | 0.029 |
| 100 | 0.0095 | 0.007 | 4.6 (1.8%) | 0.007 |

**Driving `s_token` up by six orders of magnitude (0.0095 → 950) moves
`s_measured` by only 0.007 → 0.221.** It saturates.

The reason is structural. As noise falls, the ridge `I` becomes negligible and
`omega` approaches the OLS hat-matrix diagonal, which is **scale-invariant**:
`sum omega -> rank(Psi~) <= min(N, F)`. Lowering the noise raises *every*
token's evidence together, so no token gains *relative* leverage. With 664
tokens and F=256, `mean omega <= 0.385` and hence `s_measured <= 0.63` — and the
measured saturation at `sum omega ≈ 120` says the 664-token set spans only
**~120 effective dimensions**. Set-level redundancy *is* being detected; it is
simply not expressed as a per-token duplicate discount.

## 4. Duplicate collapse — the measurement the gate turns on

`k` bit-exact copies of one biased profile column. Ideal: group mass flat in `k`
(copies carry no new evidence). Independent-vote failure: group mass linear in
`k`, i.e. `k8/k1 = 8`.

| `c` | k=1 | k=2 | k=4 | k=8 | k8/k1 | discount |
|---|---|---|---|---|---|---|
| 0.001 | 2.841 | 4.776 | 8.473 | 18.914 | 6.657 | **0.192** |
| 0.1 | 1.719 | 3.095 | 5.378 | 11.795 | 6.860 | 0.163 |
| 1 (prior) | 1.091 | 2.049 | 3.565 | 7.563 | 6.934 | 0.152 |
| 10 | 0.484 | 0.947 | 1.685 | 3.399 | 7.024 | 0.139 |
| 100 | 0.117 | 0.233 | 0.444 | 0.872 | 7.431 | 0.081 |

**Eight identical copies retain 6.66–7.43 times one copy's evidence, against a
no-collapse maximum of 8.0.** The discount is monotone in the right direction —
lower noise, more collapse — but it plateaus at **0.19 in the zero-noise limit**.
The opening example of the paper (one float distributed via real-time and
delayed-mode streams collapsing toward one) is **not reproducible at any point
in the registered family.**

Why: for a token duplicated `k` times, group mass is `k·b/(1 + k·b)` with `b`
its leverage against the rest of the set. Collapse needs `b = O(1)`; here
`b ≈ 0.03`. And `b` cannot be raised by a uniform noise rescaling, for the
scale-invariance reason in §3. The constraint is the **token-count-to-rank
ratio**, not the noise level.

## 5. RFF convergence

Kernel RMSE against the exact Gaussian, 5 basis seeds, physical length scales:

| p | 32 | 64 | 128 | **256** | 512 | 1024 |
|---|---|---|---|---|---|---|
| RMSE | 0.164 | 0.121 | 0.081 | **0.055** | 0.039 | 0.030 |

The audit's premise is confirmed: at `p = 32` the kernel error is **16.4%**
while the effects being measured are ~1% — the numerics were an order of
magnitude louder than the signal. **But `p = 256` only reaches 5.5%, still well
above 1%.** The plan's raise to 256 is a real improvement and *not* sufficient
on its own; `p ≥ 1024` (3.0%) would be needed for the numerics to sit below the
effect size, and error falls as `p^-1/2`, so this is expensive.

## 6. Exit criteria

| # | criterion | status |
|---|---|---|
| 1 | `s = O(1)`, no degeneracy warning | **FAIL** — `s_measured = 0.070` at the prior, max 0.221 across the whole family |
| 2 | `s*` registered with the rule, timestamped before the numbers | rule registered (`6f300e3`); **`s*` not selected** — see below |
| 3 | conservative refinement (P2') preserved under new units | **PASS** — mass-conservation tests green |
| 4 | RFF convergence table on record | **PASS** — §5 |

`s*` is **not selected**. The registered rule picks `s*` on validation accuracy
`J` (one-standard-error rule, three seeds); `J` needs GODAS fields and training,
and no GODAS data is present on this machine. The registration's fallback clause
applies verbatim: report the calibration curve, defer `s*`, and do **not**
substitute a geometry-only criterion. Selecting on the duplicate discount was
excluded in advance and remains excluded — doubly so now that it is the failing
measurement.

## 7. Decision point — for the PI

Outcome 3 obtains: **the discount does not respond usefully to `s`.** The work
plan's instruction for this outcome is to stop and re-derive rather than proceed
to Phase 1.

What the diagnosis got right, and what it missed:

- **Right:** the numerics were badly under-resolved (16.4% kernel error at
  `p=32`), the support measure was a normalisation artefact, and the length
  scales were accidentally anisotropic. All three are now fixed or exposed.
- **Missed:** the audit attributed the dead mechanism to the *noise-side*
  operating point. The measurements say the binding constraint is
  **structural** — 664 tokens sharing a rank-≤256 feature space give each token
  leverage `b ≈ 0.03`, and duplicate collapse needs `b = O(1)`. That cannot be
  reached by any uniform rescaling of the noise, because ridge leverage becomes
  scale-invariant in the low-noise limit.

Three re-derivation directions, cheapest first, all testable with the
geometry-only harness that already exists:

1. **Raise the rank, not the evidence.** `mean omega <= F/N`. With N=664,
   `F >= 2N ≈ 1300` is needed for `mean omega ≈ 0.5`. §5 says large `p` is
   wanted for accuracy anyway. Both point the same way, and the sweep is free.
2. **Reduce N per solve.** Run the leverage solve *within* a neighbourhood or a
   modality rather than over all 664 tokens at once. Duplicates compete locally,
   which is where redundancy actually lives; this raises `b` without touching
   the basis. Note `dfs.py` already solves locally — the two formulations may
   simply disagree about this, and that comparison is now worth running.
3. **Reconsider whether per-token discount is the right target.** `sum omega`
   *does* fall to ~120 effective dimensions from 664 tokens, so set-level
   redundancy is measured correctly. If the claim is set-level, the paper's
   demonstration should be a set-level statistic, not a per-token duplicate
   ratio that the estimator's algebra bounds away from 1.

Direction 3 is a claim change and is explicitly a PI decision. Directions 1 and
2 are intern work and can be measured this week without training.

## Caveats

- GODAS fields are absent, so the token **geometry** is rebuilt on synthetic
  all-ocean fields. `omega` is a function of geometry, noise and variable group
  — field values never enter it — so the operating point is exact for an
  all-ocean box. Land would remove tokens and lower N, which *raises* `mean
  omega ≤ F/N`; the synthetic case is therefore the dense/upper bound on
  competition and the pessimistic end for the discount.
- Single basis seed for the operating point and sweep (5 seeds for the
  convergence table). The standing three-seed rule applies to reported accuracy;
  no accuracy is reported here.
- The declared noise areas are a normalisation, not measured GODAS observation
  errors. Replacing them with real error tables is Phase 0 work that remains
  open, and is what would make `c = 1` a physical prior rather than a convention.
