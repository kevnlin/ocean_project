# Cross-check — P0 (npac_gyre)

> **What P3 cross-checks, and what it cannot.** The ladder's rows (`thin_*`, `superob_*`) are P0 rows, so the numbers below are the P0 comparison restricted to them. The plan also asks the two tracks to agree on the thinning budget, grid definition, merge radius, noise weighting and modality handling: both tracks instantiate the ladder from the same `godas_model.build_row` registry and load the same checkpoints, so those five are shared **by construction, not by agreement** — two independent implementations of them would be a stronger check and do not exist. What is independently checked is the scoring of the rows they produce.

## Provenance

| field | Track A | Track B |
|---|---|---|
| protocol version | `crosscheck_v1` | `crosscheck_v1` |
| protocol hash | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` |
| git commit | `59076c12e91177e4e84899a3a46ec356bfcb53d3` | `59076c12e91177e4e84899a3a46ec356bfcb53d3` |
| git dirty | `True` | `True` |
| seeds | `[1234, 1235, 1236]` | `[1234, 1235, 1236]` |
| created (UTC) | `2026-09-07T22:08:54Z` | `2026-09-07T22:46:45Z` |

## Data manifests

| dataset | Track A sha256 | Track B sha256 |
|---|---|---|
| argo_cohort:npac_gyre | `9196a4b50269a23f` | `9196a4b50269a23f` |
| reference_products | `373af02d9f9b7b24` | `373af02d9f9b7b24` |

## Counts (exact-match quantities)

| quantity | Track A | Track B |
|---|---:|---:|
| eval_split | development | development |
| months | 36 | 36 |
| n_profiles | 24 | 24 |
| split_protocol | main | main |
| targets | 36648 | 36648 |
| wmos | 42 | 42 |

## Acceptance (plan S6)

**Verdict: agree**

### Numerical differences above rtol (4)

- [numeric] dfs_minus_uniform.SALT.point: A=-0.005830022402817085 B=-0.00583001072092008 rel=2.004e-06
- [numeric] per_seed_rmse.en4: A={'TEMP': [0.19563481828233847, 0.19463355014688866, 0.19457230061222427], 'SALT': [1.0611249465535175, 1.3922665371530618, 1.2813356432213263]} B='<absent>' present in only one track
- [numeric] rmse.en4: A={'TEMP': 0.1949468896804838, 'SALT': 1.2449090423093019} B='<absent>' present in only one track
- [numeric] ranking_coverage: A=['en4'] B=[] rows scored by only one track — not a disagreement; the ranking comparison is restricted to the rows in common

## Warnings

- ecco does not cover any development month (product range 1999-12..2017-12); row omitted rather than scored as the climatology floor
- git working tree was dirty: the recorded commit does not fully identify the code that produced these numbers
