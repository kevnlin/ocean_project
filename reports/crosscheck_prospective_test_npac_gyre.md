# Cross-check — P7 (npac_gyre)

> **Read the disagreement below as a freeze failure, not a numerical one.** Track A scored `count_oi_expert_cbottle`, `dfs_oi_expert_cbottle`, `uniform_oi_expert_cbottle` from checkpoints the freeze record does not pin; Track B scored them from the frozen bytes. Every other row matches to float precision, and those three are exactly the rows that differ. See [`intern_prospective_test_npac_gyre.md`](intern_prospective_test_npac_gyre.md) for the hash-by-hash comparison.

## Provenance

| field | Track A | Track B |
|---|---|---|
| protocol version | `crosscheck_v1` | `crosscheck_v1` |
| protocol hash | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` |
| git commit | `59076c12e91177e4e84899a3a46ec356bfcb53d3` | `59076c12e91177e4e84899a3a46ec356bfcb53d3` |
| git dirty | `True` | `True` |
| seeds | `[1234]` | `[1234, 1235, 1236]` |
| created (UTC) | `2026-09-08T00:08:07Z` | `2026-09-07T22:57:11Z` |

## Data manifests

| dataset | Track A sha256 | Track B sha256 |
|---|---|---|
| argo_cohort:npac_gyre | `9196a4b50269a23f` | `9196a4b50269a23f` |
| reference_products | `373af02d9f9b7b24` | `373af02d9f9b7b24` |

## Counts (exact-match quantities)

| quantity | Track A | Track B |
|---|---:|---:|
| eval_split | holdout | holdout |
| months | 12 | 12 |
| n_profiles | 24 | 24 |
| split_protocol | main | main |
| targets | 12238 | 12238 |
| wmos | 30 | 30 |

## Acceptance (plan S6)

**Verdict: agree**

### Numerical differences above rtol (3)

- [numeric] per_seed_rmse.en4: A={'TEMP': [0.18798563053530606], 'SALT': [0.18410991050046713]} B='<absent>' present in only one track
- [numeric] rmse.en4: A={'TEMP': 0.18798563053530606, 'SALT': 0.18410991050046713} B='<absent>' present in only one track
- [numeric] ranking_coverage: A=['en4'] B=[] rows scored by only one track — not a disagreement; the ranking comparison is restricted to the rows in common

## Warnings

- ecco does not cover any holdout month (product range 1999-12..2017-12); row omitted rather than scored as the climatology floor
- git working tree was dirty: the recorded commit does not fully identify the code that produced these numbers
- outputs/argo_P7_npac_gyre/count_oi_expert_cbottle_s1234.pt shares the name of a frozen checkpoint but not its bytes; skipped in favour of the file the freeze record pins
- outputs/argo_P7_npac_gyre/dfs_oi_expert_cbottle_s1234.pt shares the name of a frozen checkpoint but not its bytes; skipped in favour of the file the freeze record pins
- outputs/argo_P7_npac_gyre/uniform_oi_expert_cbottle_s1234.pt shares the name of a frozen checkpoint but not its bytes; skipped in favour of the file the freeze record pins
