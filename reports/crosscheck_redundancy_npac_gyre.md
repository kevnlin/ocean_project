# Cross-check — P2 (npac_gyre)

## Provenance

| field | Track A | Track B |
|---|---|---|
| protocol version | `crosscheck_v1` | `crosscheck_v1` |
| protocol hash | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` |
| git commit | `59076c12e91177e4e84899a3a46ec356bfcb53d3` | `59076c12e91177e4e84899a3a46ec356bfcb53d3` |
| git dirty | `True` | `True` |
| seeds | `[1234, 1235, 1236]` | `[1234, 1235, 1236]` |
| created (UTC) | `2026-09-08T00:23:07Z` | `2026-09-08T00:28:28Z` |

## Data manifests

| dataset | Track A sha256 | Track B sha256 |
|---|---|---|
| argo_cohort:npac_gyre | `9196a4b50269a23f` | `9196a4b50269a23f` |
| reference_products | `373af02d9f9b7b24` | `373af02d9f9b7b24` |

## Counts (exact-match quantities)

| quantity | Track A | Track B |
|---|---:|---:|
| eval_split | development | development |
| months | 24 | 24 |
| n_profiles | 24 | 24 |

## Acceptance (plan S6)

**Verdict: agree**

### Numerical differences above rtol (60)

- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_hi: A=[0.0015252568952279222, 0.0060474975856786815, 0.0011051208396057834] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_lo: A=[-1.2480075490402252e-05, 0.001638385764171961, -9.431747246214928e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_point: A=[0.0007336335947060735, 0.003558537203206147, 0.00045614099115026674] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[False, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_hi: A=[0.001655014101640758, 0.008553360133765403, 0.0008508309017779548] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.0005348339029659837, 0.0008726923384229535, -0.0007965310078132046] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_point: A=[0.00044524818562602375, 0.0037608561506764504, 0.00015682877759837233] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, False, True] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_hi: A=[9.61918446526544e-05, 5.864235424729025e-05, 9.843811541829509e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_lo: A=[-9.248381860135768e-05, -1.633384233125368e-05, 6.965391402787014e-06] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_point: A=[9.262195266035445e-06, 2.1792931008068894e-05, 4.846726141582991e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_hi: A=[0.006248901353450823, 0.01568242793490648, 0.0016628636209490458] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_lo: A=[0.0006124091277867616, 0.0017535078754383287, -0.0035103142841633626] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_point: A=[0.003094170806784813, 0.007443033231188445, -0.0004376217241611191] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_excludes_zero: A=[True, False, True] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_hi: A=[4.9244534277545496e-05, 1.765393026758877e-05, 0.00015391874747611446] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_lo: A=[3.1502748395473293e-07, -3.3659619487455546e-05, 2.659434733976729e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_point: A=[1.526836164567058e-05, -2.1152839302374105e-06, 7.699211735368028e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_excludes_zero: A=[True, True, True] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_hi: A=[-0.008890873121296458, -0.008142509861644382, -0.006324349589445467] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_lo: A=[-0.04740329637751511, -0.03742058422986595, -0.05068206409322059] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_point: A=[-0.026041169647702866, -0.022130989119626965, -0.02722243242772282] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[True, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_hi: A=[2.441841048889254e-05, 2.6830256553853076e-06, 1.72138997940885e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_lo: A=[6.075131146975055e-06, -3.2306475028151538e-06, -7.600285377010184e-07] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_point: A=[1.4549338307912052e-05, -4.150406989200661e-07, 8.649971525553912e-06] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[False, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_hi: A=[0.00171454969513406, 0.006726479199362933, 0.0009484770079253274] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.0010491483397686852, 0.0004472035047131895, -0.0013399214052557257] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_point: A=[0.0002798492688954046, 0.0029245260782887472, 0.00011103195383904696] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_hi: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_lo: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_point: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_hi: A=[-0.00025494611264635067, 0.006498280065163082, 0.00152797332551721] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.0020965950691561944, 0.0005609874898283046, -0.0003530101239697894] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_point: A=[-0.0010933362649304113, 0.0028784079928083073, 0.00048599961008055637] B='<absent>' present in only one track
- ... and 20 more

## Warnings

- git working tree was dirty: the recorded commit does not fully identify the code that produced these numbers
