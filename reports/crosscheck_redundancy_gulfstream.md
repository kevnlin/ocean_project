# Cross-check — P2 (gulfstream)

## Provenance

| field | Track A | Track B |
|---|---|---|
| protocol version | `crosscheck_v1` | `crosscheck_v1` |
| protocol hash | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` | `c02118caadb2c2ae4367265a2fe12f9eb2f12d6158acf0688d774f280f7eeff7` |
| git commit | `59076c12e91177e4e84899a3a46ec356bfcb53d3` | `59076c12e91177e4e84899a3a46ec356bfcb53d3` |
| git dirty | `True` | `True` |
| seeds | `[1234, 1235, 1236]` | `[1234, 1235, 1236]` |
| created (UTC) | `2026-09-08T00:11:05Z` | `2026-09-08T00:20:17Z` |

## Data manifests

| dataset | Track A sha256 | Track B sha256 |
|---|---|---|
| argo_cohort:gulfstream | `3be6704343edcbe0` | `3be6704343edcbe0` |
| reference_products | `373af02d9f9b7b24` | `373af02d9f9b7b24` |

## Counts (exact-match quantities)

| quantity | Track A | Track B |
|---|---:|---:|
| eval_split | development | development |
| months | 24 | 24 |
| n_profiles | 24 | 24 |

## Acceptance (plan S6)

**Verdict: agree**

### Numerical differences above rtol (55)

- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_hi: A=[0.0009842260269690654, -0.000583900202394011, 0.0011739853605063858] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_lo: A=[-0.0010506717275361162, -0.002496804240513212, -0.0007333689624538313] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|exact|k8.ci_point: A=[-6.265707535380027e-05, -0.0015024468890465492, 0.00016171820768751122] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_hi: A=[-0.0005455540133899536, 0.0001205910505612779, 0.00011752665982057307] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.001710335621237183, -0.0005397464504000166, -0.0017207946469079167] B='<absent>' present in only one track
- [numeric] per_seed_rmse.count_expertlocal_cbottle|separated|k8.ci_point: A=[-0.0011652143217069089, -0.00021674620550726775, -0.0007178183811139549] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, True, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_hi: A=[6.8896076854566e-05, -2.869909822534597e-05, 8.563331065008646e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_lo: A=[-6.0058472478229094e-05, -0.00014960719830487134, -3.2507376415509547e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|exact|k8.ci_point: A=[4.476981194101448e-06, -8.575388678755402e-05, 1.8802551527530298e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, True, True] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_hi: A=[-0.00034314156967897276, -0.00022090219495506235, -2.933776157365841e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.002324445422667538, -0.0005515967505241276, -0.0019653397474538757] B='<absent>' present in only one track
- [numeric] per_seed_rmse.dfs_expertlocal_cbottle|separated|k8.ci_point: A=[-0.0013083540501174218, -0.0003971299736976741, -0.0009563671690802078] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_excludes_zero: A=[False, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_hi: A=[4.908587589083914e-05, 2.9056046675993167e-05, 0.00010879074450781401] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_lo: A=[-7.486735921027887e-06, -0.00030099354406685764, -1.8207581124499162e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|exact|k8.ci_point: A=[1.6880285070630663e-05, -8.428246277170093e-05, 3.488732349399282e-05] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_excludes_zero: A=[True, True, True] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_hi: A=[-0.02575605117227528, -0.014584146032504368, -0.02944929921386587] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_lo: A=[-0.05645403759935836, -0.03705157583667711, -0.07091363047424312] B='<absent>' present in only one track
- [numeric] per_seed_rmse.objective_interpolation|separated|k8.ci_point: A=[-0.03961812856857527, -0.024667615258799458, -0.048524422304599746] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_hi: A=[6.689919548089063e-06, 3.265279341593819e-06, 1.9515172953704643e-06] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_lo: A=[-1.2817307735404903e-06, -1.0949776681756784e-06, -1.8537997703474416e-06] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|exact|k8.ci_point: A=[2.8340152281591813e-06, 6.857166426899042e-07, 1.9585605659511174e-07] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_hi: A=[-0.0008222728508184865, 0.00023822920751433052, 0.0004935939870925193] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.0017294072201208919, -0.0009268470492930266, -0.0013522442673381235] B='<absent>' present in only one track
- [numeric] per_seed_rmse.superob_expertlocal_cbottle|separated|k8.ci_point: A=[-0.001292108988227736, -0.00035052571859106596, -0.00034867509693981] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_excludes_zero: A=[False, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_hi: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_lo: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|exact|k8.ci_point: A=[0.0, 0.0, 0.0] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_excludes_zero: A=[True, False, False] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_hi: A=[-0.0007351696156935469, 0.00014210141894997766, 0.00038545423989244545] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_lo: A=[-0.0016165961887000702, -0.000667433930103585, -0.0012919691620429884] B='<absent>' present in only one track
- [numeric] per_seed_rmse.thin_expertlocal_cbottle|separated|k8.ci_point: A=[-0.001193096999045129, -0.00025138176688654923, -0.00037801885905930366] B='<absent>' present in only one track
- ... and 15 more

## Warnings

- git working tree was dirty: the recorded commit does not fully identify the code that produced these numbers
