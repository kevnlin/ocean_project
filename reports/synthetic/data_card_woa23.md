# Data Card — `woa23`  (woa23_standard.zarr)

## Provenance / attributes
- **source**: World Ocean Atlas 2023
- **variable**: t_an / s_an (objectively analysed climatological mean)
- **time_meaning**: calendar month (1 = January … 12 = December)
- **clim_period**: decav91C0 (1991–2020)
- **depth_units**: m
- **temp_units**: degC
- **salt_units**: PSU

## Dimensions
- `time`: 12
- `depth`: 57
- `lat`: 180
- `lon`: 360

## Coordinates
- `depth`: 57 pts, range [0, 1.5e+03]
- `lat`: 180 pts, range [-89.5, 89.5]
- `lon`: 360 pts, range [-180, 180]
- `time`: 12 pts, range [1, 12]

## Variables (units, sample statistics)

| var | dims | units | min | mean | max | %finite |
|-----|------|-------|-----|------|-----|---------|
| SALT | (time,depth,lat,lon) | PSU | 5 | 34.6 | 40.7 | 59.1% |
| SSS | (time,lat,lon) |  | 5 | 34.1 | 40.5 | 63.4% |
| SST | (time,lat,lon) |  | -1.89 | 14.1 | 30.5 | 63.4% |
| TEMP | (time,depth,lat,lon) | degC | -2.1 | 8.62 | 31.1 | 59.1% |
