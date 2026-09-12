# Data Card — `cesm2_le_full`  (cesm2_le_full_standard.zarr)

## Provenance / attributes
- **source**: CESM2 Large Ensemble – 1°×1° regridded (jaisonk)
- **grid**: regular 1°×1° (y_regr=180, x_regr=360)
- **time_range**: 1850-01 to 2100-12
- **depth_units**: m
- **temp_units**: degC
- **salt_units**: PSU
- **SST_note**: shallowest depth level (z_t=5 m)

## Dimensions
- `lat`: 180
- `lon`: 360
- `time`: 3012
- `depth`: 60

## Coordinates
- `depth`: 60 pts, range [5, 5.38e+03]
- `lat`: 180 pts, range [-89.5, 89.5]
- `lon`: 360 pts, range [0.5, 360]
- `time`: 3012 pts, [1850-01-18 00:00:00 .. 2100-12-18 00:00:00]

## Variables (units, sample statistics)

| var | dims | units | min | mean | max | %finite |
|-----|------|-------|-----|------|-----|---------|
| MASK | (lat,lon) |  | 0 | 0.654 | 1 | 100.0% |
| SALT | (time,depth,lat,lon) | PSU | 0 | 34.5 | 44.7 | 53.7% |
| SSS | (time,lat,lon) |  | 6.92 | 33.9 | 44.7 | 65.3% |
| SST | (time,lat,lon) |  | -1.91 | 14 | 31.8 | 65.3% |
| TEMP | (time,depth,lat,lon) | degC | -2.06 | 8.04 | 31.8 | 53.7% |
