# Data Card — `cesm2`  (cesm2_standard.zarr)

## Provenance / attributes
- **source**: CESM2 Large Ensemble – single member sample
- **grid**: curvilinear POP (lat/lon are approximate placeholder)
- **time_range**: 2000-01 to 2009-12
- **depth_units**: m (converted from cm)
- **temp_units**: degC
- **salt_units**: PSU

## Dimensions
- `time`: 120
- `depth`: 30
- `lat`: 384
- `lon`: 320

## Coordinates
- `depth`: 30 pts, range [5, 378]
- `lat`: 384 pts, range [-79.5, 89.5]
- `lon`: 320 pts, range [0.5, 360]
- `time`: 120 pts, [2000-01-16 12:00:00 .. 2009-12-16 12:00:00]

## Variables (units, sample statistics)

| var | dims | units | min | mean | max | %finite |
|-----|------|-------|-----|------|-----|---------|
| SALT | (time,depth,lat,lon) | PSU | 6.88 | 34.5 | 47.4 | 66.4% |
| SSS | (time,lat,lon) |  | 6.88 | 34.1 | 47.4 | 70.1% |
| SST | (time,lat,lon) |  | -1.91 | 17.7 | 31.7 | 70.1% |
| TEMP | (time,depth,lat,lon) | degC | -1.98 | 13.3 | 32.5 | 66.4% |
