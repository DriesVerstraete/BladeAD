# F8745 RCAIDE Hanson term audit

This script independently reproduces the archived RCAIDE line-source equation sequence
from frozen arrays, including retarded geometry, raw azimuth FFT, element-force radial
integration, thickness normalization, and magnitude-only peak-pressure summation.
The archived configuration then adds a uniform 15.0 dB wing-wake
interaction adjustment to every harmonic. This adjustment is reproduced here only for
audit parity and is not part of BladeAD's production Hanson physics.

| Case | Angle | Loading peak overall | Thickness peak overall | Max archive error |
|---|---:|---:|---:|---:|
| F8745-D4-1 | 60 | 85.314 | 80.737 | 2.526155 |
| F8745-D4-1 | 90 | 82.100 | 89.105 | 2.199189 |
| F8745-D4-2 | 60 | 93.298 | 87.944 | 2.139775 |
| F8745-D4-2 | 90 | 89.462 | 99.086 | 2.767044 |
| F8745-D4-3 | 60 | 91.867 | 86.073 | 1.911370 |
| F8745-D4-3 | 90 | 88.343 | 96.414 | 2.686177 |

## BladeAD pressure-convention isolation

All rows use identical BladeAD component pressures and compare against RCAIDE with
the 15 dB adjustment removed.

| Convention | Mean absolute difference range (dB) |
|---|---:|
| coherent_rms_production | 2.528–4.624 |
| coherent_peak | 1.366–2.605 |
| magnitude_sum_peak | 1.401–5.661 |
