# F8745-D4 BladeAD tonal validation

This comparison evaluates BladeAD Lowson loading plus Barry–Magliozzi thickness noise
using the frozen RCAIDE line-source run's aerodynamic disk loads. It isolates acoustic
radiation-model differences; it is not a validation of BladeAD BEM aerodynamics.

No calibration or fixture-specific correction is applied.

| Case | Angle (deg) | MAE (dB) | Max (dB) | Overall error (dB) | Gate |
|---|---:|---:|---:|---:|---|
| F8745-D4-1 | 60 | 14.151 | 17.434 | -11.488 | FAIL |
| F8745-D4-1 | 90 | 13.320 | 15.603 | -14.168 | FAIL |
| F8745-D4-2 | 60 | 17.837 | 22.931 | -14.288 | FAIL |
| F8745-D4-2 | 90 | 13.892 | 16.963 | -15.145 | FAIL |
| F8745-D4-3 | 60 | 22.140 | 27.898 | -15.056 | FAIL |
| F8745-D4-3 | 90 | 17.132 | 18.960 | -16.779 | FAIL |

The frozen gate requires both absolute overall error and mean per-harmonic absolute
error to be no greater than 3 dB. Every case fails both criteria. BladeAD
systematically underpredicts the measured harmonics, so the current Lowson model
must not yet be used as experimental design authority for this forward-flight case.

The failure does not invalidate the HG equation/reference verification. It shows that
the current compact Lowson/Sears/thickness scope does not reproduce the mechanisms or
source representation captured by the F8745-D4 measurements and RCAIDE Hanson models.

Detailed harmonic results are in `bladead_f8745_detailed.csv`.
