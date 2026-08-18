# F8745-D4 BladeAD tonal validation

This comparison evaluates BladeAD Lowson loading plus Barry–Magliozzi thickness noise
using the frozen RCAIDE line-source run's aerodynamic disk loads. BladeAD BEM is not
used. The result measures the complete load-adapter, propagation, and acoustic-model
chain; the accompanying interface audit separates those contributions where possible.

No calibration or fixture-specific correction is applied.

| Case | Angle (deg) | MAE (dB) | Max (dB) | Overall error (dB) | Gate |
|---|---:|---:|---:|---:|---|
| F8745-D4-1 | 60 | 14.236 | 17.434 | -12.399 | FAIL |
| F8745-D4-1 | 90 | 13.434 | 15.603 | -14.763 | FAIL |
| F8745-D4-2 | 60 | 17.849 | 22.931 | -14.394 | FAIL |
| F8745-D4-2 | 90 | 13.909 | 16.963 | -15.185 | FAIL |
| F8745-D4-3 | 60 | 22.159 | 27.898 | -15.247 | FAIL |
| F8745-D4-3 | 90 | 17.160 | 18.960 | -16.872 | FAIL |

The frozen gate requires both absolute overall error and mean per-harmonic absolute
error to be no greater than 3 dB. Every case fails both criteria. BladeAD
systematically underpredicts the measured harmonics, so the current Lowson model
must not yet be used as experimental design authority for this forward-flight case.

The failure does not invalidate the HG equation/reference verification. It shows that
the current BladeAD acoustic chain does not reproduce the mechanisms or source
representation captured by the F8745-D4 measurements and RCAIDE Hanson models. The
audit rules out BEM and basic load conversion, but retains propagation convention as
an unresolved contributor.

Detailed harmonic results are in `bladead_f8745_detailed.csv`.
