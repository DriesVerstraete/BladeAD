# F8475 thickness and pressure-convention audit

## Pressure convention

| Convention | Harmonic MAE range | Overall-error range |
|---|---:|---:|
| coherent_rms_production | 4.623--9.162 dB | -9.034 to -2.060 dB |
| coherent_peak | 1.968--6.152 dB | -6.023 to +0.950 dB |
| magnitude_sum_rms | 1.080--5.911 dB | -5.843 to +0.182 dB |
| magnitude_sum_peak | 1.761--2.901 dB | -2.833 to +3.193 dB |

Production uses coherent RMS pressure, consistent with sinusoidal SPL. Peak
reporting adds exactly 3.010 dB but does not close the 90-degree deficit.
Magnitude-only component addition discards physical phase and is retained only as
a diagnostic.

## In-plane thickness scale diagnostic

| Case | Pressure multiplier required | Equivalent level |
|---:|---:|---:|
| 1 | 3.028 | +9.623 dB |
| 2 | 3.313 | +10.405 dB |
| 3 | 2.903 | +9.257 dB |

## Integration convergence

Across 25--200 chordwise stations, the maximum within-case in-plane thickness-level spread is 0.0473 dB.
Across 15--61 radial stations with each case re-matched to measured power, the maximum within-case in-plane thickness-level spread is 1.3578 dB.
Detailed values are in `f8475_thickness_integration_convergence.csv`.

## Conclusion

Neither quadrature resolution nor the peak/RMS convention explains the in-plane
deficit. The remaining issue is model-form/source-content uncertainty, not a
numerical integration or observer-label error.
