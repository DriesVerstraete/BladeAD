# F8745-D4 BladeAD Hanson line-source validation

This comparison evaluates aligned-inflow BladeAD Hanson loading plus helicoidal-surface
thickness noise using the frozen RCAIDE aerodynamic disk loads and archived F8745
airfoil thickness shape. BladeAD BEM, transverse inflow, and fixture-specific
calibration are not used.

The BladeAD production adapter uses per-blade N/m loads, normalized Fourier
coefficients, and one nondimensional radial integration. RCAIDE's archived prediction
uses its original unnormalized FFT and element-force radial convention, so code-to-code
agreement is diagnostic rather than an implementation acceptance criterion.

## Experimental comparison

| Case | Angle (deg) | MAE (dB) | Max (dB) | Overall error (dB) | Gate |
|---|---:|---:|---:|---:|---|
| F8745-D4-1 | 60 | 20.715 | 22.544 | -16.276 | FAIL |
| F8745-D4-1 | 90 | 24.080 | 26.467 | -23.141 | FAIL |
| F8745-D4-2 | 60 | 23.913 | 33.795 | -18.078 | FAIL |
| F8745-D4-2 | 90 | 24.790 | 26.554 | -24.512 | FAIL |
| F8745-D4-3 | 60 | 28.215 | 38.671 | -18.935 | FAIL |
| F8745-D4-3 | 90 | 27.992 | 31.712 | -25.873 | FAIL |

## RCAIDE combined line-source comparison

| Case | Angle | Mean vs archive | MA vs archive | Mean vs RCAIDE −15 dB | MA vs RCAIDE −15 dB |
|---|---:|---:|---:|---:|---:|
| F8745-D4-1 | 60 | -16.194 | 16.194 | -1.194 | 2.528 |
| F8745-D4-1 | 90 | -18.915 | 18.915 | -3.915 | 3.915 |
| F8745-D4-2 | 60 | -17.108 | 17.108 | -2.108 | 2.955 |
| F8745-D4-2 | 90 | -19.624 | 19.624 | -4.624 | 4.624 |
| F8745-D4-3 | 60 | -16.823 | 16.823 | -1.823 | 2.864 |
| F8745-D4-3 | 90 | -19.476 | 19.476 | -4.476 | 4.476 |

Both models now contain Hanson loading and helicoidal-surface thickness sources.
The archived RCAIDE configuration adds 15 dB to every harmonic for wing-wake
interaction. BladeAD has no corresponding empirical adjustment; the table therefore
shows comparisons both with and without that uplift.
The remaining difference includes source-shape normalization, complex versus
magnitude-only component summation, peak/RMS convention, propagation geometry, and
RCAIDE's legacy radial/Fourier scaling. The separate term audit shows that changing
only BladeAD's reporting from coherent RMS to coherent peak reduces mean absolute
code difference to 1.37–2.61 dB. Production retains physically labelled RMS pressure.

## RCAIDE legacy loading-adapter scaling with BladeAD thickness unchanged

| Case | Angle (deg) | Mean (dB) | Minimum (dB) | Maximum (dB) |
|---|---:|---:|---:|---:|
| F8745-D4-1 | 60 | -3.290 | -6.355 | -1.808 |
| F8745-D4-1 | 90 | -0.335 | -2.731 | 0.550 |
| F8745-D4-2 | 60 | -3.803 | -6.922 | -2.511 |
| F8745-D4-2 | 90 | -0.320 | -3.969 | 0.583 |
| F8745-D4-3 | 60 | -3.783 | -6.900 | -2.554 |
| F8745-D4-3 | 90 | -0.373 | -4.053 | 0.589 |

Detailed experimental harmonic errors are in
`bladead_f8745_hanson_detailed.csv`; RCAIDE summary differences are in
`bladead_f8745_hanson_rcaide_comparison.csv`; isolated adapter effects are in
`bladead_f8745_hanson_adapter_comparison.csv`.
