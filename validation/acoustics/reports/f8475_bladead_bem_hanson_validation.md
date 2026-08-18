# F8475 D-4 BladeAD BEM-to-Hanson validation

This path uses the fixture geometry, a Clark-Y ZeroD polar fitted to the RCAIDE
Re=1,000,000 XFOIL table, BladeAD BEM loads, and BladeAD Hanson acoustics. No RCAIDE
aerodynamic loads enter the calculation. Blade angle is independently adjusted to
match each Table 4 shaft power, following Weir and Powers.

Pressure was not reported in Table 4; density is inferred using 101325 Pa and the
reported temperature. Clark-Y Reynolds-number variation is not represented.

## Aerodynamics

| Case | BladeAD blade angle | Paper computed angle | Power (kW) | Thrust measured/predicted (N) | Thrust error |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.240° | 22.300° | 73.600 | 642.0 / 791.0 | +23.21% |
| 2 | 21.458° | 22.000° | 184.600 | 1907.0 / 2046.6 | +7.32% |
| 3 | 20.580° | 21.200° | 152.100 | 1500.0 / 1682.9 | +12.19% |

## Hanson tonal comparison at 4 m

| Case | Angle | Harmonic MAE | Overall error |
|---:|---:|---:|---:|
| 1 | 60° | 4.623 dB | -2.104 dB |
| 1 | 90° | 7.740 dB | -8.200 dB |
| 2 | 60° | 7.174 dB | -2.060 dB |
| 2 | 90° | 9.162 dB | -9.034 dB |
| 3 | 60° | 7.679 dB | -2.753 dB |
| 3 | 90° | 8.366 dB | -8.426 dB |

## Source-component energetic levels

| Case | Angle | Loading | Thickness | Combined |
|---:|---:|---:|---:|---:|
| 1 | 60° | 103.928 dB | 95.309 dB | 104.328 dB |
| 1 | 90° | 101.033 dB | 103.731 dB | 105.228 dB |
| 2 | 60° | 113.967 dB | 102.188 dB | 114.099 dB |
| 2 | 90° | 110.625 dB | 113.234 dB | 114.580 dB |
| 3 | 60° | 112.574 dB | 102.471 dB | 112.801 dB |
| 3 | 90° | 109.115 dB | 113.664 dB | 114.447 dB |

The observer convention is audited in `f8475_directivity_audit.md`. The
paper-to-BladeAD angle mapping is correct; at 90 degrees the axial-loading term
vanishes and thickness is the largest isolated component. The residual in-plane
deficit is therefore not corrected by mirroring or relabeling the observer.
