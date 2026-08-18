# Acoustic validation matrix

This matrix was frozen before any BladeAD tonal or broadband model prediction existed.
RCAIDE is a comparison point, not validation truth.

## Acceptance criteria for BladeAD

- Tonal: absolute overall error <= 3 dB and mean per-harmonic absolute error <= 3 dB.
- Broadband: absolute overall error <= 3 dB and mean band error <= 5 dB over bands
  materially above the measurement/background floor.
- Trends: no unexplained systematic observer-angle, RPM, or harmonic/frequency trend.
- Derivatives: relative error <= 1e-5 when well scaled; absolute error <= 1e-7 near zero,
  with convergence over at least three finite-difference step sizes.

These thresholds are not applied to force RCAIDE to pass and must not be relaxed after
seeing BladeAD results. A failed model is narrowed, extended, or rejected rather than tuned
to these fixtures.

## RCAIDE comparison results

| Case | Model | Component | Angle (deg) | N | MAE (dB) | Max (dB) | Overall error (dB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| F8745-D4-1 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 1.310 | 2.332 | -1.332 |
| F8745-D4-1 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 3.253 | 7.217 | -4.958 |
| F8745-D4-2 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 4.369 | 10.436 | -3.343 |
| F8745-D4-2 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 4.516 | 9.312 | -5.550 |
| F8745-D4-3 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 8.599 | 15.230 | -4.361 |
| F8745-D4-3 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 7.570 | 9.094 | -7.245 |
| F8745-D4-1 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 6.177 | 7.584 | -4.393 |
| F8745-D4-1 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 13.309 | 14.731 | -12.792 |
| F8745-D4-2 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 7.471 | 15.780 | -3.321 |
| F8745-D4-2 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 14.141 | 16.136 | -13.275 |
| F8745-D4-3 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 12.351 | 21.130 | -5.171 |
| F8745-D4-3 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 17.263 | 20.646 | -14.789 |
| APC-11x4-3600-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 7.150 | 14.958 | -2.418 |
| APC-11x4-4200-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 6.674 | 17.919 | -2.278 |
| APC-11x4-4800-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 6.800 | 13.213 | -2.898 |
| APC-11x4-4200-RPM | rcaide_plane_source_source_driver_angle_mapping | broadband_one_third_octave | 22.5 | 21 | 9.453 | 22.765 | -3.654 |
| APC-11x4-4200-RPM | rcaide_plane_source_source_driver_angle_mapping | broadband_one_third_octave | 45.0 | 21 | 5.296 | 16.370 | 1.262 |

Detailed signed errors are in `rcaide_vs_experiment_detailed.csv`; unrounded summary
metrics are in `rcaide_vs_experiment_summary.csv`.

## Known mapping limitation

For APC broadband data, RCAIDE labels experimental curves as 45 and 22.5 degrees but
compares them to simulated observer indices 4 and 3, whose driver angle parameters are
135 and 112.5 degrees. The rows labelled
`rcaide_plane_source_source_driver_angle_mapping` reproduce that source behaviour and
must not be interpreted as a resolved physical-angle equivalence.
