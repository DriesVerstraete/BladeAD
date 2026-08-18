# F8745-D4 load/interface parity audit

This audit separates BladeAD BEM, the RCAIDE-load adapter, propagation conventions, and
acoustic-model scope before attributing the experimental discrepancy.

## Resolved invariants

- Blade count is 2; RCAIDE total thrust and torque equal exactly B times the per-blade totals.
- Signed sectional distributions sum to the archived per-blade totals; negative root
  elements are physical entries and must not be converted station-by-station with `abs()`.
- Trapezoidal integration of `blade_dT_dr` and `blade_dQ_dr` reproduces those distributions
  to machine precision.
- Disk loads are azimuthally uniform to numerical roundoff, so the steady load harmonic is
  the complete aerodynamic information available to the current Lowson projection.
- Observer CSV positions exactly match the archived RCAIDE microphone positions and lie on
  the specified 20 m radius.

| Invariant | Maximum discrepancy |
|---|---:|
| max_thrust_distribution_total_error_n | 0.000000e+00 |
| max_torque_distribution_total_error_nm | 0.000000e+00 |
| max_thrust_gradient_quadrature_error_n | 0.000000e+00 |
| max_torque_gradient_quadrature_error_nm | 0.000000e+00 |
| max_complete_rotor_thrust_scaling_error_n | 0.000000e+00 |
| max_complete_rotor_torque_scaling_error_nm | 0.000000e+00 |
| max_azimuthal_load_spread_n | 4.476419e-13 |
| max_observer_archive_csv_difference_m | 0.000000e+00 |
| max_observer_radius_error_m | 3.552714e-15 |

## Propagation sensitivity

The archived inertial velocity is positive x and is the baseline convention. Reversing
it improves the 60-degree overall errors to roughly 5–7 dB but leaves the 90-degree
errors near 15–17 dB. Source/freestream interpretation is therefore important but cannot
explain the complete discrepancy. Detailed values are in
`bladead_f8745_convection_sensitivity.csv`.

## Component levels for the archived convention

| Case | Angle | Loading overall (dB) | Thickness overall (dB) |
|---|---:|---:|---:|
| 1 | 60 | 93.101 | 86.890 |
| 1 | 90 | 95.780 | 95.526 |
| 2 | 60 | 100.931 | 94.189 |
| 2 | 90 | 104.040 | 106.463 |
| 3 | 60 | 99.563 | 92.277 |
| 3 | 90 | 102.494 | 103.436 |

## Conclusion

BladeAD BEM is excluded from this comparison. Blade count, signed load transfer, radial
quadrature, azimuth sampling, and observer positions are now numerically closed. The
remaining gap belongs to the acoustic chain: propagation convention plus formulation/source
scope. Because credible velocity conventions do not close the gap—especially at 90 degrees—
the evidence supports a Hanson line-source implementation, but does not assign the entire
error to Lowson physics alone.
