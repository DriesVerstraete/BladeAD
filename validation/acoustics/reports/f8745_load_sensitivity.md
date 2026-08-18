# F8745-D4 aerodynamic-source sensitivity

The experimental source located for this fixture is Weir and Powers, AIAA Paper
87-0527, *Comparisons of Predicted Propeller Noise with Windtunnel and Flyover Data*.
The accessible primary-source metadata identifies the paper and test, but measured thrust,
torque/power, aerodynamic uncertainty, and sectional loading remain **not reported** in the
material recovered for this audit. The perturbations below therefore bound sensitivity;
they are not experimental uncertainty intervals.

Magnitude cases scale signed thrust and torque distributions together by ±10%. Radial
cases multiply each signed distribution by `1 + s(2r/R-1)` with `s=±0.2`, then normalize
each case and azimuth back to its original thrust and torque. Thus radial cases preserve
the archived integral loads exactly while shifting loading rootward or tipward.

| Model | Perturbation | Max |Δ overall error| (dB) | Max |Δ harmonic MAE| (dB) |
|---|---|---:|---:|
| lowson | magnitude_minus_10pct | 0.758 | 0.181 |
| lowson | magnitude_plus_10pct | 0.708 | 0.181 |
| lowson | root_shift_fixed_total | 0.111 | 0.056 |
| lowson | tip_shift_fixed_total | 0.090 | 0.045 |
| hanson_line | magnitude_minus_10pct | 0.852 | 0.646 |
| hanson_line | magnitude_plus_10pct | 0.782 | 0.627 |
| hanson_line | root_shift_fixed_total | 0.151 | 0.369 |
| hanson_line | tip_shift_fixed_total | 0.121 | 0.296 |

These bounded perturbations do not close the experimental discrepancy. They quantify
how strongly the present conclusions depend on plausible but unverified source-load
changes; they do not validate either acoustic model or the RCAIDE aerodynamic loads.
Detailed results are in `f8745_load_sensitivity.csv`.
