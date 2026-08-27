# McDonald motor-model validation

**Date:** 2026-08-27  
**Source implementation:** Shafiq Shahjahan, `mcdonald_electric_motor.py`

## Scope

- Reproduces the derived `c0`--`c3` coefficients and signed `c4`--`c12` polynomial and logarithmic
  loss terms.
- Applies the reported efficiency scaling before calculating electrical input power.
- Keeps the efficiency map separate from the speed-dependent continuous-torque constraint.
- Represents the digitised Vertiia constraint with a degree-10 Chebyshev curve over 0--4500 RPM.

## Results

- `pytest tests/motor/test_motor_models.py -q`: **4 passed**.
- Independent NumPy evaluation matches shaft power, loss, electrical power, and efficiency.
- CSDL derivatives of electrical power, efficiency, and the torque envelope pass finite-difference
  verification.
- The fitted Vertiia continuous-torque curve never exceeds any of the 32 supplied digitised data
  points. Maximum underprediction is 1.87 N m; RMS underprediction is 1.15 N m.
- Project coupling tests: **7 passed** in
  `standalone_optimisation/tests/test_hover_cruise_problem.py`, including the McDonald configuration
  and representative envelope-point checks.

## Limitations

- Positive angular speed and torque are required by the logarithmic terms.
- The Vertiia envelope is a continuous rating, not a temporary or peak rating.
- The curve must not be paired with coefficients for another motor.
- The other coefficient and torque datasets in Shahjahan's supplied file have not yet been promoted
  to named optimisation presets.
