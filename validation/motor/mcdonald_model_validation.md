# McDonald motor-model validation

**Date:** 2026-08-27  
**Source implementation:** Shafiq Shahjahan, `mcdonald_electric_motor.py`

## Scope

- Reproduces the derived `c0`--`c3` coefficients and signed `c4`--`c12` polynomial and logarithmic
  loss terms.
- Applies the reported efficiency scaling before calculating electrical input power.
- Keeps the efficiency map separate from the speed-dependent continuous-torque constraint.
- Represents the paper's scaled EMRAX 188 continuous-torque constraint with a degree-10 Chebyshev
  curve over 0--3439 RPM.

## Results

- `pytest tests/motor/test_motor_models.py -q`: **5 passed**.
- Independent NumPy evaluation matches shaft power, loss, electrical power, and efficiency.
- CSDL derivatives of electrical power, efficiency, and the torque envelope pass finite-difference
  verification.
- The fitted scaled-EMRAX curve never exceeds any of the 54 supplied digitised data points. Maximum
  underprediction is 0.82 N m; RMS underprediction is 0.43 N m.
- Project coupling tests: **7 passed** in
  `standalone_optimisation/tests/test_hover_cruise_problem.py`, including the McDonald configuration
  and representative envelope-point checks.

## Limitations

- Positive angular speed and torque are required by the logarithmic terms.
- The EMRAX envelope is a continuous rating. The paper applies 100% in cruise and 170% in hover,
  emergency hover, and transition.
- The curve must not be paired with coefficients for another motor.
- The other coefficient and torque datasets in Shahjahan's supplied file have not yet been promoted
  to named optimisation presets.
