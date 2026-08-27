# Motor-model verification

## Scope

BladeAD provides two parameter-driven, differentiable motor models in
`BladeAD.core.motor`:

- `mcdonald`: a positive-polynomial power-loss model in torque and angular speed;
- `three_constant`: a `Kv`--resistance--no-load-current equivalent circuit.

No motor-specific coefficients are distributed as defaults. Users must provide a calibration whose
power, torque, speed, and motor-size range is appropriate to their application.

## Reproducible test

From the BladeAD repository in an environment containing CSDL-alpha:

```bash
python -m pytest tests/motor/test_motor_models.py -q
```

Tested on 2026-08-27 with Python 3.11 and the SPL `rotor_design` environment:

```text
3 passed in 1.56s
```

CSDL-alpha emitted 30 NumPy deprecation warnings from its internal `setindex` operation; the motor
tests emitted no model-specific warnings or failures.

## What is checked

The tests independently reconstruct and compare:

- McDonald loss, shaft power, electrical power, and efficiency for vector speed/torque inputs;
- three-constant current, voltage, shaft power, electrical power, loss, and efficiency;
- invalid coefficients, constants, model names, and model/parameter mismatches; and
- CSDL derivatives against finite differences for electrical power and efficiency in both models,
  plus current and voltage in the three-constant model.

The derivative checks use `csdl.derivative_utils.verify_derivatives(..., raise_on_error=True)`, so a
derivative discrepancy fails the test rather than being reported passively.

## Limits of this verification

These tests verify implementation correctness and differentiability, not a particular motor
calibration. They do not establish motor-map fit error, mass prediction, thermal behaviour, or
continuous/temporary operating limits. Those require separate evidence for each supplied parameter
set.
