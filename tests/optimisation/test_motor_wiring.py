"""Motor fidelity-ladder step 1b: `evaluate_case_motor` + case validation.

The full end-to-end solve with the McDonald motor is an acceptance run, not a
unit test (a solve is ~60-90 s). Here we cover the pure motor wiring: the
placebo / mcdonald / no-motor branches, the continuous-torque envelope margin
at the known converged anchors, and the `case._validate` guard.
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.abspath(os.path.join(HERE, "..", "..", "BladeAD", "optimisation", "cases"))


def _recorder():
    import csdl_alpha as csdl
    r = csdl.Recorder(inline=True)
    r.start()
    return csdl, r


def test_placebo_branch_is_fixed_efficiency_and_no_envelope():
    csdl, r = _recorder()
    from BladeAD.optimisation.solve import evaluate_case_motor
    rpm = csdl.Variable(value=np.array([2100.0]))
    shaft = csdl.Variable(value=np.array([19000.0]))
    mp = evaluate_case_motor({"model": "placebo", "efficiency": 0.95}, rpm, shaft, overload=1.7)
    assert mp.torque_margin is None
    assert mp.efficiency == pytest.approx(0.95)
    assert float(mp.electrical_power.value[0]) == pytest.approx(19000.0 / 0.95)
    r.stop()


def test_no_motor_branch_returns_shaft_power_stand_in():
    csdl, r = _recorder()
    from BladeAD.optimisation.solve import evaluate_case_motor
    rpm = csdl.Variable(value=np.array([2100.0]))
    shaft = csdl.Variable(value=np.array([19000.0]))
    mp = evaluate_case_motor({}, rpm, shaft, overload=1.0)
    assert mp.torque_margin is None
    assert mp.efficiency == 1.0
    assert float(mp.electrical_power.value[0]) == pytest.approx(19000.0)
    r.stop()


def test_mcdonald_branch_matches_core_model_and_envelope_at_converged_anchors():
    csdl, r = _recorder()
    from BladeAD.optimisation.solve import evaluate_case_motor
    from BladeAD.core.motor import (
        evaluate_motor, SHAHJAHAN_EMRAX188_PARAMETERS,
        SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE,
    )
    # (tag, rpm, shaft_power W, overload k, expect feasible)
    anchors = [
        ("fm_hover hover", 2095.79, 19000.84, 1.7),
        ("elec_power hover", 2334.71, 21113.79, 1.7),
        ("elec_power cruise", 682.52, 6068.71, 1.0),
    ]
    for tag, rpm_v, power_v, k in anchors:
        rpm = csdl.Variable(value=np.array([rpm_v]))
        shaft = csdl.Variable(value=np.array([power_v]))
        mp = evaluate_case_motor({"model": "mcdonald"}, rpm, shaft, overload=k)

        omega = rpm_v * 2.0 * np.pi / 60.0
        q = power_v / omega
        ref = evaluate_motor(
            "mcdonald",
            csdl.Variable(value=np.array([omega])),
            csdl.Variable(value=np.array([q])),
            SHAHJAHAN_EMRAX188_PARAMETERS,
        )
        q_cont = float(SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE.evaluate(rpm).value[0])

        assert float(mp.torque.value[0]) == pytest.approx(q, rel=1e-6)
        assert float(mp.efficiency.value[0]) == pytest.approx(float(ref.efficiency.value[0]), rel=1e-6)
        assert float(mp.electrical_power.value[0]) == pytest.approx(
            float(ref.electrical_power.value[0]), rel=1e-6)
        assert 0.85 < float(mp.efficiency.value[0]) < 0.97, tag
        margin = float(mp.torque_margin.value[0])
        assert margin == pytest.approx(q / (k * q_cont), rel=1e-6)
        assert margin <= 1.0, f"{tag}: torque margin {margin:.3f} exceeds the {k}x envelope"
    r.stop()


def test_mcdonald_torque_floor_prevents_nan_at_near_zero_power():
    csdl, r = _recorder()
    from BladeAD.optimisation.solve import evaluate_case_motor
    rpm = csdl.Variable(value=np.array([250.0]))
    shaft = csdl.Variable(value=np.array([1e-3]))       # SLSQP probe to ~zero power
    mp = evaluate_case_motor({"model": "mcdonald"}, rpm, shaft, overload=1.7)
    assert np.isfinite(float(mp.electrical_power.value[0]))
    assert np.isfinite(float(mp.efficiency.value[0]))
    r.stop()


def test_validate_accepts_mcdonald_case_and_rejects_bad_overload():
    from BladeAD.optimisation.case import load_case_dict
    case = load_case_dict(os.path.join(CASES, "shahjahan_test_motor_mcdonald"))
    assert case["motor"]["model"] == "mcdonald"

    import BladeAD.optimisation.case as casemod
    good = dict(case)
    bad = dict(case)
    bad["motor"] = {"model": "mcdonald", "k_hover": -1.0}
    with pytest.raises(ValueError):
        casemod._validate(dict(bad), os.path.join(CASES, "shahjahan_test_motor_mcdonald"))
    casemod._validate(dict(good), os.path.join(CASES, "shahjahan_test_motor_mcdonald"))
