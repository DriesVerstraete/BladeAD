import csdl_alpha as csdl
import numpy as np
import pytest

from BladeAD.core.motor import (
    ChebyshevTorqueEnvelope,
    McDonaldParameters,
    ThreeConstantParameters,
    evaluate_motor,
)


def _vertiia_parameters():
    return McDonaldParameters(
        peak_efficiency=9.92744356e-1,
        peak_efficiency_rpm=4.76048797e3,
        peak_efficiency_torque=1.16569167e3,
        k0=2.69987571,
        c4=-1.44472392e-2,
        c5=-7.65017707e-2,
        c6=4.79935546e-2,
        c7=2.53643136e-4,
        c8=-3.80916375e-5,
        c9=-1.35801217e-3,
        c10=-1.78327161e-2,
        c11=1.64008569e-2,
        c12=-2.22427579e-2,
    )


def test_mcdonald_matches_shahjahan_reference_and_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(name="angular_speed", value=np.array([200.0, 300.0]))
    torque = csdl.Variable(name="torque", value=np.array([40.0, 60.0]))
    parameters = _vertiia_parameters()
    outputs = evaluate_motor("mcdonald", angular_speed, torque, parameters)

    c0, c1, c2, c3 = parameters.derived_coefficients
    omega = angular_speed.value
    q = torque.value
    expected_loss = (
        c0 + c1 * omega + c2 * omega**3 + c3 * q**2
        + parameters.c4 * q + parameters.c5 * omega * q + parameters.c6 * omega**2
        + parameters.c7 * omega * q**2 + parameters.c8 * omega**2 * q
        + parameters.c9 * omega * np.log(omega) + parameters.c10 * q * np.log(q)
        + parameters.c11 * np.log(omega) + parameters.c12 * np.log(q)
    )
    expected_shaft_power = angular_speed.value * torque.value
    expected_efficiency = expected_shaft_power / (expected_shaft_power + expected_loss)
    np.testing.assert_allclose(outputs.power_loss.value, expected_shaft_power / expected_efficiency - expected_shaft_power)
    np.testing.assert_allclose(outputs.shaft_power.value, expected_shaft_power)
    np.testing.assert_allclose(outputs.electrical_power.value, expected_shaft_power / expected_efficiency)
    np.testing.assert_allclose(outputs.efficiency.value, expected_efficiency)
    assert outputs.current is None
    assert outputs.voltage is None

    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.electrical_power, outputs.efficiency],
        [angular_speed, torque],
        1.0e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_three_constant_matches_equivalent_circuit_reference_and_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(name="angular_speed", value=np.array([200.0, 300.0]))
    torque = csdl.Variable(name="torque", value=np.array([10.0, 15.0]))
    parameters = ThreeConstantParameters(
        speed_constant=5.0,
        resistance=0.08,
        no_load_current=1.5,
    )
    outputs = evaluate_motor("three_constant", angular_speed, torque, parameters)

    expected_current = 1.5 + 5.0 * torque.value
    expected_voltage = angular_speed.value / 5.0 + expected_current * 0.08
    expected_shaft_power = angular_speed.value * torque.value
    expected_electrical_power = expected_current * expected_voltage
    np.testing.assert_allclose(outputs.current.value, expected_current)
    np.testing.assert_allclose(outputs.voltage.value, expected_voltage)
    np.testing.assert_allclose(outputs.shaft_power.value, expected_shaft_power)
    np.testing.assert_allclose(outputs.electrical_power.value, expected_electrical_power)
    np.testing.assert_allclose(outputs.power_loss.value, expected_electrical_power - expected_shaft_power)
    np.testing.assert_allclose(outputs.efficiency.value, expected_shaft_power / expected_electrical_power)

    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.electrical_power, outputs.efficiency, outputs.current, outputs.voltage],
        [angular_speed, torque],
        1.0e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_model_and_parameter_validation():
    with pytest.raises(ValueError):
        McDonaldParameters(0.0, 1000.0, 10.0, *(0.0,) * 10)
    with pytest.raises(ValueError):
        ThreeConstantParameters(speed_constant=0.0, resistance=0.1, no_load_current=1.0)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(value=np.array([200.0]))
    torque = csdl.Variable(value=np.array([10.0]))
    with pytest.raises(TypeError):
        evaluate_motor(
            "mcdonald",
            angular_speed,
            torque,
            ThreeConstantParameters(speed_constant=5.0, resistance=0.1, no_load_current=1.0),
        )
    with pytest.raises(ValueError):
        evaluate_motor(
            "unknown",
            angular_speed,
            torque,
            _vertiia_parameters(),
        )
    recorder.stop()


def test_chebyshev_torque_envelope_values_and_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    rpm = csdl.Variable(value=np.array([0.0, 2250.0, 4500.0]))
    envelope = ChebyshevTorqueEnvelope(0.0, 4500.0, (2.0, 0.5, -0.25))
    torque = envelope.evaluate(rpm)
    x = 2.0 * rpm.value / 4500.0 - 1.0
    np.testing.assert_allclose(torque.value, 2.0 + 0.5 * x - 0.25 * (2.0 * x**2 - 1.0))
    errors = csdl.derivative_utils.verify_derivatives(
        [torque], [rpm], 1.0e-6, print_results=False, raise_on_error=True
    )
    assert errors is not None
    recorder.stop()
