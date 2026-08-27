import csdl_alpha as csdl
import numpy as np
import pytest

from BladeAD.core.motor import (
    McDonaldParameters,
    ThreeConstantParameters,
    evaluate_motor,
)


def test_mcdonald_matches_polynomial_reference_and_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(name="angular_speed", value=np.array([200.0, 300.0]))
    torque = csdl.Variable(name="torque", value=np.array([40.0, 60.0]))
    parameters = McDonaldParameters(
        loss_coefficients={(0, 0): 50.0, (0, 1): 0.2, (2, 0): 0.04, (1, 1): 0.001}
    )
    outputs = evaluate_motor("mcdonald", angular_speed, torque, parameters)

    expected_loss = 50.0 + 0.2 * angular_speed.value + 0.04 * torque.value**2 + 0.001 * torque.value * angular_speed.value
    expected_shaft_power = angular_speed.value * torque.value
    np.testing.assert_allclose(outputs.power_loss.value, expected_loss)
    np.testing.assert_allclose(outputs.shaft_power.value, expected_shaft_power)
    np.testing.assert_allclose(outputs.electrical_power.value, expected_shaft_power + expected_loss)
    np.testing.assert_allclose(outputs.efficiency.value, expected_shaft_power / (expected_shaft_power + expected_loss))
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
        McDonaldParameters(loss_coefficients={})
    with pytest.raises(ValueError):
        McDonaldParameters(loss_coefficients={(0, 0): -1.0})
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
            McDonaldParameters(loss_coefficients={(0, 0): 1.0}),
        )
    recorder.stop()
