import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal import (
    compute_hanson_line_load_harmonics,
    compute_hanson_line_source_loading,
)


def _adapter(sectional_thrust):
    azimuth_values = np.arange(8) * 2.0 * np.pi / 8.0
    azimuth = np.broadcast_to(azimuth_values, (1, 3, 8)).copy()
    widths = np.broadcast_to(np.array([0.1, 0.2, 0.15])[None, :, None], (1, 3, 8)).copy()
    weights = np.ones((1, 3, 8))
    weights[:, (0, -1), :] = 0.5
    return compute_hanson_line_load_harmonics(
        sectional_thrust=sectional_thrust,
        sectional_drag=0.1 * sectional_thrust,
        radial_element_width=csdl.Variable(value=widths),
        radial_integration_weights=csdl.Variable(value=weights),
        azimuth_angle=csdl.Variable(value=azimuth),
        tip_radius=csdl.Variable(value=np.array([0.75])),
        num_blades=2,
        harmonics=(0, 1, 2),
    )


def test_hanson_adapter_recovers_normalized_complex_line_load_harmonics():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    azimuth = np.arange(8) * 2.0 * np.pi / 8.0
    widths = np.array([0.1, 0.2, 0.15])
    mean = np.array([12.0, 18.0, 9.0])
    cosine_one = np.array([2.0, -3.0, 1.5])
    sine_two = np.array([0.8, 1.2, -0.6])
    line_load = (
        mean[:, None]
        + cosine_one[:, None] * np.cos(azimuth)[None, :]
        + sine_two[:, None] * np.sin(2.0 * azimuth)[None, :]
    )
    complete_rotor_elements = 2.0 * widths[:, None] * line_load
    outputs = _adapter(csdl.Variable(value=complete_rotor_elements[None, :, :]))

    expected_real = np.stack((mean, 0.5 * cosine_one, np.zeros(3)), axis=0)[None, :, :]
    expected_imaginary = np.stack(
        (np.zeros(3), np.zeros(3), -0.5 * sine_two), axis=0
    )[None, :, :]
    np.testing.assert_allclose(outputs.axial_real.value, expected_real, atol=2e-15)
    np.testing.assert_allclose(outputs.axial_imaginary.value, expected_imaginary, atol=2e-15)
    np.testing.assert_allclose(outputs.circumferential_real.value, 0.1 * expected_real, atol=2e-15)
    np.testing.assert_allclose(
        outputs.circumferential_imaginary.value, 0.1 * expected_imaginary, atol=2e-15
    )
    np.testing.assert_allclose(
        outputs.nondimensional_radial_weights.value,
        np.array([[0.5 * 0.1 / 0.75, 0.2 / 0.75, 0.5 * 0.15 / 0.75]]),
    )
    recorder.stop()


def test_hanson_adapter_steady_load_integral_recovers_per_blade_force():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    widths = np.array([0.1, 0.2, 0.15])
    weights = np.array([0.5, 1.0, 0.5])
    per_blade_elements = np.array([3.0, 7.0, 2.0])
    complete_rotor = np.broadcast_to((2.0 * per_blade_elements)[None, :, None], (1, 3, 8)).copy()
    outputs = _adapter(csdl.Variable(value=complete_rotor))
    reconstructed = np.sum(
        outputs.axial_real.value[0, 0] * widths * weights
    )
    np.testing.assert_allclose(reconstructed, np.sum(per_blade_elements * weights))
    recorder.stop()


def test_hanson_adapter_derivative_wrt_elemental_load():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    sectional_thrust = csdl.Variable(
        name="sectional_thrust", value=np.full((1, 3, 8), 4.0)
    )
    outputs = _adapter(sectional_thrust)
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.axial_real, outputs.axial_imaginary],
        [sectional_thrust],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_hanson_adapter_to_pressure_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    sectional_thrust = csdl.Variable(
        name="sectional_thrust", value=np.full((1, 3, 8), 4.0)
    )
    loads = _adapter(sectional_thrust)
    pressure = compute_hanson_line_source_loading(
        loads.axial_real,
        loads.axial_imaginary,
        loads.circumferential_real,
        loads.circumferential_imaginary,
        csdl.Variable(value=np.array([[0.3, 0.6, 0.9]])),
        loads.nondimensional_radial_weights,
        csdl.Variable(value=np.array([190.0])),
        csdl.Variable(value=np.array([0.75])),
        csdl.Variable(value=np.array([343.0])),
        csdl.Variable(value=np.array([0.1])),
        csdl.Variable(value=np.array([[30.0]])),
        csdl.Variable(value=np.array([[1.1]])),
        csdl.Variable(value=np.array([[0.2]])),
        num_blades=2,
        modes=(1, 2),
        load_harmonics=loads.harmonic_numbers,
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [pressure.cosine_pressure, pressure.sine_pressure],
        [sectional_thrust],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
