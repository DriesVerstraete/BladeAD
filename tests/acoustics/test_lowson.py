import csdl_alpha as csdl
import numpy as np
from scipy.special import jv

from BladeAD.core.acoustics import compute_convected_distance
from BladeAD.core.acoustics.tonal import (
    LoadHarmonics,
    compute_lowson_steady_loading_pressure,
)


def _load_harmonics(thrust, drag):
    zeros = csdl.Variable(value=np.zeros_like(thrust))
    return LoadHarmonics(
        thrust_cosine=csdl.Variable(value=thrust),
        thrust_sine=zeros,
        drag_cosine=csdl.Variable(value=drag),
        drag_sine=zeros,
        harmonic_numbers=(0,),
    )


def test_lowson_steady_kernel_matches_independent_numpy_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    thrust = np.array([[[6.0, 9.0]]])
    drag = np.array([[[0.8, 1.1]]])
    radius = np.array([[0.35, 0.75]])
    omega = np.array([180.0])
    sound_speed = np.array([343.0])
    axial = np.array([[12.0, -7.0]])
    in_plane = np.array([[16.0, 24.0]])
    distance = np.sqrt(axial**2 + in_plane**2)
    outputs = compute_lowson_steady_loading_pressure(
        _load_harmonics(thrust, drag),
        csdl.Variable(value=radius),
        csdl.Variable(value=omega),
        csdl.Variable(value=axial),
        csdl.Variable(value=in_plane),
        csdl.Variable(value=distance),
        csdl.Variable(value=sound_speed),
        num_blades=2,
        modes=(1, 2),
    )

    orders = np.array([2.0, 4.0])[None, None, :, None]
    expanded_radius = radius[:, None, None, :]
    expanded_distance = distance[:, :, None, None]
    argument = (
        orders * omega[:, None, None, None] * expanded_radius
        * in_plane[:, :, None, None]
        / (sound_speed[:, None, None, None] * expanded_distance)
    )
    bessel = jv(orders, argument)
    radiation = (
        orders * omega[:, None, None, None] * axial[:, :, None, None]
        * thrust[:, :, None, :] * bessel
        / (sound_speed[:, None, None, None] * expanded_distance**2)
    )
    near_field = (
        orders * drag[:, :, None, :] * bessel
        / (expanded_radius * expanded_distance)
    )
    radiation_sign = np.array([-1.0, 1.0])[None, None, :, None]
    expected_sine = (radiation_sign * radiation - radiation_sign * near_field) / (2 * np.pi)

    np.testing.assert_allclose(outputs.bessel_argument.value, argument, rtol=1e-13)
    np.testing.assert_allclose(outputs.radial_cosine_pressure.value, 0.0, atol=1e-15)
    np.testing.assert_allclose(outputs.radial_sine_pressure.value, expected_sine, rtol=1e-12)
    np.testing.assert_allclose(outputs.sine_pressure.value, expected_sine.sum(axis=3), rtol=1e-12)
    recorder.stop()


def test_lowson_odd_acoustic_order_routes_pressure_to_cosine_component():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    outputs = compute_lowson_steady_loading_pressure(
        _load_harmonics(np.array([[[5.0]]]), np.array([[[0.4]]])),
        csdl.Variable(value=np.array([[0.6]])),
        csdl.Variable(value=np.array([160.0])),
        csdl.Variable(value=np.array([[8.0]])),
        csdl.Variable(value=np.array([[15.0]])),
        csdl.Variable(value=np.array([[17.0]])),
        csdl.Variable(value=np.array([340.0])),
        num_blades=3,
        modes=(1,),
    )
    assert outputs.acoustic_harmonic_orders == (3,)
    argument = 3.0 * 160.0 * 0.6 * 15.0 / (340.0 * 17.0)
    bessel = jv(3, argument)
    radiation = 3.0 * 160.0 * 8.0 * 5.0 * bessel / (340.0 * 17.0**2)
    near_field = 3.0 * 0.4 * bessel / (0.6 * 17.0)
    expected_cosine = (-radiation + near_field) / (2.0 * np.pi)
    np.testing.assert_allclose(outputs.radial_sine_pressure.value, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        outputs.radial_cosine_pressure.value,
        np.array([[[[expected_cosine]]]]),
        rtol=1e-12,
    )
    recorder.stop()


def test_lowson_kernel_derivative_wrt_angular_speed():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(name="angular_speed", value=np.array([175.0]))
    outputs = compute_lowson_steady_loading_pressure(
        _load_harmonics(np.array([[[7.0, 8.0]]]), np.array([[[0.5, 0.6]]])),
        csdl.Variable(value=np.array([[0.4, 0.7]])),
        angular_speed,
        csdl.Variable(value=np.array([[10.0]])),
        csdl.Variable(value=np.array([[20.0]])),
        csdl.Variable(value=np.array([[np.sqrt(500.0)]])),
        csdl.Variable(value=np.array([343.0])),
        num_blades=2,
        modes=(1, 2),
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.sine_pressure],
        [angular_speed],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_lowson_kernel_uses_supplied_convected_distance():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    physical_distance = csdl.Variable(value=np.array([[25.0]]))
    convected_distance = csdl.Variable(value=np.array([[20.0]]))
    outputs = compute_lowson_steady_loading_pressure(
        _load_harmonics(np.array([[[5.0]]]), np.array([[[0.3]]])),
        csdl.Variable(value=np.array([[0.5]])),
        csdl.Variable(value=np.array([170.0])),
        csdl.Variable(value=np.array([[15.0]])),
        csdl.Variable(value=np.array([[20.0]])),
        physical_distance,
        csdl.Variable(value=np.array([340.0])),
        num_blades=2,
        modes=(1,),
        convected_distance=convected_distance,
    )
    expected_argument = 2.0 * 170.0 * 0.5 * 20.0 / (340.0 * 20.0)
    np.testing.assert_allclose(outputs.radiation_distance.value, [[20.0]])
    np.testing.assert_allclose(outputs.bessel_argument.value, expected_argument)
    recorder.stop()


def test_lowson_pressure_derivative_wrt_source_velocity_through_convection():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    source_velocity = csdl.Variable(
        name="source_velocity", value=np.array([[25.0, 0.0, 0.0]])
    )
    observer_distance = csdl.Variable(value=np.array([[30.0]]))
    convected_distance = compute_convected_distance(
        observer_distance,
        csdl.Variable(value=np.array([[[0.8, 0.6, 0.0]]])),
        source_velocity,
        csdl.Variable(value=np.array([340.0])),
    )
    outputs = compute_lowson_steady_loading_pressure(
        _load_harmonics(np.array([[[6.0, 7.0]]]), np.array([[[0.4, 0.5]]])),
        csdl.Variable(value=np.array([[0.4, 0.7]])),
        csdl.Variable(value=np.array([180.0])),
        csdl.Variable(value=np.array([[18.0]])),
        csdl.Variable(value=np.array([[24.0]])),
        observer_distance,
        csdl.Variable(value=np.array([340.0])),
        num_blades=2,
        modes=(1,),
        convected_distance=convected_distance,
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.sine_pressure],
        [source_velocity],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
