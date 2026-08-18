import csdl_alpha as csdl
import numpy as np
from scipy.special import jv

from BladeAD.core.acoustics.tonal import compute_hanson_thickness_noise


def _inputs(thickness_to_chord, observer_angle):
    return dict(
        nondimensional_radius=csdl.Variable(value=np.array([[0.35, 0.8]])),
        radial_integration_weights=csdl.Variable(value=np.array([[0.12, 0.18]])),
        chord=csdl.Variable(value=np.array([[0.16, 0.09]])),
        thickness_to_chord=thickness_to_chord,
        normalized_thickness_shape=csdl.Variable(value=np.array([0.1, 0.8, 1.0, 0.5, 0.05])),
        chordwise_locations=csdl.Variable(value=np.array([-0.4, -0.2, 0.0, 0.2, 0.4])),
        chordwise_integration_weights=csdl.Variable(value=np.full(5, 0.2)),
        angular_speed=csdl.Variable(value=np.array([205.0])),
        tip_radius=csdl.Variable(value=np.array([0.9])),
        density=csdl.Variable(value=np.array([1.18])),
        speed_of_sound=csdl.Variable(value=np.array([341.0])),
        axial_mach_number=csdl.Variable(value=np.array([0.11])),
        observer_distance=csdl.Variable(value=np.array([[32.0]])),
        observer_polar_angle=observer_angle,
        observer_azimuth_angle=csdl.Variable(value=np.array([[0.3]])),
        num_blades=3,
        modes=(1, 2),
    )


def test_hanson_thickness_matches_complex_numpy_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    inputs = _inputs(
        csdl.Variable(value=np.array([[0.12, 0.09]])),
        csdl.Variable(value=np.array([[1.0]])),
    )
    outputs = compute_hanson_thickness_noise(**inputs)
    expected = np.zeros((1, 1, 2, 2), dtype=complex)
    z_values = (0.35, 0.8)
    chord_values = (0.16, 0.09)
    tc_values = (0.12, 0.09)
    radial_weights = (0.12, 0.18)
    shape = np.array([0.1, 0.8, 1.0, 0.5, 0.05])
    x = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    omega = 205.0
    radius = 0.9
    sound_speed = 341.0
    mach = 0.11
    theta = 1.0
    azimuth = 0.3
    distance = 32.0
    tip_mach = omega * radius / sound_speed
    convection = 1.0 - mach * np.cos(theta)
    for mode_index, mode in enumerate((1, 2)):
        n = mode * 3
        radiation = np.exp(
            1j
            * (
                n * omega * distance / (sound_speed * convection)
                + n * (azimuth - np.pi / 2.0)
            )
        )
        for radial_index, z in enumerate(z_values):
            helicoid = np.arctan(mach / (z * tip_mach))
            kx = chord_values[radial_index] / radius * (
                n * np.cos(helicoid) / z
                + n * tip_mach * np.cos(theta) * np.sin(helicoid) / convection
            )
            psi = np.sum(shape * np.exp(1j * kx * x) * 0.2)
            argument = n * z * tip_mach * np.sin(theta) / convection
            relative_mach_squared = mach**2 + (z * tip_mach) ** 2
            integrand = (
                relative_mach_squared
                * kx**2
                * tc_values[radial_index]
                * psi
                * jv(n, argument)
                * radial_weights[radial_index]
            )
            expected[0, 0, mode_index, radial_index] = (
                -1.18
                * sound_speed**2
                * 3
                * radius
                * radiation
                * integrand
                / (4.0 * np.pi * distance * convection)
            )
    np.testing.assert_allclose(outputs.radial_cosine_pressure.value, expected.real, rtol=2e-12)
    np.testing.assert_allclose(outputs.radial_sine_pressure.value, expected.imag, rtol=2e-12)
    np.testing.assert_allclose(outputs.cosine_pressure.value, expected.real.sum(axis=3))
    np.testing.assert_allclose(outputs.sine_pressure.value, expected.imag.sum(axis=3))
    recorder.stop()


def test_hanson_thickness_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    thickness = csdl.Variable(name="thickness", value=np.array([[0.12, 0.09]]))
    observer_angle = csdl.Variable(name="observer_angle", value=np.array([[1.0]]))
    outputs = compute_hanson_thickness_noise(**_inputs(thickness, observer_angle))
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.cosine_pressure, outputs.sine_pressure],
        [thickness, observer_angle],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_radially_varying_thickness_shape_matches_repeated_common_shape():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    thickness = csdl.Variable(value=np.array([[0.12, 0.09]]))
    observer_angle = csdl.Variable(value=np.array([[1.0]]))
    common_inputs = _inputs(thickness, observer_angle)
    common = compute_hanson_thickness_noise(**common_inputs)
    common_shape = common_inputs["normalized_thickness_shape"].value
    radial_inputs = dict(common_inputs)
    radial_inputs["normalized_thickness_shape"] = csdl.Variable(
        value=np.broadcast_to(common_shape, (2, len(common_shape)))
    )
    radial = compute_hanson_thickness_noise(**radial_inputs)

    np.testing.assert_allclose(radial.cosine_pressure.value, common.cosine_pressure.value)
    np.testing.assert_allclose(radial.sine_pressure.value, common.sine_pressure.value)
    recorder.stop()
