import csdl_alpha as csdl
import numpy as np
from scipy.special import jv

from BladeAD.core.acoustics.tonal import compute_hanson_line_source_loading
from BladeAD.core.acoustics.tonal import compute_hanson_retarded_geometry


def _inputs(axial_real):
    return dict(
        axial_force_real=axial_real,
        axial_force_imaginary=csdl.Variable(value=np.array([[[0.0, 0.0], [0.2, -0.1]]])),
        circumferential_force_real=csdl.Variable(value=np.array([[[1.0, 1.4], [0.3, 0.2]]])),
        circumferential_force_imaginary=csdl.Variable(value=np.array([[[0.0, 0.0], [-0.05, 0.08]]])),
        nondimensional_radius=csdl.Variable(value=np.array([[0.35, 0.8]])),
        radial_integration_weights=csdl.Variable(value=np.array([[0.3, 0.45]])),
        angular_speed=csdl.Variable(value=np.array([210.0])),
        tip_radius=csdl.Variable(value=np.array([0.9])),
        speed_of_sound=csdl.Variable(value=np.array([342.0])),
        axial_mach_number=csdl.Variable(value=np.array([0.12])),
        observer_distance=csdl.Variable(value=np.array([[35.0]])),
        observer_polar_angle=csdl.Variable(value=np.array([[1.05]])),
        observer_azimuth_angle=csdl.Variable(value=np.array([[0.4]])),
        num_blades=3,
        modes=(1, 2),
        load_harmonics=(0, 1),
    )


def test_hanson_line_source_loading_matches_complex_numpy_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    axial_real_value = np.array([[[8.0, 11.0], [0.7, -0.4]]])
    inputs = _inputs(csdl.Variable(value=axial_real_value))
    outputs = compute_hanson_line_source_loading(**inputs)

    axial = axial_real_value + 1j * inputs["axial_force_imaginary"].value
    circumferential = (
        inputs["circumferential_force_real"].value
        + 1j * inputs["circumferential_force_imaginary"].value
    )
    expected = np.zeros((1, 1, 2, 2, 2), dtype=complex)
    theta = inputs["observer_polar_angle"].value[0, 0]
    azimuth = inputs["observer_azimuth_angle"].value[0, 0]
    distance = inputs["observer_distance"].value[0, 0]
    omega = inputs["angular_speed"].value[0]
    radius = inputs["tip_radius"].value[0]
    sound_speed = inputs["speed_of_sound"].value[0]
    mach = inputs["axial_mach_number"].value[0]
    tip_mach = omega * radius / sound_speed
    convection = 1.0 - mach * np.cos(theta)
    for mode_index, mode in enumerate((1, 2)):
        n = mode * 3
        radiation = np.exp(1j * n * omega * distance / (sound_speed * convection))
        for harmonic_index, harmonic in enumerate((0, 1)):
            source_phase = np.exp(1j * (n - harmonic) * (azimuth - np.pi / 2.0))
            for radial_index, z in enumerate((0.35, 0.8)):
                argument = n * z * tip_mach * np.sin(theta) / convection
                integrand = (
                    n * z * tip_mach * np.cos(theta) / convection
                    * axial[0, harmonic_index, radial_index]
                    - (n - harmonic) * circumferential[0, harmonic_index, radial_index]
                ) * jv(n - harmonic, argument) / z
                expected[0, 0, mode_index, harmonic_index, radial_index] = (
                    1j
                    * 3
                    * radiation
                    * source_phase
                    * integrand
                    * inputs["radial_integration_weights"].value[0, radial_index]
                    / (4.0 * np.pi * distance * convection)
                )

    np.testing.assert_allclose(
        outputs.radial_harmonic_cosine_pressure.value, expected.real, rtol=2e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        outputs.radial_harmonic_sine_pressure.value, expected.imag, rtol=2e-12, atol=1e-15
    )
    np.testing.assert_allclose(outputs.cosine_pressure.value, expected.real.sum(axis=(3, 4)))
    np.testing.assert_allclose(outputs.sine_pressure.value, expected.imag.sum(axis=(3, 4)))
    recorder.stop()


def test_hanson_line_source_loading_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    axial_real = csdl.Variable(
        name="axial_force_real", value=np.array([[[8.0, 11.0], [0.7, -0.4]]])
    )
    observer_angle = csdl.Variable(name="observer_angle", value=np.array([[1.05]]))
    inputs = _inputs(axial_real)
    inputs["observer_polar_angle"] = observer_angle
    outputs = compute_hanson_line_source_loading(**inputs)
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.cosine_pressure, outputs.sine_pressure],
        [axial_real, observer_angle],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_hanson_retarded_geometry_matches_numpy_and_differentiates():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angle = csdl.Variable(name="geometric_angle", value=np.array([[2.1, 1.57]]))
    distance = csdl.Variable(value=np.array([[20.0, 20.0]]))
    mach = csdl.Variable(name="mach", value=np.array([0.22]))
    outputs = compute_hanson_retarded_geometry(distance, angle, mach)
    sine = np.sin(angle.value)
    expected_angle = np.arccos(
        np.cos(angle.value) * np.sqrt(1.0 - mach.value[:, None] ** 2 * sine**2)
        + mach.value[:, None] * sine**2
    )
    expected_distance = distance.value * sine / np.sin(expected_angle)
    np.testing.assert_allclose(outputs.polar_angle.value, expected_angle)
    np.testing.assert_allclose(outputs.distance.value, expected_distance)
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.distance, outputs.polar_angle],
        [angle, mach],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
