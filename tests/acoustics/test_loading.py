import csdl_alpha as csdl
import numpy as np
from scipy.special import jv

from BladeAD.core.acoustics.tonal import (
    LoadHarmonics,
    compute_lowson_loading_pressure,
    compute_lowson_steady_loading_pressure,
)


def _coefficients(a_t, b_t, a_d, b_d, harmonics):
    return LoadHarmonics(
        thrust_cosine=csdl.Variable(value=a_t),
        thrust_sine=csdl.Variable(value=b_t),
        drag_cosine=csdl.Variable(value=a_d),
        drag_sine=csdl.Variable(value=b_d),
        harmonic_numbers=harmonics,
    )


def test_lowson_loading_pressure_matches_equation_10_complex_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    harmonics = (0, 1, 3)
    a_t = np.array([[[5.0, 7.0], [1.2, -0.4], [0.3, 0.6]]])
    b_t = np.array([[[0.0, 0.0], [-0.7, 0.8], [0.2, -0.1]]])
    a_d = 0.12 * a_t
    b_d = 0.15 * b_t
    radius = np.array([[0.4, 0.75]])
    omega = np.array([190.0])
    axial = np.array([[18.0]])
    in_plane = np.array([[24.0]])
    distance = np.array([[30.0]])
    sound_speed = np.array([340.0])
    outputs = compute_lowson_loading_pressure(
        _coefficients(a_t, b_t, a_d, b_d, harmonics),
        csdl.Variable(value=radius),
        csdl.Variable(value=omega),
        csdl.Variable(value=axial),
        csdl.Variable(value=in_plane),
        csdl.Variable(value=distance),
        csdl.Variable(value=sound_speed),
        num_blades=2,
        modes=(1,),
    )

    expected = np.zeros((1, 1, 1, len(harmonics), 2), dtype=complex)
    n = 2
    for harmonic_index, lam in enumerate(harmonics):
        for radial_index, radial_station in enumerate(radius[0]):
            argument = n * omega[0] * radial_station * in_plane[0, 0] / (
                sound_speed[0] * distance[0, 0]
            )
            j_minus = jv(n - lam, argument)
            j_plus = jv(n + lam, argument)
            alternating = (-1) ** lam
            radiation = n * omega[0] * axial[0, 0] / (
                sound_speed[0] * distance[0, 0] ** 2
            )
            near = 1.0 / (radial_station * distance[0, 0])
            inner = (
                radiation
                * (
                    1j * a_t[0, harmonic_index, radial_index]
                    * (j_minus + alternating * j_plus)
                    - b_t[0, harmonic_index, radial_index]
                    * (j_minus - alternating * j_plus)
                )
                - near
                * (
                    1j * a_d[0, harmonic_index, radial_index]
                    * ((n - lam) * j_minus + alternating * (n + lam) * j_plus)
                    - b_d[0, harmonic_index, radial_index]
                    * ((n - lam) * j_minus - alternating * (n + lam) * j_plus)
                )
            )
            expected[0, 0, 0, harmonic_index, radial_index] = (
                1j ** (-(n - lam)) * inner / (4.0 * np.pi)
            )

    np.testing.assert_allclose(
        outputs.radial_harmonic_cosine_pressure.value, expected.real, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        outputs.radial_harmonic_sine_pressure.value, expected.imag, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(outputs.cosine_pressure.value, expected.real.sum(axis=(3, 4)))
    np.testing.assert_allclose(outputs.sine_pressure.value, expected.imag.sum(axis=(3, 4)))
    recorder.stop()


def test_general_loading_kernel_reduces_to_steady_kernel_at_lambda_zero():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    zeros = np.zeros((1, 1, 2))
    coefficients = _coefficients(
        np.array([[[6.0, 8.0]]]), zeros, np.array([[[0.5, 0.7]]]), zeros, (0,)
    )
    arguments = (
        coefficients,
        csdl.Variable(value=np.array([[0.4, 0.7]])),
        csdl.Variable(value=np.array([180.0])),
        csdl.Variable(value=np.array([[12.0, -8.0]])),
        csdl.Variable(value=np.array([[16.0, 15.0]])),
        csdl.Variable(value=np.array([[20.0, 17.0]])),
        csdl.Variable(value=np.array([343.0])),
    )
    general = compute_lowson_loading_pressure(*arguments, num_blades=2, modes=(1, 2))
    steady = compute_lowson_steady_loading_pressure(*arguments, num_blades=2, modes=(1, 2))
    np.testing.assert_allclose(general.cosine_pressure.value, steady.cosine_pressure.value)
    np.testing.assert_allclose(general.sine_pressure.value, steady.sine_pressure.value)
    recorder.stop()


def test_lowson_loading_pressure_derivative_wrt_unsteady_load():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    thrust_cosine = csdl.Variable(
        name="thrust_cosine", value=np.array([[[5.0, 6.0], [0.8, -0.3]]])
    )
    coefficients = LoadHarmonics(
        thrust_cosine=thrust_cosine,
        thrust_sine=csdl.Variable(value=np.array([[[0.0, 0.0], [0.2, 0.4]]])),
        drag_cosine=csdl.Variable(value=np.array([[[0.5, 0.6], [0.08, -0.03]]])),
        drag_sine=csdl.Variable(value=np.array([[[0.0, 0.0], [0.02, 0.04]]])),
        harmonic_numbers=(0, 1),
    )
    outputs = compute_lowson_loading_pressure(
        coefficients,
        csdl.Variable(value=np.array([[0.4, 0.7]])),
        csdl.Variable(value=np.array([180.0])),
        csdl.Variable(value=np.array([[12.0]])),
        csdl.Variable(value=np.array([[16.0]])),
        csdl.Variable(value=np.array([[20.0]])),
        csdl.Variable(value=np.array([343.0])),
        num_blades=2,
        modes=(1, 2),
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.cosine_pressure, outputs.sine_pressure],
        [thrust_cosine],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
