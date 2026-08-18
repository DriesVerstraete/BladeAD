import csdl_alpha as csdl
import numpy as np
from scipy.special import jv, yv

from BladeAD.core.acoustics.tonal import compute_sears_load_harmonics


def _evaluate_sears(angular_speed=None):
    radius = np.array([[0.35, 0.7]])
    width = np.array([[0.08, 0.08]])
    weights = np.array([[0.5, 1.0]])
    chord = np.array([[0.12, 0.08]])
    inflow = np.array([[0.12, 0.08]])
    if angular_speed is None:
        angular_speed = csdl.Variable(value=np.array([180.0]))
    outputs = compute_sears_load_harmonics(
        csdl.Variable(value=np.array([[8.0, 10.0]])),
        csdl.Variable(value=np.array([[0.8, 1.0]])),
        csdl.Variable(value=radius),
        csdl.Variable(value=width),
        csdl.Variable(value=weights),
        csdl.Variable(value=chord),
        csdl.Variable(value=inflow),
        angular_speed,
        csdl.Variable(value=np.array([1.225])),
        num_blades=2,
        harmonics=(0, 1, 3),
    )
    return outputs, radius, width, weights, chord, inflow


def test_sears_coefficients_match_independent_scipy_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    outputs, radius, width, weights, chord, inflow = _evaluate_sears()
    expected_a_t = np.zeros((1, 3, 2))
    expected_b_t = np.zeros((1, 3, 2))
    expected_a_d = np.zeros((1, 3, 2))
    expected_b_d = np.zeros((1, 3, 2))
    expected_a_t[:, 0, :] = np.array([[4.0, 5.0]])
    expected_a_d[:, 0, :] = np.array([[0.4, 0.5]])
    for output_index, harmonic in enumerate((1, 3), start=1):
        k = harmonic * chord / (2.0 * radius)
        j0, j1, y0, y1 = jv(0, k), jv(1, k), yv(0, k), yv(1, k)
        first = j1 + y0
        second = y1 - j0
        denominator = first**2 + second**2
        f_value = (j1 * first + y1 * second) / denominator
        g_value = -(y1 * y0 + j1 * j0) / denominator
        sears_real = f_value * j0 + g_value * j1
        sears_imaginary = g_value * j0 - f_value * j1 + j1
        gust = inflow * 180.0 * radius / harmonic * 0.06
        lift_per_length = 1.225 * (180.0 * radius) * chord * gust * np.pi
        scale = lift_per_length * width * weights
        expected_a_t[:, output_index, :] = sears_real * scale * np.cos(inflow)
        expected_a_d[:, output_index, :] = sears_real * scale * np.sin(inflow)
        expected_b_t[:, output_index, :] = sears_imaginary * scale * np.cos(inflow)
        expected_b_d[:, output_index, :] = sears_imaginary * scale * np.sin(inflow)
    np.testing.assert_allclose(outputs.thrust_cosine.value, expected_a_t, rtol=1e-12)
    np.testing.assert_allclose(outputs.thrust_sine.value, expected_b_t, rtol=1e-12)
    np.testing.assert_allclose(outputs.drag_cosine.value, expected_a_d, rtol=1e-12)
    np.testing.assert_allclose(outputs.drag_sine.value, expected_b_d, rtol=1e-12)
    recorder.stop()


def test_sears_coefficients_derivative_wrt_angular_speed():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(name="angular_speed", value=np.array([180.0]))
    outputs, *_ = _evaluate_sears(angular_speed)
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.thrust_cosine, outputs.thrust_sine],
        [angular_speed],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
