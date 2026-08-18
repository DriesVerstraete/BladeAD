import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics import compute_convected_distance


def test_convected_distance_matches_lowson_forward_aft_correction():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    distance = np.array([[100.0, 100.0, 50.0]])
    direction = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    velocity = np.array([[34.0, 0.0, 0.0]])
    convected = compute_convected_distance(
        csdl.Variable(value=distance),
        csdl.Variable(value=direction),
        csdl.Variable(value=velocity),
        csdl.Variable(value=np.array([340.0])),
    )
    np.testing.assert_allclose(convected.value, [[90.0, 110.0, 50.0]], atol=1e-14)
    recorder.stop()


def test_convected_distance_derivative_wrt_source_velocity():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    velocity = csdl.Variable(name="source_velocity", value=np.array([[20.0, -5.0, 2.0]]))
    convected = compute_convected_distance(
        csdl.Variable(value=np.array([[40.0, 60.0]])),
        csdl.Variable(value=np.array([[[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]]])),
        velocity,
        csdl.Variable(value=np.array([343.0])),
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [convected],
        [velocity],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
