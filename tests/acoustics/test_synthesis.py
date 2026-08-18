import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal import synthesize_lowson_rotor_pressure


def test_complete_rotor_blades_add_coherently_at_blade_passing_harmonics():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    cosine = np.array([[[1.0, 3.0]]])
    sine = np.array([[[2.0, 4.0]]])
    num_blades = 3
    outputs = synthesize_lowson_rotor_pressure(
        csdl.Variable(value=cosine),
        csdl.Variable(value=sine),
        num_blades=num_blades,
        pressure_squared_floor=1e-30,
    )
    expected_mode_squared = 0.5 * num_blades**2 * (cosine**2 + sine**2)
    np.testing.assert_allclose(outputs.rotor_cosine_pressure.value, num_blades * cosine)
    np.testing.assert_allclose(outputs.rotor_sine_pressure.value, num_blades * sine)
    np.testing.assert_allclose(outputs.mode_pressure_squared.value, expected_mode_squared)
    np.testing.assert_allclose(
        outputs.total_pressure_squared.value, expected_mode_squared.sum(axis=2)
    )
    expected_spl = 10.0 * np.log10(expected_mode_squared / (20e-6) ** 2)
    np.testing.assert_allclose(outputs.mode_spl.value, expected_spl)
    recorder.stop()


def test_coherent_blade_gain_is_twenty_log10_blade_count():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    cosine = csdl.Variable(value=np.array([[[0.2]]]))
    sine = csdl.Variable(value=np.array([[[0.1]]]))
    one_blade = synthesize_lowson_rotor_pressure(
        cosine, sine, num_blades=1, pressure_squared_floor=1e-30
    )
    four_blades = synthesize_lowson_rotor_pressure(
        cosine, sine, num_blades=4, pressure_squared_floor=1e-30
    )
    np.testing.assert_allclose(
        four_blades.mode_spl.value - one_blade.mode_spl.value,
        20.0 * np.log10(4.0),
        rtol=1e-12,
    )
    recorder.stop()


def test_rotor_synthesis_derivative_wrt_per_blade_pressure():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    cosine = csdl.Variable(name="cosine_pressure", value=np.array([[[0.2, -0.1]]]))
    outputs = synthesize_lowson_rotor_pressure(
        cosine,
        csdl.Variable(value=np.array([[[0.1, 0.3]]])),
        num_blades=3,
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.total_spl],
        [cosine],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
