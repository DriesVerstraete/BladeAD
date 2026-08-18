import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal import compute_load_harmonics


def test_load_harmonics_match_independent_discrete_fourier_reference():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    azimuth = np.arange(4) * np.pi / 2
    per_blade_thrust = np.stack(
        [10.0 + 2.0 * np.cos(azimuth) + 3.0 * np.sin(azimuth),
         20.0 - np.cos(azimuth) + 0.5 * np.sin(azimuth)]
    )[None, :, :]
    per_blade_drag = 0.2 * per_blade_thrust
    num_blades = 2
    azimuth_grid = np.broadcast_to(azimuth, per_blade_thrust.shape)
    outputs = compute_load_harmonics(
        sectional_thrust=csdl.Variable(value=num_blades * per_blade_thrust),
        sectional_drag=csdl.Variable(value=num_blades * per_blade_drag),
        azimuth_angle=csdl.Variable(value=azimuth_grid),
        num_blades=num_blades,
        harmonics=(0, 1, 2),
    )

    phase = np.asarray((0, 1, 2))[:, None] * azimuth[None, :]
    expected_thrust_cosine = np.mean(
        per_blade_thrust[:, None, :, :] * np.cos(phase)[None, :, None, :], axis=3
    )
    expected_thrust_sine = -np.mean(
        per_blade_thrust[:, None, :, :] * np.sin(phase)[None, :, None, :], axis=3
    )
    np.testing.assert_allclose(outputs.thrust_cosine.value, expected_thrust_cosine, atol=1e-14)
    np.testing.assert_allclose(outputs.thrust_sine.value, expected_thrust_sine, atol=1e-14)
    np.testing.assert_allclose(outputs.drag_cosine.value, 0.2 * expected_thrust_cosine, atol=1e-14)
    np.testing.assert_allclose(outputs.drag_sine.value, 0.2 * expected_thrust_sine, atol=1e-14)
    recorder.stop()


def test_load_harmonic_derivatives_wrt_complete_rotor_loads():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    sectional_thrust = csdl.Variable(
        name="sectional_thrust", value=np.arange(1.0, 9.0).reshape((1, 2, 4))
    )
    sectional_drag = csdl.Variable(value=np.full((1, 2, 4), 0.5))
    azimuth = csdl.Variable(
        value=np.broadcast_to(np.arange(4) * np.pi / 2, (1, 2, 4))
    )
    outputs = compute_load_harmonics(
        sectional_thrust,
        sectional_drag,
        azimuth,
        num_blades=2,
        harmonics=(0, 1, 2),
    )

    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.thrust_cosine, outputs.thrust_sine],
        [sectional_thrust],
        1e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
