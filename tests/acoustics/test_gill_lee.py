import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.broadband import compute_gill_lee_broadband


def _numpy_reference(ct, chord, radius, rpm, velocity, sound_speed, distance, angle, blades, hub, frequencies):
    dr = (1.0 - hub) * radius / (chord.size - 1)
    area = np.sum(chord) * dr
    solidity = area * blades / (np.pi * radius**2)
    weighted_chord = solidity * np.pi * radius / blades
    tip_speed = rpm * 2.0 * np.pi / 60.0 * radius - np.linalg.norm(velocity)
    tip_mach = tip_speed / sound_speed
    strouhal = frequencies * weighted_chord / tip_speed
    shift = (
        solidity * np.log10(ct)
        + 0.9 * tip_mach * solidity * (tip_mach + 3.82) * np.log10(solidity)
    )
    coordinate = strouhal - shift
    velocity_term = 10.0 * np.log10(tip_speed**7.84)
    exponent_one = -2.0 * tip_mach**2 + 2.06
    exponent_two = -ct * tip_mach * (ct - np.sin(abs(angle)) + 2.06) + 1.0
    distance_exponent = (
        4.97
        * ct
        * np.sin(abs(angle))
        * (1.5 * distance / radius * tip_mach - distance / radius + 15.0)
    )
    return velocity_term * coordinate**0.6 / (
        (coordinate + exponent_one) ** exponent_two
        + (ct * coordinate) ** distance_exponent
    )


def test_gill_lee_matches_independent_numpy_reference_and_derivatives():
    chord_value = np.array(
        [
            0.0156274, 0.0191535, 0.0223431, 0.0251964, 0.0276285,
            0.0295553, 0.0308092, 0.0316424, 0.0320548, 0.0317943,
            0.0309440, 0.0297578, 0.0277292, 0.0251962, 0.0219899,
            0.0184473, 0.0117902, 0.0048810,
        ]
    )
    frequencies = np.array([100.0, 1000.0, 10000.0])
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    ct = csdl.Variable(value=np.array([0.08]))
    chord = csdl.Variable(value=chord_value)
    rpm = csdl.Variable(value=np.array([4200.0]))
    outputs = compute_gill_lee_broadband(
        ct,
        chord,
        csdl.Variable(value=np.array([0.1397])),
        rpm,
        csdl.Variable(value=np.array([[4.91546153, 0.0, 0.0]])),
        csdl.Variable(value=np.array([343.0])),
        csdl.Variable(value=np.array([[1.905]])),
        csdl.Variable(value=np.array([[np.pi / 4.0]])),
        2,
        0.15,
        frequencies,
    )
    expected = _numpy_reference(
        0.08,
        chord_value,
        0.1397,
        4200.0,
        np.array([4.91546153, 0.0, 0.0]),
        343.0,
        1.905,
        np.pi / 4.0,
        2,
        0.15,
        frequencies,
    )
    np.testing.assert_allclose(
        outputs.one_third_octave_spl.value[0, 0], expected, rtol=1.0e-12
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.total_spl],
        [ct, chord, rpm],
        1.0e-5,
        print_results=False,
        raise_on_error=False,
    )
    for variable in (ct, chord, rpm):
        result = errors[(outputs.total_spl, variable)]
        assert np.linalg.norm(result["value"]) > 1.0e-8
        assert result["rel_error"] < 1.0e-4
    recorder.stop()


def test_gill_lee_optimizer_trial_outside_empirical_domain_remains_finite():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    outputs = compute_gill_lee_broadband(
        csdl.Variable(value=np.array([-0.02])),
        csdl.Variable(value=np.full(8, 0.05)),
        csdl.Variable(value=np.array([1.0])),
        csdl.Variable(value=np.array([500.0])),
        csdl.Variable(value=np.array([[60.0, 0.0, 0.0]])),
        csdl.Variable(value=np.array([343.0])),
        csdl.Variable(value=np.array([[25.0]])),
        csdl.Variable(value=np.array([[0.0]])),
        3,
        0.2,
        np.array([100.0, 1000.0, 10000.0]),
    )
    assert np.all(np.isfinite(outputs.one_third_octave_spl.value))
    assert np.all(np.isfinite(outputs.total_spl.value))
    recorder.stop()
