from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np


@dataclass
class GillLeeBroadbandOutputs(csdl.VariableGroup):
    frequencies: csdl.Variable
    one_third_octave_spl: csdl.Variable
    one_third_octave_pressure_squared: csdl.Variable
    total_pressure_squared: csdl.Variable
    total_spl: csdl.Variable


def _positive_guard(value, floor=1.0e-12):
    return floor + 0.5 * (value + csdl.sqrt(value**2 + 4.0 * floor**2))


def compute_gill_lee_broadband(
    thrust_coefficient,
    chord_profile,
    rotor_radius,
    rpm,
    source_velocity,
    speed_of_sound,
    observer_distance,
    observer_angle_from_plane,
    num_blades,
    norm_hub_radius,
    center_frequencies,
    reference_pressure=20.0e-6,
):
    frequencies = csdl.Variable(value=np.asarray(center_frequencies, dtype=float))
    if frequencies.shape[0] == 0 or np.any(np.asarray(center_frequencies) <= 0.0):
        raise ValueError("Gill-Lee center frequencies must be positive.")
    if num_blades <= 0:
        raise ValueError("Gill-Lee requires a positive blade count.")
    if not 0.0 <= norm_hub_radius < 1.0:
        raise ValueError("Normalized hub radius must lie in [0, 1).")

    num_nodes, num_observers = observer_distance.shape
    num_frequencies = frequencies.shape[0]
    if thrust_coefficient.shape != (num_nodes,):
        raise ValueError("Thrust coefficient must have shape (node,).")
    if rpm.shape != (num_nodes,):
        raise ValueError("RPM must have shape (node,).")
    if source_velocity.shape != (num_nodes, 3):
        raise ValueError("Source velocity must have shape (node, 3).")

    chord = chord_profile
    if chord.shape == (chord.shape[0],):
        chord = csdl.expand(chord, (num_nodes, chord.shape[0]), "r->ir")
    if chord.shape[0] != num_nodes:
        raise ValueError("Chord profile must have shape (radial,) or (node, radial).")
    num_radial = chord.shape[1]
    radius = rotor_radius
    if radius.shape == (1,) and num_nodes > 1:
        radius = csdl.expand(radius, (num_nodes,))
    if radius.shape != (num_nodes,):
        raise ValueError("Rotor radius must be scalar or have shape (node,).")
    sound_speed = speed_of_sound
    if sound_speed.shape == (1,) and num_nodes > 1:
        sound_speed = csdl.expand(sound_speed, (num_nodes,))
    if sound_speed.shape != (num_nodes,):
        raise ValueError("Speed of sound must be scalar or have shape (node,).")

    dr = (1.0 - norm_hub_radius) * radius / (num_radial - 1)
    blade_planform_area = csdl.sum(chord, axes=(1,)) * dr
    solidity = blade_planform_area * num_blades / (np.pi * radius**2)
    solidity_weighted_chord = solidity * np.pi * radius / num_blades
    velocity_magnitude = csdl.sqrt(csdl.sum(source_velocity**2, axes=(1,)) + 1.0e-24)
    tip_speed = _positive_guard(
        rpm * 2.0 * np.pi / 60.0 * radius - velocity_magnitude
    )

    shape = (num_nodes, num_observers, num_frequencies)
    frequency = csdl.expand(frequencies, shape, "f->iof")
    distance = csdl.expand(observer_distance, shape, "io->iof")
    angle = csdl.expand(observer_angle_from_plane, shape, "io->iof")
    tip_speed_expanded = csdl.expand(tip_speed, shape, "i->iof")
    radius_expanded = csdl.expand(radius, shape, "i->iof")
    sound_speed_expanded = csdl.expand(sound_speed, shape, "i->iof")
    solidity_expanded = csdl.expand(solidity, shape, "i->iof")
    coefficient = _positive_guard(
        csdl.expand(thrust_coefficient, shape, "i->iof")
    )
    weighted_chord = csdl.expand(solidity_weighted_chord, shape, "i->iof")

    tip_mach = tip_speed_expanded / sound_speed_expanded
    strouhal = frequency * weighted_chord / tip_speed_expanded
    spectral_shift = (
        solidity_expanded * csdl.log(coefficient) / np.log(10.0)
        + 0.9
        * tip_mach
        * solidity_expanded
        * (tip_mach + 3.82)
        * csdl.log(solidity_expanded)
        / np.log(10.0)
    )
    spectral_coordinate = _positive_guard(strouhal - spectral_shift)
    velocity_term = 10.0 * 7.84 * csdl.log(tip_speed_expanded) / np.log(10.0)
    exponent_one = -2.0 * tip_mach**2 + 2.06
    exponent_two = (
        -coefficient
        * tip_mach
        * (coefficient - csdl.sin(csdl.sqrt(angle**2 + 1.0e-24)) + 2.06)
        + 1.0
    )
    distance_angle_exponent = (
        4.97
        * coefficient
        * csdl.sin(csdl.sqrt(angle**2 + 1.0e-24))
        * (1.5 * distance / radius_expanded * tip_mach - distance / radius_expanded + 15.0)
    )
    numerator = velocity_term * spectral_coordinate**0.6
    denominator = (
        _positive_guard(spectral_coordinate + exponent_one) ** exponent_two
        + (coefficient * spectral_coordinate) ** distance_angle_exponent
    )
    spectrum = csdl.minimum(numerator / denominator, 300.0 * np.ones(shape), rho=1.0)
    band_pressure_squared = reference_pressure**2 * csdl.exp(
        np.log(10.0) / 10.0 * spectrum
    )
    total_pressure_squared = csdl.reshape(
        csdl.sum(band_pressure_squared, axes=(2,)), observer_distance.shape
    )
    total_spl = csdl.reshape(
        10.0 / np.log(10.0) * csdl.log(
            total_pressure_squared / reference_pressure**2
        ),
        observer_distance.shape,
    )
    return GillLeeBroadbandOutputs(
        frequencies=frequencies,
        one_third_octave_spl=spectrum,
        one_third_octave_pressure_squared=band_pressure_squared,
        total_pressure_squared=total_pressure_squared,
        total_spl=total_spl,
    )
