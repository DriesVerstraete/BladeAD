"""Differentiable Barry–Magliozzi rotor thickness noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.aggregation import pressure_squared_to_spl


@dataclass
class BarryMagliozziThicknessOutputs(csdl.VariableGroup):
    mode_rms_pressure: csdl.Variable
    mode_pressure_squared: csdl.Variable
    mode_spl: csdl.Variable
    total_pressure_squared: csdl.Variable
    total_spl: csdl.Variable
    acoustic_harmonic_orders: tuple[int, ...]


def compute_barry_magliozzi_thickness_noise(
    radial_stations: csdl.Variable,
    radial_element_width: csdl.Variable,
    chord_profile: csdl.Variable,
    thickness_to_chord: csdl.Variable,
    angular_speed: csdl.Variable,
    observer_axial_distance: csdl.Variable,
    observer_in_plane_distance: csdl.Variable,
    observer_distance: csdl.Variable,
    density: csdl.Variable,
    speed_of_sound: csdl.Variable,
    mach_number: csdl.Variable,
    num_blades: int,
    modes: Sequence[int],
    reference_pressure: float = 20.0e-6,
    pressure_squared_floor: float = 4.0e-16,
) -> BarryMagliozziThicknessOutputs:
    """Evaluate the Gill et al. Barry–Magliozzi thickness formulation."""
    if len(radial_stations.shape) != 2:
        raise ValueError("Radial stations must have shape (node, radial).")
    num_nodes, num_radial = radial_stations.shape
    if any(
        value.shape != radial_stations.shape
        for value in (radial_element_width, chord_profile, thickness_to_chord)
    ):
        raise ValueError("All radial geometry arrays must have shape (node, radial).")
    if any(
        value.shape != (num_nodes,)
        for value in (angular_speed, density, speed_of_sound, mach_number)
    ):
        raise ValueError("Node quantities must have shape (node,).")
    observer_shape = observer_distance.shape
    if len(observer_shape) != 2 or observer_shape[0] != num_nodes:
        raise ValueError("Observer geometry must have shape (node, observer).")
    if observer_axial_distance.shape != observer_shape or observer_in_plane_distance.shape != observer_shape:
        raise ValueError("All observer-geometry arrays must have identical shapes.")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    mode_numbers = tuple(int(mode) for mode in modes)
    if not mode_numbers or any(mode <= 0 for mode in mode_numbers):
        raise ValueError("Modes must be positive integers.")
    if len(set(mode_numbers)) != len(mode_numbers):
        raise ValueError("Modes must be unique.")

    num_observers = observer_shape[1]
    num_modes = len(mode_numbers)
    observer_mode_shape = (num_nodes, num_observers, num_modes)
    target_shape = observer_mode_shape + (num_radial,)
    axial = observer_axial_distance
    in_plane = observer_in_plane_distance
    mach_observer = csdl.expand(mach_number, observer_shape, "i->io")
    s0_squared = csdl.reshape(
        axial**2 + (1.0 - mach_observer**2) * in_plane**2,
        (num_nodes * num_observers,),
    )
    s0 = csdl.reshape(csdl.sqrt(s0_squared), observer_shape)

    radius = csdl.expand(radial_stations, target_shape, "ir->iomr")
    width = csdl.expand(radial_element_width, target_shape, "ir->iomr")
    chord = csdl.expand(chord_profile, target_shape, "ir->iomr")
    thickness = csdl.expand(thickness_to_chord, target_shape, "ir->iomr")
    omega = csdl.expand(angular_speed, target_shape, "i->iomr")
    sound_speed = csdl.expand(speed_of_sound, target_shape, "i->iomr")
    mach = csdl.expand(mach_number, target_shape, "i->iomr")
    y = csdl.expand(in_plane, target_shape, "io->iomr")
    s0_radial = csdl.expand(s0, target_shape, "io->iomr")
    modes_array = np.asarray(mode_numbers, dtype=float)
    mode = csdl.expand(csdl.Variable(value=modes_array), target_shape, "m->iomr")
    order_values = np.asarray([mode_value * num_blades for mode_value in mode_numbers])
    order = csdl.expand(csdl.Variable(value=order_values), target_shape, "m->iomr")
    base_argument = omega * y * radius / (sound_speed * s0_radial)
    bessel = csdl.bessel(
        order * base_argument,
        kind=1,
        order=np.broadcast_to(order_values[None, None, :, None], target_shape),
    )
    minus_orders = order_values - 1
    plus_orders = order_values + 1
    bessel_minus = csdl.bessel(
        (order - 1.0) * base_argument,
        kind=1,
        order=np.broadcast_to(minus_orders[None, None, :, None], target_shape),
    )
    bessel_plus = csdl.bessel(
        (order + 1.0) * base_argument,
        kind=1,
        order=np.broadcast_to(plus_orders[None, None, :, None], target_shape),
    )
    section_area = 0.6853 * chord**2 * thickness
    correction = (
        (1.0 - mach**2) * y * radius / (2.0 * s0_radial**2)
        * (bessel_minus - bessel_plus)
    )
    integrated = csdl.sum(section_area * (bessel + correction) * width, axes=(3,))

    rho = csdl.expand(density, observer_mode_shape, "i->iom")
    omega_mode = csdl.expand(angular_speed, observer_mode_shape, "i->iom")
    mach_mode = csdl.expand(mach_number, observer_mode_shape, "i->iom")
    x = csdl.expand(axial, observer_mode_shape, "io->iom")
    s0_mode = csdl.expand(s0, observer_mode_shape, "io->iom")
    mode = csdl.expand(csdl.Variable(value=modes_array), observer_mode_shape, "m->iom")
    factor = (
        rho * (mode * omega_mode) ** 2 * num_blades**3
        * (s0_mode + mach_mode * x) ** 2
        / (2.0 * np.pi * np.sqrt(2.0) * (1.0 - mach_mode**2) ** 2 * s0_mode**3)
    )
    mode_rms_pressure = factor * integrated
    mode_pressure_squared = mode_rms_pressure**2
    total_pressure_squared = csdl.reshape(
        csdl.sum(mode_pressure_squared, axes=(2,)), observer_shape
    )
    return BarryMagliozziThicknessOutputs(
        mode_rms_pressure=mode_rms_pressure,
        mode_pressure_squared=mode_pressure_squared,
        mode_spl=pressure_squared_to_spl(
            mode_pressure_squared, reference_pressure, pressure_squared_floor
        ),
        total_pressure_squared=total_pressure_squared,
        total_spl=csdl.reshape(
            pressure_squared_to_spl(
                total_pressure_squared, reference_pressure, pressure_squared_floor
            ),
            observer_shape,
        ),
        acoustic_harmonic_orders=tuple(int(value) for value in order_values),
    )
