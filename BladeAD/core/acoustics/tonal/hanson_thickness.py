"""Differentiable Hanson helicoidal-surface thickness noise for aligned inflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np


@dataclass
class HansonThicknessOutputs(csdl.VariableGroup):
    cosine_pressure: csdl.Variable
    sine_pressure: csdl.Variable
    radial_cosine_pressure: csdl.Variable
    radial_sine_pressure: csdl.Variable
    chordwise_wavenumber: csdl.Variable
    bessel_argument: csdl.Variable
    acoustic_harmonic_orders: tuple[int, ...]


def compute_hanson_thickness_noise(
    nondimensional_radius: csdl.Variable,
    radial_integration_weights: csdl.Variable,
    chord: csdl.Variable,
    thickness_to_chord: csdl.Variable,
    normalized_thickness_shape: csdl.Variable,
    chordwise_locations: csdl.Variable,
    chordwise_integration_weights: csdl.Variable,
    angular_speed: csdl.Variable,
    tip_radius: csdl.Variable,
    density: csdl.Variable,
    speed_of_sound: csdl.Variable,
    axial_mach_number: csdl.Variable,
    observer_distance: csdl.Variable,
    observer_polar_angle: csdl.Variable,
    observer_azimuth_angle: csdl.Variable,
    num_blades: int,
    modes: Sequence[int],
) -> HansonThicknessOutputs:
    """Evaluate the helicoidal-surface thickness source with a normalized chordwise shape."""
    radial_shape = nondimensional_radius.shape
    if len(radial_shape) != 2:
        raise ValueError("Radial arrays must have shape (node, radial).")
    if any(
        value.shape != radial_shape
        for value in (radial_integration_weights, chord, thickness_to_chord)
    ):
        raise ValueError("All radial arrays must have identical shapes.")
    num_nodes, num_radial = radial_shape
    num_chordwise = normalized_thickness_shape.shape[0]
    if len(normalized_thickness_shape.shape) != 1 or num_chordwise < 2:
        raise ValueError("Normalized thickness shape must be a one-dimensional array.")
    if chordwise_locations.shape != (num_chordwise,):
        raise ValueError("Chordwise locations must match the thickness-shape array.")
    if chordwise_integration_weights.shape != (num_chordwise,):
        raise ValueError("Chordwise weights must match the thickness-shape array.")
    for value, name in (
        (angular_speed, "Angular speed"),
        (tip_radius, "Tip radius"),
        (density, "Density"),
        (speed_of_sound, "Speed of sound"),
        (axial_mach_number, "Axial Mach number"),
    ):
        if value.shape != (num_nodes,):
            raise ValueError(f"{name} must have shape (node,).")
    observer_shape = observer_distance.shape
    if len(observer_shape) != 2 or observer_shape[0] != num_nodes:
        raise ValueError("Observer arrays must have shape (node, observer).")
    if observer_polar_angle.shape != observer_shape:
        raise ValueError("Observer polar angle must match observer distance.")
    if observer_azimuth_angle.shape != observer_shape:
        raise ValueError("Observer azimuth angle must match observer distance.")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    mode_numbers = tuple(int(value) for value in modes)
    if not mode_numbers or any(value <= 0 for value in mode_numbers):
        raise ValueError("Modes must be positive integers.")
    if len(set(mode_numbers)) != len(mode_numbers):
        raise ValueError("Modes must be unique.")

    num_observers = observer_shape[1]
    num_modes = len(mode_numbers)
    radial_target = (num_nodes, num_observers, num_modes, num_radial)
    chordwise_target = radial_target + (num_chordwise,)
    acoustic_orders = tuple(mode * num_blades for mode in mode_numbers)
    order_array = np.broadcast_to(
        np.asarray(acoustic_orders, dtype=int)[None, None, :, None], radial_target
    )
    n = csdl.expand(
        csdl.Variable(value=np.asarray(acoustic_orders, dtype=float)),
        radial_target,
        "m->iomr",
    )
    z = csdl.expand(nondimensional_radius, radial_target, "ir->iomr")
    radial_weights = csdl.expand(radial_integration_weights, radial_target, "ir->iomr")
    section_chord = csdl.expand(chord, radial_target, "ir->iomr")
    section_thickness = csdl.expand(thickness_to_chord, radial_target, "ir->iomr")
    omega = csdl.expand(angular_speed, radial_target, "i->iomr")
    radius = csdl.expand(tip_radius, radial_target, "i->iomr")
    rho = csdl.expand(density, radial_target, "i->iomr")
    sound_speed = csdl.expand(speed_of_sound, radial_target, "i->iomr")
    mach = csdl.expand(axial_mach_number, radial_target, "i->iomr")
    distance = csdl.expand(observer_distance, radial_target, "io->iomr")
    theta = csdl.expand(observer_polar_angle, radial_target, "io->iomr")
    azimuth = csdl.expand(observer_azimuth_angle, radial_target, "io->iomr")
    tip_mach = omega * radius / sound_speed
    convection = 1.0 - mach * csdl.cos(theta)
    helicoid_angle = csdl.arctan(mach / (z * tip_mach))
    chord_diameter_ratio = section_chord / (2.0 * radius)
    chordwise_wavenumber = 2.0 * chord_diameter_ratio * (
        n * csdl.cos(helicoid_angle) / z
        + n
        * tip_mach
        * csdl.cos(theta)
        * csdl.sin(helicoid_angle)
        / convection
    )
    bessel_argument = n * z * tip_mach * csdl.sin(theta) / convection
    bessel = csdl.bessel(bessel_argument, kind=1, order=order_array)

    wavenumber = csdl.expand(chordwise_wavenumber, chordwise_target, "iomr->iomrq")
    locations = csdl.expand(chordwise_locations, chordwise_target, "q->iomrq")
    shape = csdl.expand(normalized_thickness_shape, chordwise_target, "q->iomrq")
    chordwise_weights = csdl.expand(
        chordwise_integration_weights, chordwise_target, "q->iomrq"
    )
    shape_real = csdl.sum(
        shape * csdl.cos(wavenumber * locations) * chordwise_weights, axes=(4,)
    )
    shape_imaginary = csdl.sum(
        shape * csdl.sin(wavenumber * locations) * chordwise_weights, axes=(4,)
    )
    relative_mach_squared = mach**2 + (z * tip_mach) ** 2
    amplitude = (
        relative_mach_squared
        * chordwise_wavenumber**2
        * section_thickness
        * bessel
        * radial_weights
    )
    radial_real = amplitude * shape_real
    radial_imaginary = amplitude * shape_imaginary
    radiation_phase = (
        n * omega * distance / (sound_speed * convection)
        + n * (azimuth - np.pi / 2.0)
    )
    radiation_cosine = csdl.cos(radiation_phase)
    radiation_sine = csdl.sin(radiation_phase)
    scale = -rho * sound_speed**2 * num_blades * radius / (
        4.0 * np.pi * distance * convection
    )
    pressure_real = scale * (
        radial_real * radiation_cosine - radial_imaginary * radiation_sine
    )
    pressure_imaginary = scale * (
        radial_real * radiation_sine + radial_imaginary * radiation_cosine
    )
    return HansonThicknessOutputs(
        cosine_pressure=csdl.sum(pressure_real, axes=(3,)),
        sine_pressure=csdl.sum(pressure_imaginary, axes=(3,)),
        radial_cosine_pressure=pressure_real,
        radial_sine_pressure=pressure_imaginary,
        chordwise_wavenumber=chordwise_wavenumber,
        bessel_argument=bessel_argument,
        acoustic_harmonic_orders=acoustic_orders,
    )
