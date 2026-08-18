"""Differentiable Hanson line-source loading noise for aligned inflow.

The formulation follows the loading term in Hanson, "Sound from a propeller at angle of
attack: a new theoretical viewpoint," Proceedings of the Royal Society A 449, 1995. This
initial kernel is restricted to an observer and uniform inflow expressed in a rotor-axis frame;
the general angle-of-attack coordinate transform is deliberately outside this function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np


@dataclass
class HansonLineSourceLoadingOutputs(csdl.VariableGroup):
    cosine_pressure: csdl.Variable
    sine_pressure: csdl.Variable
    radial_harmonic_cosine_pressure: csdl.Variable
    radial_harmonic_sine_pressure: csdl.Variable
    bessel_argument: csdl.Variable
    acoustic_harmonic_orders: tuple[int, ...]
    load_harmonic_numbers: tuple[int, ...]


@dataclass
class HansonRetardedGeometry(csdl.VariableGroup):
    distance: csdl.Variable
    polar_angle: csdl.Variable


def compute_hanson_retarded_geometry(
    observer_distance: csdl.Variable,
    observer_polar_angle: csdl.Variable,
    axial_mach_number: csdl.Variable,
) -> HansonRetardedGeometry:
    if observer_distance.shape != observer_polar_angle.shape:
        raise ValueError("Observer distance and polar angle must have identical shapes.")
    if len(observer_distance.shape) != 2:
        raise ValueError("Observer geometry must have shape (node, observer).")
    num_nodes = observer_distance.shape[0]
    if axial_mach_number.shape != (num_nodes,):
        raise ValueError("Axial Mach number must have shape (node,).")
    mach = csdl.expand(axial_mach_number, observer_distance.shape, "i->io")
    sine = csdl.sin(observer_polar_angle)
    cosine = csdl.cos(observer_polar_angle)
    root_argument = csdl.reshape(
        1.0 - mach**2 * sine**2, (observer_distance.size,)
    )
    root = csdl.reshape(csdl.sqrt(root_argument), observer_distance.shape)
    retarded_angle = csdl.arccos(
        cosine * root + mach * sine**2
    )
    in_plane_distance = observer_distance * sine
    retarded_distance = in_plane_distance / csdl.sin(retarded_angle)
    return HansonRetardedGeometry(distance=retarded_distance, polar_angle=retarded_angle)


def compute_hanson_line_source_loading(
    axial_force_real: csdl.Variable,
    axial_force_imaginary: csdl.Variable,
    circumferential_force_real: csdl.Variable,
    circumferential_force_imaginary: csdl.Variable,
    nondimensional_radius: csdl.Variable,
    radial_integration_weights: csdl.Variable,
    angular_speed: csdl.Variable,
    tip_radius: csdl.Variable,
    speed_of_sound: csdl.Variable,
    axial_mach_number: csdl.Variable,
    observer_distance: csdl.Variable,
    observer_polar_angle: csdl.Variable,
    observer_azimuth_angle: csdl.Variable,
    num_blades: int,
    modes: Sequence[int],
    load_harmonics: Sequence[int],
) -> HansonLineSourceLoadingOutputs:
    """Return complex harmonic pressure as separate cosine and sine components.

    Force harmonics have shape ``(node, load_harmonic, radial)`` and units N/m with
    respect to dimensional radius. Radial weights integrate over nondimensional radius, matching
    Hanson's line-load convention; callers must not pre-integrate the force over span.
    """
    force_shape = axial_force_real.shape
    if len(force_shape) != 3:
        raise ValueError("Force harmonics must have shape (node, harmonic, radial).")
    if any(
        value.shape != force_shape
        for value in (
            axial_force_imaginary,
            circumferential_force_real,
            circumferential_force_imaginary,
        )
    ):
        raise ValueError("All force-harmonic arrays must have identical shapes.")
    num_nodes, num_load_harmonics, num_radial = force_shape
    if nondimensional_radius.shape != (num_nodes, num_radial):
        raise ValueError("Nondimensional radius must have shape (node, radial).")
    if radial_integration_weights.shape != (num_nodes, num_radial):
        raise ValueError("Radial integration weights must have shape (node, radial).")
    for value, name in (
        (angular_speed, "Angular speed"),
        (tip_radius, "Tip radius"),
        (speed_of_sound, "Speed of sound"),
        (axial_mach_number, "Axial Mach number"),
    ):
        if value.shape != (num_nodes,):
            raise ValueError(f"{name} must have shape (node,).")
    observer_shape = observer_distance.shape
    if len(observer_shape) != 2 or observer_shape[0] != num_nodes:
        raise ValueError("Observer arrays must have shape (node, observer).")
    if observer_polar_angle.shape != observer_shape or observer_azimuth_angle.shape != observer_shape:
        raise ValueError("All observer arrays must have identical shapes.")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")

    mode_numbers = tuple(int(value) for value in modes)
    load_numbers = tuple(int(value) for value in load_harmonics)
    if not mode_numbers or any(value <= 0 for value in mode_numbers):
        raise ValueError("Modes must be positive integers.")
    if len(set(mode_numbers)) != len(mode_numbers):
        raise ValueError("Modes must be unique.")
    if len(load_numbers) != num_load_harmonics:
        raise ValueError("Load-harmonic metadata must match the harmonic axis.")
    if any(value < 0 for value in load_numbers) or len(set(load_numbers)) != len(load_numbers):
        raise ValueError("Load harmonics must be unique non-negative integers.")

    num_observers = observer_shape[1]
    num_modes = len(mode_numbers)
    target_shape = (num_nodes, num_observers, num_modes, num_load_harmonics, num_radial)
    acoustic_orders = tuple(mode * num_blades for mode in mode_numbers)
    n_values = np.asarray(acoustic_orders, dtype=float)
    k_values = np.asarray(load_numbers, dtype=float)
    bessel_orders = np.broadcast_to(
        np.asarray(acoustic_orders, dtype=int)[None, None, :, None, None]
        - np.asarray(load_numbers, dtype=int)[None, None, None, :, None],
        target_shape,
    )

    n = csdl.expand(csdl.Variable(value=n_values), target_shape, "m->iomhr")
    k = csdl.expand(csdl.Variable(value=k_values), target_shape, "h->iomhr")
    z = csdl.expand(nondimensional_radius, target_shape, "ir->iomhr")
    weights = csdl.expand(radial_integration_weights, target_shape, "ir->iomhr")
    omega = csdl.expand(angular_speed, target_shape, "i->iomhr")
    radius = csdl.expand(tip_radius, target_shape, "i->iomhr")
    sound_speed = csdl.expand(speed_of_sound, target_shape, "i->iomhr")
    mach = csdl.expand(axial_mach_number, target_shape, "i->iomhr")
    distance = csdl.expand(observer_distance, target_shape, "io->iomhr")
    theta = csdl.expand(observer_polar_angle, target_shape, "io->iomhr")
    azimuth = csdl.expand(observer_azimuth_angle, target_shape, "io->iomhr")
    tip_mach = omega * radius / sound_speed
    convection = 1.0 - mach * csdl.cos(theta)
    argument = n * z * tip_mach * csdl.sin(theta) / convection
    bessel = csdl.bessel(argument, kind=1, order=bessel_orders)

    def expand_force(value):
        return csdl.expand(value, target_shape, "ihr->iomhr")

    axial_real = expand_force(axial_force_real)
    axial_imaginary = expand_force(axial_force_imaginary)
    circumferential_real = expand_force(circumferential_force_real)
    circumferential_imaginary = expand_force(circumferential_force_imaginary)
    axial_factor = n * z * tip_mach * csdl.cos(theta) / convection
    circumferential_factor = n - k
    integrand_real = (
        axial_factor * axial_real - circumferential_factor * circumferential_real
    ) * bessel / z
    integrand_imaginary = (
        axial_factor * axial_imaginary - circumferential_factor * circumferential_imaginary
    ) * bessel / z
    radial_real = integrand_real * weights
    radial_imaginary = integrand_imaginary * weights

    source_phase = (n - k) * (azimuth - np.pi / 2.0)
    source_cosine = csdl.cos(source_phase)
    source_sine = csdl.sin(source_phase)
    phased_real = radial_real * source_cosine - radial_imaginary * source_sine
    phased_imaginary = radial_real * source_sine + radial_imaginary * source_cosine

    radiation_phase = n * omega * distance / (sound_speed * convection)
    radiation_cosine = csdl.cos(radiation_phase)
    radiation_sine = csdl.sin(radiation_phase)
    scale = num_blades / (4.0 * np.pi * distance * convection)
    pressure_real = -scale * (
        phased_real * radiation_sine + phased_imaginary * radiation_cosine
    )
    pressure_imaginary = scale * (
        phased_real * radiation_cosine - phased_imaginary * radiation_sine
    )

    return HansonLineSourceLoadingOutputs(
        cosine_pressure=csdl.sum(pressure_real, axes=(3, 4)),
        sine_pressure=csdl.sum(pressure_imaginary, axes=(3, 4)),
        radial_harmonic_cosine_pressure=pressure_real,
        radial_harmonic_sine_pressure=pressure_imaginary,
        bessel_argument=argument,
        acoustic_harmonic_orders=acoustic_orders,
        load_harmonic_numbers=load_numbers,
    )
