"""Differentiable Lowson steady-loading pressure kernel.

This first Lowson slice evaluates the stationary-source, steady-loading (load harmonic
``lambda = 0``) terms at blade-passing harmonics.  It intentionally returns per-blade,
per-radial-station pressure components: coherent blade synthesis, moving-source convection,
and conversion to SPL belong to later layers and must not obscure kernel validation.

The sign convention follows Lowson and Ollerhead, Journal of Sound and Vibration 9(2),
1969, as represented by the independently audited LSDOlab/lsdo_acoustics implementation at
commit 7c76e0d01a71d59582d9ec3d62493dd7d37bdd69.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal.load_harmonics import LoadHarmonics


@dataclass
class LowsonSteadyLoadingPressure(csdl.VariableGroup):
    radial_cosine_pressure: csdl.Variable
    radial_sine_pressure: csdl.Variable
    cosine_pressure: csdl.Variable
    sine_pressure: csdl.Variable
    bessel_argument: csdl.Variable
    radiation_distance: csdl.Variable
    acoustic_harmonic_orders: tuple[int, ...]


def compute_lowson_steady_loading_pressure(
    load_harmonics: LoadHarmonics,
    radial_stations: csdl.Variable,
    angular_speed: csdl.Variable,
    observer_axial_distance: csdl.Variable,
    observer_in_plane_distance: csdl.Variable,
    observer_distance: csdl.Variable,
    speed_of_sound: csdl.Variable,
    num_blades: int,
    modes: Sequence[int],
    convected_distance: csdl.Variable | None = None,
) -> LowsonSteadyLoadingPressure:
    """Evaluate the per-blade Lowson ``lambda=0`` pressure at BPF harmonics.

    Parameters use SI units. Shapes are ``(node, harmonic, radial)`` for load coefficients,
    ``(node, radial)`` for dimensional radial stations, ``(node,)`` for angular speed and
    speed of sound, and ``(node, observer)`` for observer geometry. If supplied,
    ``convected_distance`` is Lowson's retarded-position distance ``S * (1 - M_0r)``.
    """
    coefficient_shape = load_harmonics.thrust_cosine.shape
    if len(coefficient_shape) != 3:
        raise ValueError("Load coefficients must have shape (node, harmonic, radial).")
    if any(
        coefficient.shape != coefficient_shape
        for coefficient in (
            load_harmonics.thrust_sine,
            load_harmonics.drag_cosine,
            load_harmonics.drag_sine,
        )
    ):
        raise ValueError("All load-harmonic coefficient arrays must have identical shapes.")

    num_nodes, _, num_radial = coefficient_shape
    if radial_stations.shape != (num_nodes, num_radial):
        raise ValueError("Radial stations must have shape (node, radial).")
    if angular_speed.shape != (num_nodes,) or speed_of_sound.shape != (num_nodes,):
        raise ValueError("Angular speed and speed of sound must each have shape (node,).")
    observer_shape = observer_distance.shape
    if len(observer_shape) != 2 or observer_shape[0] != num_nodes:
        raise ValueError("Observer distances must have shape (node, observer).")
    if observer_axial_distance.shape != observer_shape or observer_in_plane_distance.shape != observer_shape:
        raise ValueError("All observer-geometry arrays must have identical shapes.")
    if convected_distance is not None and convected_distance.shape != observer_shape:
        raise ValueError("Convected distance must match the observer-distance shape.")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    mode_numbers = tuple(int(mode) for mode in modes)
    if not mode_numbers or any(mode <= 0 for mode in mode_numbers):
        raise ValueError("Modes must be positive integers.")
    if len(set(mode_numbers)) != len(mode_numbers):
        raise ValueError("Modes must be unique.")
    if 0 not in load_harmonics.harmonic_numbers:
        raise ValueError("Steady-loading pressure requires load harmonic zero.")

    steady_index = load_harmonics.harmonic_numbers.index(0)
    thrust = load_harmonics.thrust_cosine[:, steady_index, :]
    drag = load_harmonics.drag_cosine[:, steady_index, :]
    num_observers = observer_shape[1]
    num_modes = len(mode_numbers)
    target_shape = (num_nodes, num_observers, num_modes, num_radial)

    radius = csdl.expand(radial_stations, target_shape, "ir->iomr")
    omega = csdl.expand(angular_speed, target_shape, "i->iomr")
    sound_speed = csdl.expand(speed_of_sound, target_shape, "i->iomr")
    axial = csdl.expand(observer_axial_distance, target_shape, "io->iomr")
    in_plane = csdl.expand(observer_in_plane_distance, target_shape, "io->iomr")
    radiation_distance = observer_distance if convected_distance is None else convected_distance
    distance = csdl.expand(radiation_distance, target_shape, "io->iomr")
    thrust = csdl.expand(thrust, target_shape, "ir->iomr")
    drag = csdl.expand(drag, target_shape, "ir->iomr")

    acoustic_orders = tuple(mode * num_blades for mode in mode_numbers)
    order_values = np.asarray(acoustic_orders, dtype=float)
    order = csdl.expand(csdl.Variable(value=order_values), target_shape, "m->iomr")
    bessel_argument = order * omega * radius * in_plane / (sound_speed * distance)
    bessel = csdl.bessel(bessel_argument, kind=1, order=np.broadcast_to(
        order_values[None, None, :, None], target_shape
    ))

    radiation = order * omega * axial * thrust * bessel / (sound_speed * distance**2)
    near_field = order * drag * bessel / (radius * distance)
    pressure_scale = 1.0 / (2.0 * np.pi)

    cosine_radiation_sign = np.asarray([
        (-1.0) ** ((order_value - 1) // 2) if order_value % 2 else 0.0
        for order_value in acoustic_orders
    ])
    cosine_near_sign = -cosine_radiation_sign
    sine_radiation_sign = np.asarray([
        0.0 if order_value % 2 else (-1.0) ** (order_value // 2)
        for order_value in acoustic_orders
    ])
    sine_near_sign = -sine_radiation_sign
    cosine_radiation_sign = csdl.expand(
        csdl.Variable(value=cosine_radiation_sign), target_shape, "m->iomr"
    )
    cosine_near_sign = csdl.expand(
        csdl.Variable(value=cosine_near_sign), target_shape, "m->iomr"
    )
    sine_radiation_sign = csdl.expand(
        csdl.Variable(value=sine_radiation_sign), target_shape, "m->iomr"
    )
    sine_near_sign = csdl.expand(
        csdl.Variable(value=sine_near_sign), target_shape, "m->iomr"
    )

    radial_cosine = csdl.reshape(
        pressure_scale * (
            cosine_radiation_sign * radiation + cosine_near_sign * near_field
        ),
        target_shape,
    )
    radial_sine = csdl.reshape(pressure_scale * (
        sine_radiation_sign * radiation + sine_near_sign * near_field
    ), target_shape)
    return LowsonSteadyLoadingPressure(
        radial_cosine_pressure=radial_cosine,
        radial_sine_pressure=radial_sine,
        cosine_pressure=csdl.sum(radial_cosine, axes=(3,)),
        sine_pressure=csdl.sum(radial_sine, axes=(3,)),
        bessel_argument=bessel_argument,
        radiation_distance=radiation_distance,
        acoustic_harmonic_orders=acoustic_orders,
    )
