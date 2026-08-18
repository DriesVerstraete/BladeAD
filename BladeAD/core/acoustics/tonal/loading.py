"""Lowson–Ollerhead loading-noise pressure from arbitrary load harmonics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal.load_harmonics import LoadHarmonics


@dataclass
class LowsonLoadingPressure(csdl.VariableGroup):
    radial_harmonic_cosine_pressure: csdl.Variable
    radial_harmonic_sine_pressure: csdl.Variable
    cosine_pressure: csdl.Variable
    sine_pressure: csdl.Variable
    bessel_argument: csdl.Variable
    radiation_distance: csdl.Variable
    acoustic_harmonic_orders: tuple[int, ...]
    load_harmonic_numbers: tuple[int, ...]


def compute_lowson_loading_pressure(
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
) -> LowsonLoadingPressure:
    """Evaluate equation (10) of Lowson and Ollerhead (1969), without radial force."""
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
    num_nodes, num_harmonics, num_radial = coefficient_shape
    if len(load_harmonics.harmonic_numbers) != num_harmonics:
        raise ValueError("Harmonic-number metadata must match the coefficient axis.")
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

    num_observers = observer_shape[1]
    target_shape = (num_nodes, num_observers, len(mode_numbers), num_harmonics, num_radial)
    acoustic_orders = tuple(mode * num_blades for mode in mode_numbers)
    load_orders = tuple(int(value) for value in load_harmonics.harmonic_numbers)
    if any(value < 0 for value in load_orders) or len(set(load_orders)) != len(load_orders):
        raise ValueError("Load harmonic numbers must be unique non-negative integers.")
    n_values = np.asarray(acoustic_orders, dtype=int)[None, None, :, None, None]
    lambda_values = np.asarray(load_orders, dtype=int)[None, None, None, :, None]
    n = csdl.expand(csdl.Variable(value=np.asarray(acoustic_orders, dtype=float)), target_shape, "m->iomhr")
    lam = csdl.expand(csdl.Variable(value=np.asarray(load_orders, dtype=float)), target_shape, "h->iomhr")
    radius = csdl.expand(radial_stations, target_shape, "ir->iomhr")
    omega = csdl.expand(angular_speed, target_shape, "i->iomhr")
    sound_speed = csdl.expand(speed_of_sound, target_shape, "i->iomhr")
    axial = csdl.expand(observer_axial_distance, target_shape, "io->iomhr")
    in_plane = csdl.expand(observer_in_plane_distance, target_shape, "io->iomhr")
    radiation_distance = observer_distance if convected_distance is None else convected_distance
    distance = csdl.expand(radiation_distance, target_shape, "io->iomhr")

    def expand_load(coefficient):
        return csdl.expand(coefficient, target_shape, "ihr->iomhr")

    a_t = expand_load(load_harmonics.thrust_cosine)
    b_t = expand_load(load_harmonics.thrust_sine)
    a_d = expand_load(load_harmonics.drag_cosine)
    b_d = expand_load(load_harmonics.drag_sine)
    argument = n * omega * radius * in_plane / (sound_speed * distance)
    minus_order = np.broadcast_to(n_values - lambda_values, target_shape)
    plus_order = np.broadcast_to(n_values + lambda_values, target_shape)
    j_minus = csdl.bessel(argument, kind=1, order=minus_order)
    j_plus = csdl.bessel(argument, kind=1, order=plus_order)
    lambda_sign = csdl.expand(
        csdl.Variable(value=(-1.0) ** np.asarray(load_orders)), target_shape, "h->iomhr"
    )
    j_sum = j_minus + lambda_sign * j_plus
    j_difference = j_minus - lambda_sign * j_plus
    weighted_sum = (n - lam) * j_minus + lambda_sign * (n + lam) * j_plus
    weighted_difference = (n - lam) * j_minus - lambda_sign * (n + lam) * j_plus

    radiation_constant = n * omega * axial / (sound_speed * distance**2)
    near_constant = 1.0 / (radius * distance)
    inner_real = -radiation_constant * b_t * j_difference + near_constant * b_d * weighted_difference
    inner_imaginary = radiation_constant * a_t * j_sum - near_constant * a_d * weighted_sum

    phase = np.broadcast_to((n_values - lambda_values) % 4, target_shape)
    phase_real = np.where(phase == 0, 1.0, np.where(phase == 2, -1.0, 0.0))
    phase_imaginary = np.where(phase == 1, -1.0, np.where(phase == 3, 1.0, 0.0))
    phase_real = csdl.Variable(value=phase_real)
    phase_imaginary = csdl.Variable(value=phase_imaginary)
    scale = 1.0 / (4.0 * np.pi)
    radial_cosine = scale * (phase_real * inner_real - phase_imaginary * inner_imaginary)
    radial_sine = scale * (phase_real * inner_imaginary + phase_imaginary * inner_real)
    radial_cosine = csdl.reshape(radial_cosine, target_shape)
    radial_sine = csdl.reshape(radial_sine, target_shape)
    return LowsonLoadingPressure(
        radial_harmonic_cosine_pressure=radial_cosine,
        radial_harmonic_sine_pressure=radial_sine,
        cosine_pressure=csdl.sum(radial_cosine, axes=(3, 4)),
        sine_pressure=csdl.sum(radial_sine, axes=(3, 4)),
        bessel_argument=argument,
        radiation_distance=radiation_distance,
        acoustic_harmonic_orders=acoustic_orders,
        load_harmonic_numbers=load_orders,
    )
