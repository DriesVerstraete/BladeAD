"""Differentiable Sears gust-response loading harmonics."""

from __future__ import annotations

from typing import Sequence

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal.load_harmonics import LoadHarmonics


def compute_sears_load_harmonics(
    complete_rotor_steady_thrust: csdl.Variable,
    complete_rotor_steady_drag: csdl.Variable,
    radial_stations: csdl.Variable,
    radial_element_width: csdl.Variable,
    radial_integration_weights: csdl.Variable,
    chord_profile: csdl.Variable,
    inflow_angle: csdl.Variable,
    angular_speed: csdl.Variable,
    density: csdl.Variable,
    num_blades: int,
    harmonics: Sequence[int] = tuple(range(11)),
    gust_amplification: float = 0.06,
) -> LoadHarmonics:
    """Generate steady and Sears unsteady per-blade elemental load coefficients."""
    base_shape = radial_stations.shape
    if len(base_shape) != 2:
        raise ValueError("Radial arrays must have shape (node, radial).")
    if any(
        value.shape != base_shape
        for value in (
            complete_rotor_steady_thrust,
            complete_rotor_steady_drag,
            radial_element_width,
            radial_integration_weights,
            chord_profile,
            inflow_angle,
        )
    ):
        raise ValueError("All radial inputs must have shape (node, radial).")
    num_nodes, num_radial = base_shape
    if angular_speed.shape != (num_nodes,) or density.shape != (num_nodes,):
        raise ValueError("Angular speed and density must have shape (node,).")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    harmonic_numbers = tuple(int(value) for value in harmonics)
    if not harmonic_numbers or 0 not in harmonic_numbers:
        raise ValueError("Sears harmonics must include zero.")
    if any(value < 0 for value in harmonic_numbers) or len(set(harmonic_numbers)) != len(harmonic_numbers):
        raise ValueError("Harmonics must be unique non-negative integers.")
    if gust_amplification < 0:
        raise ValueError("Gust amplification must be non-negative.")

    output_shape = (num_nodes, len(harmonic_numbers), num_radial)
    a_t = csdl.Variable(shape=output_shape, value=0.0)
    b_t = csdl.Variable(shape=output_shape, value=0.0)
    a_d = csdl.Variable(shape=output_shape, value=0.0)
    b_d = csdl.Variable(shape=output_shape, value=0.0)
    steady_thrust = complete_rotor_steady_thrust / num_blades
    steady_drag = complete_rotor_steady_drag / num_blades
    omega = csdl.expand(angular_speed, base_shape, "i->ir")
    rho = csdl.expand(density, base_shape, "i->ir")
    elemental_width = radial_element_width * radial_integration_weights

    for index, harmonic in enumerate(harmonic_numbers):
        if harmonic == 0:
            a_t = a_t.set(csdl.slice[:, index, :], steady_thrust)
            a_d = a_d.set(csdl.slice[:, index, :], steady_drag)
            continue
        reduced_frequency = harmonic * chord_profile / (2.0 * radial_stations)
        j0 = csdl.bessel(reduced_frequency, kind=1, order=0)
        j1 = csdl.bessel(reduced_frequency, kind=1, order=1)
        y0 = csdl.bessel(reduced_frequency, kind=2, order=0)
        y1 = csdl.bessel(reduced_frequency, kind=2, order=1)
        first = j1 + y0
        second = y1 - j0
        denominator = first**2 + second**2
        f_value = (j1 * first + y1 * second) / denominator
        g_value = -(y1 * y0 + j1 * j0) / denominator
        sears_real = f_value * j0 + g_value * j1
        sears_imaginary = g_value * j0 - f_value * j1 + j1
        gust_velocity = (
            inflow_angle * omega * radial_stations / harmonic * gust_amplification
        )
        lift_per_length = (
            rho * (omega * radial_stations) * chord_profile * gust_velocity * np.pi
        )
        real_element = sears_real * lift_per_length * elemental_width
        imaginary_element = sears_imaginary * lift_per_length * elemental_width
        cosine_inflow = csdl.cos(inflow_angle)
        sine_inflow = csdl.sin(inflow_angle)
        a_t = a_t.set(csdl.slice[:, index, :], real_element * cosine_inflow)
        a_d = a_d.set(csdl.slice[:, index, :], real_element * sine_inflow)
        b_t = b_t.set(csdl.slice[:, index, :], imaginary_element * cosine_inflow)
        b_d = b_d.set(csdl.slice[:, index, :], imaginary_element * sine_inflow)

    return LoadHarmonics(
        thrust_cosine=a_t,
        thrust_sine=b_t,
        drag_cosine=a_d,
        drag_sine=b_d,
        harmonic_numbers=harmonic_numbers,
    )
