"""Differentiable sectional-load Fourier coefficients for Lowson loading noise.

The coefficient convention follows Lowson and Ollerhead, "A theoretical study of helicopter
rotor noise," Journal of Sound and Vibration 9(2), 1969, and is independently implemented for
BladeAD's complete-rotor sectional-load convention. LSDOlab/lsdo_acoustics dev_csdl_alpha commit
7c76e0d01a71d59582d9ec3d62493dd7d37bdd69 was consulted for graph-structure comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np


@dataclass
class LoadHarmonics(csdl.VariableGroup):
    thrust_cosine: csdl.Variable
    thrust_sine: csdl.Variable
    drag_cosine: csdl.Variable
    drag_sine: csdl.Variable
    harmonic_numbers: tuple[int, ...]


def compute_load_harmonics(
    sectional_thrust: csdl.Variable,
    sectional_drag: csdl.Variable,
    azimuth_angle: csdl.Variable,
    num_blades: int,
    harmonics: Sequence[int] = tuple(range(11)),
) -> LoadHarmonics:
    if sectional_thrust.shape != sectional_drag.shape:
        raise ValueError("Sectional thrust and drag must have identical shapes.")
    if sectional_thrust.shape != azimuth_angle.shape:
        raise ValueError("Sectional loads and azimuth angle must have identical shapes.")
    if len(sectional_thrust.shape) != 3:
        raise ValueError("Sectional arrays must have shape (node, radial, azimuth).")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    harmonic_numbers = tuple(int(value) for value in harmonics)
    if not harmonic_numbers or any(value < 0 for value in harmonic_numbers):
        raise ValueError("Harmonic numbers must be non-negative integers.")
    if len(set(harmonic_numbers)) != len(harmonic_numbers):
        raise ValueError("Harmonic numbers must be unique.")

    num_nodes, num_radial, num_azimuthal = sectional_thrust.shape
    num_harmonics = len(harmonic_numbers)
    target_shape = (num_nodes, num_harmonics, num_radial, num_azimuthal)
    per_blade_thrust = sectional_thrust / num_blades
    per_blade_drag = sectional_drag / num_blades
    thrust_expanded = csdl.expand(per_blade_thrust, target_shape, "ira->ihra")
    drag_expanded = csdl.expand(per_blade_drag, target_shape, "ira->ihra")
    azimuth_expanded = csdl.expand(azimuth_angle, target_shape, "ira->ihra")
    harmonic_array = np.asarray(harmonic_numbers, dtype=float)
    harmonic_variable = csdl.Variable(value=harmonic_array)
    harmonic_expanded = csdl.expand(harmonic_variable, target_shape, "h->ihra")
    phase = harmonic_expanded * azimuth_expanded
    cosine = csdl.cos(phase)
    sine = csdl.sin(phase)
    normalization_values = np.asarray(
        [1.0 if harmonic == 0 else 2.0 for harmonic in harmonic_numbers]
    ) / num_azimuthal
    normalization = csdl.expand(
        csdl.Variable(value=normalization_values),
        (num_nodes, num_harmonics, num_radial),
        "h->ihr",
    )

    return LoadHarmonics(
        thrust_cosine=normalization * csdl.sum(thrust_expanded * cosine, axes=(3,)),
        thrust_sine=normalization * csdl.sum(thrust_expanded * sine, axes=(3,)),
        drag_cosine=normalization * csdl.sum(drag_expanded * cosine, axes=(3,)),
        drag_sine=normalization * csdl.sum(drag_expanded * sine, axes=(3,)),
        harmonic_numbers=harmonic_numbers,
    )
