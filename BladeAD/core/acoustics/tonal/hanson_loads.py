"""BladeAD sectional-load adapter for Hanson line-source acoustics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import csdl_alpha as csdl
import numpy as np


@dataclass
class HansonLineLoadHarmonics(csdl.VariableGroup):
    axial_real: csdl.Variable
    axial_imaginary: csdl.Variable
    circumferential_real: csdl.Variable
    circumferential_imaginary: csdl.Variable
    nondimensional_radial_weights: csdl.Variable
    harmonic_numbers: tuple[int, ...]


def compute_hanson_line_load_harmonics(
    sectional_thrust: csdl.Variable,
    sectional_drag: csdl.Variable,
    radial_element_width: csdl.Variable,
    radial_integration_weights: csdl.Variable,
    azimuth_angle: csdl.Variable,
    tip_radius: csdl.Variable,
    num_blades: int,
    harmonics: Sequence[int],
) -> HansonLineLoadHarmonics:
    """Convert complete-rotor elemental forces to normalized per-blade line-load harmonics.

    The complex Fourier convention is ``F_k = mean(F(psi) exp(-i k psi))``. Sectional forces
    include dimensional element width and all blades; returned harmonics are per-blade N/m.
    """
    shape = sectional_thrust.shape
    if len(shape) != 3:
        raise ValueError("Sectional arrays must have shape (node, radial, azimuth).")
    if any(
        value.shape != shape
        for value in (
            sectional_drag,
            radial_element_width,
            radial_integration_weights,
            azimuth_angle,
        )
    ):
        raise ValueError("All sectional arrays must have identical shapes.")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")
    harmonic_numbers = tuple(int(value) for value in harmonics)
    if not harmonic_numbers or any(value < 0 for value in harmonic_numbers):
        raise ValueError("Harmonics must be non-negative integers.")
    if len(set(harmonic_numbers)) != len(harmonic_numbers):
        raise ValueError("Harmonics must be unique.")

    num_nodes, num_radial, num_azimuthal = shape
    if tip_radius.shape != (num_nodes,):
        raise ValueError("Tip radius must have shape (node,).")
    harmonic_shape = (num_nodes, len(harmonic_numbers), num_radial, num_azimuthal)
    per_blade_axial = sectional_thrust / (num_blades * radial_element_width)
    per_blade_circumferential = sectional_drag / (num_blades * radial_element_width)
    axial = csdl.expand(per_blade_axial, harmonic_shape, "ira->ihra")
    circumferential = csdl.expand(
        per_blade_circumferential, harmonic_shape, "ira->ihra"
    )
    azimuth = csdl.expand(azimuth_angle, harmonic_shape, "ira->ihra")
    harmonic = csdl.expand(
        csdl.Variable(value=np.asarray(harmonic_numbers, dtype=float)),
        harmonic_shape,
        "h->ihra",
    )
    normalization = 1.0 / num_azimuthal
    cosine = csdl.cos(harmonic * azimuth)
    negative_sine = -csdl.sin(harmonic * azimuth)
    radial_weights = csdl.sum(radial_integration_weights, axes=(2,)) / num_azimuthal
    radius = csdl.expand(tip_radius, (num_nodes, num_radial), "i->ir")
    element_width = csdl.sum(radial_element_width, axes=(2,)) / num_azimuthal

    return HansonLineLoadHarmonics(
        axial_real=normalization * csdl.sum(axial * cosine, axes=(3,)),
        axial_imaginary=normalization * csdl.sum(axial * negative_sine, axes=(3,)),
        circumferential_real=normalization
        * csdl.sum(circumferential * cosine, axes=(3,)),
        circumferential_imaginary=normalization
        * csdl.sum(circumferential * negative_sine, axes=(3,)),
        nondimensional_radial_weights=radial_weights * element_width / radius,
        harmonic_numbers=harmonic_numbers,
    )
