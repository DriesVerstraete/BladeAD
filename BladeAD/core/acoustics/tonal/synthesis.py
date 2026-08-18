"""Coherent complete-rotor synthesis for Lowson tonal pressure."""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl

from BladeAD.core.acoustics.aggregation import pressure_squared_to_spl


@dataclass
class LowsonRotorTonalOutputs(csdl.VariableGroup):
    rotor_cosine_pressure: csdl.Variable
    rotor_sine_pressure: csdl.Variable
    mode_pressure_squared: csdl.Variable
    mode_spl: csdl.Variable
    total_pressure_squared: csdl.Variable
    total_spl: csdl.Variable


def synthesize_lowson_rotor_pressure(
    per_blade_cosine_pressure: csdl.Variable,
    per_blade_sine_pressure: csdl.Variable,
    num_blades: int,
    reference_pressure: float = 20.0e-6,
    pressure_squared_floor: float = 4.0e-16,
) -> LowsonRotorTonalOutputs:
    """Coherently add identical blades at acoustic orders ``n = mode * B``."""
    if per_blade_cosine_pressure.shape != per_blade_sine_pressure.shape:
        raise ValueError("Cosine and sine pressure arrays must have identical shapes.")
    if len(per_blade_cosine_pressure.shape) != 3:
        raise ValueError("Pressure arrays must have shape (node, observer, mode).")
    if not isinstance(num_blades, int) or num_blades <= 0:
        raise ValueError("Number of blades must be a positive integer.")

    pressure_shape = per_blade_cosine_pressure.shape
    rotor_cosine = csdl.reshape(num_blades * per_blade_cosine_pressure, pressure_shape)
    rotor_sine = csdl.reshape(num_blades * per_blade_sine_pressure, pressure_shape)
    mode_pressure_squared = csdl.reshape(
        0.5 * (rotor_cosine**2 + rotor_sine**2), pressure_shape
    )
    total_shape = pressure_shape[:2]
    total_pressure_squared = csdl.reshape(
        csdl.sum(mode_pressure_squared, axes=(2,)), total_shape
    )
    total_spl = csdl.reshape(
        pressure_squared_to_spl(
            total_pressure_squared, reference_pressure, pressure_squared_floor
        ),
        total_shape,
    )
    return LowsonRotorTonalOutputs(
        rotor_cosine_pressure=rotor_cosine,
        rotor_sine_pressure=rotor_sine,
        mode_pressure_squared=mode_pressure_squared,
        mode_spl=pressure_squared_to_spl(
            mode_pressure_squared, reference_pressure, pressure_squared_floor
        ),
        total_pressure_squared=total_pressure_squared,
        total_spl=total_spl,
    )
