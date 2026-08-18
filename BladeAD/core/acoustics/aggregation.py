from __future__ import annotations

import csdl_alpha as csdl


def energetic_sum(*pressure_squared_components):
    if not pressure_squared_components:
        raise ValueError("At least one pressure-squared component is required.")
    total = pressure_squared_components[0]
    for component in pressure_squared_components[1:]:
        total = total + component
    return total


def pressure_squared_to_spl(
    pressure_squared,
    reference_pressure: float = 20.0e-6,
    pressure_squared_floor: float = 4.0e-16,
):
    if reference_pressure <= 0:
        raise ValueError("Reference pressure must be positive.")
    if pressure_squared_floor <= 0:
        raise ValueError("Pressure-squared floor must be positive.")
    ratio = (pressure_squared + pressure_squared_floor) / reference_pressure**2
    return 10.0 / csdl.log(10.0) * csdl.log(ratio)
