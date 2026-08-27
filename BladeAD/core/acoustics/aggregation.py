from __future__ import annotations

import csdl_alpha as csdl
import numpy as np


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


def smooth_maximum_spl(
    observer_spl,
    observer_axis: int = 1,
    maximum_bias_db: float = 0.5,
    reference_spl_db: float = 100.0,
):
    if maximum_bias_db <= 0.0:
        raise ValueError("Maximum smooth-maximum bias must be positive.")
    if observer_axis < 0:
        observer_axis += len(observer_spl.shape)
    if observer_axis < 0 or observer_axis >= len(observer_spl.shape):
        raise ValueError("Observer axis is outside the SPL array dimensions.")
    num_observers = observer_spl.shape[observer_axis]
    if num_observers < 1:
        raise ValueError("At least one observer is required.")
    output_shape = tuple(
        size
        for axis, size in enumerate(observer_spl.shape)
        if axis != observer_axis
    )
    if num_observers == 1:
        return csdl.reshape(observer_spl, output_shape)
    rho = np.log(num_observers) / maximum_bias_db
    exponentials = csdl.exp(rho * (observer_spl - reference_spl_db))
    summed = csdl.reshape(
        csdl.sum(exponentials, axes=(observer_axis,)), output_shape
    )
    return reference_spl_db + csdl.log(summed) / rho
