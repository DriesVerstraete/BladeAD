from __future__ import annotations

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.var_groups import AcousticObserverData


def _as_variable(value):
    if isinstance(value, csdl.Variable):
        return value
    return csdl.Variable(value=np.asarray(value, dtype=float))


def evaluate_observer_geometry(
    observers: AcousticObserverData,
    source_origin,
    thrust_axis,
):
    if observers.frame not in {"rotor_local", "inertial"}:
        raise ValueError("Observer frame must be 'rotor_local' or 'inertial'.")

    positions = _as_variable(observers.positions)
    origin = _as_variable(source_origin)
    axis = _as_variable(thrust_axis)
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError("Observer positions must have shape (num_observers, 3).")
    if origin.shape != (3,) or axis.shape != (3,):
        raise ValueError("Source origin and thrust axis must each have shape (3,).")

    num_observers = positions.shape[0]
    origin_expanded = csdl.expand(origin, positions.shape, "j->ij")
    axis_norm = csdl.sqrt(csdl.sum(axis**2))
    unit_axis = axis / axis_norm
    axis_expanded = csdl.expand(unit_axis, positions.shape, "j->ij")
    displacement = positions - origin_expanded
    distance = csdl.sqrt(csdl.sum(displacement**2, axes=(1,)))
    if distance.size == 1:
        distance_expanded = csdl.expand(distance, positions.shape)
    else:
        distance_expanded = csdl.expand(distance, positions.shape, "i->ij")
    direction = displacement / distance_expanded
    axis_cosine = csdl.sum(direction * axis_expanded, axes=(1,))

    if observers.names is not None and len(observers.names) != num_observers:
        raise ValueError("Observer names must match the number of observer positions.")
    return distance, direction, axis_cosine
