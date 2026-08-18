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


def evaluate_observer_geometry_nodes(
    observers: AcousticObserverData,
    source_origin: csdl.Variable,
    thrust_axis: csdl.Variable,
):
    """Evaluate observer geometry for node-dependent rotor origins and axes."""
    positions = _as_variable(observers.positions)
    if len(positions.shape) != 2 or positions.shape[1] != 3:
        raise ValueError("Observer positions must have shape (num_observers, 3).")
    if len(source_origin.shape) != 2 or source_origin.shape[1] != 3:
        raise ValueError("Source origin must have shape (node, 3).")
    if thrust_axis.shape != source_origin.shape:
        raise ValueError("Thrust axis must match the source-origin shape.")
    if observers.frame not in {"rotor_local", "inertial"}:
        raise ValueError("Observer frame must be 'rotor_local' or 'inertial'.")

    num_nodes = source_origin.shape[0]
    num_observers = positions.shape[0]
    vector_shape = (num_nodes, num_observers, 3)
    positions = csdl.expand(positions, vector_shape, "oj->ioj")
    origins = csdl.expand(source_origin, vector_shape, "ij->ioj")
    axes = csdl.expand(thrust_axis, vector_shape, "ij->ioj")
    axis_norm = csdl.sqrt(csdl.sum(thrust_axis**2, axes=(1,)))
    unit_axes = axes / csdl.expand(axis_norm, vector_shape, "i->ioj")
    displacement = positions - origins
    distance_squared = csdl.reshape(
        csdl.sum(displacement**2, axes=(2,)),
        (num_nodes * num_observers,),
    )
    distance = csdl.reshape(
        csdl.sqrt(distance_squared),
        (num_nodes, num_observers),
    )
    direction = displacement / csdl.expand(distance, vector_shape, "io->ioj")
    axial_distance = csdl.reshape(
        csdl.sum(displacement * unit_axes, axes=(2,)),
        (num_nodes, num_observers),
    )
    in_plane_squared = csdl.reshape(
        distance**2 - axial_distance**2 + 1.0e-12,
        (num_nodes * num_observers,),
    )
    in_plane_distance = csdl.reshape(
        csdl.sqrt(in_plane_squared),
        (num_nodes, num_observers),
    )
    return distance, direction, axial_distance, in_plane_distance
