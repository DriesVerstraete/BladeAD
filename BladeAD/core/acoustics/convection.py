"""Moving-source convection correction for rotor acoustics."""

from __future__ import annotations

import csdl_alpha as csdl


def compute_convected_distance(
    observer_distance: csdl.Variable,
    observer_direction: csdl.Variable,
    source_velocity: csdl.Variable,
    speed_of_sound: csdl.Variable,
) -> csdl.Variable:
    """Return Lowson's retarded-position distance ``S * (1 - M_0r)``.

    Direction vectors point from the source to each observer. Inputs have shapes
    ``(node, observer)``, ``(node, observer, 3)``, ``(node, 3)``, and ``(node,)``.
    """
    if len(observer_distance.shape) != 2:
        raise ValueError("Observer distance must have shape (node, observer).")
    num_nodes, num_observers = observer_distance.shape
    if observer_direction.shape != (num_nodes, num_observers, 3):
        raise ValueError("Observer direction must have shape (node, observer, 3).")
    if source_velocity.shape != (num_nodes, 3):
        raise ValueError("Source velocity must have shape (node, 3).")
    if speed_of_sound.shape != (num_nodes,):
        raise ValueError("Speed of sound must have shape (node,).")

    velocity = csdl.expand(
        source_velocity, (num_nodes, num_observers, 3), "ij->ikj"
    )
    radial_velocity = csdl.reshape(
        csdl.sum(velocity * observer_direction, axes=(2,)),
        (num_nodes, num_observers),
    )
    sound_speed = csdl.expand(
        speed_of_sound, (num_nodes, num_observers), "i->ij"
    )
    return csdl.reshape(
        observer_distance * (1.0 - radial_velocity / sound_speed),
        (num_nodes, num_observers),
    )
