from __future__ import annotations

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.observers import evaluate_observer_geometry
from BladeAD.core.acoustics.var_groups import (
    AcousticObserverData,
    RotorAcousticOutputs,
    RotorAcousticSettings,
)


def evaluate_rotor_acoustics(
    rotor_inputs,
    rotor_outputs,
    observers: AcousticObserverData,
    settings: RotorAcousticSettings | None = None,
):
    settings = settings or RotorAcousticSettings()
    if settings.tonal_enabled or settings.broadband_enabled:
        raise NotImplementedError(
            "Tonal and broadband models are not available in the acoustic foundation."
        )
    if not settings.modes or any(int(mode) != mode or mode <= 0 for mode in settings.modes):
        raise ValueError("Acoustic modes must be positive integers.")

    mesh = rotor_inputs.mesh_parameters
    distance, direction, axis_cosine = evaluate_observer_geometry(
        observers=observers,
        source_origin=mesh.thrust_origin,
        thrust_axis=mesh.thrust_vector,
    )
    rpm = rotor_inputs.rpm
    if not isinstance(rpm, csdl.Variable):
        rpm = csdl.Variable(value=np.asarray(rpm, dtype=float))
    if rpm.shape == ():
        rpm = rpm.reshape((1,))
    if len(rpm.shape) != 1:
        raise ValueError("RPM must be scalar or one-dimensional.")

    modes = csdl.Variable(value=np.asarray(settings.modes, dtype=float))
    fundamental = rpm * mesh.num_blades / 60.0
    fundamental_expanded = csdl.expand(
        fundamental, (rpm.shape[0], modes.shape[0]), "i->ij"
    )
    modes_expanded = csdl.expand(modes, fundamental_expanded.shape, "j->ij")
    blade_passing_frequencies = fundamental_expanded * modes_expanded

    return RotorAcousticOutputs(
        observer_distance=distance,
        observer_direction=direction,
        observer_axis_cosine=axis_cosine,
        blade_passing_frequencies=blade_passing_frequencies,
    )
