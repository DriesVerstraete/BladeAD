"""Shared SPL rotor-design acoustic wiring for the BladeAD/CSDL rotor optimisers.

Encapsulates the 2026-08-26 Gate-E acoustic proxy decision in ONE place instead
of copy-pasting it per optimiser script:

  - the ``RotorAcousticSettings`` preset -- Lowson loading + Barry--Magliozzi
    thickness + Gill--Lee broadband, energetically combined, A-weighting on;
  - the two fixed observers -- hover 25 m directly below the disc (rotor-local
    polar 180 deg from +thrust), cruise 500 ft (152.4 m) directly below (polar
    90 deg from the horizontal cruise thrust axis);
  - the vehicle-level cruise constraint  OSPL + 10*log10(n_rotors) <= cap_db.

The acoustic *physics* is entirely in the BladeAD fork
(``evaluate_rotor_acoustics``, the 2026-08-26 local patch). This module is SPL
*application policy* only and is deliberately NOT pushed into BladeAD -- the
fork is a general-purpose rotor library and must not hard-code "62 dB at 500 ft"
or an SPL vehicle rotor count.

Both observer position vectors assume the optimiser builds the rotor with
``thrust_vector = [1, 0, 0]`` (hover: thrust "up" along +x; cruise: thrust
"forward" along +x), matching every optimiser script in this project.

Refs: ``decisions/2026-08-26-acoustic-demonstrator-objective.md``;
``999-software/bladead_repo/local-patches.md`` (2026-08-26 Gill--Lee entry).
"""
from dataclasses import dataclass

import numpy as np
import csdl_alpha as csdl

from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    evaluate_rotor_acoustics,
)

HOVER_OBSERVER_M = 25.0        # directly below, ISA hover altitude (Shahjahan 2024)
CRUISE_OBSERVER_M = 152.4      # 500 ft directly below
CRUISE_NOISE_CAP_DB = 62.0    # Shahjahan cruise requirement (adopted as the SPL proxy)
NUM_AZIMUTHAL_ACOUSTIC = 8    # azimuthal resolution the tonal model needs (verified value)

# Generic 4-point chordwise thickness distribution for the Barry--Magliozzi
# thickness term. Only its *scale* (thickness_to_chord, per station) is airfoil-
# specific and comes from the caller; the chordwise shape is a rounded-profile
# default shared across the MH-series sections used here.
_THICKNESS_SHAPE = np.array([0.0, 1.0, 0.6, 0.0])
_THICKNESS_CHORDWISE_LOC = np.array([-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5])
_THICKNESS_CHORDWISE_WEIGHT = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])


def spl_acoustic_settings():
    """The Gate-E acoustic settings preset (2026-08-26 decision)."""
    return RotorAcousticSettings(
        modes=(1, 2, 3),
        load_harmonics=(0,),
        tonal_model="lowson",
        tonal_enabled=True,
        thickness_enabled=True,
        sears_enabled=False,
        broadband_enabled=True,
        a_weighting_enabled=True,
    )


def thickness_to_chord_profile(section_boundaries_r_over_R, t_over_c, r_over_R_stations):
    """Per-station t/c for the thickness-noise term, from a piecewise section spec.

    ``section_boundaries_r_over_R`` has ``len(t_over_c) + 1`` entries in r/R.
    Interpolated linearly between section mid-points (flat beyond the ends) so the
    distribution feeding the acoustic model is smooth rather than stepped. Never
    hard-code the t/c here -- pass ``design["airfoil"]["t_over_c"]`` and
    ``["section_boundaries_r_over_r"]`` from the optimiser config so it can't go
    stale if the airfoil stack changes.
    """
    bounds = np.asarray(section_boundaries_r_over_R, dtype=float)
    tc = np.asarray(t_over_c, dtype=float)
    if bounds.shape[0] != tc.shape[0] + 1:
        raise ValueError(
            f"section_boundaries_r_over_R (len {bounds.shape[0]}) must have one more "
            f"entry than t_over_c (len {tc.shape[0]})"
        )
    midpoints = 0.5 * (bounds[:-1] + bounds[1:])
    return np.interp(np.asarray(r_over_R_stations, dtype=float), midpoints, tc)


def acoustic_mesh_fields(thickness_to_chord_per_station):
    """``RotorMeshParameters`` kwargs for the thickness-noise term. Splat into the
    constructor only when acoustics is active (they carry no cost otherwise but
    are meaningless without ``num_azimuthal = NUM_AZIMUTHAL_ACOUSTIC``)."""
    tc = np.asarray(thickness_to_chord_per_station, dtype=float).reshape(-1)
    return {
        "thickness_to_chord": csdl.Variable(value=tc),
        "normalized_thickness_shape": csdl.Variable(value=_THICKNESS_SHAPE),
        "thickness_shape_chordwise_locations": csdl.Variable(value=_THICKNESS_CHORDWISE_LOC),
        "thickness_shape_chordwise_weights": csdl.Variable(value=_THICKNESS_CHORDWISE_WEIGHT),
    }


@dataclass
class SplAcousticBundle:
    hover: object                       # RotorAcousticOutputs at the 25 m observer
    cruise: object                      # RotorAcousticOutputs at the 152.4 m observer
    hover_ospl: csdl.Variable           # unweighted total OSPL, dB
    cruise_single_ospl: csdl.Variable   # single-rotor unweighted OSPL, dB
    cruise_vehicle_ospl: csdl.Variable  # + 10*log10(n_rotors), dB -- the constrained quantity
    hover_ospl_a_weighted: csdl.Variable
    n_rotors: int
    settings: RotorAcousticSettings


def add_spl_hover_cruise_acoustics(
    hover_inputs,
    hover_outputs,
    cruise_inputs,
    cruise_outputs,
    n_rotors,
    *,
    cruise_cap_db=CRUISE_NOISE_CAP_DB,
    hover_noise_max_db=None,
    constrain_cruise=True,
):
    """Evaluate hover + cruise rotor acoustics at the Gate-E observers and set the
    vehicle cruise-noise constraint.

    Parameters
    ----------
    hover_inputs, hover_outputs, cruise_inputs, cruise_outputs
        The ``RotorAnalysisInputs`` / BEM outputs for each design point. The BEM
        mesh must have been built with ``num_azimuthal = NUM_AZIMUTHAL_ACOUSTIC``
        and the ``acoustic_mesh_fields(...)`` kwargs.
    n_rotors : int
        Active identical rotors on the vehicle, aggregated incoherently
        (``+10*log10(n_rotors)``) for the cruise constraint. Caller's decision.
    cruise_cap_db : float
        Vehicle cruise OSPL cap (default 62 dB, the Shahjahan proxy).
    hover_noise_max_db : float or None
        If given, also constrain single-rotor hover OSPL <= this (an
        epsilon-constraint direction for a noise Pareto sweep).
    constrain_cruise : bool
        Set False to evaluate acoustics for reporting only, without adding the
        cruise constraint to the optimisation.
    """
    settings = spl_acoustic_settings()
    hover_ac = evaluate_rotor_acoustics(
        hover_inputs,
        hover_outputs,
        AcousticObserverData(
            positions=csdl.Variable(value=np.array([[-HOVER_OBSERVER_M, 0.0, 0.0]]))
        ),
        settings,
    )
    cruise_ac = evaluate_rotor_acoustics(
        cruise_inputs,
        cruise_outputs,
        AcousticObserverData(
            positions=csdl.Variable(value=np.array([[0.0, 0.0, -CRUISE_OBSERVER_M]]))
        ),
        settings,
    )
    cruise_vehicle_ospl = cruise_ac.total_spl + 10.0 * np.log10(n_rotors)
    if constrain_cruise:
        cruise_vehicle_ospl.set_as_constraint(upper=cruise_cap_db, scaler=0.1)
    if hover_noise_max_db is not None:
        hover_ac.total_spl.set_as_constraint(upper=hover_noise_max_db, scaler=0.1)
    return SplAcousticBundle(
        hover=hover_ac,
        cruise=cruise_ac,
        hover_ospl=hover_ac.total_spl,
        cruise_single_ospl=cruise_ac.total_spl,
        cruise_vehicle_ospl=cruise_vehicle_ospl,
        hover_ospl_a_weighted=hover_ac.total_spl_a_weighted,
        n_rotors=n_rotors,
        settings=settings,
    )
