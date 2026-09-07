"""Shahjahan (2024) Case 1 -- FIXED-PITCH proprotor (FPP) variant.

Thin override of `shahjahan_case1` (no duplication -> cannot drift): the only
change is `rotor.fixed_pitch = True` -- one shared rigid collective for hover,
cruise and OEI; only the per-condition rpm varies. Matches Shahjahan's FPP
case (Figs 11b/12b), where the fronts are wider and the OEI thrust margin
harder to reach than the VPP case, because the blade twist can't be re-pitched
between conditions.

`bounds.pitch_deg` is the shared collective range; `seed.pitch_deg` its start.
The VPP `hover_pitch_deg` / `cruise_pitch_deg` bounds and seeds are ignored when
fixed_pitch is set.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["rotor"]["fixed_pitch"] = True
CASE["bounds"]["pitch_deg"] = (0.0, 60.0)

# FPP is over-constrained by the flat cl_max = 0.8 -- no single fixed blade +
# collective satisfies hover cl <= 0.8 AND cruise efficiency (2026-09-07 check).
# Use the radius-dependent section-Cl_max stall constraint instead (each MH
# airfoil at its `airfoil.clmax_ref_reynolds`, x this margin).
CASE["constraints"].pop("cl_max", None)
CASE["constraints"]["stall_margin"] = 0.9

# Seed by inverse design: the FPP blend (`seed.fpp_blend_blade`) -- hover-ideal
# chord + hover/cruise twist blend -- picked automatically because
# rotor.fixed_pitch is set. Off the hover-biased local corner a pure hover seed
# lands in.
CASE["seed"] = "inverse-design"
