"""Shahjahan Case 1 (FPP) -- NSGA-II population-method arm, OEI pair H
(hover electrical power vs OEI thrust margin).

One of the two pairwise OEI subproblems behind the 3-objective Case 1 front
(the hover_elec x cruise_elec pair is `shahjahan_case1_fpp_nsga2`). Basin
scout for the Pareto-tracer anchors, so the constraint set MUST match the
3-objective epsilon front: the hard OEI survivability floor
`min_oei_thrust_margin = 1.143` is KEPT (dropping it lets NSGA-II settle in a
low-margin basin the front never visits).

Thin override of the phase-1 NSGA-II base -- same rotor (num_radial 32),
stall_margin 0.99, operators, pop / gen / seed, seed prescale. Only the
objective pair changes; the OEI design variable + floor that phase 1 stripped
are restored. `nsga2_runner` auto-detects the OEI arm from the objective set
(adds the `oei_rpm` DV and the OEI stall / torque / margin-floor constraints).
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_nsga2.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

# restore the OEI design variable + survivability floor that phase 1 popped
CASE["bounds"]["oei_rpm"] = (200.0, 3440.0)
CASE["constraints"]["min_oei_thrust_margin"] = 1.143

# OEI pair H: hover electrical power (min) vs OEI thrust margin (max).
# case._validate requires exactly one 'proxy'; the NSGA-II runner reads `goal`
# only (min directly, max negated), so the role is cosmetic for the run.
CASE["objectives"] = [
    {"name": "hover_elec", "result_key": "hover_electrical_power", "goal": "min",
     "role": "proxy", "proxy_min_key": "hover_electrical_power",
     "label": "hover electrical power (W)"},
    {"name": "oei_margin", "result_key": "oei_thrust_margin", "goal": "max",
     "role": "floor", "label": "OEI thrust margin"},
]

CASE["nsga2"] = dict(CASE["nsga2"], save_tag="oeiH_run1", hot_start=False)
