"""Shahjahan Case 1 FPP -- Pareto-TRACER validation, hover_elec x oei_margin
edge (2026-09-08).

`shahjahan_case1_fpp_explore_sm099` constraint set (num_radial 32, stall_margin
0.99, OEI floor 1.143, `oei_rpm` DV), objective pair reduced to
[hover_elec (proxy, minimised), oei_margin (floor, maximised)] -- the
hover-power vs OEI-margin edge of the 3-objective Case-1 front.

Driven by `tracer_run.py`. Compare to
`shahjahan_case1_fpp_explore_sm099/pareto_front_hover_elec__oei_margin_*.pkl`.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore_sm099.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["objectives"] = [
    {"name": "hover_elec", "result_key": "hover_electrical_power", "goal": "min",
     "role": "proxy", "proxy_min_key": "hover_electrical_power",
     "label": "hover electrical power (W)"},
    {"name": "oei_margin", "result_key": "oei_thrust_margin", "goal": "max",
     "role": "floor", "label": "OEI thrust margin"},
]
