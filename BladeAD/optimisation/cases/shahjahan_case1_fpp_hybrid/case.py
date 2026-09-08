"""Shahjahan Case 1 FPP -- overnight stress test of the generic hybrid driver
(`BladeAD.optimisation.hybrid.run_hybrid`), 2026-09-08.

Objectives inherited unchanged from `shahjahan_case1` via the sm099 explore
variant: hover electrical power (cap), cruise electrical power (proxy), OEI
thrust margin (floor). num_radial 32, stall_margin 0.99, OEI floor 1.143.

The hybrid block drives a full run: NSGA-II scout -> 3 clustered anchors ->
per anchor a primary `(cruise_elec, hover_elec)` stack over auto-bracketed
`oei_margin` levels + a cross-check `(cruise_elec, oei_margin)` stack over
auto-bracketed `hover_elec` levels -> goal-aware non-dominated merge.

Separate dir so the run's artefacts do not touch the manual tracer/orthogonal
reference pkls in `shahjahan_case1_fpp_tracer_hc/`.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore_sm099.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["hybrid"] = {
    "n_anchors": 3,
    "scout_pop": 48,
    "scout_gens": 60,
    "scout_objectives": None,       # all 3 objectives are BEM-computable here
    "scout_front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "seed": 1,
    "n_points": 12,
    "cluster_method": "kmeans",
    "force_scout": False,
    "with_acoustics": False,
    "cross_check": True,
    "levels": {},                   # auto-bracket both non-proxy axes
    "auto_levels": 6,
}
