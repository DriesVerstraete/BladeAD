"""Shahjahan Case 1 FPP -- Pareto-TRACER validation, hover_elec x cruise_elec
edge (2026-09-08).

Identical constraint set / objectives to `shahjahan_case1_fpp_explore_sm099`
(num_radial 32, stall_margin 0.99, OEI floor 1.143, cruise_elec proxy) -- same
as the `shahjahan_case1_fpp_reseed` case. Separate directory so the tracer's
`stage1_*` / seed artefacts do not mix with the re-seed run's pool.

Driven by `tracer_run.py` -> `BladeAD.optimisation.pareto_tracer.trace`:
NSGA-II basin seed -> seeded endpoint anchors -> predictor--corrector trace.
Compare to the corrected edge in `shahjahan_case1_fpp_reseed/`.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore_sm099.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
# hybrid driver config -- explicit pinned levels (from the manual tracer runs)
# rather than "auto", so a re-run reproduces the reference surface exactly.
CASE["hybrid"] = {
    "n_anchors": 3, "scout_pop": 48, "scout_gens": 60, "seed": 1,
    "scout_front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "n_points": 12, "cluster_method": "kmeans", "force_scout": False,
    "with_acoustics": False, "cross_check": True,
    "levels": {"oei_margin": [1.143, 1.33, 1.52, 1.70, 1.85],
               "hover_elec": [19300.0, 20000.0, 21000.0]},
}
