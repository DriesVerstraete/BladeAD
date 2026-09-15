"""Shahjahan Case 1 FPP -- FULLY-UNSEEDED LHS ACOUSTIC BASIN SCOUT.

`roadmap.md` "Later / deferred" (PI ask, 2026-09-11): the seeded scout
(`shahjahan_case1_fpp_noise_scout`) and the dense hybrid RUN 2 (checked
2026-09-14, `worklog.md`) both floor at ~72.4 dB, ~0.34 dB below the
2026-09-10 ep/tracer front (72.75 dB) -- marginal, not a distinct quieter
basin. This is the real test of whether a disconnected quiet basin exists
NOWHERE NEAR any known geometry: NO front seeding at all, pure LHS fill
(`seed_clone_frac 0`, `front_seed_from None`) plus the single inverse-design
lifeline blade `nsga2_runner` always seeds as population member 0.

Objectives unchanged from the seeded scout: `{cruise_elec, cruise_noise}`,
`hover_elec` a loose 25 kW constraint, thrust `T >= T_target` + KS overshoot
penalty (`reference_nsga2_equality_constraint_limitation`).

Population sized 20x n_var (PI, 2026-09-14): this case has 13 design
variables (chord_cps + twist_cps B-spline CPs + hover_rpm + cruise_rpm, no
OEI slot -- noise case drops it) -> pop_size 260. Generations NOT scaled up
(basin scout wants breadth, not depth) -- n_gen 100.

Deep-copies `shahjahan_case1_fpp_noise_scout` (which itself deep-copies
`shahjahan_case1_fpp_noise`: num_radial 32, stall_margin 0.99, OEI dropped,
acoustic block n_rotors 8, sweep.maxiter 400). Needs `forward.evaluate(
with_acoustics=True)` (branch `noise-objective` @`0ab9d4d` and later).
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_noise_scout.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

# Fully unseeded: no front geometries, no clone-around-seed fraction -- pure
# LHS fill (+ the inverse-design lifeline nsga2_runner always adds as member
# 0, regardless of pop_size).
CASE["nsga2"]["front_seed_from"] = None
CASE["nsga2"]["seed_clone_frac"] = 0.0
CASE["nsga2"]["seed_clone_sigma"] = 0.0

# 20 x n_var (13 vars -> 260), generations left at the scout default (100,
# not 200 -- breadth over depth for a basin scout).
CASE["nsga2"]["pop_size"] = 260
CASE["nsga2"]["n_gen"] = 100

CASE["nsga2"]["save_tag"] = "lhs_run1"
CASE["nsga2"]["hot_start"] = False
