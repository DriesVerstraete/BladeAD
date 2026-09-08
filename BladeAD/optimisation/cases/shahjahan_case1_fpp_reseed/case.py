"""Shahjahan Case 1 FPP -- epsilon-constraint re-trace SEEDED FROM THE NSGA-II
FRONT GEOMETRY (2026-09-08).

Identical constraint set to `shahjahan_case1_fpp_explore_sm099` (num_radial 32,
stall_margin 0.99, OEI floor 1.143). Separate directory only so the re-seeded
`stage1_*` / anchor artefacts do not mix with the original explore run's warm
pool. Driven by `reseed_epsilon_check.py`, not the orchestrator.

Purpose: the original `hover_elec__cruise_elec` edge front bottoms out at
cruise electrical power ~9278 W; NSGA-II (run1) found a fully-feasible design
(OEI margin 1.385) at ~8311 W, ~10% lower. Test whether SLSQP, seeded from that
geometry, confirms the lower-cruise-power basin -- i.e. whether the original
epsilon front is truncated by a locally-converged `cruise_elec` anchor.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore_sm099.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
