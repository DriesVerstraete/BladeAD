"""Shahjahan Case 1 (FPP) -- NSGA-II population-method arm (roadmap step 4).

Same rotor / operating points / airfoil stack / motor as `shahjahan_case1_fpp`,
with the `shahjahan_case1_fpp_explore` knobs (num_radial 32) for a cheaper
per-eval cost. Differences from the epsilon-constraint case:

  * PHASE 1 objectives: hover electrical power vs cruise electrical power only
    (2-objective). OEI thrust margin is added later as a third objective
    (`objectives` + a `bounds['oei_rpm']` entry -- see shahjahan_case1_fpp).
  * stall_margin 0.99 (matches the sm099 sensitivity variant -- incipient
    stall, still inside the NeuralFoil table's fitted alpha range).
  * roles ('proxy'/'cap') are only there to satisfy `case._validate`; the
    NSGA-II runner reads `goal` only (both minimised) and poses thrust as a
    lower-bound constraint + KS overshoot penalty
    (`reference_nsga2_equality_constraint_limitation`).

The `nsga2` block is read by `run.py` / `nsga2_runner.run`, not by `_validate`.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

CASE["rotor"]["num_radial"] = 32
CASE["constraints"]["stall_margin"] = 0.99
CASE["constraints"]["thrust_tolerance_n"] = 5.0     # post-solve check only; NSGA-II uses the KS penalty

# PHASE 1: 2-objective (hover power vs cruise power). Drop OEI for now.
CASE["objectives"] = [
    {"name": "hover_elec",  "result_key": "hover_electrical_power",  "goal": "min", "role": "cap",
     "label": "hover electrical power (W)"},
    {"name": "cruise_elec", "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
     "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
]
CASE["constraints"].pop("min_oei_thrust_margin", None)   # no OEI objective in phase 1
CASE["bounds"].pop("oei_rpm", None)

# Real run (2026-09-08). pop_size = 5*D (D = 13 DVs); 100 generations, stop /
# restart as needed (`hot_start: True` resumes from
# results/optimisation_history_<name>_<save_tag>.pkl -- keep save_tag + pop_size
# fixed across a resume). Initial population is seeded from the inverse-design
# FPP blend + Gaussian clones (seed_clone_frac of the pop) with a one-shot
# hover-rpm sqrt(T*/T) prescale -- parity with the SLSQP seed; pure LHS cannot
# reach hover thrust (2026-09-08 smoke). Penalty softened from the smoke
# (mu 10->5, cap 1.0->0.5): cruise *undershoot* is the binding constraint,
# overshoot self-closes through the min-power objective, so a steep KS cliff is
# not needed and blocks SBX stepping-stones.
CASE["nsga2"] = {
    "pop_size": 65,
    "n_gen": 100,
    "seed": 1,
    "n_workers": 1,          # 1 -> serial (MPI singleton path); >1 -> multiprocess.Pool (untested)
    "eta_c": 15.0,
    "eta_m": 20.0,
    "p_c": 0.9,
    "p_m": None,             # None -> 1/n_var
    "ks_rho": 100.0,
    "overshoot_mu": 5.0,
    "overshoot_ks_cap": 0.5,
    "seed_clone_frac": 0.5,
    "seed_clone_sigma": 0.08,
    "save_tag": "run1",      # fixed (not a timestamp) so hot_start can find the history file
    "hot_start": False,      # flip to True + re-run the same command to resume
}
