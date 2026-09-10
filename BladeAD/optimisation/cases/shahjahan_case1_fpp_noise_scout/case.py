"""Shahjahan Case 1 FPP -- NSGA-II ACOUSTIC BASIN SCOUT.

Question: the ep/tracer hybrid noise front (`shahjahan_case1_fpp_noise`,
72 pts) floors at cruise_noise = 72.75 dB and its quiet end is bound-pinned on
hover power -- is the whole pipeline basin-trapped, i.e. is there a distinct
geometry basin that reaches meaningfully below 72.75 dB? The Step-1 reconcile
found the infeasible FPP cold seed at 66.8 dB vehicle (~6 dB under the front),
so a quieter feasible region plausibly exists.

This is a population-method BASIN SCOUT, not a front tracer. It is EXPECTED to
collapse toward the utopia corner -- that is fine. The deliverable is the set of
DISTINCT collapsed geometries (mine `results/optimisation_history_*` via
`scratchpad/basin_compare.py`), to be handed to the gradient tracer as anchors
for a follow-on dense hybrid run, plus a check whether any collapsed geometry
beats 72.75 dB.

Objectives (PI, 2026-09-11): **`{cruise_noise, cruise_elec}`** -- the widest
genuine trade (quiet cruise ⇒ geometry that cruises badly). `hover_elec` is a
LOOSE CONSTRAINT here, not an objective: `constraints.max_hover_elec_w`
(read by `RotorNSGA2Setup._constraints` as `hover_elec_cap`). Thrust stays a
`T >= T_target` lower bound + KS overshoot penalty
(`reference_nsga2_equality_constraint_limitation`).

Deep-copies `shahjahan_case1_fpp_noise` (num_radial 32, stall_margin 0.99,
OEI dropped, `acoustic` block n_rotors 8, `sweep.maxiter` 400). Needs the
`forward.evaluate(with_acoustics=True)` path (branch `noise-objective` @`0ab9d4d`
and later).
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_noise.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

# hover_elec: objective -> loose constraint. Front hover_elec span is
# 19.53-22.90 kW; 25 kW (~9% over the max) only rejects absurd hover designs,
# it does not shape the trade. Tune here if the scout wanders hover-expensive.
CASE["constraints"]["max_hover_elec_w"] = 25000.0

# 2-objective: cruise noise (cap) vs cruise electrical power (proxy). Drop the
# hover_elec objective slot.
CASE["objectives"] = [
    {"name": "cruise_elec",  "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
     "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
    {"name": "cruise_noise", "result_key": "acoustics.cruise_vehicle_ospl_db", "goal": "min", "role": "cap",
     "label": "cruise vehicle noise (dB)"},
]

# The hybrid block is inherited but unused by the NSGA-II runner; leave it.

# Real scout (2026-09-11). pop x gen sized for a basin map, not a front trace.
# n_workers > 1: the acoustic forward eval is ~10-100x a BEM eval
# (`multiprocess.Pool`; Codex 2-worker smoke ~2.0x speedup at ~0.17 s/eval).
# front_seed_from: the sm099 feasible edge geometries ONLY (~16 pts) -- a spread
# feasible starting set so the population can hold a front it cannot build
# (2026-08-31). Deliberately NOT the `shahjahan_case1_fpp_noise` front: those
# are all in the ONE basin this scout exists to escape, and there are ~200 of
# them -- seeding from them would flood the pop (pop_size 96) and kill
# exploration. The seed prescale + LHS fill provide the rest.
CASE["nsga2"] = {
    "pop_size": 96,
    "n_gen": 200,
    "seed": 1,
    "n_workers": 4,
    "eta_c": 15.0,
    "eta_m": 20.0,
    "p_c": 0.9,
    "p_m": None,
    "ks_rho": 100.0,
    "overshoot_mu": 5.0,
    "overshoot_ks_cap": 0.5,
    "seed_clone_frac": 0.5,
    "seed_clone_sigma": 0.08,
    "front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "save_tag": "run1",
    "hot_start": False,
}
