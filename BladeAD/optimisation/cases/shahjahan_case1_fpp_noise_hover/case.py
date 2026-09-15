"""Shahjahan Case 1 FPP -- 3-objective with HOVER NOISE in place of cruise
noise (PI-settled 2026-09-11/-14, the original Case 1N intent). Deep-copies
`shahjahan_case1_fpp_noise` (itself: num_radial 32, stall_margin 0.99, OEI
dropped) and swaps the acoustic objective from the cruise condition to the
hover condition.

Objectives (PI, 2026-09-11): `{hover_elec, cruise_elec, hover_noise}` --
`hover_elec` kept as a live swept objective (not just a scout constraint) so
the surface answers "is the quiet-hover end also hover-efficient?". Widest
genuine trade expected: quiet hover -> fat/high-solidity/high-twist FPP blade
-> poor cruise prop; hover_noise vs hover_elec only weakly opposed -> short
front on that pair. `cruise_noise` is no longer a live objective here, so it
reverts to a fixed default cap (`case["acoustic"]["cruise_noise_cap_db"]`,
62 dB) via `solve.run`'s normal acoustic-cap resolution -- unchanged code
path, confirmed by reading `solve.py`'s `_cn_is_objective` branch before this
case was written.

Blocker cleared 2026-09-14: `nsga2_runner.py`'s `hover_noise is
diagnostic-only` `ValueError` guard lifted -- `forward.evaluate` already
always computes `out["hover_noise"]` when `with_acoustics`, no other
plumbing needed. Hover-noise validity: not yet a dedicated systematic
diagnostic across the full feasible region, but every hover OSPL seen so far
on this FPP rotor class (2026-09-10 reconcile, 2026-09-14 gen4 pushes) is
physical (53-97 dB) -- no 314 dB-style blowup. Watch for one on this run.

No known hover-noise front exists yet -- `scout_front_seed_from` points at
the sm099 feasible edge (generic, not noise-specific), same as this project's
FIRST cruise-noise hybrid run before the RUN-2-era front-seeded rescue.
`levels: {}` + `auto_levels` auto-brackets both hover_elec and hover_noise;
n_anchors/n_points/auto_levels use the standing DENSE config (PI,
2026-09-11) since this run doubles as a pipeline robustness test.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_noise.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

# Swap the acoustic objective: cruise_noise -> hover_noise.
CASE["objectives"] = [
    {"name": "hover_elec",  "result_key": "hover_electrical_power", "goal": "min", "role": "cap",
     "label": "hover electrical power (W)"},
    {"name": "cruise_elec", "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
     "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
    {"name": "hover_noise", "result_key": "acoustics.hover_ospl_db", "goal": "min", "role": "cap",
     "label": "hover vehicle noise (dB)"},
]

# The inherited `constraints.max_hover_elec_w` cap is meaningless now that
# hover_elec is itself a swept objective (it was there to stop a cruise-noise
# scout chasing an absurd hover design) -- drop it.
CASE["constraints"].pop("max_hover_elec_w", None)

# The inherited `cruise_noise_cap_db=62` was calibrated for the cruise-noise
# case where cruise_noise WAS the driven objective -- here it's a leftover
# hard constraint that fights the hover-noise-optimized blade family.
# Diagnosed 2026-09-14 (smoke anchor-2 trace, unmuffled: BEM inner solver
# converges cleanly every step, no ill-posed-geometry chatter -- SLSQP simply
# prioritises the 62 dB cap over thrust: seed started near-feasible
# (hover_thrust +4%, cruise_thrust +3%) and the 400-iter result ends with
# hover_thrust -19%, cruise_thrust -32%, both max_cl blown, cruise_noise
# pinned exactly at 62.006 -- a genuine constraint conflict, not a
# convergence-budget issue. Loosened well above anything seen this session
# (loudest cruise number all session ~78 dB) so it doesn't bind.
CASE["acoustic"]["cruise_noise_cap_db"] = 85.0

CASE["hybrid"] = {
    "n_anchors": 3,
    "scout_pop": 96,
    "scout_gens": 150,
    "scout_workers": 4,
    "scout_objectives": ["cruise_elec", "hover_noise"],   # = active_objectives order
    "scout_front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "scout_kwargs": {
        "overshoot_mu": 2.0,
        "seed_clone_frac": 0.35,
        "seed_clone_sigma": 0.15,
        "eta_m": 6.0,
        "eta_c": 12.0,
    },
    "seed": 1,
    "n_points": 18,
    "cluster_method": "kmeans",
    "force_scout": True,
    "with_acoustics": True,
    "cross_check": True,
    "levels": {},
    "auto_levels": 6,
}
