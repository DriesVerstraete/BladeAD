"""Shahjahan Case 1 FPP -- 3-objective with CRUISE NOISE in place of the OEI
thrust margin (our "Case 1N minus transition", roadmap step 6 precursor).

Objectives: hover electrical power (cap) / cruise electrical power (proxy) /
cruise vehicle OSPL (cap).  Same rotor / airfoil stack / motor as
`shahjahan_case1_fpp_explore_sm099` (num_radial 32, stall_margin 0.99); the OEI
slot and its `min_oei_thrust_margin` floor are dropped, `cruise_noise` added.

Hover noise is deliberately NOT an objective here: Gill--Lee returns a
non-physical hover OSPL (~314 dB) on this Shahjahan-class rotor -- out of its
APC-11x4 fit envelope, no domain guard. Cruise vehicle noise (~64 dB vs the
62 dB cap) is in range and physical. See the rotor-optimisation project
`status.md` "Hover broadband acoustic model -- out of validity envelope" and
`briefs/noise-objective-deblock-steps.md`.

`case["acoustic"]` (n_rotors 8, cruise_noise_cap_db 62) is inherited from
`shahjahan_case1`: the constrained quantity is the vehicle OSPL
(single-rotor + 10*log10(8)), matching how `cruise_vehicle_ospl` is defined.

First run uses `levels: {}` + auto-bracketing -- the single-objective anchors
find the achievable cruise-noise span. Pin explicit `levels` for the production
run once that span is known (acoustic bracketing solves are ~300 s each).
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore_sm099.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)

CASE["constraints"].pop("min_oei_thrust_margin", None)

CASE["objectives"] = [
    {"name": "hover_elec",   "result_key": "hover_electrical_power",  "goal": "min", "role": "cap",
     "label": "hover electrical power (W)"},
    {"name": "cruise_elec",  "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
     "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
    {"name": "cruise_noise", "result_key": "acoustics.cruise_vehicle_ospl_db", "goal": "min", "role": "cap",
     "label": "cruise vehicle noise (dB)"},
]

CASE["hybrid"] = {
    "n_anchors": 2,
    "scout_pop": 48,
    "scout_gens": 60,
    "scout_objectives": ["hover_elec", "cruise_elec"],   # BEM kernel cannot evaluate acoustics
    "scout_front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "seed": 1,
    "n_points": 6,
    "cluster_method": "kmeans",
    "force_scout": False,
    "with_acoustics": True,
    "cross_check": True,
    "levels": {},                                        # auto-bracket hover_elec + cruise_noise
    "auto_levels": 3,
}
