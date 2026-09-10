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

# Loose hover-power cap. Only read by `RotorNSGA2Setup._constraints` (the scout
# phase), where `scout_objectives` no longer includes hover_elec -- it bounds
# hover power so the acoustic scout does not chase a quiet cruise blade that
# hovers absurdly. `solve.run` / the tracer ignore this key entirely; hover_elec
# stays a real swept objective there, and 25 kW is slack vs the 19.5-22.9 kW
# front span.
CASE["constraints"]["max_hover_elec_w"] = 25000.0

# The acoustic solve resolves the rotor azimuthally (num_azimuthal =
# NUM_AZIMUTHAL_ACOUSTIC, not 1), so the coupled hover+cruise BEM+motor+acoustic
# problem needs more SLSQP headroom than the non-acoustic FPP chain's 200 to
# close thrust / stall / torque from a warm start.
CASE["sweep"]["maxiter"] = 400

CASE["objectives"] = [
    {"name": "hover_elec",   "result_key": "hover_electrical_power",  "goal": "min", "role": "cap",
     "label": "hover electrical power (W)"},
    {"name": "cruise_elec",  "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
     "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
    {"name": "cruise_noise", "result_key": "acoustics.cruise_vehicle_ospl_db", "goal": "min", "role": "cap",
     "label": "cruise vehicle noise (dB)"},
]

# DENSE one-shot run (PI, 2026-09-11) -- acoustic scout + dense tracer surface
# in one command, meant to run overnight. `scout_objectives` is now the ACOUSTIC
# pair (a BEM-only scout is noise-blind and cannot find a quieter basin -- the
# whole reason for the forward.evaluate acoustics port); `hover_elec` is bounded
# by `constraints.max_hover_elec_w` above during the scout, and remains a full
# swept objective in the tracer stage. `n_points` / `auto_levels` / `n_anchors`
# bumped ~3x over the 72-pt 2026-09-10 run. `force_scout` -- do NOT reuse that
# run's BEM-only cached scout.
CASE["hybrid"] = {
    "n_anchors": 3,
    "scout_pop": 96,
    "scout_gens": 150,
    "scout_workers": 4,
    "scout_objectives": ["cruise_elec", "cruise_noise"],   # = active_objectives order
    "scout_front_seed_from": ["../shahjahan_case1_fpp_explore_sm099"],
    "seed": 1,
    "n_points": 18,
    "cluster_method": "kmeans",
    "force_scout": True,
    "with_acoustics": True,
    "cross_check": True,
    "levels": {},                                        # auto-bracket hover_elec + cruise_noise
    "auto_levels": 6,
}
