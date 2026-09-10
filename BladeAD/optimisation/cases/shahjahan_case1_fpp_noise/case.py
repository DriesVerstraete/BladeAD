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

# DENSE one-shot run (PI, 2026-09-11). Acoustic scout + dense tracer surface in
# one overnight command. `scout_objectives` = the ACOUSTIC pair (a BEM-only
# scout is noise-blind). `hover_elec` bounded during the scout by
# `constraints.max_hover_elec_w`; full swept objective in the tracer stage.
#
# RUN 2 (00:35, after run 1 killed at gen 45): run 1 seeded the scout from the
# sm099 EFFICIENT edge -> it collapsed to a high-RPM SKINNY blade at ~76.7 dB,
# ~4 dB LOUDER than the 2026-09-10 tracer floor (72.75), never near the quiet
# fat-low-RPM family. Fix (the "test = can we find a different basin"): seed the
# scout from a SPREAD of the tracer's own noise front (72.04 -> 81 dB), soften
# the KS overshoot penalty (mu 10 -> 2 -- it was crushing RPM-reduction moves)
# and widen mutation, so the population holds the known front and probes for
# anything BELOW it. Morning check: min feasible cruise_noise over the whole
# scout history vs 72.75.
CASE["hybrid"] = {
    "n_anchors": 3,
    "scout_pop": 96,
    "scout_gens": 150,
    "scout_workers": 4,
    "scout_objectives": ["cruise_elec", "cruise_noise"],   # = active_objectives order
    "scout_front_seed_from": [
        "stage1_opt-cruise_elec_grid-cosine32_ac_pinned.pkl",                                   # 72.04
        "stage1_opt-cruise_elec_grid-cosine32_ac_eps-cruise_noise=72.7488.pkl",                  # 72.75
        "stage1_opt-cruise_elec_grid-cosine32_ac_eps-cruise_noise=73.0640_eps-hover_elec=22521.1856.pkl",
        "stage1_opt-cruise_elec_grid-cosine32_ac_eps-cruise_noise=74.0058_eps-hover_elec=21025.4307.pkl",
        "stage1_opt-cruise_elec_grid-cosine32_ac_eps-hover_elec=21025.4307.pkl",                 # 74.58
        "stage1_opt-cruise_noise_grid-cosine32_eps-hover_elec=19529.6757.pkl",                   # 76.78
        "stage1_opt-cruise_elec_grid-cosine32_ac_eps-hover_elec=19514.8203.pkl",                 # 77.37
        "stage1_opt-hover_elec_grid-cosine32_ac.pkl",                                            # 81.11
    ],
    "scout_kwargs": {
        "overshoot_mu": 2.0,        # was 10 -- softened; RPM-reduction moves survive
        "seed_clone_frac": 0.35,
        "seed_clone_sigma": 0.15,
        "eta_m": 6.0,               # wider polynomial mutation (basin-hop)
        "eta_c": 12.0,
    },
    "seed": 1,
    "n_points": 18,
    "cluster_method": "kmeans",
    "force_scout": True,
    "with_acoustics": True,
    "cross_check": True,
    "levels": {},                                        # auto-bracket hover_elec + cruise_noise
    "auto_levels": 6,
}
