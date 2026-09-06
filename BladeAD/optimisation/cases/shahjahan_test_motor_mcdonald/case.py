"""Shahjahan (2024) proprotor -- motor fidelity-ladder step 1b.

Identical to `cases/shahjahan_test_motor/` (the step-1a placebo case) except the
motor is the real McDonald loss map for the paper-scaled EMRAX-188
(`BladeAD.core.motor.SHAHJAHAN_EMRAX188_PARAMETERS`, reproduces Shahjahan's
`emrax188cc_scaled5` block exactly -- peak torque scaled to 170 Nm, max rpm
6500/1.89 = 3439). The continuous-torque envelope
(`SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE`, ~90 Nm continuous line) is enforced as
a constraint at both design points: Q <= k * Q_continuous(rpm) with k = 1.7 in
hover (Shahjahan raises the continuous line to 170% for hover / emergency hover
/ transition) and k = 1.0 in cruise.

Pre-checks (2026-09-07): at the converged FM-0.80 and eta-front anchors the
hover point sits at ~0.52 of the 1.7x envelope (the asb-track 65% overshoot did
NOT reproduce here); cruise is the binding case at ~0.90 of the 1.0x line.
McDonald eta ~ 0.944-0.950 at these points, ~= the 0.95 placebo, so 1b's value
is the torque-envelope constraint and eta varying across the front, not a level
shift.

Non-SI keys carry their unit as a suffix.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "airfoil_tables"))

CASE = {
    "rotor": {
        "n_blades": 3,
        "radius_m": 0.75,
        "hub_radius_m": 0.1125,
        "n_chord_cps": 5,
        "n_twist_cps": 5,
        "bspline_order": 4,
        "radial_distribution": "cosine",       # "cosine" (Chebyshev cell-centred) | "uniform"
        "num_radial": 48,
    },
    "operating": {
        "hover":  {"altitude_m": 25.0,  "airspeed_m_s": 0.5,  "thrust_n": 999.375},
        "cruise": {"altitude_m": 500.0, "airspeed_m_s": 41.9, "thrust_n": 121.875},
    },
    "bounds": {
        "chord_m": (0.015, 0.350),             # physical limits are on the profile (taper, Cl)
        "twist_deg": (-15.0, 85.0),
        "hover_rpm": (200.0, 3440.0),
        "cruise_rpm": (200.0, 3440.0),
        "hover_pitch_deg": (0.0, 60.0),        # rigid collective offset on the shared twist B-spline
        "cruise_pitch_deg": (0.0, 60.0),
    },
    "constraints": {
        "cl_max": 0.8,                          # stall-margin design cap + BEM robustness
        "taper": (0.15, 0.9),
        "thrust_tolerance_n": 1.0,              # post-solve check only
        # reserved (structure step 3, not built): "max_utilization <= 1"
    },
    "airfoil": {
        "section_boundaries_r_over_r": [0.0, 0.22, 0.55, 0.85, 1.0],
        "names": ["MH126", "MH113", "MH115", "MH121"],
        "t_over_c": [0.250, 0.147, 0.111, 0.088],
        "table_dir": _TABLE_DIR,
    },
    "acoustic": {
        "n_rotors": 8,
        "cruise_noise_cap_db": 62.0,
    },
    "seed": {
        "chord_cps_m": [0.122, 0.200, 0.125, 0.075, 0.044],   # spec section 8 blade
        "twist_cps_deg": [40.0, 28.0, 20.0, 15.0, 11.0],
        "hover_rpm": 1850.0,
        "cruise_rpm": 965.0,
        "hover_pitch_deg": 0.0,
        "cruise_pitch_deg": 28.0,
    },
    # motor step 1b: real McDonald loss map + continuous-torque envelope.
    # k_hover / k_cruise default to 1.7 / 1.0 if omitted.
    "motor": {"model": "mcdonald", "k_hover": 1.7, "k_cruise": 1.0},
    "objectives": [
        {"name": "fm_hover",        "result_key": "figure_of_merit",         "goal": "max", "role": "floor", "label": "hover FM"},
        {"name": "electrical_power", "result_key": "cruise_electrical_power",  "goal": "min", "role": "proxy", "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
        {"name": "hover_noise",     "result_key": "acoustics.hover_ospl_db",  "goal": "min", "role": "cap",   "label": "hover noise (dB)"},
    ],
    "sweep": {
        "n_points": 12,          # pairwise 2-objective edge fronts
        "n_points_3obj": 45,     # nobj picker sweep (optional gap-fill, not run by default)
        "grid_fm": 8,            # epsilon-grid 3-obj lattice: FM-floor divisions
        "grid_noise": 8,         # epsilon-grid 3-obj lattice: noise-cap divisions
        "maxiter": 60,           # a clean solve is ~33 iters; 60 caps thrashers at ~3.6 min
        "ftol": 1e-6,
    },
    "cruise_anchor_fm_floor_frac": 0.65,
}
