"""Shahjahan (2024) proprotor -- the reference case for the `rotor_pareto`
workflow. Config only: geometry, operating points, bounds, seed, airfoil stack,
acoustics, objective set and sweep budgets. All physics/solver logic lives in
the case-agnostic package.

Values carried over verbatim from the frozen
`code/optimize_shahjahan_proprotor/optimize_shahjahan_stage1.py` (`design` /
`constraints` / `solver` dicts) and its 2026-09-06 decision doc:

  3 blades, R 0.75 m, hub 0.1125 m (norm hub 0.15); cosine-48 aero grid.
  hover  999.375 N / 0.5 m/s / 25 m ;  cruise 121.875 N / 41.9 m/s / 500 m.
  4-section MH126/MH113/MH115/MH121 NeuralFoil B-spline composite + Mach
  correction; n_rotors = 8, vehicle cruise OSPL + 10*log10(8) <= 62 dB.

Acceptance target (N_POINTS=12): reproduce the 2026-09-06 anchors --
  max_fm   FM 0.800 / eta 0.633
  max_eta  FM ~ 0.68 / eta ~ 0.85
  min_noise FM 0.704 / eta 0.518
Then raise sweep.n_points_3obj to ~45 for the definitive 3-objective front.

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
    # objective slots (3-entry dict spec form -- the acceptance case for the
    # objective-slots refactor). Exactly one slot must be role "proxy".
    # 2-objective case: one edge, no grid, no nobj.
    "objectives": [
        {"name": "fm_hover",   "result_key": "figure_of_merit",   "goal": "max", "role": "floor", "label": "hover FM"},
        {"name": "eta_cruise", "result_key": "cruise_efficiency", "goal": "max", "role": "proxy", "proxy_min_key": "cruise_power", "label": "cruise efficiency"},
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
