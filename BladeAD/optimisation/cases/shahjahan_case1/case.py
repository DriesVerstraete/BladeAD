"""Shahjahan (2024) -- Case 1: the 3-objective proprotor benchmark WITHOUT
transition (their VPP / FPP cases, Figs 11-12).

Objectives, all on a powertrain / thrust basis (NOT FM / eta):
  - hover electrical input power   (minimise)
  - cruise electrical input power  (minimise)   <- the swept proxy
  - OEI ("emergency hover") thrust margin  (maximise)

OEI thrust margin (paper footnote 2): "the ratio between the thrust generated
at the emergency hover condition to the thrust required to hover." One motor
out, the remaining rotors must still hover. Here:
  margin = T_available_at_OEI / T_hover_per_rotor        (denominator = 999.375 N)
Emergency hover shares nominal hover's collective (Shahjahan Table 2: collective
is a single hover-vs-cruise offset), so `oei_rpm` is the ONLY OEI design
variable -- the hover blade, spun up -- bounded by the x1.7 motor
continuous-torque envelope and cl_max; no thrust equality (thrust is the
objective). N/(N-1) = 8/7 = 1.143 is the true OEI-survivability threshold, but
NO hard floor is applied for now (set `constraints.min_oei_thrust_margin` to add
one) -- read survivability off the front, tighten later.

Paper baseline (Tables 4-5): MTOM 815 kg -> 7995 N hover / 8 rotors = 999.375 N
per rotor; 975 N cruise / 8 = 121.875 N; hover & emergency hover both at 25 m,
cruise 41.9 m/s at 500 m. Rotor geometry / airfoil stack carried over from
`shahjahan_test_motor_mcdonald` (motor step 1b). Airfoil tables are the
production xxxlarge NeuralFoil set. `cl_max = 0.8` is our sectional stall-margin
/ BEM-robustness cap (Shahjahan states no explicit rotor sectional Cl limit;
Table 4's C_Lmax 1.2 is the aircraft wing, not the blade).

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
        "radial_distribution": "cosine",
        "num_radial": 48,
    },
    "operating": {
        "hover":  {"altitude_m": 25.0,  "airspeed_m_s": 0.5,  "thrust_n": 999.375},
        "cruise": {"altitude_m": 500.0, "airspeed_m_s": 41.9, "thrust_n": 121.875},
        # OEI hover reuses the hover altitude / airspeed; its thrust is the
        # objective, so no target here (denominator = operating.hover.thrust_n).
    },
    "bounds": {
        "chord_m": (0.015, 0.350),
        "twist_deg": (-15.0, 85.0),
        "hover_rpm": (200.0, 3440.0),
        "cruise_rpm": (200.0, 3440.0),
        "oei_rpm": (200.0, 3440.0),           # motor map ceiling = 6500/1.89 = 3439 rpm
        "hover_pitch_deg": (0.0, 60.0),
        "cruise_pitch_deg": (0.0, 60.0),
    },
    "constraints": {
        "cl_max": 0.8,
        "taper": (0.15, 0.9),
        "thrust_tolerance_n": 1.0,
        # OEI survivability: one motor out, N-1 rotors carry the aircraft -> each
        # must make N/(N-1) = 8/7 = 1.143 of nominal per-rotor hover thrust. The
        # explore sweep showed margin is near-binary (~0.76 min-hover-power corner
        # vs ~1.9 everywhere else) and Shahjahan reports 1.3-1.7, so this floor
        # cuts only the non-survivable corner and costs ~nothing.
        "min_oei_thrust_margin": 1.143,
    },
    "airfoil": {
        "section_boundaries_r_over_r": [0.0, 0.22, 0.55, 0.85, 1.0],
        "names": ["MH126", "MH113", "MH115", "MH121"],
        "t_over_c": [0.250, 0.147, 0.111, 0.088],
        "table_dir": _TABLE_DIR,
        # Reference Reynolds per airfoil for the section-Cl_max stall constraint
        # (only used when constraints.stall_margin is set). Chosen from the
        # sectional-Re range across converged VPP + FPP blades (2026-09-07):
        # root ~0.5 M, mid-blade ~1 M, tip drops to ~0.4 M in VPP cruise and
        # MH121 is the most Re-sensitive airfoil, so the tip gets a conservative
        # 3e5. Derivation + revisit criteria: decisions/2026-09-07-section-clmax-stall-constraint.md.
        "clmax_ref_reynolds": [5.0e5, 1.0e6, 1.0e6, 3.0e5],   # MH126 / MH113 / MH115 / MH121
    },
    "acoustic": {
        "n_rotors": 8,
        "cruise_noise_cap_db": 62.0,
    },
    "seed": {
        "chord_cps_m": [0.122, 0.200, 0.125, 0.075, 0.044],
        "twist_cps_deg": [40.0, 28.0, 20.0, 15.0, 11.0],
        "hover_rpm": 1850.0,
        "cruise_rpm": 965.0,
        "oei_rpm": 2400.0,          # feasible on the crude seed blade; optimiser walks it up to ~2900-3400 as geometry sharpens
        "hover_pitch_deg": 0.0,
        "cruise_pitch_deg": 28.0,
    },
    "motor": {"model": "mcdonald", "k_hover": 1.7, "k_cruise": 1.0, "k_oei": 1.7},
    "objectives": [
        {"name": "hover_elec",  "result_key": "hover_electrical_power",  "goal": "min", "role": "cap",
         "label": "hover electrical power (W)"},
        {"name": "cruise_elec", "result_key": "cruise_electrical_power", "goal": "min", "role": "proxy",
         "proxy_min_key": "cruise_electrical_power", "label": "cruise electrical power (W)"},
        {"name": "oei_margin",  "result_key": "oei_thrust_margin",       "goal": "max", "role": "floor",
         "label": "OEI thrust margin"},
    ],
    "sweep": {
        "n_points": 12,
        "n_points_3obj": 45,
        "grid_fm": 8,
        "grid_noise": 8,
        "maxiter": 120,        # 3-point (hover+cruise+OEI) coupled solve needs more headroom than the 2-point cases
        "ftol": 1e-6,
    },
    "cruise_anchor_fm_floor_frac": 0.65,
}
