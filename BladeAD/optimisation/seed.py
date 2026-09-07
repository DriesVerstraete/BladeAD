"""Deterministic inverse-design cold-start seed for `solve.py`.

A physics-grounded starting blade for a given operating point -- replaces
hand-tuned seeds and ill-posed cold optimiser anchors. Moved (2026-09-06) from
`code/optimize_shahjahan_proprotor/inverse_design_seed.py`; the core is
unchanged, only the case glue is generalised
(`shahjahan_seed_inputs()` -> `seed_inputs_from_case(case)`).

Two closed-form methods, no optimiser, no airfoil model, no CSDL:

  mil_blade()          -- Adkins & Liebeck (1994) minimum-induced-loss propeller
                          design. For a forward-flight / cruise point.
  hover_ideal_blade()  -- momentum theory + blade-element ideal twist
                          atan(v_i / Omega r). For an axial-hover point.
  design_seed_blade()  -- unified entry with a Cl ramp.

Keeping the core decoupled from any CSDL airfoil model / graph is deliberate --
the seed generator must be structurally incapable of the Re-clamp / divergence
failures it exists to prevent. `fit_bspline_cps()` is the one helper that
imports BladeAD (its B-spline basis matrix), kept at the bottom.

Signals to monitor (returned in the result dict):
  converged / feasible  -- zeta fixed point settled / thrust achievable at a
                           physical section Cl. `feasible=False` even at cl_max
                           => the operating point itself is the problem (the
                           orchestrator's pre-flight gate aborts on this).
  reynolds              -- local Re per station.

Adkins & Liebeck, "Design of Optimum Propellers", J. Propulsion and Power
10(5), 1994.
"""
import os
import pickle

import numpy as np

# algorithm params (NOT problem constraints): the Cl ramp start/step and the
# drag coefficient for the MIL circulation. Documented, tunable.
CL_RAMP_START = 0.6
CL_RAMP_STEP = 0.05
MIL_CD = 0.011          # MH-series section ballpark at design Re
SEED_N_STATIONS = 41


# ---------------------------------------------------------------------------
# Adkins & Liebeck (1994) minimum-induced-loss propeller design
# ---------------------------------------------------------------------------

def mil_blade(*, n_blades, radius_m, hub_radius_m, omega_rad_s, airspeed_m_s,
              density, thrust_n, n_stations=41, cl_design=0.6, cd=0.011,
              lift_slope_per_rad=2.0 * np.pi, zeta_tol=1e-7, max_iter=60):
    """Adkins-Liebeck MIL blade for a propeller operating point.

    Returns dict: r_m, chord_m, twist_rad, zeta, converged, n_iter,
    reynolds, alpha_rad, thrust_check_n, phi_rad.

    `cl_design` / `cd` may be scalars or length-`n_stations` arrays. `cd` may
    also be a callable cd(cl, re) -> array.
    """
    R, rh, B = float(radius_m), float(hub_radius_m), int(n_blades)
    V, rho, Om = float(airspeed_m_s), float(density), float(omega_rad_s)
    if V <= 1.0:
        raise ValueError("mil_blade is a propeller method; use hover_ideal_blade for V ~ 0")

    xi = np.linspace(rh / R, 1.0, n_stations)          # r/R
    r = xi * R
    x = Om * r / V                                     # local speed ratio
    lam = V / (Om * R)                                 # advance ratio V/(Omega R)
    cl = np.broadcast_to(np.asarray(cl_design, float), xi.shape).copy()
    Tc_target = 2.0 * thrust_n / (rho * V ** 2 * np.pi * R ** 2)

    zeta = 0.0
    converged = False
    for it in range(1, max_iter + 1):
        phi_t = np.arctan(lam * (1.0 + zeta / 2.0))
        phi = np.arctan(np.tan(phi_t) / xi)
        f = (B / 2.0) * (1.0 - xi) / max(np.sin(phi_t), 1e-9)
        F = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        F = np.maximum(F, 1e-6)
        G = F * x * np.cos(phi) * np.sin(phi)

        if callable(cd):
            W_approx = V / np.sin(phi)
            re_approx = rho * W_approx * (0.05 * R) / 1.81e-5
            cd_arr = np.asarray(cd(cl, re_approx), float)
        else:
            cd_arr = np.broadcast_to(np.asarray(cd, float), xi.shape)
        eps = cd_arr / cl

        a = (zeta / 2.0) * np.cos(phi) ** 2 * (1.0 - eps * np.tan(phi))
        a_prime = (zeta / (2.0 * x)) * np.cos(phi) * np.sin(phi) * (1.0 + eps / np.tan(phi))
        W = V * (1.0 + a) / np.sin(phi)

        Wc = 4.0 * np.pi * lam * G * V * R * zeta / (cl * B)
        chord = np.where(zeta > 0, Wc / W, 0.05 * R)

        I1p = 4.0 * xi * G * (1.0 - eps * np.tan(phi))
        I2p = lam * (I1p / (2.0 * xi)) * (1.0 + eps / np.tan(phi)) * np.sin(phi) * np.cos(phi)
        I1 = np.trapz(I1p, xi)
        I2 = np.trapz(I2p, xi)
        disc = I1 ** 2 - 4.0 * I2 * Tc_target
        if disc < 0.0 or I2 <= 0.0:
            return {"r_m": r, "chord_m": chord, "twist_rad": np.arctan(np.tan(phi_t) / xi),
                    "zeta": zeta, "converged": False, "n_iter": it,
                    "reason": "target thrust infeasible (disc<0)",
                    "reynolds": rho * W * chord / 1.81e-5, "phi_rad": phi,
                    "alpha_rad": cl / lift_slope_per_rad,
                    "thrust_check_n": float(0.5 * rho * V ** 2 * np.pi * R ** 2 * (I1 * zeta - I2 * zeta ** 2))}
        zeta_new = (I1 - np.sqrt(disc)) / (2.0 * I2)
        if abs(zeta_new - zeta) < zeta_tol:
            zeta = zeta_new
            converged = True
            break
        zeta = zeta_new

    alpha = cl / lift_slope_per_rad
    twist = alpha + phi
    Tc = I1 * zeta - I2 * zeta ** 2
    return {
        "r_m": r, "chord_m": chord, "twist_rad": twist, "phi_rad": phi,
        "alpha_rad": alpha, "zeta": float(zeta), "converged": converged, "n_iter": it,
        "reynolds": rho * W * chord / 1.81e-5,
        "thrust_check_n": float(0.5 * rho * V ** 2 * np.pi * R ** 2 * Tc),
    }


# ---------------------------------------------------------------------------
# Momentum theory + blade-element ideal twist -- hover
# ---------------------------------------------------------------------------

def hover_ideal_blade(*, n_blades, radius_m, hub_radius_m, omega_rad_s, density,
                      thrust_n, n_stations=41, cl_design=0.6, tip_loss=True,
                      lift_slope_per_rad=2.0 * np.pi):
    """Ideal-twist hover blade: phi(r) = atan(v_i / Omega r) with a uniform
    induced velocity v_i from momentum theory, chord sized so each station runs
    at `cl_design`.

    Returns dict: r_m, chord_m, twist_rad, v_i, thrust_check_n, reynolds.
    """
    R, rh, B = float(radius_m), float(hub_radius_m), int(n_blades)
    rho, Om = float(density), float(omega_rad_s)
    A = np.pi * (R ** 2 - rh ** 2)
    v_i = np.sqrt(thrust_n / (2.0 * rho * A))

    xi = np.linspace(rh / R, 1.0, n_stations)
    r = xi * R
    U_t = Om * r
    phi = np.arctan2(v_i, U_t)
    W = np.sqrt(U_t ** 2 + v_i ** 2)

    if tip_loss:
        f = (B / 2.0) * (1.0 - xi) / np.maximum(np.sin(phi), 1e-6)
        Fl = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f), 0.0, 1.0))
        Fl = np.maximum(Fl, 1e-3)
    else:
        Fl = np.ones_like(xi)

    integrand_shape = B * 0.5 * rho * W ** 2 * cl_design * np.cos(phi) * Fl
    taper_shape = (1.0 - 0.6 * xi)
    T_unit = np.trapz(integrand_shape * taper_shape, r)
    c_scale = thrust_n / max(T_unit, 1e-9)
    chord = np.maximum(c_scale * taper_shape, 1e-3)

    alpha = cl_design / lift_slope_per_rad
    twist = alpha + phi
    T_check = np.trapz(B * 0.5 * rho * W ** 2 * chord * cl_design * np.cos(phi) * Fl, r)
    return {
        "r_m": r, "chord_m": chord, "twist_rad": twist, "v_i": float(v_i),
        "reynolds": rho * W * chord / 1.81e-5,
        "thrust_check_n": float(T_check), "converged": True, "n_iter": 1,
    }


# ---------------------------------------------------------------------------
# Cl ramp: a design that fails at Cl=0.6 may just need a more loaded blade.
# ---------------------------------------------------------------------------

def _enforce_taper(chord, taper_bounds):
    if taper_bounds is None:
        return chord
    lo, hi = taper_bounds
    root = float(chord[0])
    out = np.maximum(chord, lo * root)
    if out[-1] > hi * root:
        out[-1] = hi * root
    return out


def design_seed_blade(kind, *, cl_start=0.6, cl_max=0.8, cl_step=0.05,
                      solidity_max=1.2, taper_bounds=(0.15, 0.9), **kw):
    """Unified seed entry. kind = 'hover' | 'cruise'.

    Ramps cl_design from cl_start to cl_max. For 'cruise' (mil_blade) it stops
    at the first zeta-converged blade; for 'hover' (hover_ideal_blade) at the
    first blade whose peak local solidity B*c/(2*pi*r) <= solidity_max. The
    chord profile is then clamped so tip/root taper lands inside `taper_bounds`.

    Returns the blade dict plus: cl_used, feasible (False only if cl_max still
    failed => thrust genuinely exceeds what the disk supports at a physical Cl),
    taper.
    """
    solver = mil_blade if kind == "cruise" else hover_ideal_blade
    cls = np.round(np.arange(cl_start, cl_max + 1e-9, cl_step), 4)
    last = None
    for cl in cls:
        d = solver(cl_design=float(cl), **kw)
        last = d
        if kind == "cruise":
            ok = d["converged"]
        else:
            r, c, B = d["r_m"], d["chord_m"], kw["n_blades"]
            ok = float(np.max(B * c / (2.0 * np.pi * r))) <= solidity_max
        if ok:
            chord = _enforce_taper(d["chord_m"], taper_bounds)
            return {**d, "chord_m": chord, "cl_used": float(cl), "feasible": True,
                    "taper": float(chord[-1] / chord[0])}
    chord = _enforce_taper(last["chord_m"], taper_bounds)
    return {**last, "chord_m": chord, "cl_used": float(cls[-1]), "feasible": False,
            "taper": float(chord[-1] / chord[0]),
            "note": f"{kind}: infeasible even at cl_max={cl_max} -- thrust exceeds "
                    "what the disk supports at a physical section Cl"}


def fpp_blend_blade(*, hover_kw, cruise_kw, hover_weight=0.4,
                    taper_bounds=(0.15, 0.9)):
    """Fixed-pitch (FPP) seed -- one blade that must hover AND cruise.

    Chord comes from the hover-ideal blade (it has to carry the hover
    solidity; the cruise-MIL blade is razor-thin and cannot make hover
    thrust). Twist keeps the hover-ideal *mean* pitch (so hover sections stay
    near cl_design, not stalled) and blends in the cruise-MIL blade's extra
    *washout* -- `1 - hover_weight` of the way -- so the built-in root-tip
    gradient lets the tip approach cruise pitch while the root hovers
    (Shahjahan's "large blade pitch change per unit span"). The shared
    collective DV then trims the mean for the compromise. Not an optimum, a
    non-diverging start off the hover-biased local corner a pure hover seed
    lands in.

    Returns the hover-ideal blade dict with `twist_rad` replaced by the blend,
    plus `feasible` (both single-point blades feasible), `taper`, `_blend`.
    """
    h = design_seed_blade("hover", **hover_kw)
    c = design_seed_blade("cruise", **cruise_kw)
    w = float(np.clip(hover_weight, 0.0, 1.0))
    tw_h = h["twist_rad"]
    tw_c = np.interp(h["r_m"], c["r_m"], c["twist_rad"])
    twist = w * tw_h + (1.0 - w) * tw_c   # blade angle blend; cl overshoot in hover is
                                          # expected and the optimiser trims it (FPP penalty)
    chord = _enforce_taper(h["chord_m"], taper_bounds)
    return {**h, "chord_m": chord, "twist_rad": twist,
            "cl_used": h["cl_used"], "feasible": bool(h["feasible"] and c["feasible"]),
            "taper": float(chord[-1] / chord[0]),
            "_blend": {"hover_weight": w, "hover_cl": h["cl_used"], "cruise_cl": c["cl_used"]}}


# ---------------------------------------------------------------------------
# Case glue: every quantity read from the case dict, nothing re-typed
# ---------------------------------------------------------------------------

def seed_inputs_from_case(case):
    """kwargs for design_seed_blade('hover'|'cruise', **kw), all derived from a
    case dict (`case.py`'s CASE). Reads rotor geometry, operating points,
    constraints["cl_max"] / ["taper"]. RPM comes from the case's seed guess when
    the seed is hand-specified, else from the rpm-bound midpoint."""
    from .case import isa_atmos_density

    rot = {"n_blades": case["rotor"]["n_blades"],
           "radius_m": case["rotor"]["radius_m"],
           "hub_radius_m": case["rotor"]["hub_radius_m"]}
    c = case["constraints"]
    # the seed's Cl ramp cap: the flat cl_max if set, else a mid-section-airfoil
    # ballpark for the section-Cl_max mode (the seed is only a starting blade).
    cl_ramp_cap = c.get("cl_max", 1.2)
    common = dict(rot, n_stations=SEED_N_STATIONS, cl_start=CL_RAMP_START,
                  cl_step=CL_RAMP_STEP, cl_max=cl_ramp_cap, taper_bounds=tuple(c["taper"]))

    seed = case["seed"]
    if isinstance(seed, dict):
        hover_rpm = seed["hover_rpm"]
        cruise_rpm = seed["cruise_rpm"]
    else:
        hover_rpm = 0.5 * sum(case["bounds"]["hover_rpm"])
        cruise_rpm = 0.5 * sum(case["bounds"]["cruise_rpm"])

    hov = case["operating"]["hover"]
    cru = case["operating"]["cruise"]
    hover_kw = dict(common,
                    omega_rad_s=hover_rpm / 60.0 * 2 * np.pi,
                    density=isa_atmos_density(hov["altitude_m"]), thrust_n=hov["thrust_n"])
    cruise_kw = dict(common, cd=MIL_CD,
                     omega_rad_s=cruise_rpm / 60.0 * 2 * np.pi,
                     airspeed_m_s=cru["airspeed_m_s"],
                     density=isa_atmos_density(cru["altitude_m"]), thrust_n=cru["thrust_n"])
    return {"hover": hover_kw, "cruise": cruise_kw,
            "targets": {"hover": hov["thrust_n"], "cruise": cru["thrust_n"]}}


def preflight_feasibility(case):
    """Run design_seed_blade for both operating points. Returns
    {point: blade_dict}. The orchestrator aborts if any `feasible` is False."""
    inp = seed_inputs_from_case(case)
    return {name: design_seed_blade(name, **inp[name]) for name in ("hover", "cruise")}


def write_inverse_design_seed(ctx, kind):
    """Write an `initial_result_from`-format pkl from the deterministic
    inverse-design blade for `kind` ('hover' | 'cruise' | 'fpp') into the case
    dir; return its path. Cached by filename. 'fpp' is the fixed-pitch blend
    (`fpp_blend_blade`); the shared collective is 0 (the blade angle is the full
    angle) and both rpm guesses come from the bound midpoints, not the extremes."""
    out = ctx.out(f"seed_inverse_design_{kind}.pkl")
    if os.path.exists(out):
        return out
    case = ctx.case
    inp = seed_inputs_from_case(case)
    if kind == "fpp":
        blade = fpp_blend_blade(
            hover_kw=inp["hover"], cruise_kw=inp["cruise"],
            hover_weight=case["rotor"].get("fpp_seed_hover_weight", 0.4),
            taper_bounds=tuple(case["constraints"]["taper"]))
    else:
        blade = design_seed_blade(kind, **inp[kind])
    r_over_R = blade["r_m"] / case["rotor"]["radius_m"]
    hub_frac = case["rotor"]["hub_radius_m"] / case["rotor"]["radius_m"]
    norm = (r_over_R - hub_frac) / (1.0 - hub_frac)
    n_cp = case["rotor"]["n_chord_cps"]
    order = case["rotor"]["bspline_order"]
    hover_rpm_guess = (case["seed"]["hover_rpm"] if isinstance(case["seed"], dict)
                       else 0.5 * sum(case["bounds"]["hover_rpm"]))
    cruise_rpm_guess = (case["seed"]["cruise_rpm"] if isinstance(case["seed"], dict)
                        else 0.5 * sum(case["bounds"]["cruise_rpm"]))
    seed = {
        "chord_cps": fit_bspline_cps(blade["chord_m"], norm, n_cp, order),
        "twist_cps": fit_bspline_cps(blade["twist_rad"], norm, n_cp, order),   # radians
        # a cruise-shaped blade held to the hover thrust equality needs all the RPM it can get;
        # the fpp blend already carries hover solidity so it uses the midpoint guess.
        "rpm": (float(case["bounds"]["hover_rpm"][1]) if kind == "cruise"
                else float(hover_rpm_guess)),
        "cruise_rpm": float(cruise_rpm_guess),
        "oei_rpm": float(hover_rpm_guess),
        "hover_pitch_deg": 0.0,     # MIL / ideal / blend twist is already the full blade angle
        "cruise_pitch_deg": 0.0,
        "pitch_deg": 0.0,          # FPP shared collective (blend twist is the full angle)
        "_inverse_design_kind": kind, "_feasible": blade["feasible"],
    }
    with open(out, "wb") as f:
        pickle.dump(seed, f)
    return out


# ---------------------------------------------------------------------------
# Fit a spanwise profile onto B-spline control points (least squares)
# ---------------------------------------------------------------------------

def fit_bspline_cps(profile_values, sample_locations, n_cp, order=4):
    """Least-squares control points so BsplineParameterization reproduces
    `profile_values` at `sample_locations` (normalized 0..1). Uses BladeAD's own
    B-spline basis matrix so the fit matches the optimiser's evaluation."""
    from BladeAD.utils.parameterization import get_bspline_mtx
    B = get_bspline_mtx(n_cp, len(profile_values), order,
                        sample_locations=np.asarray(sample_locations, float)).toarray()
    cps, *_ = np.linalg.lstsq(B, np.asarray(profile_values, float), rcond=None)
    return cps
