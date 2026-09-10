"""One BEM forward evaluation of a rotor design point -- NO optimiser, NO
`set_as_objective`/`set_as_constraint`, NO SLSQP.

The population-method eval kernel (`nsga2_runner.py` and any future EA/DE/CMA
driver). Given a full design vector it builds the CSDL/BEM graph once, inline,
reads every quantity an objective or constraint could name, tears the recorder
down, and returns a flat dict of Python floats.

The pure, side-effect-free helpers (`_isa_atmos`, `_spec_from_case`,
`_airfoil_model`, `_section_clmax_profile`, `_bem_point`, `_op`,
`evaluate_case_motor`, the scaler constants) are imported from `solve.py` so
there is exactly one copy. Only the driver differs: `solve.run()` wires the
graph into modopt; `evaluate()` just reads `.value`.

    TECH DEBT: `solve.run()` still builds its own copy of this same graph
    (lines ~362-540). Reconciling the two -- `solve.run()` calling a shared
    builder -- is tracked in `todos.md` (2026-09-08). Kept separate for now so
    the epsilon-constraint workflow carries zero risk from the NSGA-II work.

Run in the `rotor_design` conda env.
"""
import numpy as np
import csdl_alpha as csdl

from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters, AtmosStates  # noqa: F401
from BladeAD.utils.parameterization import BsplineParameterization

from . import solve as _s
from .acoustics import (
    NUM_AZIMUTHAL_ACOUSTIC,
    acoustic_mesh_fields,
    add_spl_hover_cruise_acoustics,
    thickness_to_chord_profile,
)


def _stall_ratio_value(bem_out, clmax_profile, num_radial):
    """Peak sectional cl / section-Cl_max over the blade (a plain float). Mirrors
    `solve._stall_ratio` but reads `.value` directly -- no csdl graph node."""
    cl = np.asarray(bem_out.sectional_lift_coefficient.value)
    n = int(cl.size)
    k = n // num_radial
    if k >= 1 and k * num_radial == n:
        prof = np.repeat(clmax_profile, k)
    else:
        prof = np.full(n, float(clmax_profile.min()))
    return float(np.max(cl.reshape(-1) / prof.reshape(-1)))


def _failed_result(case, oei):
    """Sentinel for a candidate the forward model cannot evaluate (Re/Mach out
    of table range even after clamping, non-convergent inflow, NaN). Power is
    huge, thrust zero, stall ratio and margins far past their caps -- so the
    EA ranks it last but the run continues (population methods probe extremes,
    `feedback_bem_surrogate_table_domain_margins`)."""
    hov = case["operating"]["hover"]["thrust_n"]
    cru = case["operating"]["cruise"]["thrust_n"]
    d = {
        "hover_power": 1e12, "cruise_power": 1e12,
        "hover_electrical_power": 1e12, "cruise_electrical_power": 1e12,
        "figure_of_merit": 0.0, "cruise_efficiency": 0.0,
        "hover_thrust": 0.0, "cruise_thrust": 0.0,
        "hover_thrust_target": hov, "cruise_thrust_target": cru,
        "hover_thrust_overshoot": 0.0, "cruise_thrust_overshoot": 0.0,
        "hover_stall_ratio": 10.0, "cruise_stall_ratio": 10.0,
        "cl_cap": float(case["constraints"].get("stall_margin", case["constraints"].get("cl_max", 1.0))),
        "taper": 0.0, "twist_washout_deg": -90.0,
        "hover_torque_margin": 10.0, "cruise_torque_margin": 10.0,
        "hover_rpm": float("nan"), "cruise_rpm": float("nan"), "collective_deg": float("nan"),
        "_failed": True,
    }
    if oei:
        d.update(oei_rpm=float("nan"), oei_thrust=0.0, oei_thrust_margin=0.0,
                 oei_stall_ratio=10.0, oei_torque_margin=10.0,
                 oei_power=1e12, oei_electrical_power=1e12)
    return d


def evaluate(case, x, *, oei=False, with_acoustics=False):
    try:
        return _evaluate(case, x, oei=oei, with_acoustics=with_acoustics)
    except Exception as exc:                     # noqa: BLE001  -- any solver failure
        d = _failed_result(case, oei)
        d["_error"] = f"{type(exc).__name__}: {exc}"
        return d


def _evaluate(case, x, *, oei=False, with_acoustics=False):
    """One forward BEM eval.

    `x` -- dict with keys: chord_cps_m (n_chord_cps,), twist_cps_deg
    (n_twist_cps,), hover_rpm (), cruise_rpm (), collective_deg () [shared
    rigid offset; FPP reuses it for cruise + OEI, VPP would need a second key
    but every current case is FPP], and oei_rpm () when `oei`.

    Returns a flat dict of floats -- power (shaft + electrical), thrust, the
    thrust-overshoot fractions, figure of merit / efficiency, per-condition
    peak stall ratio, taper, twist washout (deg), motor torque margins, and
    (when `oei`) the OEI thrust margin + its stall ratio + torque margin.
    Never raises on a merely bad design -- clamps like `solve.run()` does.
    """
    spec = _s._spec_from_case(case)
    num_radial, norm_stations = _s._radial_grid(spec)
    hub_frac = spec.hub_radius / spec.radius
    if norm_stations is None:
        _ns = 1.0 / num_radial / 2.0 + np.linspace(0.0, 1.0 - 1.0 / num_radial, num_radial)
    else:
        _ns = np.asarray(norm_stations, dtype=float)
    r_over_R = hub_frac + (1.0 - hub_frac) * _ns

    hover = _s._op(case["operating"]["hover"])
    cruise = _s._op(case["operating"]["cruise"])
    constraints = case["constraints"]
    motor_cfg = case.get("motor") or {}

    _stall_margin = constraints.get("stall_margin")
    clmax_profile = (_s._section_clmax_profile(case, r_over_R)
                     if _stall_margin is not None else None)
    cl_cap = float(_stall_margin) if _stall_margin is not None else float(constraints["cl_max"])

    chord_cp0 = np.asarray(x["chord_cps_m"], dtype=float)
    twist_cp0 = np.deg2rad(np.asarray(x["twist_cps_deg"], dtype=float))
    collective0 = np.deg2rad(float(x["collective_deg"]))

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        chord_cps = csdl.Variable(value=chord_cp0)
        twist_cps = csdl.Variable(value=twist_cp0)
        hover_rpm = csdl.Variable(value=np.array([float(x["hover_rpm"])]))
        cruise_rpm = csdl.Variable(value=np.array([float(x["cruise_rpm"])]))
        collective = csdl.Variable(value=np.array([collective0]))

        param = BsplineParameterization(
            num_radial=spec.num_radial, num_cp=spec.n_chord_cps, order=spec.bspline_order,
            sample_locations=(None if norm_stations is None else _ns))
        chord_profile = param.evaluate_radial_profile(chord_cps)
        twist_profile = param.evaluate_radial_profile(twist_cps)

        # population methods probe far past the operating envelope -- always
        # clamp (SLSQP only clamps for acoustics/OEI).
        airfoil_model = _s._airfoil_model(case, clamp_reynolds=True)

        naz = NUM_AZIMUTHAL_ACOUSTIC if with_acoustics else 1
        if with_acoustics:
            tc = thickness_to_chord_profile(
                case["airfoil"]["section_boundaries_r_over_r"], case["airfoil"]["t_over_c"],
                r_over_R)
        hover_in, hover_out = _s._bem_point(chord_profile, twist_profile + collective, hover_rpm,
                                     spec, hover, airfoil_model, norm_stations=norm_stations,
                                     num_azimuthal=naz,
                                     extra_mesh_fields=(acoustic_mesh_fields(tc)
                                                        if with_acoustics else None))
        cruise_in, cruise_out = _s._bem_point(chord_profile, twist_profile + collective, cruise_rpm,
                                      spec, cruise, airfoil_model, norm_stations=norm_stations,
                                      num_azimuthal=naz,
                                      extra_mesh_fields=(acoustic_mesh_fields(tc)
                                                         if with_acoustics else None))
        oei_out = None
        if oei:
            oei_rpm = csdl.Variable(value=np.array([float(x["oei_rpm"])]))
            _, oei_out = _s._bem_point(chord_profile, twist_profile + collective, oei_rpm,
                                       spec, hover, airfoil_model, norm_stations=norm_stations,
                                       num_azimuthal=1)

        hover_motor = _s.evaluate_case_motor(motor_cfg, hover_rpm, hover_out.total_power,
                                             overload=motor_cfg.get("k_hover", 1.7))
        cruise_motor = _s.evaluate_case_motor(motor_cfg, cruise_rpm, cruise_out.total_power,
                                              overload=motor_cfg.get("k_cruise", 1.0))

        chord_v = np.asarray(chord_profile.value).reshape(-1)
        twist_v = np.asarray(twist_profile.value).reshape(-1)
        taper = float(chord_v[-1] / chord_v[0])
        washout_deg = float(np.rad2deg(twist_v[0] - twist_v[-1]))

        def _mval(m):
            v = m.torque_margin
            return None if v is None else float(np.asarray(v.value).reshape(-1)[0])

        def _ratio(bem_out):
            if clmax_profile is not None:
                return _stall_ratio_value(bem_out, clmax_profile, num_radial)
            return float(np.max(np.asarray(bem_out.sectional_lift_coefficient.value)))

        T_h = float(hover_out.total_thrust.value[0])
        T_c = float(cruise_out.total_thrust.value[0])
        out = {
            "hover_power": float(hover_out.total_power.value[0]),
            "cruise_power": float(cruise_out.total_power.value[0]),
            "hover_electrical_power": float(hover_motor.electrical_power.value[0]),
            "cruise_electrical_power": float(cruise_motor.electrical_power.value[0]),
            "figure_of_merit": float(hover_out.figure_of_merit.value[0]),
            "cruise_efficiency": float(cruise_out.efficiency.value[0]),
            "hover_thrust": T_h,
            "cruise_thrust": T_c,
            "hover_thrust_target": hover.thrust,
            "cruise_thrust_target": cruise.thrust,
            "hover_thrust_overshoot": max(0.0, T_h / hover.thrust - 1.0),
            "cruise_thrust_overshoot": max(0.0, T_c / cruise.thrust - 1.0),
            "hover_stall_ratio": _ratio(hover_out),
            "cruise_stall_ratio": _ratio(cruise_out),
            "cl_cap": cl_cap,
            "taper": taper,
            "twist_washout_deg": washout_deg,
            "hover_torque_margin": _mval(hover_motor),
            "cruise_torque_margin": _mval(cruise_motor),
            "hover_rpm": float(hover_rpm.value[0]),
            "cruise_rpm": float(cruise_rpm.value[0]),
            "collective_deg": float(np.rad2deg(collective0)),
        }
        if with_acoustics:
            acoustics = add_spl_hover_cruise_acoustics(
                hover_in, hover_out, cruise_in, cruise_out,
                n_rotors=case["acoustic"]["n_rotors"], constrain_cruise=False)
            out.update({
                "cruise_noise": float(acoustics.cruise_vehicle_ospl.value[0]),
                "cruise_single_noise": float(acoustics.cruise_single_ospl.value[0]),
                "hover_noise": float(acoustics.hover_ospl.value[0]),
                "cruise_tonal_noise": float(acoustics.cruise.tonal_spl.value[0]),
                "cruise_broadband_noise": float(acoustics.cruise.broadband_spl.value[0]),
            })
        if oei:
            oei_motor = _s.evaluate_case_motor(
                motor_cfg, oei_rpm, oei_out.total_power,
                overload=motor_cfg.get("k_oei", motor_cfg.get("k_hover", 1.7)))
            T_oei = float(oei_out.total_thrust.value[0])
            out.update({
                "oei_rpm": float(oei_rpm.value[0]),
                "oei_thrust": T_oei,
                "oei_thrust_margin": T_oei / hover.thrust,
                "oei_stall_ratio": _ratio(oei_out),
                "oei_torque_margin": _mval(oei_motor),
                "oei_power": float(oei_out.total_power.value[0]),
                "oei_electrical_power": float(oei_motor.electrical_power.value[0]),
            })
        return out
    finally:
        try:
            recorder.stop()
        except Exception:
            pass
