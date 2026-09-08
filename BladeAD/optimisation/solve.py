"""One BEM + acoustic optimisation for a rotor case.

Case-agnostic port (2026-09-06) of
`code/optimize_shahjahan_proprotor/optimize_shahjahan_stage1.py`: the physics,
DV set (5 chord CP + 5 twist CP + per-point RPM + per-point rigid collective
offset), constraints and objective modes are identical; the only change is that
everything is read from a `case` dict instead of module globals, and outputs go
into the case directory.

Objective identity is spec-driven (`case.parse_objectives`). `run()` takes:
  optimize  -- the objective the solver drives directly (default: the `proxy`).
               A `proxy` -> minimise its `proxy_min_key` (a well-conditioned
               stand-in, e.g. cruise_power for cruise efficiency). A `floor`/`cap`
               objective -> maximise / minimise its own `result_key` directly.
  epsilons  -- {objective_name: value} epsilon-constraints on the OTHER non-proxy
               objectives:  `floor` -> result_key >= value,  `cap` -> <= value.
               An acoustic cap is enforced inside the acoustics helper.

The working scalarisation for the coupled FM-eta trade is therefore
`optimize=None` (proxy = min cruise power) s.t. `epsilons={'fm_hover': x}`
(+ optional `hover_noise` cap) -- `reference_coupled_pareto_minpower_direction`.

Run switches are function kwargs (no argparse). `run_case_dir(case_dir, **opts)`
is the subprocess entry the sweep layer calls; `run(case, out_dir, seed_dict,
**opts)` is the in-process form.

Run in the `rotor_design` conda env (BladeAD fork `neuralfoil-csdl-pass1`).
"""
import csv
import functools
import os
import pickle
from dataclasses import dataclass

import numpy as np
import csdl_alpha as csdl

from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters, AtmosStates
from BladeAD.utils.parameterization import BsplineParameterization
from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.airfoil.composite_airfoil_model import CompositeAirfoilModel

from BladeAD.core.airfoil.mach_corrected_bspline_airfoil_model import MachCorrectedBSplineAirfoilModel
from .case import Context, cosine_stations

from .acoustics import (
    add_spl_hover_cruise_acoustics,
    acoustic_mesh_fields,
    thickness_to_chord_profile,
    NUM_AZIMUTHAL_ACOUSTIC,
)

_CRUISE_P_SCALER = 1e-4   # brings ~8 kW cruise power to O(1)
_RPM_TO_RAD = 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class MotorPoint:
    """One design point through the case motor model. `torque_margin` is
    Q / (k * Q_continuous(rpm)) -- feasible when <= 1 -- or None for `placebo`
    / no-motor (which carry no speed-dependent envelope)."""
    electrical_power: object
    efficiency: object
    torque: object
    torque_margin: object


def evaluate_case_motor(motor_cfg, rpm, shaft_power, *, overload):
    """Map a rotor design point (`rpm`, `shaft_power` as csdl scalars) to its
    motor electrical draw + continuous-torque margin.

      placebo            -- fixed efficiency, no envelope (torque_margin None).
      mcdonald / emrax188 -- McDonald loss map for the paper-scaled EMRAX-188
                            (`SHAHJAHAN_EMRAX188_PARAMETERS`) + its
                            continuous-torque envelope; `overload` is the
                            multiplier on the continuous line permitted here
                            (Shahjahan 2024: 1.7 for hover / emergency hover /
                            transition, 1.0 for cruise).
    """
    omega = rpm * _RPM_TO_RAD
    torque = shaft_power / omega
    model = (motor_cfg or {}).get("model")
    if model == "placebo":
        eta = float(motor_cfg.get("efficiency", 0.95))
        return MotorPoint(shaft_power / eta, eta, torque, None)
    if model in ("mcdonald", "emrax188"):
        from BladeAD.core.motor import (
            evaluate_motor,
            SHAHJAHAN_EMRAX188_PARAMETERS,
            SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE,
        )
        # McDonald loss map takes log(torque)/log(omega): floor torque at 1 Nm
        # so an SLSQP probe to a near-zero-power point cannot produce a NaN.
        safe_torque = csdl.maximum(torque, csdl.Variable(value=np.array([1.0])))
        out = evaluate_motor("mcdonald", omega, safe_torque, SHAHJAHAN_EMRAX188_PARAMETERS)
        q_continuous = SHAHJAHAN_EMRAX188_CONTINUOUS_TORQUE.evaluate(rpm)
        return MotorPoint(out.electrical_power, out.efficiency, torque,
                          torque / (float(overload) * q_continuous))
    # no motor configured -> shaft power as a stand-in (the electrical_power
    # objective is guarded by case._validate, so this never feeds an objective).
    return MotorPoint(shaft_power * 1.0, 1.0, torque, None)


@dataclass(frozen=True)
class OperatingPoint:
    altitude: float          # m
    airspeed: float          # m/s
    thrust: float            # N


@dataclass(frozen=True)
class RotorSpec:
    n_blades: int
    radius: float
    hub_radius: float
    n_chord_cps: int
    n_twist_cps: int
    bspline_order: int
    num_radial: int
    radial_distribution: str = "uniform"


def _isa_atmos(altitude_m):
    """ISA troposphere state (valid to 11 km) as a BladeAD AtmosStates."""
    temperature = 288.15 - 0.0065 * altitude_m
    pressure = 101325.0 * (temperature / 288.15) ** 5.25588
    density = pressure / (287.058 * temperature)
    speed_of_sound = np.sqrt(1.4 * 287.058 * temperature)
    viscosity = (1.716e-5 * (temperature / 273.15) ** 1.5
                 * (273.15 + 110.4) / (temperature + 110.4))          # Sutherland
    return AtmosStates(density=density, speed_of_sound=speed_of_sound,
                       temperature=temperature, pressure=pressure,
                       dynamic_viscosity=viscosity)


def _radial_grid(spec):
    if spec.radial_distribution == "uniform":
        return spec.num_radial, None
    if spec.radial_distribution == "cosine":
        return spec.num_radial, cosine_stations(spec.num_radial)
    raise SystemExit(f"unhandled radial_distribution {spec.radial_distribution!r}")


def _spec_from_case(case):
    r = case["rotor"]
    return RotorSpec(
        n_blades=r["n_blades"], radius=r["radius_m"], hub_radius=r["hub_radius_m"],
        n_chord_cps=r["n_chord_cps"], n_twist_cps=r["n_twist_cps"],
        bspline_order=r["bspline_order"], num_radial=r["num_radial"],
        radial_distribution=r.get("radial_distribution", "uniform"))


def _airfoil_model(case, clamp_reynolds):
    af = case["airfoil"]
    models = [MachCorrectedBSplineAirfoilModel(
        table_path=os.path.join(af["table_dir"], f"{name}_neuralfoil_table.csv"),
        t_over_c=tc, clamp_reynolds=clamp_reynolds)
        for name, tc in zip(af["names"], af["t_over_c"])]
    return CompositeAirfoilModel(
        sections=af["section_boundaries_r_over_r"], airfoil_models=models, smoothing=False)


@functools.lru_cache(maxsize=64)
def _airfoil_table_clmax(table_path, ref_reynolds):
    """Pre-stall Cl_max (max CL over alpha in [-5, 20] deg) of one NeuralFoil
    polar table, at the nearest tabulated Reynolds node to `ref_reynolds`,
    floored at 2e5 (below that the thick MH root section hits a
    laminar-separation Cl_max cliff that is not a real rotor design limit)."""
    re_all, al_all, cl_all = [], [], []
    with open(table_path, newline="") as f:
        for row in csv.DictReader(f):
            re_all.append(float(row["reynolds"]))
            al_all.append(float(row["alpha"]))
            cl_all.append(float(row["CL"]))
    re_all, al_all, cl_all = np.array(re_all), np.array(al_all), np.array(cl_all)
    nodes = np.unique(re_all)
    target = float(np.clip(ref_reynolds, max(2e5, nodes.min()), nodes.max()))
    node = nodes[np.argmin(np.abs(nodes - target))]
    m = (re_all == node) & (al_all >= -5.0) & (al_all <= 20.0)
    return float(cl_all[m].max())


def _section_clmax_profile(case, r_over_R):
    """Per-radial-station raw section Cl_max (NOT margin-reduced). Each airfoil's
    Cl_max is read at its `airfoil.clmax_ref_reynolds` (a list, one per airfoil;
    default 7e5 each). How those reference Reynolds numbers were chosen -- and
    when to revisit them for a different rotor/mission -- is in the rotor-
    optimisation project's decisions/2026-09-07-section-clmax-stall-constraint.md."""
    af = case["airfoil"]
    bounds = np.asarray(af["section_boundaries_r_over_r"], dtype=float)
    names = af["names"]
    ref_re = af.get("clmax_ref_reynolds", [7e5] * len(names))
    per_airfoil = [
        _airfoil_table_clmax(os.path.join(af["table_dir"], f"{n}_neuralfoil_table.csv"), float(re))
        for n, re in zip(names, ref_re)]
    r = np.asarray(r_over_R, dtype=float)
    idx = np.clip(np.searchsorted(bounds, r, side="right") - 1, 0, len(names) - 1)
    return np.array([per_airfoil[i] for i in idx])


def _bem_point(chord_profile, twist_profile, rpm_var, spec, op, airfoil_model,
               norm_stations=None, num_azimuthal=1, extra_mesh_fields=None):
    """One BEM design point -> (RotorAnalysisInputs, BEMOutputs). twist_profile
    already carries any pitch offset."""
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.array([0.0, 0.0, 0.0])),
        chord_profile=chord_profile,
        twist_profile=twist_profile,
        radius=csdl.Variable(value=spec.radius),
        num_radial=spec.num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=spec.n_blades,
        norm_hub_radius=spec.hub_radius / spec.radius,
        norm_radial_stations=norm_stations,
        **(extra_mesh_fields or {}),
    )
    inputs = RotorAnalysisInputs(
        rpm=rpm_var,
        mesh_velocity=csdl.Variable(value=np.array([[op.airspeed, 0.0, 0.0]])),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = _isa_atmos(op.altitude)
    out = BEMModel(num_nodes=1, airfoil_model=airfoil_model,
                   integration_scheme="trapezoidal").evaluate(inputs=inputs)
    return inputs, out


def _op(entry, thrust_scale=1.0):
    return OperatingPoint(altitude=entry["altitude_m"], airspeed=entry["airspeed_m_s"],
                          thrust=entry["thrust_n"] * thrust_scale)


# ---------------------------------------------------------------------------

def run_case_dir(case_dir, **opts):
    """Subprocess entry: load the case, resolve the default cold-start seed,
    run, write the result pkl into the case dir, print 'Saved to <path>'."""
    ctx = Context.load(case_dir)
    seed = ctx.case["seed"]
    if isinstance(seed, str):                       # "inverse-design"
        if not opts.get("initial_result_from") and not opts.get("pin_geometry_from"):
            from . import seed as seedmod
            kind = "fpp" if ctx.case["rotor"].get("fixed_pitch") else "hover"
            opts["initial_result_from"] = seedmod.write_inverse_design_seed(ctx, kind)
        seed_dict = None
    else:
        seed_dict = seed
    return run(ctx.case, ctx.case_dir, seed_dict, **opts)


# quantity name (result-dict dotted key or proxy_min_key) -> objective scaler
_OBJ_SCALER = {"cruise_power": _CRUISE_P_SCALER,
               "cruise_electrical_power": _CRUISE_P_SCALER,
               "hover_power": _CRUISE_P_SCALER,
               "hover_electrical_power": _CRUISE_P_SCALER,
               # match the acoustic constraint scaler (0.1) when cruise noise is
               # the directly-optimised objective (~64 dB -> ~6.4).
               "acoustics.cruise_vehicle_ospl_db": 0.1}


def _dotted(d, path):
    cur = d
    for k in path.split("."):
        cur = cur[k]
    return cur


def run(case, out_dir, seed_dict, specs=None, epsilons=None, optimize=None,
        with_acoustics=False, initial_result_from=None, pin_geometry_from=None,
        hover_thrust_scale=1.0):
    """One BEM + acoustic optimisation for the case.

    Objective identity comes from `specs` (`case.parse_objectives`, resolved by
    the caller or here). `optimize` names the objective the solver drives
    directly (default: the `proxy`); `epsilons` = {objective_name: value} are the
    active epsilon-constraints on the OTHER non-proxy objectives for this solve.
    """
    from .case import parse_objectives, epsilon_tag

    specs = specs or parse_objectives(case["objectives"])
    live = [s for s in specs if s.role != "reserved"]
    by_name = {s.name: s for s in live}
    proxy = next(s for s in live if s.role == "proxy")
    epsilons = {k: float(v) for k, v in (epsilons or {}).items()}
    for k in epsilons:
        if k not in by_name:
            raise SystemExit(f"epsilon on {k!r} is not an objective of this case")
        # the proxy may still carry a BOUND while being optimised directly
        # (nobj's free-non-proxy legs use this); a non-proxy is a floor/cap.
    target = optimize or proxy.name
    if target not in by_name:
        raise SystemExit(f"optimize={target!r} is not an objective of this case")
    tspec = by_name[target]

    # an acoustic objective (as target or epsilon, or -- since the sweep always
    # runs with_acoustics -- whenever it is a live objective) needs the acoustic
    # graph built so its value can be read back.
    acoustics_active = (with_acoustics or tspec.is_acoustic
                        or any(by_name[k].is_acoustic for k in epsilons))
    # Acoustic slots are constrained INSIDE add_spl_hover_cruise_acoustics (a
    # csdl acoustic quantity is not in `result_vars` for the generic epsilon
    # loop, which skips `is_acoustic` slots -- see below). Resolve here, before
    # that builder call, what each acoustic knob should be for THIS solve:
    #   - a pinned acoustic epsilon  -> a per-solve cap fed to the builder
    #   - an acoustic slot as the swept target -> drive it via the objective,
    #     do NOT also hard-cap it (constrain_cruise=False)
    #   - neither -> the case-default cap, unchanged (every existing case)
    _CRUISE_OSPL_KEY = "acoustics.cruise_vehicle_ospl_db"
    _HOVER_OSPL_KEY = "acoustics.hover_ospl_db"
    cruise_noise_cap = None
    constrain_cruise_noise = True
    hover_noise_max = None
    if acoustics_active:
        cruise_noise_cap = float(case["acoustic"]["cruise_noise_cap_db"])
        for _n in epsilons:
            _rk = by_name[_n].result_key
            if _rk == _CRUISE_OSPL_KEY:
                cruise_noise_cap = float(epsilons[_n])  # pinned cruise-noise level
            elif _rk == _HOVER_OSPL_KEY:
                hover_noise_max = float(epsilons[_n])   # pinned hover-noise level
        if tspec.result_key == _CRUISE_OSPL_KEY:
            constrain_cruise_noise = False              # swept target drives it
    constraints = case["constraints"]
    sweep = case["sweep"]
    spec = _spec_from_case(case)
    num_radial, norm_stations = _radial_grid(spec)

    hub_frac = spec.hub_radius / spec.radius
    if norm_stations is None:
        _ns = 1.0 / num_radial / 2.0 + np.linspace(0.0, 1.0 - 1.0 / num_radial, num_radial)
    else:
        _ns = np.asarray(norm_stations, dtype=float)
    r_over_R = hub_frac + (1.0 - hub_frac) * _ns

    hover = _op(case["operating"]["hover"], hover_thrust_scale)
    cruise = _op(case["operating"]["cruise"])
    bounds = case["bounds"]

    # fixed-pitch proprotor (Shahjahan FPP): one shared rigid collective for
    # hover, cruise and OEI -- only the per-condition rpm varies. Default off
    # (VPP: independent hover_pitch + cruise_pitch, as before).
    fixed_pitch = bool(case["rotor"].get("fixed_pitch", False))

    # --- resolve the starting DVs ---
    if pin_geometry_from is not None or initial_result_from is not None:
        src = pin_geometry_from if pin_geometry_from is not None else initial_result_from
        with open(src, "rb") as f:
            prev = pickle.load(f)
        chord_cp0 = np.asarray(prev["chord_cps"], dtype=float)
        twist_cp0 = np.asarray(prev["twist_cps"], dtype=float)
        hover_rpm0 = float(prev["rpm"])
        cruise_rpm0 = float(prev.get("cruise_rpm", 0.5 * sum(bounds["cruise_rpm"])))
        _hp_seed = prev.get("pitch_deg", prev.get("hover_pitch_deg", 0.0)) if fixed_pitch \
            else prev.get("hover_pitch_deg", 0.0)
        hover_pitch0 = np.deg2rad(float(_hp_seed))
        cruise_pitch0 = np.deg2rad(float(prev.get("cruise_pitch_deg", 0.0)))
        oei_rpm0 = float(prev.get("oei_rpm") or hover_rpm0)
    else:
        if seed_dict is None:
            raise SystemExit("no seed: case['seed'] is 'inverse-design' but no "
                             "initial_result_from was resolved")
        chord_cp0 = np.array(seed_dict["chord_cps_m"], dtype=float)
        twist_cp0 = np.deg2rad(np.array(seed_dict["twist_cps_deg"], dtype=float))
        hover_rpm0 = seed_dict["hover_rpm"]
        cruise_rpm0 = seed_dict["cruise_rpm"]
        _hp_seed = seed_dict.get("pitch_deg", seed_dict["hover_pitch_deg"]) if fixed_pitch \
            else seed_dict["hover_pitch_deg"]
        hover_pitch0 = np.deg2rad(_hp_seed)
        cruise_pitch0 = np.deg2rad(seed_dict["cruise_pitch_deg"])
        oei_rpm0 = seed_dict.get("oei_rpm", seed_dict["hover_rpm"])

    pinned = pin_geometry_from is not None
    # OEI (one-engine-inoperative) hover thrust-margin slot -- Shahjahan 2024
    # "emergency hover": one motor out, the remaining rotors must still hover.
    # margin = T_available_at_OEI / T_hover_per_rotor (denominator = the hover
    # thrust target). Emergency hover is a V=0 condition sharing nominal hover's
    # collective (Shahjahan Table 2: collective is a single hover-vs-cruise
    # offset), so `oei_rpm` is the ONLY OEI design variable -- the blade is the
    # hover blade, spun up. Capped by the x1.7 motor torque envelope and cl_max;
    # no thrust equality (thrust IS the objective).
    oei_active = any(s.result_key == "oei_thrust_margin" for s in live)
    oei_max_cl = None
    oei_thrust = oei_torque = oei_torque_margin = oei_motor_eff = oei_thrust_margin = None

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    chord_cps = csdl.Variable(value=chord_cp0)
    twist_cps = csdl.Variable(value=twist_cp0)
    hover_rpm = csdl.Variable(value=np.array([hover_rpm0]))
    cruise_rpm = csdl.Variable(value=np.array([cruise_rpm0]))
    hover_pitch = csdl.Variable(value=np.array([hover_pitch0]))
    # FPP: cruise (and OEI) reuse the same collective variable as hover.
    cruise_pitch = hover_pitch if fixed_pitch else csdl.Variable(value=np.array([cruise_pitch0]))
    _pitch_b = bounds["pitch_deg"] if fixed_pitch else bounds["hover_pitch_deg"]

    if not pinned:
        chord_cps.set_as_design_variable(lower=bounds["chord_m"][0], upper=bounds["chord_m"][1], scaler=5.0)
        twist_cps.set_as_design_variable(lower=np.deg2rad(bounds["twist_deg"][0]),
                                         upper=np.deg2rad(bounds["twist_deg"][1]), scaler=2.0)
        hover_rpm.set_as_design_variable(lower=bounds["hover_rpm"][0], upper=bounds["hover_rpm"][1], scaler=1e-3)
        hover_pitch.set_as_design_variable(lower=np.deg2rad(_pitch_b[0]),
                                           upper=np.deg2rad(_pitch_b[1]), scaler=2.0)
    cruise_rpm.set_as_design_variable(lower=bounds["cruise_rpm"][0], upper=bounds["cruise_rpm"][1], scaler=1e-3)
    if not fixed_pitch:
        cruise_pitch.set_as_design_variable(lower=np.deg2rad(bounds["cruise_pitch_deg"][0]),
                                            upper=np.deg2rad(bounds["cruise_pitch_deg"][1]), scaler=2.0)

    if oei_active:
        _oei_rpm_b = bounds.get("oei_rpm", bounds["hover_rpm"])
        oei_rpm = csdl.Variable(value=np.array([oei_rpm0]))
        oei_rpm.set_as_design_variable(lower=_oei_rpm_b[0], upper=_oei_rpm_b[1], scaler=1e-3)

    param = BsplineParameterization(num_radial=spec.num_radial, num_cp=spec.n_chord_cps,
                                    order=spec.bspline_order,
                                    sample_locations=(None if norm_stations is None else _ns))
    chord_profile = param.evaluate_radial_profile(chord_cps)
    twist_profile = param.evaluate_radial_profile(twist_cps)

    # an OEI thrust-maximising point probes the airfoil-table Re edge during the
    # search (root sections can fall below the table's Re floor) -- clamp rather
    # than crash (feedback_bem_surrogate_table_domain_margins).
    airfoil_model = _airfoil_model(case, clamp_reynolds=(acoustics_active or oei_active))
    naz = NUM_AZIMUTHAL_ACOUSTIC if acoustics_active else 1
    if acoustics_active:
        tc_profile = thickness_to_chord_profile(
            case["airfoil"]["section_boundaries_r_over_r"], case["airfoil"]["t_over_c"], r_over_R)
        hover_extra, cruise_extra = acoustic_mesh_fields(tc_profile), acoustic_mesh_fields(tc_profile)
    else:
        hover_extra = cruise_extra = None

    from ._quiet import muffled as _muffled, QUIET_SOLVER as _QS
    with _muffled(_QS):
        hover_in, hover_out = _bem_point(chord_profile, twist_profile + hover_pitch, hover_rpm, spec,
                                         hover, airfoil_model, norm_stations=norm_stations,
                                         num_azimuthal=naz, extra_mesh_fields=hover_extra)
        cruise_in, cruise_out = _bem_point(chord_profile, twist_profile + cruise_pitch, cruise_rpm, spec,
                                           cruise, airfoil_model, norm_stations=norm_stations,
                                           num_azimuthal=naz, extra_mesh_fields=cruise_extra)

        oei_out = None
        if oei_active:
            # emergency hover = the hover blade (same collective) spun to oei_rpm
            oei_op = OperatingPoint(altitude=hover.altitude, airspeed=hover.airspeed, thrust=hover.thrust)
            _, oei_out = _bem_point(chord_profile, twist_profile + hover_pitch, oei_rpm, spec,
                                    oei_op, airfoil_model, norm_stations=norm_stations,
                                    num_azimuthal=1, extra_mesh_fields=None)

    acoustic_bundle = None
    if acoustics_active:
        acoustic_bundle = add_spl_hover_cruise_acoustics(
            hover_in, hover_out, cruise_in, cruise_out,
            n_rotors=case["acoustic"]["n_rotors"],
            cruise_cap_db=cruise_noise_cap,
            hover_noise_max_db=hover_noise_max,
            constrain_cruise=constrain_cruise_noise)

    taper = chord_profile[-1] / chord_profile[0]
    twist_washout = twist_profile[0] - twist_profile[-1]
    hover_max_cl = csdl.maximum(hover_out.sectional_lift_coefficient)     # raw, for reporting
    cruise_max_cl = csdl.maximum(cruise_out.sectional_lift_coefficient)

    # Stall constraint: either a flat `constraints.cl_max` cap on the peak
    # sectional Cl (legacy, still the default), or -- when `constraints.stall_margin`
    # is set -- a radius-dependent one: max_i(cl_i / section_Cl_max_i) <= stall_margin,
    # with section Cl_max from each airfoil's polar table (`_section_clmax_profile`).
    _stall_margin = constraints.get("stall_margin")
    if _stall_margin is None:
        hover_cl_con, cruise_cl_con = hover_max_cl, cruise_max_cl
        _cl_upper = float(constraints["cl_max"])
        _clmax_profile = None
    else:
        _clmax_profile = _section_clmax_profile(case, r_over_R)

        def _stall_ratio(bem_out):
            cl = bem_out.sectional_lift_coefficient
            n = int(np.asarray(cl.value).size)
            k = n // num_radial
            prof = (np.repeat(_clmax_profile, k) if k >= 1 and k * num_radial == n
                    else np.full(n, float(_clmax_profile.min())))
            # cl is (num_nodes, num_radial, num_azimuthal); np.repeat's C-order
            # layout is already row-major (num_radial, num_azimuthal), so a plain
            # reshape to cl.shape lines the section Cl_max up per station.
            return csdl.maximum(cl / csdl.Variable(value=prof.reshape(cl.shape)))

        hover_cl_con, cruise_cl_con = _stall_ratio(hover_out), _stall_ratio(cruise_out)
        _cl_upper = float(_stall_margin)

    if not pinned:
        hover_out.total_thrust.set_as_constraint(equals=hover.thrust, scaler=1e-3)
        taper.set_as_constraint(lower=constraints["taper"][0], upper=constraints["taper"][1], scaler=1.0)
        twist_washout.set_as_constraint(lower=0.0, scaler=1.0)
        hover_cl_con.set_as_constraint(upper=_cl_upper, scaler=1.0)
    cruise_out.total_thrust.set_as_constraint(equals=cruise.thrust, scaler=1e-3)
    cruise_cl_con.set_as_constraint(upper=_cl_upper, scaler=1.0)

    # --- motor: shaft torque/speed -> electrical input power ---
    # step 1a is a PLACEBO (fixed efficiency); step 1b is the real McDonald
    # loss map for the paper-scaled EMRAX-188 (`mcdonald` / alias `emrax188`),
    # plus its speed-dependent continuous-torque envelope enforced as a
    # constraint (Q <= k * Q_continuous(rpm), k = 1.7 hover / 1.0 cruise --
    # Shahjahan 2024 raises the continuous line to 170% for hover/transition).
    motor_cfg = case.get("motor") or {}
    hover_motor = evaluate_case_motor(motor_cfg, hover_rpm, hover_out.total_power,
                                      overload=motor_cfg.get("k_hover", 1.7))
    cruise_motor = evaluate_case_motor(motor_cfg, cruise_rpm, cruise_out.total_power,
                                       overload=motor_cfg.get("k_cruise", 1.0))
    hover_torque, cruise_torque = hover_motor.torque, cruise_motor.torque
    hover_motor_eff, cruise_motor_eff = hover_motor.efficiency, cruise_motor.efficiency
    hover_electrical_power = hover_motor.electrical_power
    cruise_electrical_power = cruise_motor.electrical_power
    hover_torque_margin, cruise_torque_margin = hover_motor.torque_margin, cruise_motor.torque_margin
    if cruise_motor_eff is None or isinstance(cruise_motor_eff, float):
        eta_motor = float(cruise_motor_eff) if cruise_motor_eff is not None else 1.0
    else:
        eta_motor = float(cruise_motor_eff.value[0])   # representative scalar (back-compat field)
    if hover_torque_margin is not None and not pinned:
        hover_torque_margin.set_as_constraint(upper=1.0, scaler=1.0)
    if cruise_torque_margin is not None:
        cruise_torque_margin.set_as_constraint(upper=1.0, scaler=1.0)

    if oei_active:
        oei_motor = evaluate_case_motor(motor_cfg, oei_rpm, oei_out.total_power,
                                        overload=motor_cfg.get("k_oei", motor_cfg.get("k_hover", 1.7)))
        oei_torque, oei_motor_eff = oei_motor.torque, oei_motor.efficiency
        oei_torque_margin = oei_motor.torque_margin
        oei_thrust = oei_out.total_thrust
        oei_thrust_margin = oei_thrust / hover.thrust        # denominator = hover thrust target
        oei_max_cl = csdl.maximum(oei_out.sectional_lift_coefficient)     # raw, for reporting
        oei_cl_con = oei_max_cl if _stall_margin is None else _stall_ratio(oei_out)
        oei_cl_con.set_as_constraint(upper=_cl_upper, scaler=1.0)
        if oei_torque_margin is not None:
            oei_torque_margin.set_as_constraint(upper=1.0, scaler=1.0)
        _min_oei_margin = constraints.get("min_oei_thrust_margin")
        if _min_oei_margin is not None:
            oei_thrust_margin.set_as_constraint(lower=float(_min_oei_margin), scaler=1.0)

    # the physical result quantities each objective slot can name
    result_vars = {
        "figure_of_merit": hover_out.figure_of_merit,
        "cruise_efficiency": cruise_out.efficiency,
        "cruise_power": cruise_out.total_power,
        "hover_power": hover_out.total_power,
        "cruise_electrical_power": cruise_electrical_power,
        "hover_electrical_power": hover_electrical_power,
    }
    if oei_thrust_margin is not None:
        result_vars["oei_thrust_margin"] = oei_thrust_margin
    if acoustic_bundle is not None:
        result_vars["acoustics.hover_ospl_db"] = acoustic_bundle.hover_ospl
        result_vars["acoustics.cruise_vehicle_ospl_db"] = acoustic_bundle.cruise_vehicle_ospl

    # --- the objective the solver drives directly ---
    if tspec.role == "proxy":
        qkey = tspec.proxy_min_key
        # a distinct stand-in quantity (e.g. cruise_power for eta) is always
        # MINIMISED; if the proxy minimises its own reported quantity, use goal.
        sign = 1.0 if qkey != tspec.result_key else (-1.0 if tspec.goal == "max" else 1.0)
    else:
        qkey = tspec.result_key
        sign = -1.0 if tspec.goal == "max" else 1.0
    if qkey not in result_vars:
        raise SystemExit(f"objective {target!r} names result quantity {qkey!r}, "
                         f"which solve.py does not compute (reserved / not built)")
    objective = sign * result_vars[qkey] * _OBJ_SCALER.get(qkey, 1.0)
    objective.set_as_objective(scaler=1.0)

    # --- epsilon-constraints on the other non-proxy objectives ---
    # (an acoustic cap is enforced inside add_spl_hover_cruise_acoustics via the
    #  hover_noise_max_db argument above -- do not double-constrain it here.)
    for name, val in epsilons.items():
        s = by_name[name]
        if s.is_acoustic:
            continue
        var = result_vars[s.result_key]
        low_side = s.role == "floor" or (s.role == "proxy" and s.goal == "max")
        if low_side:
            kw = {"lower": val}
            if s.result_key == "figure_of_merit":
                kw["upper"] = 1.0                 # FM in [floor, 1] -- exact prior behaviour
            var.set_as_constraint(scaler=1.0, **kw)
        else:                                     # cap, or a min-goal proxy bound
            var.set_as_constraint(upper=val, scaler=1.0)

    print(f"\nBefore: FM={hover_out.figure_of_merit.value[0]:.4f}  "
          f"eta={cruise_out.efficiency.value[0]:.4f}  "
          f"hover_thrust={hover_out.total_thrust.value[0]:.1f}  "
          f"cruise_thrust={cruise_out.total_thrust.value[0]:.1f}  "
          f"taper={taper.value[0]:.4f}  hover_max_cl={hover_max_cl.value[0]:.4f}")

    import modopt
    from modopt import CSDLAlphaProblem
    from ._quiet import muffled, QUIET_SOLVER
    # PySimulator construction re-runs the graph once, and SLSQP re-evaluates
    # the BEM every iteration -> the `bracketed_search converged` / csdl
    # non-convergence dump lines. Silence them by default (flip
    # `_quiet.QUIET_SOLVER` to see them). The constraint verification +
    # "Saved to" report below stays visible.
    with muffled(QUIET_SOLVER):
        sim = csdl.experimental.PySimulator(recorder=recorder)
        prob = CSDLAlphaProblem(problem_name="rotor_pareto_solve", simulator=sim)
        modopt.SLSQP(prob, solver_options={"maxiter": sweep["maxiter"],
                                           "ftol": sweep.get("ftol", 1e-6)}).solve()

    fm = float(hover_out.figure_of_merit.value[0])
    eta = float(cruise_out.efficiency.value[0])

    def _tip_scalars(bem_out):
        nr = num_radial

        def _rad(v):
            a = np.asarray(v)
            return a.reshape(nr, -1).mean(axis=1) if a.size != nr else a.reshape(-1)

        dT = _rad(bem_out.sectional_thrust.value)
        dQ = _rad(bem_out.sectional_torque.value)
        dr_raw = np.asarray(bem_out.radial_element_width.value)
        dr = np.full(nr, float(dr_raw)) if dr_raw.size == 1 else _rad(dr_raw)
        dQ_dr = dQ / dr
        k = int(np.argmax(dQ_dr))
        tip = r_over_R >= 0.9
        re = _rad(bem_out.sectional_reynolds_number.value)
        aoa = np.rad2deg(_rad(bem_out.sectional_angle_of_attack.value))
        return {
            "T_tip_over_T_total": float(dT[tip].sum() / dT.sum()),
            "peak_dQ_dr": float(dQ_dr[k]),
            "peak_dQ_dr_r_over_R": float(r_over_R[k]),
            "tip_reynolds_min": float(re[tip].min()), "tip_reynolds_max": float(re[tip].max()),
            "tip_aoa_deg_min": float(aoa[tip].min()), "tip_aoa_deg_max": float(aoa[tip].max()),
        }
    hover_tip = _tip_scalars(hover_out)
    cruise_tip = _tip_scalars(cruise_out)

    tol = constraints["thrust_tolerance_n"]
    checks = {
        "hover thrust matches target": abs(hover_out.total_thrust.value[0] - hover.thrust) < tol,
        "cruise thrust matches target": abs(cruise_out.total_thrust.value[0] - cruise.thrust) < tol,
        "taper in bounds": constraints["taper"][0] - 1e-3 <= taper.value[0] <= constraints["taper"][1] + 1e-3,
        "twist washout >= 0": twist_washout.value[0] >= -1e-4,
        "hover max_cl <= limit": float(hover_cl_con.value[0]) <= _cl_upper + 1e-3,
        "cruise max_cl <= limit": float(cruise_cl_con.value[0]) <= _cl_upper + 1e-3,
    }
    if hover_torque_margin is not None:
        checks["hover motor torque within envelope"] = float(hover_torque_margin.value[0]) <= 1.0 + 1e-3
        checks["cruise motor torque within envelope"] = float(cruise_torque_margin.value[0]) <= 1.0 + 1e-3
    if oei_thrust_margin is not None:
        checks["oei max_cl <= limit"] = float(oei_cl_con.value[0]) <= _cl_upper + 1e-3
        if oei_torque_margin is not None:
            checks["oei motor torque within envelope"] = float(oei_torque_margin.value[0]) <= 1.0 + 1e-3
        _mom = constraints.get("min_oei_thrust_margin")
        if _mom is not None:
            checks["oei thrust margin >= floor"] = float(oei_thrust_margin.value[0]) >= float(_mom) - 1e-4
    acoustics = None
    if acoustic_bundle is not None:
        def _s(v):
            return None if v is None else float(np.asarray(v.value).reshape(-1)[0])
        # the cap actually applied this solve (a pinned acoustic epsilon or the
        # swept-target case both move it off the static case default)
        cap = cruise_noise_cap
        acoustics = {
            "n_rotors": case["acoustic"]["n_rotors"],
            "hover_ospl_db": _s(acoustic_bundle.hover_ospl),
            "hover_ospl_a_weighted_db": _s(acoustic_bundle.hover_ospl_a_weighted),
            "hover_tonal_spl_db": _s(acoustic_bundle.hover.tonal_spl),
            "hover_broadband_spl_db": _s(acoustic_bundle.hover.broadband_spl),
            "cruise_single_ospl_db": _s(acoustic_bundle.cruise_single_ospl),
            "cruise_vehicle_ospl_db": _s(acoustic_bundle.cruise_vehicle_ospl),
            "cruise_tonal_spl_db": _s(acoustic_bundle.cruise.tonal_spl),
            "cruise_broadband_spl_db": _s(acoustic_bundle.cruise.broadband_spl),
            "cruise_noise_cap_db": cap,
            "hover_noise_max_db": hover_noise_max,
        }
        if constrain_cruise_noise:
            checks["cruise vehicle noise <= cap"] = acoustics["cruise_vehicle_ospl_db"] <= cap + 1e-3
        _cap_txt = f"cap {cap}" if constrain_cruise_noise else "no cap -- swept objective"
        print(f"\nNoise: hover OSPL {acoustics['hover_ospl_db']:.2f} dB "
              f"({acoustics['hover_ospl_a_weighted_db']:.2f} dBA)  "
              f"cruise vehicle {acoustics['cruise_vehicle_ospl_db']:.2f} dB ({_cap_txt})")

    # --- per-slot objective values + generic epsilon-constraint checks ---
    _reported = {
        "figure_of_merit": fm, "cruise_efficiency": eta,
        "cruise_power": float(cruise_out.total_power.value[0]),
        "hover_power": float(hover_out.total_power.value[0]),
        "cruise_electrical_power": float(cruise_electrical_power.value[0]),
        "hover_electrical_power": float(hover_electrical_power.value[0]),
        "oei_thrust_margin": (float(oei_thrust_margin.value[0])
                              if oei_thrust_margin is not None else float("nan")),
        "acoustics": acoustics,
    }
    objective_values = {}
    for s in live:
        try:
            objective_values[s.name] = float(_dotted(_reported, s.result_key))
        except (KeyError, TypeError):
            objective_values[s.name] = float("nan")
    for name, val in epsilons.items():
        s, got = by_name[name], objective_values[name]
        if s.role == "floor" or (s.role == "proxy" and s.goal == "max"):
            checks[f"{name} >= epsilon"] = got >= val - 1e-4
        else:
            checks[f"{name} <= epsilon"] = got <= val + 1e-3

    print("\nConstraint verification:")
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    result = {
        "objective_mode": f"optimize:{target}",
        "optimize": target, "epsilons": epsilons, "objective_values": objective_values,
        "hover_thrust_scale": hover_thrust_scale, "hover_thrust_target": hover.thrust,
        "pin_geometry_from": pin_geometry_from, "initial_result_from": initial_result_from,
        "chord_cps": np.asarray(chord_cps.value), "twist_cps": np.asarray(twist_cps.value),
        "chord_profile": np.asarray(chord_profile.value), "twist_profile": np.asarray(twist_profile.value),
        "radial_distribution": spec.radial_distribution, "num_radial": num_radial,
        "radial_stations_r_over_R": np.asarray(r_over_R),
        "hover_tip_scalars": hover_tip, "cruise_tip_scalars": cruise_tip,
        "rpm": float(hover_rpm.value[0]),
        "cruise_rpm": float(cruise_rpm.value[0]),
        "hover_pitch_deg": float(np.rad2deg(hover_pitch.value[0])),
        "cruise_pitch_deg": float(np.rad2deg(cruise_pitch.value[0])),
        "fixed_pitch": fixed_pitch,
        "hover_thrust": float(hover_out.total_thrust.value[0]),
        "cruise_thrust": float(cruise_out.total_thrust.value[0]),
        "figure_of_merit": fm,
        "cruise_efficiency": eta,
        "hover_power": float(hover_out.total_power.value[0]),
        "cruise_power": float(cruise_out.total_power.value[0]),
        "motor_model": motor_cfg.get("model"),
        "motor_efficiency": eta_motor,
        "hover_motor_efficiency": (float(hover_motor_eff.value[0])
                                   if hasattr(hover_motor_eff, "value") else hover_motor_eff),
        "cruise_motor_efficiency": (float(cruise_motor_eff.value[0])
                                    if hasattr(cruise_motor_eff, "value") else cruise_motor_eff),
        "hover_electrical_power": float(hover_electrical_power.value[0]),
        "cruise_electrical_power": float(cruise_electrical_power.value[0]),
        "hover_torque_nm": float(hover_torque.value[0]),
        "cruise_torque_nm": float(cruise_torque.value[0]),
        "hover_torque_margin": (float(hover_torque_margin.value[0])
                                if hover_torque_margin is not None else None),
        "cruise_torque_margin": (float(cruise_torque_margin.value[0])
                                 if cruise_torque_margin is not None else None),
        "oei_rpm": (float(oei_rpm.value[0]) if oei_active else None),
        "oei_pitch_deg": (float(np.rad2deg(hover_pitch.value[0])) if oei_active else None),  # = hover collective
        "oei_thrust": (float(oei_thrust.value[0]) if oei_thrust is not None else None),
        "oei_thrust_margin": (float(oei_thrust_margin.value[0])
                              if oei_thrust_margin is not None else None),
        "oei_torque_nm": (float(oei_torque.value[0]) if oei_torque is not None else None),
        "oei_torque_margin": (float(oei_torque_margin.value[0])
                              if oei_torque_margin is not None else None),
        "oei_motor_efficiency": (float(oei_motor_eff.value[0])
                                 if hasattr(oei_motor_eff, "value") else oei_motor_eff),
        "taper": float(taper.value[0]),
        "twist_washout_deg": float(np.rad2deg(twist_washout.value[0])),
        "hover_max_sectional_cl": float(hover_max_cl.value[0]),
        "cruise_max_sectional_cl": float(cruise_max_cl.value[0]),
        "stall_margin": (float(_stall_margin) if _stall_margin is not None else None),
        "section_clmax_profile": (list(map(float, _clmax_profile)) if _clmax_profile is not None else None),
        "hover_stall_ratio": float(hover_cl_con.value[0]),
        "cruise_stall_ratio": float(cruise_cl_con.value[0]),
        "acoustics_active": acoustics_active,
        "hover_noise_max": hover_noise_max,
        "acoustics": acoustics,
        "constraint_checks": checks,
    }

    mode_tag = f"opt-{target}"
    if spec.radial_distribution != "uniform":
        mode_tag += f"_grid-{spec.radial_distribution}{num_radial}"
    if acoustics_active and not tspec.is_acoustic:
        mode_tag += "_ac"
    if hover_thrust_scale != 1.0:
        mode_tag += f"_tscale-{hover_thrust_scale:.2f}"
    mode_tag += epsilon_tag(epsilons)
    if pinned:
        mode_tag += "_pinned"

    out_path = os.path.join(out_dir, f"stage1_{mode_tag}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nFM={fm:.4f}  eta_cruise={eta:.4f}  hover_power={result['hover_power']:.1f} W  "
          f"cruise_power={result['cruise_power']:.1f} W")
    print(f"Saved to {out_path}")
    return result
