"""CSDL-alpha adapter: drive `optimisation_framework`'s generic Pareto-front
tracer with `solve.run` as the corrector, for ONE pairwise objective edge.

Workflow (per 2-objective subproblem):

  1. NSGA-II on the pair  -> a feasible BASIN geometry (external, done already).
  2. seeded single-objective extremes at BOTH ends of the edge
     (`sweep_common.seeded_anchor`, warm-started from the NSGA basin) -> the eps
     range + the start anchor.
  3. THIS module -- predictor--corrector continuation between the two endpoints,
     replacing the fixed-linspace / adaptive-midpoint stepping of `edges.py`.

The generic loop lives in `optimisation_framework.gradient.pareto_tracer`; this
module supplies:

  * `corrector(x, eps)` -- one `min_power_solve(ctx, {sweep: eps}, warm=<seed>)`
    (a subprocess `solve.run_case_dir`, process isolation -- `solve.run` leaves a
    live `csdl.Recorder`), warm-started from a seed pkl written from the
    predicted design vector `x`.
  * `_pack` / `_unpack` -- the flat design vector the predictor extrapolates,
    <-> the `initial_result_from` pkl keys `solve.run` reads.

The generic tracer parametrises the front by `eps` on objective 1 and minimises
objective 0. Here objective 0 = the case PROXY, objective 1 = the SWEEP slot (the
non-proxy member of the pair). `predictor="bfgs"` needs a `jac(x)` callback
(CSDL-alpha reverse-mode) -- NOT YET IMPLEMENTED (Step 2b).

No argparse (workspace style): a `tracer_run.py` in each case dir builds a
`TraceConfig` and calls `trace(cfg)`.
"""
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np

from optimisation_framework.gradient.pareto_tracer import CorrectorResult, trace_front

from .case import Context
from .sweep_common import (
    min_power_solve, nsga_geometry_to_seed, objective_value, proxy_spec,
    seeded_anchor, spec_by_name,
)

_RESULT_KEY = {"cruise_elec": "cruise_electrical_power",
               "hover_elec": "hover_electrical_power",
               "fm_hover": "figure_of_merit", "eta_cruise": "cruise_efficiency",
               "oei_margin": "oei_thrust_margin"}


@dataclass
class TraceConfig:
    case_dir: str
    name_a: str                 # the objective pair (order = plot axes only)
    name_b: str
    nsga_seed: str              # path to an NSGA-II result / history pkl (basin geometry)
    n_points: int = 14
    predictor: str = "secant"   # "secant" | "none" | "bfgs" (bfgs -> jac, Step 2b)
    tag: str = "run1"
    fixed_epsilons: dict | None = None   # extra {obj_name: value} held on every solve
                                         # (a 3rd objective pinned -> an interior LAYER)
    with_acoustics: bool = False         # True computes the acoustic graph -- required
                                         # when an objective is a noise slot; ~15x slower


def _pack(data, *, fixed_pitch):
    x = [*np.asarray(data["chord_cps"], float).ravel(),
         *np.asarray(data["twist_cps"], float).ravel(),
         float(data["rpm"]), float(data["cruise_rpm"])]
    if fixed_pitch:
        x.append(float(data.get("pitch_deg", data.get("hover_pitch_deg"))))
    else:
        x += [float(data["hover_pitch_deg"]), float(data["cruise_pitch_deg"])]
    if data.get("oei_rpm") is not None:
        x.append(float(data["oei_rpm"]))
    return np.array(x, float)


def _unpack(x, template, *, fixed_pitch):
    d = dict(template)
    n_chord = len(np.asarray(template["chord_cps"]).ravel())
    n_twist = len(np.asarray(template["twist_cps"]).ravel())
    i = 0
    d["chord_cps"] = np.asarray(x[i:i + n_chord], float); i += n_chord
    d["twist_cps"] = np.asarray(x[i:i + n_twist], float); i += n_twist
    d["rpm"] = float(x[i]); i += 1
    d["cruise_rpm"] = float(x[i]); i += 1
    if fixed_pitch:
        d["pitch_deg"] = d["hover_pitch_deg"] = float(x[i]); i += 1
    else:
        d["hover_pitch_deg"] = float(x[i]); i += 1
        d["cruise_pitch_deg"] = float(x[i]); i += 1
    if template.get("oei_rpm") is not None:
        d["oei_rpm"] = float(x[i]); i += 1
    return d


def _make_corrector(ctx, proxy_name, sweep_name, seed_template, *, fixed_pitch, tag,
                    fixed_epsilons=None, with_acoustics=False):
    seed_path = ctx.out(f"_tracer_seed_{tag}.pkl")
    fixed_epsilons = dict(fixed_epsilons or {})

    def corrector(x, eps):
        with open(seed_path, "wb") as f:
            pickle.dump(_unpack(np.asarray(x, float), seed_template, fixed_pitch=fixed_pitch), f)
        t0 = time.time()
        try:
            _pkl, data = min_power_solve(ctx, {sweep_name: float(eps), **fixed_epsilons},
                                         warm=seed_path, with_acoustics=with_acoustics)
        except RuntimeError as exc:
            print(f"  eps={eps:10.4g}  FAILED ({time.time() - t0:.0f}s): {exc}", flush=True)
            return CorrectorResult(x=np.asarray(x, float), f=np.array([np.nan, np.nan]),
                                   success=False, message=str(exc).splitlines()[0])
        f = np.array([objective_value(data, proxy_name), objective_value(data, sweep_name)])
        print(f"  eps={eps:10.4g}  {proxy_name}={f[0]:10.4g}  {sweep_name}={f[1]:10.4g}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        return CorrectorResult(x=_pack(data, fixed_pitch=fixed_pitch), f=f, success=True,
                               message="ok")

    return corrector


def _jac_not_implemented(x):
    raise NotImplementedError(
        "predictor='bfgs' needs a CSDL-alpha reverse-mode jac(x) -> (df, dg). "
        "That is tracer Step 2b -- use predictor='secant'.")


def trace(cfg: TraceConfig):
    ctx = Context.load(cfg.case_dir)
    fixed_pitch = bool(ctx.case["rotor"].get("fixed_pitch"))
    proxy = proxy_spec(ctx)
    pair = (cfg.name_a, cfg.name_b)
    if proxy.name not in pair:
        raise SystemExit(f"one of {pair} must be the case proxy ({proxy.name!r})")
    sweep = next(n for n in pair if n != proxy.name)
    print(f"edge {cfg.name_a} x {cfg.name_b}: proxy={proxy.name}  sweep={sweep} "
          f"({spec_by_name(ctx, sweep).role})  n_points={cfg.n_points}  predictor={cfg.predictor}",
          flush=True)

    seed_pkl = nsga_geometry_to_seed(
        pickle.load(open(cfg.nsga_seed, "rb")),
        ctx.out(f"_tracer_nsga_seed_{cfg.tag}.pkl"), fixed_pitch=fixed_pitch)

    fixed_eps = dict(cfg.fixed_epsilons or {})
    if fixed_eps:
        print(f"fixed epsilons (interior layer): {fixed_eps}", flush=True)
    acoustics = bool(getattr(cfg, "with_acoustics", False))
    print("anchor A: sweep-slot single-objective extreme (COLD -- thrust eq. active) ...", flush=True)
    sw_pkl, sw_data = seeded_anchor(ctx, sweep, seed_pkl, epsilons=fixed_eps, with_acoustics=acoustics)
    print("anchor B: proxy single-objective extreme (seeded from NSGA basin) ...", flush=True)
    ot_pkl, ot_data = seeded_anchor(ctx, proxy.name, seed_pkl, epsilons=fixed_eps, with_acoustics=acoustics)

    eps_start = objective_value(sw_data, sweep)     # sweep at its own extreme
    eps_end = objective_value(ot_data, sweep)       # sweep at the proxy's extreme
    f_start = np.array([objective_value(sw_data, proxy.name), eps_start])
    f_end = np.array([objective_value(ot_data, proxy.name), eps_end])
    print(f"\nendpoints: {sweep} {eps_start:.4g} -> {eps_end:.4g}   "
          f"{proxy.name} {f_start[0]:.4g} -> {f_end[0]:.4g}\n", flush=True)

    corrector = _make_corrector(ctx, proxy.name, sweep, sw_data,
                                fixed_pitch=fixed_pitch, tag=cfg.tag, fixed_epsilons=fixed_eps,
                                with_acoustics=acoustics)
    t0 = time.time()
    res = trace_front(
        corrector, x0=_pack(sw_data, fixed_pitch=fixed_pitch),
        f_start=f_start, f_end=f_end,
        jac=_jac_not_implemented if cfg.predictor == "bfgs" else None,
        n_points=cfg.n_points, predictor=cfg.predictor,
    )
    print(f"\ntraced {len(res.points)} points in {time.time() - t0:.0f}s"
          + (f"   messages: {res.messages}" if res.messages else ""))

    # edges.py-format front pkl, for compare_nsga2_vs_epsilon / plotting
    front = []
    for p in res.points:
        row = {cfg.name_a: float(p.f[0] if cfg.name_a == proxy.name else p.f[1]),
               cfg.name_b: float(p.f[0] if cfg.name_b == proxy.name else p.f[1]),
               proxy.name: float(p.f[0]), sweep: float(p.f[1]),
               "eps": float(p.eps), "predictor": p.predictor_used}
        row.update({k: float(v) for k, v in fixed_eps.items()})   # pinned 3rd-obj value
        front.append(row)
    front.sort(key=lambda d: d[cfg.name_a])
    out = ctx.out(f"pareto_front_{cfg.name_a}__{cfg.name_b}_tracer_{cfg.tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": (cfg.name_a, cfg.name_b),
                     "n_points": cfg.n_points, "predictor": cfg.predictor,
                     "fixed_epsilons": fixed_eps, "front": front,
                     "messages": res.messages,
                     "sweep_anchor_pkl": sw_pkl, "proxy_anchor_pkl": ot_pkl}, f)
    print(f"saved {out}")
    return res


def trace_interior(case_dir, nsga_seed, oei_levels, *, n_points=12, tag="int"):
    """The 3-objective interior as a stack of `hover_elec x cruise_elec` traces,
    each at a fixed `oei_margin >= level`. Minimising cruise power drives
    `oei_margin` down to its bound, so each layer sits at `oei_margin ~= level`.

    Writes `pareto_surface_3obj_tracer_<tag>.pkl` (all layers' points in one
    `front` list, each carrying `hover_elec` / `cruise_elec` / `oei_margin`)."""
    pts = []
    for i, lvl in enumerate(oei_levels):
        print(f"\n{'=' * 70}\nLAYER {i}/{len(oei_levels) - 1}  oei_margin >= {lvl:.3f}\n{'=' * 70}",
              flush=True)
        res = trace(TraceConfig(
            case_dir=case_dir, name_a="hover_elec", name_b="cruise_elec",
            nsga_seed=nsga_seed, n_points=n_points, predictor="secant",
            tag=f"{tag}_L{i}_oei{lvl:.2f}", fixed_epsilons={"oei_margin": float(lvl)}))
        for p in res.points:
            pts.append({"hover_elec": float(p.f[1]), "cruise_elec": float(p.f[0]),
                        "oei_margin": float(lvl), "layer": i})
        if res.messages:
            print(f"  layer {i} messages: {res.messages}", flush=True)

    ctx = Context.load(case_dir)
    out = ctx.out(f"pareto_surface_3obj_tracer_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": ("hover_elec", "cruise_elec", "oei_margin"),
                     "oei_levels": [float(v) for v in oei_levels],
                     "n_points_per_layer": n_points, "front": pts}, f)
    print(f"\nsaved {out}  ({len(pts)} surface points, {len(oei_levels)} layers)")

    hc = ctx.out(f"pareto_front_hover_elec__cruise_elec_tracer_{tag}_L0_oei{oei_levels[0]:.2f}.pkl")
    try:
        overlay(out, surface_pkl=out, hc_pkl=hc if os.path.exists(hc) else None,
                out_path=ctx.out(f"pareto_overlay_tracer_{tag}.png"),
                title=f"{os.path.basename(case_dir)}: 3-obj tracer surface vs edges [{tag}]")
    except Exception as exc:                              # noqa: BLE001 -- plotting is optional
        print(f"overlay plot skipped: {exc}")
    return out


def overlay(ref_case_dir_or_pkl, *, surface_pkl, hc_pkl=None, ho_pkl=None, co_pkl=None,
            out_path, title=None):
    """SPL 2x2 overlay for a tracer run -- interior surface (grey) + the 3
    pairwise edges (blue). All inputs are explicit paths; generalises across
    runs (point at any run's pkls). `hc_pkl` / `ho_pkl` / `co_pkl` default to the
    newest `pareto_front_<pair>_tracer_*.pkl` in the matching tracer case dir
    (sibling of `tracer_hc`). `ref_case_dir_or_pkl` supplies only the 3-objective
    spec -- any 3-obj tracer case dir (or a surface pkl inside one) works."""
    import glob
    from .plot import render_overlay

    ref = ref_case_dir_or_pkl
    case_dir = ref if os.path.isdir(ref) else os.path.dirname(ref)
    if not os.path.isfile(os.path.join(case_dir, "case.py")):
        case_dir = os.path.join(os.path.dirname(case_dir), "shahjahan_case1_fpp_tracer_hc")
    ctx = Context.load(case_dir)

    cases_root = os.path.dirname(os.path.dirname(os.path.abspath(surface_pkl)))

    def _newest(dirname, a, b):
        for pat in (f"pareto_front_{a}__{b}_tracer_*.pkl", f"pareto_front_{b}__{a}_tracer_*.pkl"):
            m = sorted(glob.glob(os.path.join(dirname, pat)), key=os.path.getmtime)
            if m:
                return m[-1]
        return None

    hc_pkl = hc_pkl or _newest(os.path.join(cases_root, "shahjahan_case1_fpp_tracer_hc"),
                               "hover_elec", "cruise_elec")
    ho_pkl = ho_pkl or _newest(os.path.join(cases_root, "shahjahan_case1_fpp_tracer_ho"),
                               "hover_elec", "oei_margin")
    co_pkl = co_pkl or _newest(os.path.join(cases_root, "shahjahan_case1_fpp_tracer_co"),
                               "cruise_elec", "oei_margin")
    edge_pkls = {pair: pkl for pair, pkl in (
        (("hover_elec", "cruise_elec"), hc_pkl),
        (("hover_elec", "oei_margin"), ho_pkl),
        (("cruise_elec", "oei_margin"), co_pkl)) if pkl}

    return render_overlay(ctx, surface_pkl, edge_pkls, title=title,
                          out_name=os.path.basename(out_path))
