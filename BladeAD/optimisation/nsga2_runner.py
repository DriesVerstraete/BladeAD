"""NSGA-II population-method rotor optimisation -- the population-method arm of
the Shahjahan Case-1 benchmark (roadmap step 4).

Reuses the lab's `optimisation_framework` (spl-optimisation) NSGA-II verbatim.
The rotor-specific parts are only:

  * design vector <-> B-spline CP / rpm / collective  (`_x_from_var_dict`)
  * one BEM forward eval per candidate                (`forward.evaluate`)
  * objective + constraint posing                     (`RotorNSGA2Setup`)

Thrust handling -- NSGA-II cannot hold a thrust *equality* front
(`reference_nsga2_equality_constraint_limitation`,
`results/2026-08-31-survival-mechanisms-for-thrust-equality.md`). Here:

  * `T >= T_target` is a ONE-SIDED lower-bound constraint (a half-space, not the
    razor band that collapsed in Aug).
  * thrust *overshoot* is folded into the objectives as a scale-free
    multiplicative KS penalty:  f_i <- f_i * (1 + mu * KS_rho(overshoots)).
    The min-power objectives already punish overshoot; this makes it sharp so
    the front sits on the equality instead of drifting high.
  * raw (un-penalised) power is reported from the per-individual `performance`
    payload -- NO RPM re-solve.

Run in the `rotor_design` conda env.  Entry: `run(case_dir, **nsga2_opts)`.
"""
import csv
import os
import pickle
import time
from datetime import datetime

import numpy as np

from .case import load_case_dict, active_objectives
from . import forward
from . import seed as seedmod

# result_key -> the key `forward.evaluate` returns it under
_RESULT_KEY_MAP = {
    "hover_electrical_power": "hover_electrical_power",
    "cruise_electrical_power": "cruise_electrical_power",
    "hover_power": "hover_power",
    "cruise_power": "cruise_power",
    "figure_of_merit": "figure_of_merit",
    "cruise_efficiency": "cruise_efficiency",
    "oei_thrust_margin": "oei_thrust_margin",
}

_VAR_GROUPS = ("chord_cps_m", "twist_cps_deg", "hover_rpm", "cruise_rpm", "collective_deg")


from ._quiet import muffled as _muffle_cm     # shared fd-level solver-output suppressor


def _muffle():
    """Silence the BEM inner-solver chatter for one forward eval (the
    per-generation NSGA-II progress line prints outside it). Forward-eval
    failures are caught + returned as a sentinel inside `forward.evaluate`, so
    nothing diagnostic is lost."""
    return _muffle_cm(True)


def _ks(values, rho, cap):
    """Numerically-stable KS aggregate of `values`, with the log(n)/rho zero
    floor removed so an all-zero input gives exactly 0, then clamped to
    [0, cap]."""
    v = np.asarray(values, dtype=float)
    m = float(np.max(v))
    ks = m + np.log(np.sum(np.exp(rho * (v - m)))) / rho
    ks -= np.log(len(v)) / rho
    return float(np.clip(ks, 0.0, cap))


class RotorNSGA2Setup:
    """`optimisation_framework.Setup`-compatible (duck-typed -- we only need
    `set_variables` / `set_constraints` / `set_objectives` / `set_pareto` /
    `obj_func`, all called by `Setup.do` which we replicate in `run`)."""

    def __init__(self, case, seed_x, *, ks_rho, overshoot_mu, overshoot_ks_cap,
                 washout_ref_deg=5.0, objective_names=None):
        self.case = case
        self.seed_x = seed_x
        self.ks_rho = float(ks_rho)
        self.overshoot_mu = float(overshoot_mu)
        self.overshoot_ks_cap = float(overshoot_ks_cap)
        self.washout_ref_deg = float(washout_ref_deg)

        self.objs = active_objectives(case)
        if objective_names is not None:
            keep = list(objective_names)
            self.objs = [s for s in self.objs if s.name in keep]
            if {s.name for s in self.objs} != set(keep):
                raise ValueError(f"objective_names {keep} not all present in case "
                                 f"objectives {[s.name for s in active_objectives(case)]}")
        self.obj_names = [s.name for s in self.objs]
        self.oei = any(s.result_key == "oei_thrust_margin" for s in self.objs)

        b = case["bounds"]
        r = case["rotor"]
        self.groups = [
            ("chord_cps_m", r["n_chord_cps"], b["chord_m"][0], b["chord_m"][1]),
            ("twist_cps_deg", r["n_twist_cps"], b["twist_deg"][0], b["twist_deg"][1]),
            ("hover_rpm", 1, b["hover_rpm"][0], b["hover_rpm"][1]),
            ("cruise_rpm", 1, b["cruise_rpm"][0], b["cruise_rpm"][1]),
            ("collective_deg", 1, b["pitch_deg"][0], b["pitch_deg"][1]),
        ]
        if self.oei:
            ob = b.get("oei_rpm", b["hover_rpm"])
            self.groups.append(("oei_rpm", 1, ob[0], ob[1]))

        self._con_names = self._constraint_names()
        self.n_constraints = len(self._con_names)
        self.n_objectives = len(self.objs)

    # -- constraint bookkeeping ------------------------------------------------
    def _constraint_names(self):
        names = ["thrust_hover_lb", "thrust_cruise_lb", "stall_hover", "stall_cruise",
                 "taper_lo", "taper_hi", "washout"]
        # torque margins exist only for a real motor model (placebo / none -> None)
        m = (self.case.get("motor") or {}).get("model")
        if m in ("mcdonald", "emrax188"):
            names += ["torque_hover", "torque_cruise"]
        if self.oei:
            names += ["stall_oei"]
            if m in ("mcdonald", "emrax188"):
                names += ["torque_oei"]
            if self.case["constraints"].get("min_oei_thrust_margin") is not None:
                names += ["oei_margin_floor"]
        return names

    def _constraints(self, r):
        """Violation-form vector (g <= 0 satisfied), same order as
        `_constraint_names`. Relative / O(1) scaled."""
        c = self.case["constraints"]
        cap = r["cl_cap"]
        lo, hi = c["taper"]
        g = {
            "thrust_hover_lb": (r["hover_thrust_target"] - r["hover_thrust"]) / r["hover_thrust_target"],
            "thrust_cruise_lb": (r["cruise_thrust_target"] - r["cruise_thrust"]) / r["cruise_thrust_target"],
            "stall_hover": r["hover_stall_ratio"] / cap - 1.0,
            "stall_cruise": r["cruise_stall_ratio"] / cap - 1.0,
            "taper_lo": (lo - r["taper"]) / lo,
            "taper_hi": (r["taper"] - hi) / hi,
            "washout": -r["twist_washout_deg"] / self.washout_ref_deg,
        }
        if "torque_hover" in self._con_names:
            g["torque_hover"] = (r["hover_torque_margin"] or 0.0) - 1.0
            g["torque_cruise"] = (r["cruise_torque_margin"] or 0.0) - 1.0
        if self.oei:
            g["stall_oei"] = r["oei_stall_ratio"] / cap - 1.0
            if "torque_oei" in self._con_names:
                g["torque_oei"] = (r["oei_torque_margin"] or 0.0) - 1.0
            if "oei_margin_floor" in self._con_names:
                fl = float(c["min_oei_thrust_margin"])
                g["oei_margin_floor"] = (fl - r["oei_thrust_margin"]) / fl
        return np.array([g[n] for n in self._con_names], dtype=float)

    # -- optimisation_framework hooks ---------------------------------------
    def set_variables(self, prob, **kw):
        for name, n, lo, hi in self.groups:
            v0 = np.asarray(self.seed_x[name], dtype=float)
            v0 = np.clip(v0, lo, hi)
            prob.add_var_group(name, n, "c", lower=lo, upper=hi,
                               value=(v0 if v0.size == n else float(v0)))

    def set_constraints(self, prob, **kw):
        prob.add_con_group("g", self.n_constraints, lower=None, upper=0.0)

    def set_objectives(self, prob, **kw):
        for s in self.objs:
            prob.add_obj(s.name)

    def set_pareto(self, prob, **kw):
        pass

    def obj_func(self, x_dict, **kw):
        with _muffle():
            r = forward.evaluate(self.case, x_dict, oei=self.oei)
        ks = _ks([r["hover_thrust_overshoot"], r["cruise_thrust_overshoot"]],
                 self.ks_rho, self.overshoot_ks_cap)
        factor = 1.0 + self.overshoot_mu * ks

        raw_obj, pen_obj = [], []
        for s in self.objs:
            val = r[_RESULT_KEY_MAP[s.result_key]]
            m = val if s.goal == "min" else -val
            raw_obj.append(m)
            pen_obj.append(m * factor)

        cons = self._constraints(r)
        perf = dict(r)
        perf.update(_raw_obj=raw_obj, _pen_obj=pen_obj, _overshoot_ks=ks,
                    _penalty_factor=factor, _cons=cons.tolist(),
                    _con_names=self._con_names, _obj_names=self.obj_names)
        return np.array(pen_obj, dtype=float), cons, perf


# ---------------------------------------------------------------------------

def _make_seeded_lhs(seed_vecs, xl, xu, n_clones, sigma, base_cls):
    """LHS initial population with members 0..m-1 = the given seed blades
    (the inverse-design blade, plus any `front_seed_from` geometries), then
    `n_clones` small Gaussian perturbations of randomly-chosen seeds, the rest
    pure LHS. Pure LHS alone leaves NSGA-II unable to reach hover thrust in a
    small budget (2026-09-08 smoke); a spread of feasible front geometries lets
    a population method *hold* a front it cannot *build* (2026-08-31)."""
    span = np.where(xu > xl, xu - xl, 1.0)
    S = np.clip((np.atleast_2d(np.asarray(seed_vecs, float)) - xl) / span, 0.0, 1.0)

    class SeededLHS(base_cls):
        def _do(self, dim, n_samples, seed=None):
            super()._do(dim, n_samples, seed)
            rng = np.random.default_rng(seed)
            m = min(len(S), n_samples)
            self.x[:m] = S[:m]
            k = min(int(n_clones), max(0, n_samples - m))
            for i in range(m, m + k):
                base = S[rng.integers(0, m)] if m else self.x[i]
                self.x[i] = np.clip(base + rng.normal(0.0, sigma, size=dim), 0.0, 1.0)

    return SeededLHS()


def _dv_vec_from_stage1(pkl_path, setup):
    """Flat DV vector (prob order) from a `solve.py` `stage1_*.pkl` geometry:
    chord CP (m), twist CP (rad -> deg), hover/cruise rpm, FPP collective (deg),
    and oei_rpm when the case has an OEI objective."""
    g = pickle.load(open(pkl_path, "rb"))
    d = {
        "chord_cps_m": np.ravel(np.asarray(g["chord_cps"], float)),
        "twist_cps_deg": np.rad2deg(np.ravel(np.asarray(g["twist_cps"], float))),
        "hover_rpm": float(g["rpm"]),
        "cruise_rpm": float(g.get("cruise_rpm", np.mean(setup.case["bounds"]["cruise_rpm"]))),
        "collective_deg": float(g.get("hover_pitch_deg", g.get("pitch_deg", 0.0))),
        "oei_rpm": float(g.get("oei_rpm") or g["rpm"]),
    }
    return _seed_vec(setup, d)


def _prescale_seed_rpm(case, seed_x, oei):
    """One forward eval of the seed blade, then rescale HOVER rpm by
    sqrt(T_hover_target / T_hover_seed) (thrust ~ rpm^2 at fixed geometry) so the
    seeded member-0 lands on its hover thrust lower bound instead of ~2x over.
    Cruise rpm is left alone -- the coarse FPP blend blade goes through zero
    thrust between the overshoot rpm and any lower value, so an rpm^2 rescale
    there produces negative thrust; the seed's cruise overshoot is lower-bound
    feasible (just KS-penalised) and NSGA-II walks cruise_rpm down as the
    geometry sharpens. Clipped to the rpm bounds."""
    b = case["bounds"]
    with _muffle():
        r = forward.evaluate(case, seed_x, oei=oei)
    if r.get("_failed") or not (r["hover_thrust"] > 1.0):
        return seed_x
    out = dict(seed_x)
    scale = float(np.clip((r["hover_thrust_target"] / r["hover_thrust"]) ** 0.5, 0.6, 1.6))
    out["hover_rpm"] = float(np.clip(seed_x["hover_rpm"] * scale, *b["hover_rpm"]))
    out["oei_rpm"] = out["hover_rpm"]
    return out


def _seed_vec(setup, seed_x):
    """Flat design vector in `prob` DV order from the seed dict."""
    out = []
    for name, n, _lo, _hi in setup.groups:
        v = np.atleast_1d(np.asarray(seed_x[name], float))
        out.extend(v.tolist() if v.size == n else [float(v)] * n)
    return np.array(out, float)


def _seed_x_from_case(case_dir, case):
    """Cold-start design vector -- the inverse-design FPP blend blade (same
    seed the SLSQP workflow uses). Only sets `add_var_group(value=)`; the LHS
    sampler still fills gen-0 from the box."""
    from .case import Context
    ctx = Context.load(case_dir)
    sp = seedmod.write_inverse_design_seed(ctx, "fpp")
    d = pickle.load(open(sp, "rb"))
    return {
        "chord_cps_m": np.asarray(d["chord_cps"], float),
        "twist_cps_deg": np.rad2deg(np.asarray(d["twist_cps"], float)),
        "hover_rpm": float(d["rpm"]),
        "cruise_rpm": float(d["cruise_rpm"]),
        "collective_deg": float(d.get("pitch_deg", 0.0)),
        "oei_rpm": float(d.get("oei_rpm", d["rpm"])),
    }


def _nondominated(F):
    """Indices of the non-dominated rows of F (minimisation, small n)."""
    n = len(F)
    keep = []
    for i in range(n):
        dom = False
        for j in range(n):
            if j == i:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                dom = True
                break
        if not dom:
            keep.append(i)
    return keep


def run(case_dir, *, pop_size=16, n_gen=8, seed=1, n_workers=1,
        algorithm="nsga2", n_ref_partitions=None,
        moead_neighbours=15, moead_nr=2,
        eta_c=15.0, eta_m=20.0, p_c=0.9, p_m=None,
        ks_rho=100.0, overshoot_mu=10.0, overshoot_ks_cap=1.0,
        seed_clone_frac=0.5, seed_clone_sigma=0.08,
        front_seed_from=None, save_tag=None, hot_start=False,
        objective_names=None):
    """Run NSGA-II (or NSGA-III, `algorithm="nsga3"`) on the case. History
    auto-pickled per generation to
    `<case_dir>/results/optimisation_history_<name>_<tag>.pkl`; the final front +
    raw values go to `<case_dir>/nsga2_result_<tag>.pkl` and
    `nsga2_front_raw_<tag>.csv`.

    `hot_start=True` resumes from `results/optimisation_history_<name>_<tag>.pkl`
    (same `save_tag` and `pop_size` as the original run) and continues to
    `n_gen` -- so raising `n_gen` and re-running with `hot_start=True` extends a
    stopped run in place. For restartability set a fixed `save_tag` in the case
    `nsga2` block (a timestamp is used only when it is None).

    `front_seed_from`: a list of `solve.py` `stage1_*.pkl` paths (or directories
    to glob `stage1_*.pkl` from) whose geometries seed the initial population
    (members after the inverse-design blade), so NSGA-II starts already holding
    a spread feasible front."""
    import matplotlib
    matplotlib.use("Agg")
    from optimisation_framework.optimisation.model.problem import Problem
    from optimisation_framework.optimisation.optimise import minimise
    from optimisation_framework.optimisation.algorithms.nsga2 import NSGA2
    from optimisation_framework.optimisation.algorithms.nsga3 import NSGA3
    from optimisation_framework.optimisation.algorithms.moead import MOEAD
    from optimisation_framework.optimisation.algorithms.moead_de import MOEADDE
    from optimisation_framework.optimisation.util.reference_directions import UniformReferenceDirection
    from optimisation_framework.optimisation.operators.crossover.simulated_binary_crossover \
        import SimulatedBinaryCrossover
    from optimisation_framework.optimisation.operators.mutation.polynomial_mutation \
        import PolynomialMutation

    class _VerboseMixin:
        """Replaces the framework's `n_gen | n_eval` line with a per-generation
        feasibility line (true non-dominated count on the RAW objectives among
        feasible -- `ind.rank` after survival does not match a clean ND sort).
        Also makes the per-generation history write atomic (tmp + os.replace)
        so a concurrent live-plot reader never sees a half-written pkl.
        Constructed with `print=False` so only this line shows."""

        def write_history(self, write_format):
            sn = getattr(self, "save_name", None)   # MOEAD.__init__ does not capture it
            nm = self.problem.name if sn is None else f"{self.problem.name}_{sn}"
            os.makedirs("./results", exist_ok=True)
            final = f"./results/optimisation_history_{nm}.pkl"
            tmp = f"{final}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"history": self.history, "eval_history": self.eval_history},
                            f, pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, final)

        def each_iteration(self):
            pop = self.population
            cs = np.array([float(getattr(ind, "cons_sum", 0.0)) for ind in pop])
            feas = cs <= 1e-6
            nfeas = int(feas.sum())
            # true non-dominated count on the RAW objectives among feasible
            # (`ind.rank` after survival does not match a clean ND sort here).
            front = 0
            if nfeas:
                F = np.array([ind.performance["_raw_obj"] for ind, ok in zip(pop, feas) if ok])
                front = len(_nondominated(F))
            print(f"gen {self.n_gen:3d} | evals {self.evaluator.n_eval:5d} | "
                  f"feasible {nfeas:3d}/{len(pop)} ({100 * nfeas / max(1, len(pop)):3.0f}%) | "
                  f"min_cv {cs.min():.3f} | median_cv {float(np.median(cs)):.3f} | front {front}")
            super().each_iteration()

    class _NSGA2Verbose(_VerboseMixin, NSGA2):
        pass

    class _NSGA3Verbose(_VerboseMixin, NSGA3):
        pass

    class _MOEADVerbose(_VerboseMixin, MOEAD):
        pass

    class _MOEADDEVerbose(_VerboseMixin, MOEADDE):
        pass

    case_dir = os.path.abspath(case_dir)
    case = load_case_dict(case_dir)
    tag = save_tag or datetime.now().strftime("%Y%m%dT%H%M%S")

    seed_x = _seed_x_from_case(case_dir, case)
    _scout_objs = active_objectives(case) if objective_names is None else \
        [s for s in active_objectives(case) if s.name in set(objective_names)]
    _oei = any(s.result_key == "oei_thrust_margin" for s in _scout_objs)
    seed_x = _prescale_seed_rpm(case, seed_x, _oei)
    setup = RotorNSGA2Setup(case, seed_x, ks_rho=ks_rho, overshoot_mu=overshoot_mu,
                            overshoot_ks_cap=overshoot_ks_cap,
                            objective_names=objective_names)
    name = f"rotor_nsga2_{os.path.basename(case_dir)}"

    print(f"=== {algorithm.upper()} {name} ===")
    print(f"objectives: {setup.obj_names}   constraints ({setup.n_constraints}): {setup._con_names}")
    print(f"vars: {[(g[0], g[1]) for g in setup.groups]}  (n_var={sum(g[1] for g in setup.groups)})")
    print(f"algorithm={algorithm} pop={pop_size} gen={n_gen} seed={seed} workers={n_workers} "
          f"ks_rho={ks_rho} overshoot_mu={overshoot_mu} overshoot_ks_cap={overshoot_ks_cap} "
          f"tag={tag} hot_start={hot_start}")

    cwd0 = os.getcwd()
    os.chdir(case_dir)
    t0 = time.time()
    try:
        prob = Problem(name, setup.obj_func,
                       map_internally=(n_workers > 1), n_processors=max(1, n_workers))
        setup.set_variables(prob)
        setup.set_constraints(prob)
        setup.set_objectives(prob)
        prob.finalise()

        from optimisation_framework.optimisation.operators.sampling.latin_hypercube_sampling \
            import LatinHypercubeSampling
        import glob as _glob
        seed_vecs = [_seed_vec(setup, seed_x)]
        front_paths = []
        for src in (front_seed_from or []):
            front_paths += (sorted(_glob.glob(os.path.join(src, "stage1_*.pkl")))
                            if os.path.isdir(src) else [src])
        used = []
        for p in front_paths:
            try:
                g = pickle.load(open(p, "rb"))
            except Exception:
                continue
            if not all((g.get("constraint_checks") or {"_": True}).values()):
                continue                                  # skip an infeasible / edge stage1
            seed_vecs.append(_dv_vec_from_stage1(p, setup))
            used.append(os.path.basename(p))
        if front_paths:
            print(f"front seeds: {len(used)}/{len(front_paths)} feasible used  "
                  + ", ".join(used))
        sampling = _make_seeded_lhs(
            seed_vecs, np.asarray(prob.x_lower, float), np.asarray(prob.x_upper, float),
            n_clones=round(seed_clone_frac * pop_size), sigma=seed_clone_sigma,
            base_cls=LatinHypercubeSampling)

        common = dict(n_population=pop_size, max_gen=n_gen, sampling=sampling,
                      crossover=SimulatedBinaryCrossover(eta=eta_c, prob=p_c),
                      mutation=PolynomialMutation(eta=eta_m, prob=p_m),
                      plot=False, save_results=False, save_name=tag, print=False)
        if algorithm == "nsga2":
            algo = _NSGA2Verbose(**common)
        elif algorithm in ("nsga3", "moead", "moead_de"):
            n_part = int(n_ref_partitions or pop_size - 1)
            ref_dirs = UniformReferenceDirection(setup.n_objectives, n_partitions=n_part).do()
            print(f"algorithm={algorithm}  n_ref_partitions={n_part}  n_ref_dirs={len(ref_dirs)}")
            if algorithm == "nsga3":
                algo = _NSGA3Verbose(ref_dirs=ref_dirs, **common)
            elif algorithm == "moead":
                algo = _MOEADVerbose(ref_dirs=ref_dirs, n_neighbours=moead_neighbours,
                                     constraint_handling="cdp", **common)
            else:  # moead_de -- DE variation + restricted replacement (nr)
                _c = {k: v for k, v in common.items() if k != "crossover"}  # DE crossover forced
                algo = _MOEADDEVerbose(ref_dirs=ref_dirs, n_neighbours=moead_neighbours,
                                       constraint_handling="cdp", update_scheme="sync",
                                       nr=moead_nr, **_c)
        else:
            raise SystemExit(f"unknown algorithm {algorithm!r} (nsga2 | nsga3 | moead | moead_de)")
        algo.save_name = tag        # MOEAD/MOEADDE __init__ do not capture save_name
        minimise(prob, algo, seed=seed, save_history=True, hot_start=hot_start)
    finally:
        os.chdir(cwd0)
    wall = time.time() - t0

    final = algo.history[-1]
    rows = []
    for ind in final:
        perf = ind.performance
        rows.append({
            "cons_sum": float(getattr(ind, "cons_sum", np.nan)),
            "feasible": float(getattr(ind, "cons_sum", 1.0)) <= 1e-6,
            "raw_obj": perf["_raw_obj"], "pen_obj": perf["_pen_obj"],
            "overshoot_ks": perf["_overshoot_ks"], "penalty_factor": perf["_penalty_factor"],
            "var_dict": {k: np.asarray(v).tolist() for k, v in ind.var_dict.items()},
            "perf": dict(perf),
        })

    n_failed = sum(1 for r in rows if r["perf"].get("_failed"))
    feas = [r for r in rows if r["feasible"]]
    front_idx = []
    if feas:
        F = np.array([r["raw_obj"] for r in feas])
        front_idx = _nondominated(F)
    front = [feas[i] for i in front_idx]

    # compact per-generation snapshot for the progress plot (raw objective
    # cloud + feasibility per individual) -- floats only, ~n_gen*pop*(n_obj+2).
    n_obj = len(setup.objs)
    generations = []
    for g, pop in enumerate(algo.history):
        ro, cvv, ksv = [], [], []
        for ind in pop:
            pf = ind.performance
            ro.append(pf.get("_raw_obj", [np.nan] * n_obj))
            cvv.append(float(getattr(ind, "cons_sum", np.nan)))
            ksv.append(float(pf.get("_overshoot_ks", np.nan)))
        generations.append({
            "gen": g,
            "raw_obj": np.asarray(ro, float),
            "cons_sum": np.asarray(cvv, float),
            "overshoot_ks": np.asarray(ksv, float),
        })

    result = {
        "case_dir": case_dir, "name": name, "tag": tag,
        "obj_names": setup.obj_names, "con_names": setup._con_names,
        "pop_size": pop_size, "n_gen": n_gen, "seed": seed, "n_workers": n_workers,
        "ks_rho": ks_rho, "overshoot_mu": overshoot_mu, "overshoot_ks_cap": overshoot_ks_cap,
        "n_eval": int(algo.evaluator.n_eval), "wall_s": wall,
        "n_final": len(rows), "n_feasible": len(feas), "n_front": len(front),
        "n_failed_final": n_failed,
        "final_population": rows, "front": front,
        "generations": generations,
        "epsilon_case_dir": os.path.join(os.path.dirname(case_dir),
                                         "shahjahan_case1_fpp_explore_sm099"),
        "history_pkl": os.path.join(case_dir, "results",
                                    f"optimisation_history_{name}_{tag}.pkl"),
    }
    out_pkl = os.path.join(case_dir, f"nsga2_result_{tag}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)

    csv_path = os.path.join(case_dir, f"nsga2_front_raw_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        cols = (["idx"] + setup.obj_names
                + ["hover_electrical_power", "cruise_electrical_power",
                   "hover_thrust_overshoot", "cruise_thrust_overshoot",
                   "hover_stall_ratio", "cruise_stall_ratio", "taper", "twist_washout_deg",
                   "hover_rpm", "cruise_rpm", "collective_deg"]
                + (["oei_thrust_margin", "oei_rpm"] if setup.oei else []))
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(front):
            p = r["perf"]
            row = [i] + [p[_RESULT_KEY_MAP[s.result_key]] for s in setup.objs]
            row += [p.get(k) for k in cols[1 + len(setup.objs):]]
            w.writerow(row)

    png = None
    try:
        from . import plot_nsga2_progress as _pp
        png = _pp.plot(out_pkl, filter_feasible=True)
    except Exception as exc:                     # noqa: BLE001
        print(f"progress plot skipped: {type(exc).__name__}: {exc}")

    print(f"\n=== done in {wall / 60:.1f} min  ({result['n_eval']} evals) ===")
    print(f"final pop {len(rows)}  feasible {len(feas)}  front {len(front)}  failed {n_failed}")
    for r in front:
        print("  " + "  ".join(f"{n}={v:.1f}" for n, v in zip(setup.obj_names, r["raw_obj"]))
              + f"   ks={r['overshoot_ks']:.3f}")
    print(f"\nSaved {out_pkl}\nSaved {csv_path}" + (f"\nSaved {png}" if png else ""))
    return result
