"""Shared plumbing for the case-driven Pareto sweep (`edges.py`, `nobj.py`,
`orchestrator.py`).

Case-agnostic, objective-slot-driven (2026-09-06 objective-slots refactor of the
2026-09-06 port). Same machinery:

* Each solve runs in its own subprocess (`python -c "from rotor_pareto_slots
  import solve; solve.run_case_dir(...)"` -- process isolation without a CLI)
  with `with_acoustics=True`, so every front lives in the same feasible space
  (vehicle cruise OSPL + 10*log10(N) <= cap enforced everywhere).
* **`min_power_solve` (optimise the `proxy` s.t. epsilon-constraints on the
  other slots) is the ONLY working direction** for the coupled trade
  (`reference_coupled_pareto_minpower_direction`).
* `objective_anchor(name)` gives one slot's single-objective extreme:
  `floor`/`cap` slots maximise/minimise directly; the `proxy` anchor is the
  loosest-feasible-floor endpoint (geometric probe DOWN + ~3 bisections).
* Anchors are solved once and cached by slot name in a **per-case**
  `_anchor_registry.json` inside the case directory.
* A **cross-sweep warm pool** (`nearest_warm`) globs the converged `stage1_*`
  pkls, parses their `_eps-<name>=<value>` tags, and returns the nearest so
  every interior solve starts from a nearby feasible point.
"""
import glob
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

from .case import parse_objectives, parse_epsilon_tag

# every sweep solve enforces the vehicle cruise-noise cap
BASE_OPTS = {"with_acoustics": True}


# ---------------------------------------------------------------------------
# subprocess solve
# ---------------------------------------------------------------------------

def run_solve(ctx, opts):
    """Subprocess `solve.run_case_dir(case_dir, **BASE_OPTS, **opts)`. Parse the
    real 'Saved to <path>' line. Raise if any constraint check failed."""
    full = {**BASE_OPTS, **opts}
    code = (
        "from BladeAD.optimisation import solve; "
        "solve.run_case_dir(r'{case_dir}', **{opts!r})"
    ).format(case_dir=ctx.case_dir, opts=full)
    print(f"  $ run({full})", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, "-c", code], cwd=ctx.case_dir,
                            capture_output=True, text=True)
    dt = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(f"solve failed (exit {result.returncode}, {dt:.0f}s): {full}")
    saved = [ln for ln in result.stdout.splitlines() if ln.startswith("Saved to ")]
    if not saved:
        print(result.stdout[-4000:])
        raise RuntimeError(f"no 'Saved to <path>' line in solve stdout ({dt:.0f}s)")
    pkl_path = saved[-1][len("Saved to "):].strip()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    checks = data["constraint_checks"]
    bad = [n for n, ok in checks.items() if not ok]
    print(f"  -> {os.path.basename(pkl_path)}  ({dt:.0f}s"
          f"{', FAILED: ' + ', '.join(bad) if bad else ''})", flush=True)
    if bad:
        raise RuntimeError(
            f"solve at {os.path.basename(pkl_path)} converged with FAILED constraints: {bad}")
    return pkl_path, data


# ---------------------------------------------------------------------------
# per-case anchor registry
# ---------------------------------------------------------------------------

def clear_anchor_registry(ctx):
    try:
        os.remove(ctx.registry_path)
    except FileNotFoundError:
        pass


def registered_anchor(ctx, key, solver):
    """(pkl_path, data) for a named shared anchor. `solver()` -> (pkl_path,
    data), run once per key per registry; later fronts reuse the cached pkl."""
    try:
        with open(ctx.registry_path) as f:
            reg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        reg = {}
    if key in reg and os.path.exists(reg[key]):
        with open(reg[key], "rb") as f:
            print(f"  anchor '{key}': reusing {os.path.basename(reg[key])}")
            return reg[key], pickle.load(f)
    pkl_path, data = solver()
    reg[key] = pkl_path
    with open(ctx.registry_path, "w") as f:
        json.dump(reg, f, indent=2)
    return pkl_path, data


# ---------------------------------------------------------------------------
# spec helpers
# ---------------------------------------------------------------------------

def specs(ctx):
    return parse_objectives(ctx.case["objectives"])


def live_specs(ctx):
    return [s for s in specs(ctx) if s.role != "reserved"]


def proxy_spec(ctx):
    return next(s for s in live_specs(ctx) if s.role == "proxy")


def spec_by_name(ctx, name):
    return next(s for s in specs(ctx) if s.name == name)


def objective_value(data, name):
    """Read one objective slot's value from a solve result dict."""
    return data["objective_values"][name]


# ---------------------------------------------------------------------------
# the generic solve + warm pool
# ---------------------------------------------------------------------------

def nearest_warm(ctx, epsilons):
    """Path of the converged `stage1_*` pkl in the case dir whose epsilon tags
    are closest to `epsilons`, or None if the pool is empty. Cheap: parses the
    filename tag, no unpickling.

    Metric = the frozen `rotor_pareto` absolute one, generalised: per shared key
    `|tag[k] - eps[k]| * w`, with `w = 1.0` for O(1)-scale slots (`|eps| <= 2`:
    FM, efficiency, thrust-margin ratio) and `w = 0.02` for large-scale slots
    (noise in dB) -- matching the frozen `|Δfm| + 0.02*|Δnoise|`. A tiny
    keyset-mismatch penalty (0.1/key) is a tiebreak only, NOT a dominant term
    (the 5.0/key penalty regressed grid coverage ~2x -- 2026-09-07)."""
    best, best_d = None, None
    for path in glob.glob(ctx.out("stage1_*.pkl")):
        tags = parse_epsilon_tag(os.path.basename(path))
        if not tags:
            continue
        shared = set(tags) & set(epsilons)
        if not shared and epsilons:
            continue
        d = sum(abs(tags[k] - epsilons[k]) * (1.0 if abs(epsilons[k]) <= 2.0 else 0.02)
                for k in shared)
        d += 0.1 * len(set(epsilons) ^ set(tags))          # tiebreak on mismatched key sets
        if best_d is None or d < best_d:
            best, best_d = path, d
    return best


def _solve(ctx, optimize, epsilons, warm, with_acoustics=None):
    """`warm`: a pkl path, None (auto-pick nearest), or False (no warm start).
    `with_acoustics`: None keeps the BASE_OPTS default (True); pass False to
    skip the acoustic graph (much faster -- used by the Pareto tracer, whose
    reference fronts' anchors were also run acoustics-off)."""
    opts = {}
    if optimize is not None:
        opts["optimize"] = optimize
    if epsilons:
        opts["epsilons"] = {k: round(float(v), 6) for k, v in epsilons.items()}
    if warm is None:
        warm = nearest_warm(ctx, epsilons or {})
    if warm:
        opts["initial_result_from"] = warm
    if with_acoustics is not None:
        opts["with_acoustics"] = bool(with_acoustics)
    return run_solve(ctx, opts)


def min_power_solve(ctx, epsilons, warm=None, with_acoustics=None):
    """The working scalarisation: optimise the `proxy` (min cruise power) subject
    to `epsilons` = {objective_name: value}. Raises RuntimeError on a failed /
    past-the-feasibility-wall solve."""
    return _solve(ctx, None, epsilons, warm, with_acoustics)


def direct_solve(ctx, optimize, epsilons=None, warm=None, with_acoustics=None):
    """Optimise a non-proxy objective `optimize` directly (max a `floor`, min a
    `cap`), optionally subject to `epsilons` on the remaining slots."""
    return _solve(ctx, optimize, epsilons or {}, warm, with_acoustics)


# ---------------------------------------------------------------------------
# anchors -- one per objective slot, registry-cached by slot name
# ---------------------------------------------------------------------------

def objective_anchor(ctx, name):
    """(pkl, data) for the single-objective extreme of slot `name` (it free, all
    other slots unconstrained). Registry key = `name`.

      floor / cap : maximise / minimise the slot's own quantity directly.
      proxy       : the loosest-feasible-floor endpoint -- probe the (first)
                    non-proxy `floor` slot's floor DOWN from its own extreme
                    until min_power_solve raises, then bisect. Unconstrained
                    min-power if the case has no `floor` slot.
    """
    s = spec_by_name(ctx, name)
    if s.role != "proxy":
        # anchors do NOT warm-start (match the frozen hover-only / acoustic-only
        # anchors: a warm start can pull SLSQP to a different local extreme).
        return registered_anchor(ctx, name, lambda: direct_solve(ctx, name, warm=False))
    return registered_anchor(ctx, name, lambda: _proxy_anchor_impl(ctx))


def _vardict_from_flat(v):
    """NSGA-II framework `Individual.var` (flat) -> the `var_dict` layout.
    Layout: chord_cps[5], twist_cps_deg[5], hover_rpm, cruise_rpm, collective_deg,
    [oei_rpm]."""
    v = np.asarray(v, float).ravel()
    d = {"chord_cps_m": v[0:5], "twist_cps_deg": v[5:10],
         "hover_rpm": float(v[10]), "cruise_rpm": float(v[11]),
         "collective_deg": float(v[12])}
    if v.size > 13:
        d["oei_rpm"] = float(v[13])
    return d


def nsga_geometry_to_seed(obj, out_path, *, fixed_pitch):
    """Write an `initial_result_from`-format seed pkl from an NSGA-II artefact.

    `obj` may be a result pkl dict (`front` / `final_population` of point dicts
    with `var_dict`), a raw `var_dict`, an optimisation-history pkl (last
    generation's first feasible individual), or an already-seed-shaped dict.
    Converts the NSGA var names/units (`chord_cps_m`, `twist_cps_deg` in DEGREES,
    `hover_rpm`, `collective_deg`) to what `solve.run(initial_result_from=...)`
    reads (`chord_cps`, `twist_cps` in RADIANS, `rpm`, `pitch_deg` /
    `hover_pitch_deg`, `oei_rpm`)."""
    if isinstance(obj, dict) and "chord_cps" in obj:            # already seed-shaped
        vd = None
    elif isinstance(obj, dict) and obj.get("front"):
        vd = obj["front"][0]["var_dict"]
    elif isinstance(obj, dict) and obj.get("final_population"):
        vd = obj["final_population"][0]["var_dict"]
    elif isinstance(obj, dict) and "chord_cps_m" in obj:        # a raw var_dict
        vd = obj
    else:                                                       # history pkl / list
        hist = obj["history"] if isinstance(obj, dict) else obj
        pop = hist[-1]
        pick = min((i for i in pop if float(getattr(i, "cons_sum", 1.0)) <= 1e-6),
                   key=lambda i: float(i.objectives[0]) if hasattr(i, "objectives")
                   else float(i.performance["_raw_obj"][0]), default=pop[0])
        vd = _vardict_from_flat(pick.var)

    if vd is None:
        d = dict(obj)
    else:
        g = lambda k: np.ravel(np.asarray(vd[k], float))
        d = {"chord_cps": g("chord_cps_m"),
             "twist_cps": np.deg2rad(g("twist_cps_deg")),
             "rpm": float(g("hover_rpm")[0]), "cruise_rpm": float(g("cruise_rpm")[0]),
             "oei_rpm": (float(g("oei_rpm")[0]) if "oei_rpm" in vd else None)}
        coll = float(g("collective_deg")[0])
        if fixed_pitch:
            d["pitch_deg"] = d["hover_pitch_deg"] = coll
        else:
            d["hover_pitch_deg"] = coll
            d["cruise_pitch_deg"] = (float(g("cruise_pitch_deg")[0])
                                     if "cruise_pitch_deg" in vd else coll)
    with open(out_path, "wb") as f:
        pickle.dump(d, f)
    return out_path


def seeded_anchor(ctx, name, seed_pkl, epsilons=None, with_acoustics=None):
    """Single-objective extreme of slot `name`. Returns (pkl_path, data).

    Unlike `objective_anchor`: no registry, no proxy-anchor bisection.

    A `proxy` extreme (the unconstrained min-power end -- historically the one
    that a cold SLSQP anchor truncated) is warm-started from `seed_pkl`, an
    `initial_result_from`-format geometry (e.g. an NSGA-II basin). A non-proxy
    extreme (`cap` min / `floor` max, thrust equalities active -- well-posed and
    where a warm start from a far point tends to diverge) is run COLD, matching
    `objective_anchor`. Pass `seed_pkl=False` to force the proxy end cold too."""
    s = spec_by_name(ctx, name)
    eps = dict(epsilons or {})
    if s.role == "proxy":
        return min_power_solve(ctx, eps, warm=seed_pkl, with_acoustics=with_acoustics)
    return direct_solve(ctx, name, eps, warm=False, with_acoustics=with_acoustics)


def _proxy_anchor_impl(ctx):
    floors = [s for s in live_specs(ctx) if s.role == "floor"]
    proxy = proxy_spec(ctx)
    if not floors:
        return min_power_solve(ctx, {})
    fl = floors[0]
    fl_pkl, fl_data = objective_anchor(ctx, fl.name)
    x_max = objective_value(fl_data, fl.name)
    # O(1)-scale slots (FM, efficiency, thrust-margin ratio) probe on the
    # historical absolute step; large-scale slots scale it.
    unit_scale = x_max <= 2.0
    x_floor_min = 0.30 if unit_scale else 0.30 * x_max

    feas = []                 # (floor, proxy_value, pkl, data), descending floor
    infeas = None
    step = 0.02 if unit_scale else 0.02 * x_max
    x = x_max - step
    warm = fl_pkl
    while x > x_floor_min:
        try:
            pkl, data = min_power_solve(ctx, {fl.name: x}, warm=warm)
        except RuntimeError as exc:
            print(f"  {proxy.name} anchor: {fl.name}>={x:.4f} INFEASIBLE ({exc})")
            infeas = x
            break
        pv = objective_value(data, proxy.name)
        print(f"  {proxy.name} anchor: {fl.name}>={x:.4f} feasible  "
              f"{proxy.name}={pv:.4f}  {fl.name}={objective_value(data, fl.name):.4f}")
        if feas and pv - feas[-1][1] < 5e-3:
            print(f"  {proxy.name} anchor: plateaued -> floor slack, at natural extreme")
            return pkl, data
        feas.append((x, pv, pkl, data)); warm = pkl
        step *= 2.0
        x -= step
    if not feas:
        raise RuntimeError(f"{proxy.name} anchor: no feasible {fl.name} floor below {x_max:.4f}")
    if infeas is None:
        return feas[-1][2], feas[-1][3]
    lo, hi = feas[-1][0], infeas          # lo feasible, hi infeasible, lo > hi
    best = (feas[-1][2], feas[-1][3])
    for _ in range(3):
        mid = 0.5 * (lo + hi)
        try:
            pkl, data = min_power_solve(ctx, {fl.name: mid}, warm=best[0])
            print(f"  {proxy.name} anchor: {fl.name}>={mid:.4f} feasible")
            lo, best = mid, (pkl, data)
        except RuntimeError:
            print(f"  {proxy.name} anchor: {fl.name}>={mid:.4f} infeasible")
            hi = mid
    return best


