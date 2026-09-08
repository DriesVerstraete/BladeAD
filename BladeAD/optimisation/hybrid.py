"""Scout a design basin with a population method, then refine the Pareto surface
with the gradient front tracer.

Generic over the objective set. Any case whose ``CASE["objectives"]`` parse to
one ``proxy`` slot plus one or more non-proxy ``floor`` / ``cap`` slots works --
the driver reads the specs and builds the stacks from them; nothing here is
hard-coded to a particular objective.

Construction (N objectives, one of them the proxy):

* **primary stack** -- for the first non-proxy objective ``A``: sweep the edge
  ``(proxy, A)`` with the gradient tracer, once per pinned-level combination of
  every *other* non-proxy objective.
* **cross-check stack** (optional, ``cross_check=True``) -- the same, with ``A``
  swapped for each of the other non-proxy objectives in turn. Two independent
  constructions of the same manifold; disagreement localises an under-resolved
  region (see ``results/2026-09-08-3obj-surface-orthogonal-crosscheck.md``).

The two stacks are merged under a goal-aware non-dominated filter. A partial
surface pkl is rewritten after every layer so a long run that dies mid-way still
leaves a usable surface plus every per-edge pkl the tracer wrote.

``CASE["hybrid"]`` (all keys optional, defaults below)::

    n_anchors        int    anchors handed from the scout to the refiner
    scout_pop        int    NSGA-II population
    scout_gens       int    NSGA-II generations
    scout_objectives list   restrict the scout to these objective names (the BEM
                            population kernel cannot evaluate an acoustic slot);
                            None -> all case objectives
    seed             int    RNG seed (scout + k-means)
    n_points         int    tracer points per edge trace
    cluster_method   str    "kmeans" anchor picker
    force_scout      bool   ignore a cached scout pkl
    with_acoustics   bool   run the acoustic graph in the refine stage (~15x slower)
    cross_check      bool   build the cross-check stack too
    levels           dict   {objective_name: [v, ...]  OR  "auto"} -- the pinned
                            levels for each non-proxy objective. "auto" (or a
                            missing entry) brackets that objective from single-
                            objective anchors and picks `auto_levels` interior
                            values.
    auto_levels      int    N interior levels when an entry is "auto"

No argparse (workspace style): a ``hybrid_run.py`` in the case dir calls
``run_hybrid(case_dir)``.
"""
import os
import pickle
from itertools import combinations, product

import numpy as np

from . import nsga2_runner, pareto_tracer
from .case import Context, parse_objectives
from .sweep_common import (
    nsga_geometry_to_seed, objective_value, seeded_anchor,
)

DEFAULTS = {
    "n_anchors": 3,
    "scout_pop": 48,
    "scout_gens": 60,
    "scout_objectives": None,
    "scout_front_seed_from": None,   # dir(s) of feasible stage1_*.pkl to seed the
                                     # initial scout population (faster feasibility)
    "seed": 1,
    "n_points": 12,
    "cluster_method": "kmeans",
    "force_scout": False,
    "with_acoustics": False,
    "cross_check": True,
    "levels": {},
    "auto_levels": 4,
}

DRY_RUN_OVERRIDES = {
    "scout_pop": 24,
    "scout_gens": 6,
    "n_points": 3,
    "auto_levels": 2,
}


# --------------------------------------------------------------------------- #
# anchor selection                                                            #
# --------------------------------------------------------------------------- #

def _objective_vector(point, obj_names):
    raw = point.get("perf", {}).get("_raw_obj", point.get("raw_obj"))
    return np.asarray(raw, float)[:len(obj_names)]


def _pick_anchors(result, n, *, obj_names, seed=1, cluster_method="kmeans"):
    """Return up to `n` scout point-dicts to seed the refiner from.

    A well-spread non-dominated set is clustered (k-means, one representative per
    cluster). A collapsed set -- the common case for a weakly-conflicting rotor
    front -- is padded with the whole-feasible objective-space extremes so the
    anchors still span the box.
    """
    if isinstance(result, (str, os.PathLike)):
        result = pickle.load(open(result, "rb"))
    feasible = [p for p in result.get("final_population", [])
                if p.get("feasible", float(p.get("cons_sum", 1)) <= 1e-6)]
    nd = result.get("front") or feasible
    if not feasible or not nd or n <= 0:
        return []

    all_x = np.array([_objective_vector(p, obj_names) for p in feasible])
    lo = all_x.min(0)
    span = np.ptp(all_x, axis=0)
    span[span == 0] = 1.0

    def norm(xs):
        return (np.asarray(xs) - lo) / span

    def add(picks, p):
        if not any(p.get("var_dict") == q.get("var_dict") for q in picks):
            picks.append(p)

    # 1. the per-objective extremes of the feasible set -- guarantees the anchors
    #    span every objective axis.
    picks = []
    for j in range(len(obj_names)):
        for idx in (int(np.argmin(all_x[:, j])), int(np.argmax(all_x[:, j]))):
            if len(picks) < n:
                add(picks, feasible[idx])

    # 2. fill any remaining slots with k-means cluster representatives of the
    #    non-dominated set (interior spread). A collapsed ND set (fewer than the
    #    remaining slots, distinct) contributes nothing and step 1 already
    #    covered the box.
    slots = n - len(picks)
    if slots > 0 and cluster_method == "kmeans" and len(nd) >= slots:
        nd_x = norm([_objective_vector(p, obj_names) for p in nd])
        try:
            from sklearn.cluster import KMeans
            centers = KMeans(n_clusters=slots, random_state=seed, n_init=10).fit(nd_x).cluster_centers_
            for c in centers:
                add(picks, nd[int(np.argmin(np.linalg.norm(nd_x - c, axis=1)))])
        except ImportError:
            for i in range(slots):
                add(picks, nd[i * len(nd) // max(1, slots)])
    return picks[:n]


# --------------------------------------------------------------------------- #
# goal-aware non-dominated merge                                              #
# --------------------------------------------------------------------------- #

def _min_vector(point, specs):
    return np.array([(-point[s.name] if s.goal == "max" else point[s.name]) for s in specs], float)


def _dominates(a, b, specs):
    av = _min_vector(a, specs)
    bv = _min_vector(b, specs)
    return np.all(av <= bv) and np.any(av < bv)


def _merge(points, specs):
    keep = []
    for i, p in enumerate(points):
        if not any(_dominates(q, p, specs) for j, q in enumerate(points) if j != i):
            keep.append(p)
    return keep


# --------------------------------------------------------------------------- #
# pinned-level resolution                                                     #
# --------------------------------------------------------------------------- #

def _auto_levels(ctx, spec, seed_pkl, n, with_acoustics, proxy_data):
    """Bracket objective `spec` between its own single-objective extreme and its
    value at the proxy optimum, then take `n` interior levels."""
    _, own = seeded_anchor(ctx, spec.name, seed_pkl, with_acoustics=with_acoustics)
    a = float(objective_value(own, spec.name))
    b = float(objective_value(proxy_data, spec.name))
    lo, hi = min(a, b), max(a, b)
    pad = 0.1 * (hi - lo)
    return list(np.linspace(lo + pad, hi - pad, n))


def _resolve_levels(ctx, non_proxy, cfg, seed_pkl, proxy_data):
    levels = {}
    for spec in non_proxy:
        want = cfg["levels"].get(spec.name, "auto")
        if isinstance(want, str) and want == "auto":
            levels[spec.name] = _auto_levels(ctx, spec, seed_pkl, cfg["auto_levels"],
                                             cfg["with_acoustics"], proxy_data)
        else:
            levels[spec.name] = [float(v) for v in want]
        print(f"  levels[{spec.name}] = "
              + ", ".join(f"{v:.4g}" for v in levels[spec.name]), flush=True)
    return levels


# --------------------------------------------------------------------------- #
# plot                                                                        #
# --------------------------------------------------------------------------- #

def _plot_surface(front, specs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s.name for s in specs]
    pairs = list(combinations(range(len(names)), 2))
    ncol = min(3, len(pairs)) or 1
    nrow = (len(pairs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, (i, j) in enumerate(pairs):
        ax = axes.flat[k]
        ax.set_visible(True)
        xi = [p[names[i]] for p in front]
        yj = [p[names[j]] for p in front]
        ck = [p[names[3 - i - j]] for p in front] if len(names) == 3 else None
        sc = ax.scatter(xi, yj, c=ck, s=28, cmap="viridis", edgecolor="0.3", linewidth=0.3)
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.grid(True, alpha=0.3)
        if ck is not None:
            fig.colorbar(sc, ax=ax, label=names[3 - i - j])
    fig.suptitle(f"hybrid surface -- {len(front)} non-dominated points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #

def _run_scout(ctx, case_dir, cfg, dry_run):
    scout_path = ctx.out("nsga2_result_hybrid_scout.pkl")
    if os.path.exists(scout_path) and not cfg["force_scout"]:
        print(f"reusing cached scout {scout_path}", flush=True)
        return pickle.load(open(scout_path, "rb"))
    kwargs = dict(pop_size=cfg["scout_pop"], n_gen=cfg["scout_gens"], seed=cfg["seed"],
                  algorithm="nsga2", save_tag="hybrid_scout",
                  objective_names=cfg["scout_objectives"])
    seed_from = cfg["scout_front_seed_from"]
    if dry_run and not seed_from:
        seed_from = ["../shahjahan_case1_fpp_explore_sm099"]
    if seed_from:
        kwargs["front_seed_from"] = [ctx.out(p) if p.startswith("..") else p for p in seed_from]
    nsga2_runner.run(case_dir, **kwargs)
    return pickle.load(open(scout_path, "rb"))


def run_hybrid(case_dir, *, dry_run=False):
    ctx = Context.load(case_dir)
    cfg = dict(DEFAULTS)
    cfg.update(ctx.case.get("hybrid", {}))
    if dry_run:
        cfg.update(DRY_RUN_OVERRIDES)

    specs = parse_objectives(ctx.case["objectives"])
    specs = [s for s in specs if s.role != "reserved"]
    proxy = next(s for s in specs if s.role == "proxy")
    non_proxy = [s for s in specs if s.role != "proxy"]
    names = [s.name for s in specs]
    fixed_pitch = bool(ctx.case["rotor"].get("fixed_pitch"))
    print(f"hybrid: proxy={proxy.name}  non_proxy={[s.name for s in non_proxy]}  "
          f"acoustics={cfg['with_acoustics']}  cross_check={cfg['cross_check']}", flush=True)

    scout = _run_scout(ctx, case_dir, cfg, dry_run)
    if not scout.get("final_population") or scout.get("n_feasible", 0) < 1:
        raise RuntimeError("scout did not produce a feasible population")
    anchors = _pick_anchors(scout, cfg["n_anchors"], obj_names=cfg["scout_objectives"] or names,
                            seed=cfg["seed"], cluster_method=cfg["cluster_method"])
    print(f"anchors: {len(anchors)}", flush=True)

    # which non-proxy objectives are swept (get their own edge stack)
    swept_specs = non_proxy if cfg["cross_check"] else non_proxy[:1]

    partial_path = ctx.out("pareto_surface_hybrid_partial.pkl")
    primary_pts, cross_pts = [], []
    levels = {}
    for k, anchor in enumerate(anchors):
        seed_path = ctx.out(f"_hybrid_anchor_{k}.pkl")
        nsga_geometry_to_seed(anchor.get("var_dict", anchor), seed_path, fixed_pitch=fixed_pitch)

        if not levels:
            # bracket the auto levels from the first anchor whose proxy solve
            # converges (a poor anchor seed can diverge it).
            try:
                _, proxy_data = seeded_anchor(ctx, proxy.name, seed_path,
                                              with_acoustics=cfg["with_acoustics"])
                levels = _resolve_levels(ctx, non_proxy, cfg, seed_path, proxy_data)
            except Exception as exc:                       # noqa: BLE001
                print(f"  anchor {k} level-bracket solve failed "
                      f"({type(exc).__name__}); trying the next anchor", flush=True)
                if k == len(anchors) - 1 and not levels:
                    raise RuntimeError(
                        "no anchor could bracket the auto levels; set explicit "
                        "CASE['hybrid']['levels']") from exc
                continue

        for si, swept in enumerate(swept_specs):
            pinned = [s for s in non_proxy if s.name != swept.name]
            combos = list(product(*[levels[s.name] for s in pinned])) or [()]
            for ci, combo in enumerate(combos):
                fixed_eps = {s.name: float(v) for s, v in zip(pinned, combo)}
                tag = f"hybrid_a{k}_s{si}_{ci}"
                print(f"\n=== anchor {k} | sweep {swept.name} | pinned {fixed_eps} ===", flush=True)
                # A bad anchor geometry can diverge an endpoint anchor solve, or a
                # pinned level can be infeasible -- neither should kill the run;
                # skip this edge and keep going. RuntimeError = anchor/corrector
                # solve failed; SystemExit = trace() rejected the pair.
                try:
                    res = pareto_tracer.trace(pareto_tracer.TraceConfig(
                        case_dir=case_dir, name_a=proxy.name, name_b=swept.name,
                        nsga_seed=seed_path, n_points=cfg["n_points"], predictor="secant",
                        tag=tag, fixed_epsilons=fixed_eps or None,
                        with_acoustics=cfg["with_acoustics"]))
                except (RuntimeError, SystemExit) as exc:
                    print(f"  edge SKIPPED ({type(exc).__name__}): "
                          f"{str(exc).splitlines()[0]}", flush=True)
                    continue
                except Exception as exc:                   # noqa: BLE001
                    print(f"  edge SKIPPED (unexpected {type(exc).__name__}): "
                          f"{str(exc).splitlines()[0]}", flush=True)
                    continue
                bucket = primary_pts if si == 0 else cross_pts
                for p in res.points:
                    row = {proxy.name: float(p.f[0]), swept.name: float(p.f[1])}
                    row.update(fixed_eps)
                    bucket.append(row)
                merged_so_far = _merge(primary_pts + cross_pts, specs)
                with open(partial_path, "wb") as f:
                    pickle.dump({"objective_labels": tuple(names), "front": merged_so_far,
                                 "partial": True}, f)
                print(f"  partial: {len(primary_pts) + len(cross_pts)} pts "
                      f"-> {len(merged_so_far)} nd  ({partial_path})", flush=True)

    all_pts = primary_pts + cross_pts
    merged = _merge(all_pts, specs)
    primary_ids = {id(p) for p in primary_pts}
    provenance = {
        "primary_in": len(primary_pts),
        "cross_in": len(cross_pts),
        "merged_nd": len(merged),
        "from_primary": sum(id(p) in primary_ids for p in merged),
        "from_cross": sum(id(p) not in primary_ids for p in merged),
    }
    out = ctx.out("pareto_surface_hybrid_run.pkl")
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": tuple(names), "n_points_per_edge": cfg["n_points"],
                     "levels": {s.name: [float(v) for v in levels.get(s.name, [])] for s in non_proxy},
                     "provenance": provenance, "front": merged}, f)
    try:
        _plot_surface(merged, specs, ctx.out("pareto_overlay_hybrid_run.png"))
    except Exception as exc:                       # noqa: BLE001 -- plot is optional
        print(f"plot skipped: {exc}", flush=True)
    print(f"\nsaved {out}\n  provenance {provenance}", flush=True)
    return out
