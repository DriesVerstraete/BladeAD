"""Scout a rotor basin with NSGA-II and refine it with the Pareto tracer."""
import os
import pickle
from itertools import combinations

import numpy as np

from . import nsga2_runner, pareto_tracer
from .case import Context, parse_objectives
from .sweep_common import nsga_geometry_to_seed

DEFAULTS = {"n_anchors": 3, "scout_pop": 48, "scout_gens": 60, "seed": 1,
            "oei_levels": [1.143, 1.33, 1.52, 1.70, 1.85],
            "n_points": 12, "cluster_method": "kmeans", "force_scout": False,
            "cross_check": True, "cross_levels": [19300.0, 20000.0, 21000.0]}


def _objective_vector(point, obj_names):
    raw = point.get("perf", {}).get("_raw_obj", point.get("raw_obj"))
    return np.asarray(raw, float)[:len(obj_names)]


def _pick_anchors(result_pkl, n, *, obj_names, seed=1, cluster_method="kmeans"):
    result = pickle.load(open(result_pkl, "rb")) if isinstance(result_pkl, (str, os.PathLike)) else result_pkl
    feasible = [p for p in result.get("final_population", []) if p.get("feasible", float(p.get("cons_sum", 1)) <= 1e-6)]
    nd = result.get("front") or feasible
    if not feasible or not nd or n <= 0:
        return []
    all_x = np.array([_objective_vector(p, obj_names) for p in feasible])
    lo, span = all_x.min(0), np.ptp(all_x, axis=0)
    span[span == 0] = 1.0
    norm = lambda xs: (np.asarray(xs) - lo) / span
    nd_x = norm([_objective_vector(p, obj_names) for p in nd])
    selected = []
    if cluster_method == "kmeans" and len(nd) >= n:
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n, random_state=seed, n_init=10).fit(nd_x).cluster_centers_
            selected = [nd[int(np.argmin(np.linalg.norm(nd_x - c, axis=1)))] for c in labels]
        except ImportError:
            selected = [nd[i * len(nd) // n] for i in range(n)]
    else:
        selected = list(nd)
    if len(selected) > 1:
        spacing = [np.linalg.norm(norm(_objective_vector(a, obj_names)) -
                                   norm(_objective_vector(b, obj_names)))
                   for a, b in combinations(selected, 2)]
        if min(spacing) < 0.05:
            selected = []
    unique = []
    for p in selected:
        if not any(p.get("var_dict") == q.get("var_dict") for q in unique): unique.append(p)
    if len(unique) >= n:
        required = []
        for j in range(len(obj_names)):
            for extreme in (int(np.argmin(all_x[:, j])), int(np.argmax(all_x[:, j]))):
                candidate = feasible[extreme]
                if not any(candidate.get("var_dict") == q.get("var_dict") for q in required):
                    required.append(candidate)
        return (required + [q for q in unique if not any(q.get("var_dict") == r.get("var_dict") for r in required)])[:n]
    # Add whole-feasible objective extremes when the ND front has collapsed.
    for j in range(len(obj_names)):
        idx = int(np.argmin(all_x[:, j]))
        if not any(feasible[idx].get("var_dict") == q.get("var_dict") for q in unique): unique.append(feasible[idx])
        idx = int(np.argmax(all_x[:, j]))
        if not any(feasible[idx].get("var_dict") == q.get("var_dict") for q in unique): unique.append(feasible[idx])
    return unique[:n]


def _dominates(a, b, specs):
    av = [a[s.name] for s in specs]; bv = [b[s.name] for s in specs]
    aa = [-v if s.goal == "max" else v for v, s in zip(av, specs)]
    bb = [-v if s.goal == "max" else v for v, s in zip(bv, specs)]
    return all(x <= y for x, y in zip(aa, bb)) and any(x < y for x, y in zip(aa, bb))


def _merge(points, specs):
    return [p for i, p in enumerate(points) if not any(_dominates(q, p, specs) for j, q in enumerate(points) if i != j)]


def run_hybrid(case_dir, *, dry_run=False):
    ctx = Context.load(case_dir)
    cfg = dict(DEFAULTS)
    cfg.update(ctx.case.get("hybrid", {}))
    if dry_run:
        cfg.update(scout_pop=24, scout_gens=6, oei_levels=[1.143, 1.52],
                   cross_levels=[19300.0, 20500.0], n_points=3)
    scout_path = ctx.out("nsga2_result_hybrid_scout.pkl")
    if os.path.exists(scout_path) and not cfg["force_scout"]:
        print(f"reusing cached scout {scout_path}")
    else:
        kwargs = dict(pop_size=cfg["scout_pop"], n_gen=cfg["scout_gens"],
                      seed=cfg["seed"], algorithm="nsga2", save_tag="hybrid_scout")
        if dry_run:
            kwargs["front_seed_from"] = [ctx.out("../shahjahan_case1_fpp_explore_sm099")]
        nsga2_runner.run(case_dir, **kwargs)
    scout = pickle.load(open(scout_path, "rb"))
    if not scout.get("final_population") or scout.get("n_feasible", 0) < 1: raise RuntimeError("scout did not produce a feasible population")
    specs = parse_objectives(ctx.case["objectives"]); names = [s.name for s in specs]
    anchors = _pick_anchors(scout, cfg["n_anchors"], obj_names=names, seed=cfg["seed"], cluster_method=cfg["cluster_method"])
    primary = []
    cross = []
    for k, anchor in enumerate(anchors):
        seed_path = ctx.out(f"_hybrid_anchor_{k}.pkl")
        nsga_geometry_to_seed(anchor, seed_path, fixed_pitch=bool(ctx.case["rotor"].get("fixed_pitch")))
        if len(names) == 3:
            pkl = pareto_tracer.trace_interior(case_dir, seed_path, cfg["oei_levels"], n_points=cfg["n_points"], tag=f"hybrid_a{k}")
            primary.extend(pickle.load(open(pkl, "rb"))["front"])
            if cfg["cross_check"]:
                for level in cfg["cross_levels"]:
                    try:
                        result = pareto_tracer.trace(pareto_tracer.TraceConfig(
                            case_dir, "cruise_elec", "oei_margin", seed_path,
                            n_points=cfg["n_points"], predictor="secant",
                            tag=f"hybrid_a{k}_x{level:.0f}",
                            fixed_epsilons={"hover_elec": float(level)}))
                    except SystemExit as exc:
                        print(f"layer {level} SKIPPED: {exc}")
                        continue
                    for point in result.points:
                        cross.append({"hover_elec": float(level),
                                      "cruise_elec": float(point.f[0]),
                                      "oei_margin": float(point.f[1])})
        else:
            r = pareto_tracer.trace(pareto_tracer.TraceConfig(case_dir, names[0], names[1], seed_path, n_points=cfg["n_points"], tag=f"hybrid_a{k}"))
            primary.extend([{names[0]: float(p.f[0]), names[1]: float(p.f[1])} for p in r.points])
    out = ctx.out("pareto_surface_hybrid_run.pkl")
    points = primary + cross
    merged = _merge(points, specs)
    primary_ids = {id(point) for point in primary}
    provenance = {"primary_in": len(primary), "cross_in": len(cross),
                  "merged_nd": len(merged),
                  "from_primary": sum(id(point) in primary_ids for point in merged),
                  "from_cross": sum(id(point) not in primary_ids for point in merged)}
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": tuple(names),
                     "oei_levels": cfg["oei_levels"] if len(names) == 3 else [],
                     "n_points_per_layer": cfg["n_points"], "front": merged,
                     "provenance": provenance}, f)
    pareto_tracer.overlay(case_dir, surface_pkl=out, out_path=ctx.out("pareto_overlay_hybrid_run.png"), title=f"{os.path.basename(case_dir)}: hybrid")
    print(f"saved {out}: {len(points)} -> {len(merged)} points")
    return out
