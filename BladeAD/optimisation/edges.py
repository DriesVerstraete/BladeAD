"""One generic pairwise 2-objective edge front.

`run_edge(ctx, name_a, name_b)` traces the trade-off between two objective slots
as a monotone 1-parameter warm-chained sweep. Objective-agnostic replacement for
the frozen `run_fm_eta_edge` / `run_fm_noise_edge` / `run_eta_noise_edge`.

Direction picking (from the slot roles):

* one of the pair is the `proxy`  -> sweep the OTHER slot's epsilon, each point
  = `min_power_solve` (optimise the proxy s.t. that epsilon).
* neither is the proxy (floor x cap)  -> sweep `name_a`'s epsilon, each point =
  `direct_solve` optimising `name_b` (max a floor / min a cap) s.t. that epsilon.

The sweep runs from the sweep-slot's own single-objective extreme to the value
it takes at the other slot's extreme. All points run with acoustics active, so
the endpoints are exactly the shared `objective_anchor`s.

`adaptive=True` (auto for a proxy-vs-floor pair -- the sharp-knee case): instead
of a fixed linspace, repeatedly solve at the epsilon midpoint of the adjacent
solved pair with the largest normalised gap in the two pair objectives, warm
from the nearer neighbour. A never-cleared `infeasible_gaps` set stops the
picker re-probing a feasibility wall (todos.md 2026-09-06).

Each call returns (front, out_pkl_path); front is a list of per-point dicts, each
carrying every live objective slot's value by name, plus `eps` and `pkl_path`.
"""
import pickle
from datetime import datetime

import numpy as np

from .sweep_common import (
    live_specs, proxy_spec, spec_by_name, objective_anchor,
    min_power_solve, direct_solve, objective_value,
)


def _point(ctx, data, pkl, eps=None, anchor=None):
    p = {s.name: objective_value(data, s.name) for s in live_specs(ctx)}
    p["pkl_path"] = pkl
    if eps is not None:
        p["eps"] = float(eps)
    if anchor:
        p["anchor"] = anchor
    return p


def _save(ctx, name, labels, n_points, front):
    tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ctx.out(f"pareto_front_{name}_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": labels, "n_points": n_points, "front": front}, f)
    print(f"\nSaved {out}")
    return out


def _largest_norm_gap(pts, name_a, name_b, exclude):
    """Adjacent pair (sorted by eps) with the largest normalised
    sqrt((da/a_range)^2 + (db/b_range)^2), skipping excluded eps-midpoints."""
    ordered = sorted(pts, key=lambda p: p["eps"])
    ra = (max(p[name_a] for p in ordered) - min(p[name_a] for p in ordered)) or 1.0
    rb = (max(p[name_b] for p in ordered) - min(p[name_b] for p in ordered)) or 1.0
    best, best_g = None, -1.0
    for a, b in zip(ordered[:-1], ordered[1:]):
        key = round(0.5 * (a["eps"] + b["eps"]), 5)
        if key in exclude:
            continue
        g = (((a[name_a] - b[name_a]) / ra) ** 2 + ((a[name_b] - b[name_b]) / rb) ** 2) ** 0.5
        if g > best_g:
            best, best_g = (a, b), g
    return best


def run_edge(ctx, name_a, name_b, adaptive=None):
    n_points = ctx.sweep["n_points"]
    proxy = proxy_spec(ctx)
    pair = (name_a, name_b)
    non_proxy = [n for n in pair if n != proxy.name]

    if len(non_proxy) == 1:                       # proxy vs one other -> sweep the other
        sweep, other, driver = non_proxy[0], proxy.name, None
    else:                                         # floor x cap -> sweep a, drive b
        sweep, other, driver = name_a, name_b, name_b
    if adaptive is None:
        adaptive = (proxy.name in pair
                    and any(spec_by_name(ctx, n).role == "floor" for n in pair))

    sw_pkl, sw_data = objective_anchor(ctx, sweep)
    ot_pkl, ot_data = objective_anchor(ctx, other)
    hi = objective_value(sw_data, sweep)          # sweep slot at its own extreme
    lo = objective_value(ot_data, sweep)          # sweep slot at the other's extreme
    role = spec_by_name(ctx, sweep).role
    print(f"\nedge {name_a} vs {name_b}: sweep {sweep} ({role}) {hi:.4f} -> {lo:.4f}"
          f"  driver={driver or proxy.name}  adaptive={adaptive}  budget {n_points}")

    def _solve_at(eps, warm):
        if driver is None:
            return min_power_solve(ctx, {sweep: eps}, warm=warm)
        return direct_solve(ctx, driver, {sweep: eps}, warm=warm)

    front = [_point(ctx, sw_data, sw_pkl, anchor=sweep),
             _point(ctx, ot_data, ot_pkl, anchor=other)]

    if not adaptive:
        warm = sw_pkl
        for eps in np.linspace(hi, lo, n_points + 2)[1:-1]:
            try:
                pkl, data = _solve_at(eps, warm)
            except RuntimeError as exc:
                print(f"  {sweep}~{eps:.4f} infeasible, skip ({exc})")
                continue
            warm = pkl
            front.append(_point(ctx, data, pkl, eps=eps))
            print(f"  {sweep}~{eps:.4f}  ->  " + "  ".join(
                f"{n}={front[-1][n]:.4f}" for n in (name_a, name_b)))
    else:
        solved = [{**front[0], "eps": hi}, {**front[1], "eps": lo}]
        excluded, infeasible_gaps = set(), set()
        for i in range(n_points):
            pick = _largest_norm_gap(solved, name_a, name_b, excluded | infeasible_gaps)
            if pick is None:
                print(f"  all gaps exhausted at interior solve {i}/{n_points}")
                break
            a, b = pick
            eps = 0.5 * (a["eps"] + b["eps"])
            key = round(eps, 5)
            if any(abs(eps - p["eps"]) < 1e-3 for p in solved):
                excluded.add(key)
                continue
            warm = a["pkl_path"] if abs(a["eps"] - eps) <= abs(b["eps"] - eps) else b["pkl_path"]
            try:
                pkl, data = _solve_at(eps, warm)
            except RuntimeError as exc:
                print(f"  {sweep}~{eps:.4f} infeasible, gap retired ({exc})")
                infeasible_gaps.add(key)
                continue
            solved.append({**_point(ctx, data, pkl, eps=eps), "eps": float(eps)})
            excluded.clear()                       # front changed -> stale gap ids; infeasible_gaps kept
            print(f"  [{i + 1}/{n_points}] {sweep}~{eps:.4f}  ->  " + "  ".join(
                f"{n}={solved[-1][n]:.4f}" for n in (name_a, name_b)))
        front = solved

    front.sort(key=lambda p: p[name_a])
    labels = (spec_by_name(ctx, name_a).label, spec_by_name(ctx, name_b).label)
    out = _save(ctx, f"{name_a}__{name_b}", labels, n_points, front)
    return front, out
