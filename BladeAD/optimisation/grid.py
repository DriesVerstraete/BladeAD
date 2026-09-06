"""Structured 2-D epsilon-grid 3-objective front -- the primary method for a
3-objective case (PI 2026-09-06, after the shared `run_sweep_n` picker failed
four ways). Objective-slot-driven (2026-09-06 refactor).

A 3-objective case has one `proxy` slot and two non-proxy slots. The front is
parameterised by epsilons on the two non-proxy slots:

    optimise the proxy   s.t.   f_col-slot epsilon   AND   f_stair-slot epsilon

with the proxy value and the blade geometry falling out. Every node is one
`min_power_solve`.

* `f_col` -- the `floor` non-proxy slot. Its lattice lines come from
  `_lattice_lines` (index-quantiles of the pairwise-edge points' own values, so
  a knee-adaptive edge's clustering propagates into the grid; uniform w/o edges).
* `f_stair` -- the other non-proxy slot. Per `f_col` value: one *unconstrained*
  solve gives that blade's natural `f_stair` value `v_nat`; then `num_stair`
  epsilons are swept from just inside `v_nat` toward `f_stair`'s global extreme
  (min for a `cap`, loosen for a `floor`) -- so every stair node genuinely bites.

**Staircase, not a rectangle**: a rectangular lattice collapses every loose node
onto the same point. Warm-chained within a column; the column's infeasible tail
(an epsilon too tight to hold `f_col` + the thrust equalities) ends it early --
but not until a stair epsilon has been feasible and then a tighter one fails
(2026-09-06 first-step-retry fix).

Resumable: each node's `stage1_*` pkl is written by the solver, so a killed run
re-uses completed nodes (`reuse_nodes=True`).

N > 3 (>2 non-proxy slots) is OUT OF SCOPE here -- the orchestrator routes it to
`nobj`. `run_grid_front` raises if called with >2 non-proxy slots.
"""
import glob
import pickle
from datetime import datetime

import numpy as np

from .case import epsilon_tag
from .sweep_common import (
    live_specs, proxy_spec, objective_anchor, min_power_solve, objective_value,
)


def _point(ctx, data, pkl, eps=None, anchor=None):
    p = {s.name: objective_value(data, s.name) for s in live_specs(ctx)}
    p["pkl_path"] = pkl
    if eps:
        p["eps"] = {k: float(v) for k, v in eps.items()}
    if anchor:
        p["anchor"] = anchor
    return p


def _existing_node(ctx, eps):
    """A previously-solved node matching `eps` (dict) to tag precision that
    converged feasibly, else None."""
    tag = epsilon_tag(eps)
    for path in glob.glob(ctx.out(f"stage1_*{tag}*.pkl")):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            continue
        if all(data.get("constraint_checks", {}).values()) and data.get("objective_values"):
            return path, data
    return None


def _lattice_lines(lo, hi, n, edge_values):
    """n lattice lines on [lo, hi] -- at index-quantiles of the sorted unique
    `edge_values` clamped to [lo, hi] if enough are given, else uniform."""
    vals = sorted({float(v) for v in edge_values if lo - 1e-9 <= v <= hi + 1e-9})
    if len(vals) < max(3, n):
        return np.linspace(lo, hi, n)
    vals = sorted(set(round(v, 6) for v in ([lo] + vals + [hi])))
    idx = np.unique(np.linspace(0, len(vals) - 1, n).round().astype(int))
    return np.array([vals[i] for i in idx])


def run_grid_front(ctx, edge_fronts=(), reuse_nodes=True):
    live = live_specs(ctx)
    proxy = proxy_spec(ctx)
    non_proxy = [s for s in live if s.role != "proxy"]
    if len(non_proxy) != 2:
        raise ValueError(
            f"run_grid_front needs exactly 2 non-proxy slots, got {len(non_proxy)} "
            f"({[s.name for s in non_proxy]}); route N>3 to nobj.run_nobj_front")

    floors = [s for s in non_proxy if s.role == "floor"]
    f_col = floors[0] if floors else non_proxy[0]
    f_stair = non_proxy[1] if non_proxy[0] is f_col else non_proxy[0]
    n_col = ctx.sweep.get("grid_fm", 8)
    n_stair = ctx.sweep.get("grid_noise", 8)

    anchors = {s.name: objective_anchor(ctx, s.name) for s in live}
    col_vals = [objective_value(d, f_col.name) for _, d in anchors.values()]
    stair_vals = [objective_value(d, f_stair.name) for _, d in anchors.values()]
    col_lo, col_hi = min(col_vals), max(col_vals)
    if f_col.role == "cap":
        col_lo, col_hi = col_hi, col_lo
    stair_extreme = min(stair_vals) if f_stair.role == "cap" else max(stair_vals)
    # loose end of the stair slot -- passed on the uncapped column solve so a
    # slack `f_stair` inequality stays in the SLSQP graph (frozen rotor_pareto
    # passed `noise_hi`; dropping it fails the marginal lowest-f_col column --
    # 2026-09-07). A cap's loose end is its max, a floor's is its min.
    stair_loose = max(stair_vals) if f_stair.role == "cap" else min(stair_vals)

    edge_col = [p[f_col.name] for fr in edge_fronts for p in fr if f_col.name in p]
    col_lines = _lattice_lines(col_lo, col_hi, n_col, edge_col)
    spacing = "edge-quantile" if edge_col else "uniform"
    print(f"\n[grid] staircase: {len(col_lines)} {f_col.name} lines ({spacing}) x up to "
          f"{n_stair} {f_stair.name} epsilons  |  {f_stair.name} extreme {stair_extreme:.3f}")
    print(f"  {f_col.name} lines {np.round(col_lines, 4).tolist()}")

    front = [_point(ctx, d, p, anchor=name) for name, (p, d) in anchors.items()]
    n_est = len(col_lines) * (n_stair + 1)
    n_visited = n_solved = n_reused = n_infeasible = 0

    def _node(eps, warm, why):
        nonlocal n_visited, n_solved, n_reused, n_infeasible
        n_visited += 1
        tag = f"[grid {n_visited}/~{n_est}]"
        hit = _existing_node(ctx, eps) if reuse_nodes else None
        if hit:
            n_reused += 1
            print(f"  {tag} {eps} {why}  ->  {proxy.name}="
                  f"{objective_value(hit[1], proxy.name):.4f}  (reused)")
            return hit
        try:
            pkl, data = min_power_solve(ctx, eps, warm=warm)
        except RuntimeError:
            n_infeasible += 1
            print(f"  {tag} {eps} {why} infeasible")
            return None
        n_solved += 1
        print(f"  {tag} {eps} {why}  ->  " + "  ".join(
            f"{s.name}={objective_value(data, s.name):.4f}" for s in non_proxy))
        return pkl, data

    for cv in col_lines:
        got = _node({f_col.name: float(cv), f_stair.name: float(stair_loose)},
                    anchors[f_col.name][0], "uncapped")
        if got is None:
            continue
        pkl, data = got
        v_nat = objective_value(data, f_stair.name)
        front.append(_point(ctx, data, pkl, eps={f_col.name: float(cv)}))
        if abs(v_nat - stair_extreme) < 1.0:
            continue
        col_warm, col_had_feasible = None, False
        for sv in np.linspace(v_nat, stair_extreme, n_stair + 1)[1:]:
            eps = {f_col.name: float(cv), f_stair.name: float(sv)}
            got = _node(eps, col_warm, f"{f_stair.name}~{sv:.2f}")
            if got is None:
                if col_had_feasible:
                    break
                continue
            pkl, data = got
            col_warm, col_had_feasible = pkl, True
            front.append(_point(ctx, data, pkl, eps=eps))

    # dedup exact-coincident points
    seen, uniq = set(), []
    for p in front:
        k = tuple(round(p[s.name], 5) for s in live)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    uniq.sort(key=lambda p: tuple(p[s.name] for s in non_proxy))

    tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ctx.out(f"pareto_front_3obj_grid_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "objective_labels": tuple(s.name for s in live),
            "method": "epsilon-grid-staircase", "grid": [n_col, n_stair],
            "col_slot": f_col.name, "stair_slot": f_stair.name,
            "col_lines": np.round(col_lines, 6).tolist(),
            "n_solved": n_solved, "n_reused": n_reused, "n_infeasible": n_infeasible,
            "front": uniq}, f)
    print(f"\n[grid] {len(uniq)} points ({n_solved} solved, {n_reused} reused, "
          f"{n_infeasible} infeasible nodes)\n[grid] saved {out}")
    return uniq, out
