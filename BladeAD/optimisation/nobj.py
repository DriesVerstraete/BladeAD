"""The N-objective Pareto front via the picker-based `run_sweep_n` loop.

Objective-slot-driven (2026-09-06 refactor). Kept as:
  * the N > 3 path (the epsilon-grid staircase in `grid.py` is 2-D only), and
  * an optional gap-fill second pass after the grid for a 3-objective case
    (`three_obj_method="grid+nobj"`).

Every objective slot is mapped to a maximise-all sign convention
(`sign = +1` for a `goal:"max"` slot, `-1` for `"min"`) so `FrontPointN` /
`is_dominated_n` / `pick_next_target_n` work unchanged.

Interior sub-problems all route through `min_power_solve` (optimise the `proxy`
s.t. epsilon-constraints):
  * free slot == proxy      -> epsilons on every other slot at its picked target.
  * free slot != proxy      -> epsilons on every other slot at its picked target,
                               PLUS a LOOSE bound on the freed slot (the front's
                               current min for a floor / max for a cap) so
                               min_power_solve stays well-posed without the freed
                               slot's target over-constraining the shared
                               geometry (2026-09-06: tight interpolated floors
                               thrashed ~1/3 of free-FM legs to maxiter).

All the picker guards from the 2026-09-06 rewrite are retained
(`project_pareto_sweep_gap_picker_repeat_bug`): consumed-pair (never re-cleared),
near-duplicate skip, sentinel-region skip, budget = real points added, `it_cap`.
"""
import os
import pickle
from datetime import datetime

from ._sweep import (
    FrontPointN, pick_next_target_n, is_dominated_n, FrontExhausted,
)

from .sweep_common import (
    live_specs, proxy_spec, objective_anchor, min_power_solve, objective_value,
)

_SENTINEL = "INFEASIBLE_SUBPROBLEM"


def run_nobj_front(ctx, edge_fronts=()):
    n_sweep = ctx.sweep["n_points_3obj"]
    live = live_specs(ctx)
    proxy = proxy_spec(ctx)
    n = len(live)
    names = [s.name for s in live]
    sign = [1.0 if s.goal == "max" else -1.0 for s in live]
    proxy_idx = names.index(proxy.name)

    def _obj_tuple(data):
        return tuple(sign[k] * objective_value(data, names[k]) for k in range(n))

    def _fp(pkl_path, data, target_kind=None, target_values=None):
        return FrontPointN(objectives=_obj_tuple(data), target_kind=target_kind,
                           target_values=target_values or {},
                           meta={"pkl_path": pkl_path, "result": data})

    # --- anchors (one per slot) ---
    anchors = {name: objective_anchor(ctx, name) for name in names}
    front = [_fp(p, d) for p, d in anchors.values()]

    # --- seed the payoff cloud with the pairwise-edge points ---
    seen, seeds = set(), []
    for fr in edge_fronts:
        for pt in fr:
            path = pt.get("pkl_path")
            if not path or path == _SENTINEL or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            with open(path, "rb") as f:
                seeds.append(_fp(path, pickle.load(f)))
    for s in seeds:
        if not is_dominated_n(s, front):
            front = [p for p in front if not is_dominated_n(p, [s])]
            front.append(s)
    print(f"[nobj] anchors + {len(seeds)} edge seeds -> {len(front)}-point starting front")

    def _sentinel(free_index, other_targets):
        return FrontPointN(objectives=(-1e9,) * n, target_kind=free_index,
                           target_values=dict(other_targets), meta={"pkl_path": _SENTINEL})

    def solve_epsilon_constraint(free_index, other_targets):
        epsilons = {names[k]: sign[k] * tv for k, tv in other_targets.items()}
        if free_index != proxy_idx:
            raw = [sign[free_index] * p.objectives[free_index] for p in front]
            loose = min(raw) if live[free_index].role == "floor" else max(raw)
            epsilons[names[free_index]] = loose
        try:
            pkl, data = min_power_solve(ctx, epsilons)
        except RuntimeError as exc:
            print(f"  free[{free_index}] {other_targets} -> INFEASIBLE, sentinel ({exc})")
            return _sentinel(free_index, other_targets)
        return _fp(pkl, data, target_kind=free_index, target_values=dict(other_targets))

    NEAR_DUP = 5e-3
    SENTINEL_SKIP = 0.02

    def _ranges():
        return [max(1e-9, max(p.objectives[k] for p in front) - min(p.objectives[k] for p in front))
                for k in range(n)]

    def _near_dup(cand, ranges):
        return any(all(abs(a - b) / r < NEAR_DUP for a, b, r in zip(p.objectives, cand.objectives, ranges))
                   for p in front)

    def _near_sentinel(free_index, other_targets, ranges):
        for s_free, s_targets in sentinel_targets:
            if s_free == free_index and set(s_targets) == set(other_targets) and all(
                    abs(other_targets[k] - s_targets[k]) / ranges[k] < SENTINEL_SKIP for k in other_targets):
                return True
        return False

    excluded_pairs, tried, sentinel_targets = set(), set(), []
    made, it, it_cap = 0, 0, 8 * n_sweep
    while made < n_sweep and it < it_cap:
        it += 1
        try:
            free_index, other_targets, pair_key = pick_next_target_n(
                front, excluded_pairs=excluded_pairs, return_pair=True)
        except FrontExhausted:
            print(f"[nobj] all real gaps exhausted, stopping at {made}/{n_sweep} points")
            break
        excluded_pairs.add(pair_key)
        target_key = (free_index, tuple(sorted((k, round(v, 4)) for k, v in other_targets.items())))
        if target_key in tried:
            continue
        if _near_sentinel(free_index, other_targets, _ranges()):
            continue
        tried.add(target_key)
        print(f"[nobj] point {made + 1}/{n_sweep} (iter {it}): free obj[{free_index}]"
              f"={names[free_index]}, constrain {other_targets}")
        candidate = solve_epsilon_constraint(free_index, other_targets)
        print(f"  objectives={tuple(round(v, 4) for v in candidate.objectives)}")
        if candidate.meta.get("pkl_path") == _SENTINEL:
            sentinel_targets.append((free_index, dict(other_targets)))
            continue
        if is_dominated_n(candidate, front) or _near_dup(candidate, _ranges()):
            print("  -> dominated / near-duplicate, dropped")
            continue
        front = [p for p in front if not is_dominated_n(p, [candidate])]
        front.append(candidate)
        made += 1
    else:
        if it >= it_cap:
            print(f"[nobj] iteration cap ({it_cap}) hit at {made}/{n_sweep} points")
    front.sort(key=lambda p: p.objectives[0])

    print("\nFinal N-objective front:")
    hdr = "  ".join(f"{nm:>12}" for nm in names)
    print(f"{'target':>10}  {hdr}")
    for pt in front:
        tgt = "(anchor)" if pt.target_kind is None else f"free[{pt.target_kind}]"
        vals = "  ".join(f"{sign[k] * pt.objectives[k]:>12.4f}" for k in range(n))
        print(f"{tgt:>10}  {vals}")

    tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ctx.out(f"pareto_front_nobj_{tag}.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "objective_labels": tuple(names),
            "n_sweep_points": n_sweep,
            "front": [{**{names[k]: sign[k] * pt.objectives[k] for k in range(n)},
                       "target_kind": pt.target_kind, "target_values": pt.target_values,
                       "pkl_path": pt.meta["pkl_path"]} for pt in front]}, f)
    print(f"\nSaved N-objective front to {out}")
    return front, out
