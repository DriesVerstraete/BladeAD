"""Generic epsilon-constraint Pareto front sweep: payoff-table anchors +
adaptive gap-bisection, for a SEQUENCE of independent single-objective
solves (not population-based).

Vendored 2026-09-07 into `BladeAD.optimisation` from AircraftDesign's
`optimisation_framework/optimisation/util/pareto_front_sweep.py` (the lazy
`uncrowded_hypervolume` import now points at the sibling `_hypervolume` module).
Originally ported 2026-08-12 from the SPL rotor-optimisation project
(`06-rotor-optimisation/code/rotor_bridge/pareto_front_sweep.py`) after being
proven twice on a real problem there
(a real gap-metric normalization bug found and fixed, then a symmetric
direction-flip capability added and validated) -- see that project's
`decisions/2026-08-11-multipoint-pareto-front-strategy.md` and
`decisions/2026-08-12-epsilon-constraint-pareto-front-implemented.md` for
the full research trail and real-world validation behind this design.

Bi-objective only (2 objectives). Deliberately NOT a population-based
Algorithm (unlike CMAES/COMOCMAES in this same `algorithms/` family) --
this is a sequential orchestrator that calls an arbitrary external
single-objective solve (SLSQP, a closed-form solve, anything) once per
front point. It does not use `Problem`/`Evaluator`/`Algorithm` at all.

It only needs FOUR callbacks that each run ONE scalar-objective-plus-
constraint problem and return a FrontPoint:
  - solve_obj1_only()                -> FrontPoint  (payoff-table anchor #1,
        maximize obj1, obj2 unconstrained)
  - solve_obj2_only(pin_from)        -> FrontPoint  (anchor #3 when
        pin_from=None; the anchor #1 REFINEMENT step when pin_from is given
        -- maximize obj2 alone, whatever obj1's solve fixed held constant)
  - solve_epsilon_constraint_obj2(target) -> FrontPoint  (maximize obj1
        subject to obj2 >= target)
  - solve_epsilon_constraint_obj1(target) -> FrontPoint  (the SYMMETRIC
        flip: maximize obj2 subject to obj1 >= target. Real motivation, not
        a hypothetical: a front can be steep in EITHER objective in
        different regions. Constraining the steep objective directly and
        freeing the flat one is better-conditioned there than the reverse;
        this callback is what makes that possible.)

obj1 is the objective free-maximized by solve_epsilon_constraint_obj2. obj2
is free-maximized by solve_epsilon_constraint_obj1. Which one gets
constrained at each sweep point is decided PER GAP by `pick_next_target`
below, not fixed for the whole sweep.

Gap-selection is a swappable function (`pick_next_target`) -- 2D normalized-
Euclidean-gap bisection today (real, tested, working, direction-aware).

**2026-09-05: N-objective (N >= 2) extension added below** (`FrontPointN`,
`is_dominated_n`, `pick_next_target_n`, `run_sweep_n`) -- the
hypervolume-contribution scorer this docstring used to flag as a future
drop-in, now built for a real rotor-project 3rd-objective use case. Reuses
`uncrowded_hypervolume.py`'s already-tested UHVI math. The bi-objective
functions above are untouched and still the ones existing callers use;
the N-objective functions are a separate, parallel API in this same file,
not a replacement. See that section's own comment for what is deliberately
NOT carried over (the lexicographic anchor-refinement step).
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# `uncrowded_hypervolume` imports `pygmo`, a heavy native dependency (pulls in pagmo/ipopt/MKL)
# that has caused real, disruptive breakage in at least one lean environment that consumes this
# module (`rotor_design`, 2026-09-05: installing pygmo there bumped numpy to an incompatible
# version AND introduced an OpenMP-runtime conflict crashing every csdl_alpha run). Deferred to
# inside `pick_next_target_n` itself so that any caller using ONLY the bi-objective API above
# (the vast majority of existing callers) never needs pygmo importable at all.


@dataclasses.dataclass
class FrontPoint:
    """One solved point on the front. `target_kind` is "obj1"/"obj2" for a
    real sweep point (which objective was epsilon-constrained to
    `target_value`) or None for the two anchors. `meta` carries whatever the
    caller wants attached (a result path, a full result dict, ...) -- this
    module never inspects it."""
    obj1: float
    obj2: float
    target_kind: Optional[str] = None
    target_value: Optional[float] = None
    meta: dict = dataclasses.field(default_factory=dict)


class FrontExhausted(Exception):
    """Raised by `pick_next_target`/`pick_next_target_n` when every real gap
    between existing front points has already been tried and excluded --
    the sweep has nothing genuinely new left to sample. `run_sweep`/
    `run_sweep_n` catch this and stop early rather than loop forever
    re-proposing a saturated gap (the 2026-09-05 repeat-target bug)."""


def _pair_key(*points) -> tuple:
    """Stable identity for the adjacent/originating pair that produced a
    gap, order-independent, robust to tiny float noise. Used as the key in
    a sweep's `excluded_pairs` memory."""
    coords = tuple(
        tuple(round(v, 9) for v in (p.obj1, p.obj2))
        if isinstance(p, FrontPoint)
        else tuple(round(v, 9) for v in p.objectives)
        for p in points
    )
    return tuple(sorted(coords))


def is_dominated(candidate: FrontPoint, front: List[FrontPoint]) -> bool:
    """True if some point in `front` is >= candidate on BOTH objectives, and
    strictly greater on at least one (maximize-both convention -- whichever
    objective is epsilon-constrained during a given solve, the front itself
    should still be Pareto-clean)."""
    for p in front:
        if p.obj1 >= candidate.obj1 and p.obj2 >= candidate.obj2 \
                and (p.obj1 > candidate.obj1 or p.obj2 > candidate.obj2):
            return True
    return False


def pick_next_target(
    front: List[FrontPoint],
    excluded_pairs: Optional[set] = None,
    return_pair: bool = False,
):
    """2D Euclidean-gap bisection, NORMALIZED by each objective's own
    front-wide range before computing distance -- un-normalized raw units
    silently let whichever objective happens to span a larger absolute
    range dominate the gap metric (confirmed real on the rotor project's
    front prop_rotor case: obj1 spanned ~0.02, obj2 spanned ~0.12, so raw
    distance was effectively pure obj2-bisection and under-resolved a
    visually steep, obj1-dominated segment). Sort the front by obj2, find
    the largest gap between adjacent points in NORMALIZED space.

    DIRECTION-AWARE: within that largest gap, also decide WHICH objective
    to epsilon-constrain -- whichever one has the LARGER normalized delta
    across the gap is the locally steep one; constraining it directly (and
    freeing the other) is the better-conditioned solve. Returns
    (direction, target_value) where direction is "obj2" (constrain obj2,
    maximize obj1) or "obj1" (the symmetric flip: constrain obj1, maximize
    obj2). Swappable -- replace with a hypervolume-contribution selector
    for 3+ objectives (see module docstring).

    `excluded_pairs`: a set of `_pair_key(a, b)` values whose gap has
    already been tried and led nowhere this sweep -- skipped when scanning,
    so a saturated gap is never re-proposed (2026-09-05 repeat-target bug
    fix). If every gap is excluded, raises `FrontExhausted`.
    `return_pair`: when True, append the winning gap's `_pair_key` to the
    return tuple so a caller (`run_sweep`) can record it on failure. Off by
    default to keep the historical 2-tuple return for other callers."""
    if len(front) < 2:
        raise ValueError("need at least 2 front points (the two anchors) to pick a gap")
    excluded_pairs = excluded_pairs or set()
    ordered = sorted(front, key=lambda p: p.obj2)
    obj1_vals = [p.obj1 for p in ordered]
    obj2_vals = [p.obj2 for p in ordered]
    obj1_range = max(obj1_vals) - min(obj1_vals) or 1.0  # guard against a degenerate flat front
    obj2_range = max(obj2_vals) - min(obj2_vals) or 1.0
    best_gap, best_a, best_b = -1.0, None, None
    for a, b in zip(ordered[:-1], ordered[1:]):
        if _pair_key(a, b) in excluded_pairs:
            continue
        gap = (((a.obj1 - b.obj1) / obj1_range) ** 2 + ((a.obj2 - b.obj2) / obj2_range) ** 2) ** 0.5
        if gap > best_gap:
            best_gap, best_a, best_b = gap, a, b
    if best_a is None:
        raise FrontExhausted("every gap between existing front points has been tried and excluded")
    d1_norm = abs(best_a.obj1 - best_b.obj1) / obj1_range
    d2_norm = abs(best_a.obj2 - best_b.obj2) / obj2_range
    pair_key = _pair_key(best_a, best_b)
    if d1_norm > d2_norm:
        result = ("obj1", 0.5 * (best_a.obj1 + best_b.obj1))
    else:
        result = ("obj2", 0.5 * (best_a.obj2 + best_b.obj2))
    return (*result, pair_key) if return_pair else result


def run_sweep(
    solve_obj1_only: Callable[[], FrontPoint],
    solve_obj2_only: Callable[[Optional[object]], FrontPoint],
    solve_epsilon_constraint_obj2: Callable[[float], FrontPoint],
    solve_epsilon_constraint_obj1: Callable[[float], FrontPoint],
    obj1_anchor_pin_getter: Callable[[FrontPoint], object],
    n_sweep_points: int,
) -> List[FrontPoint]:
    """Runs the full payoff-table + gap-bisection procedure. Returns the
    final Pareto front, sorted by obj2 ascending (anchors + sweep points,
    dominated candidates dropped).

    obj1_anchor_pin_getter: given the obj1-only anchor's FrontPoint, returns
    whatever `solve_obj2_only` needs to run the refinement step (e.g. a
    result-file path, or the anchor point itself) -- a callback because the
    "pin" representation is entirely up to the caller's solve functions,
    this module never inspects it.
    """
    print("[pareto_front_sweep] Step A: obj1-only anchor (maximize obj1)")
    anchor_obj1 = solve_obj1_only()
    print(f"  obj1={anchor_obj1.obj1:.4f}  obj2={anchor_obj1.obj2:.4f}")

    print("[pareto_front_sweep] Step B: refinement (pin obj1's solution, maximize obj2)")
    pin = obj1_anchor_pin_getter(anchor_obj1)
    obj2_lo_point = solve_obj2_only(pin)
    print(f"  obj1={obj2_lo_point.obj1:.4f}  obj2={obj2_lo_point.obj2:.4f}  "
          f"(TRUE lexicographic anchor, not the weakly-efficient one off step A)")

    print("[pareto_front_sweep] Step C: obj2-only anchor (maximize obj2 alone)")
    obj2_hi_point = solve_obj2_only(None)
    print(f"  obj1={obj2_hi_point.obj1:.4f}  obj2={obj2_hi_point.obj2:.4f}")

    front: List[FrontPoint] = [obj2_lo_point, obj2_hi_point]
    excluded_pairs: set = set()

    for i in range(n_sweep_points):
        try:
            direction, target, pair_key = pick_next_target(
                front, excluded_pairs=excluded_pairs, return_pair=True)
        except FrontExhausted:
            print(f"[pareto_front_sweep] all real gaps exhausted, stopping early "
                  f"at sweep point {i}/{n_sweep_points}")
            break
        if direction == "obj2":
            print(f"[pareto_front_sweep] Sweep point {i + 1}/{n_sweep_points}: "
                  f"constrain obj2 >= {target:.4f} (maximize obj1)")
            candidate = solve_epsilon_constraint_obj2(target)
        else:
            print(f"[pareto_front_sweep] Sweep point {i + 1}/{n_sweep_points}: "
                  f"constrain obj1 >= {target:.4f} (maximize obj2)  [FLIPPED -- locally steep in obj1]")
            candidate = solve_epsilon_constraint_obj1(target)
        print(f"  obj1={candidate.obj1:.4f}  obj2={candidate.obj2:.4f}")
        if is_dominated(candidate, front):
            print("  -> dominated by existing front point, dropped (gap excluded from re-selection)")
            excluded_pairs.add(pair_key)
            continue
        front = [p for p in front if not is_dominated(p, [candidate])]
        front.append(candidate)
        excluded_pairs.clear()  # front changed -> old gap identities no longer meaningful

    front.sort(key=lambda p: p.obj2)
    return front


# ── N-objective (N >= 2) extension, 2026-09-05 ──────────────────────────────
#
# The bi-objective machinery above stays untouched (existing callers/tests
# unaffected). This is the "hypervolume-contribution scorer... a natural
# drop-in for 3+ objectives" the module docstring named but deliberately
# deferred until a real use case existed (the rotor project's 3rd-objective
# work). Reuses `uncrowded_hypervolume.py`'s already-tested UHVI math rather
# than re-deriving hypervolume geometry.
#
# Deliberately NOT included: the bi-objective module's lexicographic
# "Step B" anchor refinement (pin one objective's solution, re-optimize a
# free nuisance variable for a TRUE, not weakly-efficient, anchor). For
# N>=3 the refinement order across the other N-1 objectives is ambiguous
# without problem-specific knowledge -- flagged here as a known, deliberate
# gap versus the 2-objective module, not an oversight. Each of the N
# single-objective anchors here is only as good as `solve_single_objective`
# makes it; if the caller's own free variables need a lexicographic
# refinement, that is the caller's responsibility for now.

@dataclasses.dataclass
class FrontPointN:
    """One solved point on an N-objective front (N >= 2), maximize-all
    convention (matching `FrontPoint` above). `target_kind` is the index of
    the objective that was epsilon-constrained during THIS solve, or None
    for a single-objective anchor. `target_values` carries the epsilon
    targets applied to every OTHER (non-`target_kind`) objective. `meta`
    carries whatever the caller wants attached; never inspected here."""
    objectives: Tuple[float, ...]
    target_kind: Optional[int] = None
    target_values: Dict[int, float] = dataclasses.field(default_factory=dict)
    meta: dict = dataclasses.field(default_factory=dict)


def is_dominated_n(candidate: FrontPointN, front: List[FrontPointN]) -> bool:
    """N-dimensional generalization of `is_dominated`: True if some point in
    `front` is >= candidate in EVERY objective, and strictly greater in at
    least one."""
    for p in front:
        if all(pv >= cv for pv, cv in zip(p.objectives, candidate.objectives)) \
                and any(pv > cv for pv, cv in zip(p.objectives, candidate.objectives)):
            return True
    return False


def _objective_ranges(front: List[FrontPointN]) -> List[float]:
    n_obj = len(front[0].objectives)
    ranges = []
    for k in range(n_obj):
        vals = [p.objectives[k] for p in front]
        ranges.append(max(vals) - min(vals) or 1.0)  # guard a degenerate flat front
    return ranges


def pick_next_target_n(
    front: List[FrontPointN],
    excluded_pairs: Optional[set] = None,
    return_pair: bool = False,
):
    """Hypervolume-contribution generalization of `pick_next_target` for
    N >= 2 objectives.

    1. Every pairwise midpoint (in raw objective space) between existing
       front points is a candidate "region we'd like a real solve to land
       near" -- the direct N-D generalization of the bi-objective module's
       "bisect the largest gap" idea, since a single global sort order (the
       2D method's `sorted(front, key=obj2)`) does not exist once N >= 3.
    2. Each candidate midpoint is scored by `uncrowded_hypervolume_improvement`
       (maximize convention here -- objectives and the reference point are
       negated before the call, since that function is minimize-convention)
       against the CURRENT front as the archive. The highest-scoring midpoint
       is treated as the most under-covered gap.
    3. Within that gap's originating pair, the objective with the SMALLEST
       normalized spread is freed (maximized) and every other objective is
       constrained to the midpoint's value there -- the direct generalization
       of the bi-objective module's direction flip (constrain the locally
       steep objective(s), free the locally flat one).

    Returns (free_index, other_targets) where `other_targets` maps every
    objective index except `free_index` to its epsilon-constraint target.

    `excluded_pairs`/`return_pair`: same contract as `pick_next_target` --
    a set of `_pair_key` values whose originating pair has already been
    tried and led nowhere this sweep (skipped here), and an opt-in third
    return element carrying the winning pair's key. Raises `FrontExhausted`
    when every pair is excluded.
    """
    from ._hypervolume import uncrowded_hypervolume_improvement
    if len(front) < 2:
        raise ValueError("need at least 2 front points (the anchors) to pick a gap")
    excluded_pairs = excluded_pairs or set()
    n_obj = len(front[0].objectives)
    ranges = _objective_ranges(front)
    archive = np.array([p.objectives for p in front], dtype=float)

    # Gap-SCORING runs in front-wide-range-NORMALIZED space (2026-09-05 fix) -- un-normalized
    # raw units let whichever objective happens to span the largest absolute range dominate
    # every hypervolume-improvement comparison, regardless of real trade-off structure (same
    # failure mode `pick_next_target`'s own docstring already documents for the bi-objective
    # sibling; confirmed here empirically: a 3-objective front with noise spanning ~8 dB vs FM
    # spanning ~0.09 and cruise efficiency ~0.31 picked noise-involving gaps almost exclusively).
    # `other_targets` is still returned in RAW units below (`mid`, not `mid_norm`) -- callers
    # pass it straight to a real epsilon-constraint solve, which needs actual units.
    mins = archive.min(axis=0)
    ranges_arr = np.array(ranges)
    archive_norm = (archive - mins) / ranges_arr
    nadir_norm = archive_norm.min(axis=0)
    reference_point = -(nadir_norm - 0.1)  # negated (minimize) convention; margin is 0.1 in every
                                            # normalized dim now, since each dim's normalized range is 1.0

    best_score, best_pair, best_mid = -np.inf, None, None
    for i in range(len(front)):
        for j in range(i + 1, len(front)):
            a, b = front[i], front[j]
            if _pair_key(a, b) in excluded_pairs:
                continue
            mid = tuple(0.5 * (av + bv) for av, bv in zip(a.objectives, b.objectives))
            mid_norm = (np.array(mid) - mins) / ranges_arr
            score = uncrowded_hypervolume_improvement(
                point=-mid_norm, archive_points=-archive_norm, reference_point=reference_point,
            )
            if score > best_score:
                best_score, best_pair, best_mid = score, (a, b), mid

    if best_pair is None:
        raise FrontExhausted("every pair between existing front points has been tried and excluded")
    a, b = best_pair
    deltas_norm = [abs(a.objectives[k] - b.objectives[k]) / ranges[k] for k in range(n_obj)]
    free_index = int(np.argmin(deltas_norm))
    other_targets = {k: best_mid[k] for k in range(n_obj) if k != free_index}
    if return_pair:
        return free_index, other_targets, _pair_key(a, b)
    return free_index, other_targets


def run_sweep_n(
    n_objectives: int,
    solve_single_objective: Callable[[int], FrontPointN],
    solve_epsilon_constraint: Callable[[int, Dict[int, float]], FrontPointN],
    n_sweep_points: int,
) -> List[FrontPointN]:
    """N-objective (N >= 2) payoff-table + hypervolume-gap-selection sweep.
    See the module-level comment above this section for what is
    deliberately NOT included versus the bi-objective `run_sweep` (no
    lexicographic anchor refinement).

    solve_single_objective(i): maximizes objective i alone, every other
    objective unconstrained -- one anchor per objective, `n_objectives`
    anchors total.
    solve_epsilon_constraint(free_index, other_targets): maximizes
    objective `free_index` subject to objective k >= other_targets[k] for
    every other k.
    """
    print(f"[pareto_front_sweep] N={n_objectives}-objective anchors")
    front: List[FrontPointN] = []
    for i in range(n_objectives):
        anchor = solve_single_objective(i)
        print(f"  anchor obj[{i}]: objectives={tuple(round(v, 4) for v in anchor.objectives)}")
        front.append(anchor)
    excluded_pairs: set = set()

    for i in range(n_sweep_points):
        try:
            free_index, other_targets, pair_key = pick_next_target_n(
                front, excluded_pairs=excluded_pairs, return_pair=True)
        except FrontExhausted:
            print(f"[pareto_front_sweep] all real gaps exhausted, stopping early "
                  f"at sweep point {i}/{n_sweep_points}")
            break
        print(f"[pareto_front_sweep] Sweep point {i + 1}/{n_sweep_points}: "
              f"free obj[{free_index}], constrain {other_targets}")
        candidate = solve_epsilon_constraint(free_index, other_targets)
        print(f"  objectives={tuple(round(v, 4) for v in candidate.objectives)}")
        if is_dominated_n(candidate, front):
            print("  -> dominated by existing front point, dropped (gap excluded from re-selection)")
            excluded_pairs.add(pair_key)
            continue
        front = [p for p in front if not is_dominated_n(p, [candidate])]
        front.append(candidate)
        excluded_pairs.clear()  # front changed -> old gap identities no longer meaningful

    front.sort(key=lambda p: p.objectives[-1])
    return front
