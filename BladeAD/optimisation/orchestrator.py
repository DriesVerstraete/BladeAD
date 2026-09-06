"""`run_case(case_dir)` -- the whole pipeline for one case, everything written
back into the case directory. Objective-slot-driven (2026-09-06 refactor): the
stages are derived from `case.parse_objectives`, not a hardcoded objective set.

Stages (as printed):
  [0] pre-flight feasibility gate -- `seed.preflight_feasibility` for each
      operating point; ABORT before any sweep if a point is infeasible.
  [1] one single-objective anchor per live objective slot (`objective_anchor`),
      cached in a per-case `_anchor_registry.json` (registry key = slot name).
  [2] one `run_edge` per C(N, 2) pair of live slots.
  [3] the interior front (>= 3 live slots):
       N == 3 : "grid" (default, `grid.run_grid_front`) | "nobj"
                (`nobj.run_nobj_front`) | "grid+nobj".
       N  > 3 : always `nobj.run_nobj_front` (the grid staircase is 2-D only --
                see notes/2026-09-06-many-objective-front-methods-websearch.md
                and todos.md for the open (N-1)-D vs nobj question).
  [4] the overlay plot.

Reserved slots (role "reserved", e.g. "electrical_power") are listed with a NOTE
and skipped -- no solver code yet.
"""
import glob
import itertools
import os
import pickle
import time

from .case import Context, active_objectives, reserved_objectives
from . import seed as seedmod
from . import edges as edgesmod
from . import grid as gridmod
from . import nobj as nobjmod
from .sweep_common import objective_anchor


def _latest_edge_pkl(ctx, name):
    m = sorted(glob.glob(ctx.out(f"pareto_front_{name}_*.pkl")), key=os.path.getmtime)
    return m[-1] if m else None


def run_case(case_dir, do_plot=True, reuse_edges=True, three_obj_method="grid"):
    t0 = time.time()
    ctx = Context.load(case_dir)
    active = active_objectives(ctx.case)
    names = [s.name for s in active]
    print(f"=== run_case {ctx.case_dir} ===")
    print(f"active objectives: {names}  (proxy: "
          f"{next(s.name for s in active if s.role == 'proxy')})")
    for s in reserved_objectives(ctx.case):
        print(f"  NOTE: objective {s.name!r} reserved (no solver code yet) -- skipped")

    # --- stage 0: pre-flight feasibility gate ---
    print("\n[0] pre-flight feasibility gate")
    blades = seedmod.preflight_feasibility(ctx.case)
    for name, b in blades.items():
        print(f"  {name}: feasible={b['feasible']} cl_used={b['cl_used']} "
              f"thrust_check={b['thrust_check_n']:.1f} N (target "
              f"{ctx.case['operating'][name]['thrust_n']:.1f})")
        if not b["feasible"]:
            raise SystemExit(
                f"ABORT: operating point '{name}' is infeasible even at cl_max -- "
                f"{b.get('note', 'thrust exceeds what the disk supports at a physical Cl')}. "
                "Fix the case before sweeping.")

    # --- stage 1: one anchor per slot ---
    print("\n[1] single-objective anchors")
    for name in names:
        pkl, data = objective_anchor(ctx, name)
        vals = "  ".join(f"{s.name}={data['objective_values'].get(s.name, float('nan')):.4f}"
                         for s in active)
        print(f"  {name}: {vals}")

    # --- stage 2: pairwise edge fronts ---
    print("\n[2] pairwise 2-objective edge fronts")
    edge_fronts, edge_pkls = [], {}
    for a, b in itertools.combinations(names, 2):
        tag = f"{a}__{b}"
        cached = _latest_edge_pkl(ctx, tag) if reuse_edges else None
        if cached:
            with open(cached, "rb") as f:
                front = pickle.load(f)["front"]
            out = cached
            print(f"\n  --- edge {a} vs {b}: reusing {os.path.basename(cached)} "
                  f"({len(front)} pts) ---")
        else:
            print(f"\n  --- edge {a} vs {b} ---")
            front, out = edgesmod.run_edge(ctx, a, b)
        edge_fronts.append(front)
        edge_pkls[frozenset((a, b))] = out

    # --- stage 3: interior front ---
    three_obj_pkl = None
    n = len(active)
    if n >= 3:
        method = "nobj" if n > 3 else three_obj_method
        if n > 3 and three_obj_method != "nobj":
            print(f"\n[3] interior front: N={n} > 3 -> forcing method 'nobj' "
                  f"(requested {three_obj_method!r})")
        else:
            print(f"\n[3] interior front (method: {method})")
        if method in ("grid", "grid+nobj"):
            _, three_obj_pkl = gridmod.run_grid_front(ctx, edge_fronts=edge_fronts)
        if method in ("nobj", "grid+nobj"):
            _, three_obj_pkl = nobjmod.run_nobj_front(ctx, edge_fronts=edge_fronts)
        if method not in ("grid", "nobj", "grid+nobj"):
            raise SystemExit(f"unknown three_obj_method {three_obj_method!r}")
    else:
        print("\n[3] interior front: skipped (< 3 active objectives)")

    # --- stage 4: overlay plot ---
    if do_plot and three_obj_pkl is not None:
        print("\n[4] overlay plot")
        from . import plot as plotmod
        plotmod.render_overlay(ctx, three_obj_pkl, edge_pkls)

    print(f"\n=== done in {(time.time() - t0) / 60:.1f} min ===")
    return {"edge_pkls": edge_pkls, "three_obj_pkl": three_obj_pkl}
