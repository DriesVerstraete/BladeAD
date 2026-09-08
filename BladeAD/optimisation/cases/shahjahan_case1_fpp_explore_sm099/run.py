"""Edges-only driver: stages 0-2 of the run_case pipeline (pre-flight gate,
single-objective anchors, the C(N,2) pairwise 2-objective fronts), then stop --
the interior grid is skipped by design for this sensitivity variant. Acoustics
forced OFF (patched into `sweep_common.BASE_OPTS` before any subprocess solve).
"""
import itertools
import os
import sys
import time

import BladeAD.optimisation.sweep_common as _sc
_sc.BASE_OPTS = {"with_acoustics": False}


class _Stamp:
    def __init__(self, s): self.s = s; self._bol = True
    def write(self, t):
        for c in t.splitlines(keepends=True):
            if self._bol and c.strip():
                self.s.write(time.strftime("[%H:%M:%S] "))
            self.s.write(c); self._bol = c.endswith("\n")
        return len(t)
    def flush(self): self.s.flush()


sys.stdout = _Stamp(sys.stdout); sys.stderr = _Stamp(sys.stderr)

from BladeAD.optimisation.case import Context, active_objectives
from BladeAD.optimisation import seed as seedmod
from BladeAD.optimisation import edges as edgesmod
from BladeAD.optimisation.sweep_common import objective_anchor

if __name__ == "__main__":
    print(f"BASE_OPTS = {_sc.BASE_OPTS}  (edges-only variant)")
    ctx = Context.load(os.path.dirname(os.path.abspath(__file__)))
    names = [s.name for s in active_objectives(ctx.case)]
    print(f"active objectives: {names}")

    print("\n[0] pre-flight feasibility gate")
    blades = seedmod.preflight_feasibility(ctx.case)
    for name, b in blades.items():
        print(f"  {name}: feasible={b['feasible']} cl_used={b['cl_used']} "
              f"thrust_check={b['thrust_check_n']:.1f} N")
        if not b["feasible"]:
            raise SystemExit(f"ABORT: operating point '{name}' infeasible even at cl_max")

    print("\n[1] single-objective anchors")
    for name in names:
        _, data = objective_anchor(ctx, name)
        print("  " + name + ": "
              + "  ".join(f"{k}={v:.4f}" for k, v in data["objective_values"].items()))

    print("\n[2] pairwise 2-objective edge fronts")
    for a, b in itertools.combinations(names, 2):
        print(f"\n  --- edge {a} vs {b} ---")
        edgesmod.run_edge(ctx, a, b)

    print("\n[done] edges only -- interior grid skipped by design")
