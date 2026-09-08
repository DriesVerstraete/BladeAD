"""Orthogonal cross-check of the 3-objective interior surface.

The `interior_run.py` surface stacks `hover_elec x cruise_elec` traces over
`oei_margin` levels. This one does the complementary construction: stacks
`cruise_elec x oei_margin` traces over pinned `hover_elec` levels. If the two
surfaces agree after a non-dominated merge, the interior is confirmed complete;
where they disagree localises an under-resolved region.

Run direct (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u \
      BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_hc/orthogonal_run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_hc/orthogonal.log 2>&1
"""
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASES = os.path.dirname(_HERE)

HOVER_LEVELS = [19300.0, 20000.0, 21000.0]   # W -- where the hover x cruise stack is thin
N_POINTS = 12
NSGA_SEED = os.path.join(_CASES, "shahjahan_case1_fpp_nsga2", "nsga2_result_run1.pkl")
TAG = "ortho_run1"


class _Stamp:
    def __init__(self, s):
        self.s = s
        self._bol = True

    def write(self, t):
        for c in t.splitlines(keepends=True):
            if self._bol and c.strip():
                self.s.write(time.strftime("[%H:%M:%S] "))
            self.s.write(c)
            self._bol = c.endswith("\n")
        return len(t)

    def flush(self):
        self.s.flush()


sys.stdout = _Stamp(sys.stdout)
sys.stderr = _Stamp(sys.stderr)

from BladeAD.optimisation.pareto_tracer import TraceConfig, overlay, trace
from BladeAD.optimisation.case import Context

if __name__ == "__main__":
    pts = []
    for i, lvl in enumerate(HOVER_LEVELS):
        print(f"\n{'=' * 70}\nLAYER {i}/{len(HOVER_LEVELS) - 1}  hover_elec <= {lvl:.0f} W"
              f"\n{'=' * 70}", flush=True)
        try:
            res = trace(TraceConfig(
                case_dir=_HERE, name_a="cruise_elec", name_b="oei_margin",
                nsga_seed=NSGA_SEED, n_points=N_POINTS, predictor="secant",
                tag=f"{TAG}_L{i}_hov{lvl:.0f}",
                fixed_epsilons={"hover_elec": float(lvl)}))
        except SystemExit as exc:
            print(f"  layer {i} SKIPPED: {exc}", flush=True)
            continue
        for p in res.points:
            pts.append({"hover_elec": float(lvl), "cruise_elec": float(p.f[0]),
                        "oei_margin": float(p.f[1]), "layer": i})
        if res.messages:
            print(f"  layer {i} messages: {res.messages}", flush=True)

    ctx = Context.load(_HERE)
    out = ctx.out(f"pareto_surface_3obj_{TAG}.pkl")
    with open(out, "wb") as f:
        pickle.dump({"objective_labels": ("hover_elec", "cruise_elec", "oei_margin"),
                     "hover_levels": [float(v) for v in HOVER_LEVELS],
                     "n_points_per_layer": N_POINTS, "front": pts}, f)
    print(f"\nsaved {out}  ({len(pts)} surface points, {len(HOVER_LEVELS)} layers)")

    try:
        overlay(out, surface_pkl=out,
                out_path=ctx.out(f"pareto_overlay_{TAG}.png"),
                title=f"{os.path.basename(_HERE)}: orthogonal cross-check surface [{TAG}]")
    except Exception as exc:                              # noqa: BLE001
        print(f"overlay plot skipped: {exc}")
