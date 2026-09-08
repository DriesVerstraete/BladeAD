"""3-objective interior as a stack of `hover_elec x cruise_elec` tracer traces,
each at a fixed `oei_margin` level. Run direct (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u \
      BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_hc/interior_run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_hc/interior.log 2>&1

Levels stop at 1.85 -- the cruise_elec x oei_margin edge diverges past ~1.87.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CASES = os.path.dirname(_HERE)


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

from BladeAD.optimisation.pareto_tracer import trace_interior

if __name__ == "__main__":
    trace_interior(
        case_dir=_HERE,
        nsga_seed=os.path.join(_CASES, "shahjahan_case1_fpp_nsga2", "nsga2_result_run1.pkl"),
        oei_levels=[1.143, 1.33, 1.52, 1.70, 1.85],
        n_points=12,
        tag="run1",
    )
