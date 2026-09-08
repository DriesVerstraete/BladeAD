"""Pareto-tracer: hover_elec x oei_margin edge, seeded from the oeiH NSGA-II basin. Per-line HH:MM:SS timestamps; run direct (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u \
      BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_ho/tracer_run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_tracer_ho/tracer.log 2>&1
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

from BladeAD.optimisation.pareto_tracer import TraceConfig, trace

if __name__ == "__main__":
    trace(TraceConfig(
        case_dir=_HERE,
        name_a="hover_elec", name_b="oei_margin",
        nsga_seed=os.path.join(_CASES, "shahjahan_case1_fpp_nsga2_oeiH",
                               "nsga2_result_oeiH_run1.pkl"),
        n_points=14, predictor="secant", tag="ho_run1",
    ))
