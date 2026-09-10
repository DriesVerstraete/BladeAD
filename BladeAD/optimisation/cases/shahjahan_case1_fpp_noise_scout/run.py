"""NSGA-II acoustic basin scout for Shahjahan Case 1 FPP -- reads the `nsga2`
block from `case.py` and calls `nsga2_runner.run`. Per-line HH:MM:SS stamps;
the per-generation line shows feasible count + true ND-front size.

Run (NOT `conda run`, NOT `| tail` -- this project's solves auto-background):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise_scout/run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise_scout/run1.log 2>&1

Budget: pop 96 x 200 gen ~= 19 200 acoustic forward evals. At ~0.35 s/eval
serial / 4 workers (~2x effective from the Codex smoke) ~= 25-60 min; allow up
to ~1.5 h if acoustic evals run slow. Expected to COLLAPSE -- that is the point;
the deliverable is the distinct collapsed geometries in
`results/optimisation_history_*_run1.pkl` for `scratchpad/basin_compare.py`.

`hot_start: True` in case.py + re-run resumes from the history pkl (keep
`save_tag` + `pop_size` fixed).
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


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

from BladeAD.optimisation.case import load_case_dict
from BladeAD.optimisation import nsga2_runner

if __name__ == "__main__":
    opts = dict(load_case_dict(_HERE).get("nsga2", {}))
    print(f"nsga2 opts = {opts}")
    nsga2_runner.run(_HERE, **opts)
