"""Generic hybrid run for Shahjahan Case 1 FPP with cruise noise as the 3rd
objective (in place of OEI thrust margin).

Direct invocation (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/hybrid_run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/hybrid.log 2>&1

Set DRY_RUN = True for the smoke (cuts scout_pop/gens, n_points, auto_levels).
"""
import os
import sys
import time

DRY_RUN = False

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Stamp:
    def __init__(self, stream):
        self.stream = stream
        self.beginning = True

    def write(self, text):
        for line in text.splitlines(keepends=True):
            if self.beginning and line.strip():
                self.stream.write(time.strftime("[%H:%M:%S] "))
            self.stream.write(line)
            self.beginning = line.endswith("\n")
        return len(text)

    def flush(self):
        self.stream.flush()


sys.stdout = _Stamp(sys.stdout)
sys.stderr = _Stamp(sys.stderr)

from BladeAD.optimisation.hybrid import run_hybrid

if __name__ == "__main__":
    run_hybrid(_HERE, dry_run=DRY_RUN)
