"""Diagnostic: re-run the smoke's failed anchor-2 bracket solve with the
solver's own chatter UNMUFFLED (`solve.QUIET_SOLVER = False`) to see whether
the inner BEM solve is converging cleanly (SLSQP outer loop just needs more
iterations) or the inner nonlinear solver itself is failing along the way
(geometry moving into an ill-posed region -- corrupted gradients, SLSQP
"converges" far from feasible for a different reason). 2026-09-14.

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise_hover/diagnose_anchor2_trace.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise_hover/diagnose_anchor2_trace.log 2>&1

Same anchor-2 seed that took 1126 s unmuffled-adjacent in the smoke -- budget
up to ~20-25 min.
"""
import os
import sys
import time

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

from BladeAD.optimisation import solve

solve.QUIET_SOLVER = False  # unmuffle: see BEM inner-solver + SLSQP chatter

if __name__ == "__main__":
    t0 = time.time()
    seed_path = os.path.join(_HERE, "_hybrid_anchor_2.pkl")
    print(f"seed = {seed_path}  (the smoke's anchor 2 -- 1126 s, 6 constraints FAILED)")
    try:
        result = solve.run_case_dir(_HERE, with_acoustics=True, initial_result_from=seed_path)
        print(f"=== done in {(time.time() - t0):.1f} s ===")
        print(f"constraint_checks = {result.get('constraint_checks')}")
    except Exception as e:
        print(f"=== raised after {(time.time() - t0):.1f} s: {e!r} ===")
