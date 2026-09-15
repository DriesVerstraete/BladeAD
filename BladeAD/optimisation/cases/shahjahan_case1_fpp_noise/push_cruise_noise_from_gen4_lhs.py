"""Single-obj gradient anchor push, seeded from the fully-unseeded LHS basin
scout's gen-4 individual (`shahjahan_case1_fpp_noise_scout_lhs`, 2026-09-14,
cruise_noise=68.9 dB raw/unretrimmed, cv<=1e-6). That case has a free
`collective_deg` 13th DV the base ep/tracer pipeline (fixed_pitch=True, no
separate collective DV) doesn't have -- `seed_gen4_lhs.pkl` folds collective
into `twist_cps` (twist_new = twist_old + collective_old, pitch_deg=0.0) so
the seed lands in this case's own DOF convention (PI, 2026-09-14).

`optimize="cruise_noise"`, no epsilons -- a genuinely free single-objective
push (hover_elec/cruise_elec unconstrained beyond the case's normal physical
constraints: thrust match, stall, taper, washout, torque margins). Answers:
does gradient descent from this low-twist family reach a real front point
below the known 72.41 dB floor, or does it converge back into a known basin?

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/push_cruise_noise_from_gen4_lhs.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/push_gen4_lhs.log 2>&1

Single cold acoustic SLSQP solve -- budget up to ~10-15 min.
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

if __name__ == "__main__":
    t0 = time.time()
    seed_path = os.path.join(_HERE, "seed_gen4_lhs.pkl")
    print(f"seed = {seed_path}")
    result = solve.run_case_dir(
        _HERE,
        optimize="cruise_noise",
        with_acoustics=True,
        initial_result_from=seed_path,
    )
    print(f"=== done in {(time.time() - t0):.1f} s ===")
    print(f"objective_values = {result.get('objective_values')}")
    print(f"acoustics = {result.get('acoustics')}")
    print(f"constraint_checks = {result.get('constraint_checks')}")
