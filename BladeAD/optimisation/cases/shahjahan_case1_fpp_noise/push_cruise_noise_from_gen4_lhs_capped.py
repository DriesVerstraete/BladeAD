"""Same anchor push as `push_cruise_noise_from_gen4_lhs.py`, but with
`hover_elec` epsilon-capped at 25 kW -- the same cap the LHS/seeded scouts
used -- for a fair, apples-to-apples comparison against the known 72.41 dB
front (whose hover_elec span is 19.5-22.9 kW). The uncapped push reached
68.81 dB cruise_noise but at hover_elec=51.3 kW, ~2.2-2.6x the known front's
range and well outside the vehicle's real hover power budget (2026-09-14).

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/push_cruise_noise_from_gen4_lhs_capped.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/push_gen4_lhs_capped.log 2>&1

Single cold acoustic SLSQP solve -- budget up to ~10-15 min (uncapped twin
converged in 29.7 s; the added epsilon constraint may cost a few more
iterations).
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
        epsilons={"hover_elec": 25000.0},
    )
    print(f"=== done in {(time.time() - t0):.1f} s ===")
    print(f"objective_values = {result.get('objective_values')}")
    print(f"acoustics = {result.get('acoustics')}")
    print(f"constraint_checks = {result.get('constraint_checks')}")
