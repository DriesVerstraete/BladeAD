"""One real NSGA-II scout for the noise case -- the population the acoustic
sweep (`acoustic_scout_sweep.py`, Option D) then evaluates for cruise noise.

Scout objectives are `hover_elec` x `cruise_elec` only (the BEM kernel cannot
evaluate acoustics); the scout is the `hover_elec`/`cruise_elec` basin map, and
the acoustic sweep checks post-hoc whether any of those basins is also quiet.

Output `nsga2_result_hybrid_scout.pkl` is written to this case dir with the same
tag `run_hybrid` expects, so a later full hybrid run reuses this scout instead
of regenerating it (unless `hybrid["force_scout"]`).

Direct invocation (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/scout_run.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/scout_run.log 2>&1
"""
import os
import sys
import time

N_WORKERS = 4

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEED_DIR = os.path.normpath(os.path.join(_HERE, "..", "shahjahan_case1_fpp_explore_sm099"))


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

from BladeAD.optimisation import nsga2_runner
from BladeAD.optimisation.cases.shahjahan_case1_fpp_noise.case import CASE

if __name__ == "__main__":
    cfg = CASE["hybrid"]
    print(f"scout: pop {cfg['scout_pop']} x gen {cfg['scout_gens']}, "
          f"objectives {cfg['scout_objectives']}, workers {N_WORKERS}, "
          f"seeded from {os.path.basename(_SEED_DIR)}", flush=True)
    nsga2_runner.run(
        _HERE,
        pop_size=cfg["scout_pop"],
        n_gen=cfg["scout_gens"],
        seed=cfg["seed"],
        n_workers=N_WORKERS,
        algorithm="nsga2",
        save_tag="hybrid_scout",
        objective_names=cfg["scout_objectives"],
        front_seed_from=[_SEED_DIR],
    )
    print("scout done -> nsga2_result_hybrid_scout.pkl", flush=True)
