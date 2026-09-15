"""Step 5 of `notes/2026-09-15-transition-point-scoping.md` -- exercise the
new `case["transition"]` wiring in `solve.py` end-to-end (not the standalone
smoke script's manual construction). Pins the known 72.41 dB front geometry
(chord/twist/hover_rpm/hover_pitch frozen), leaves `cruise_rpm` AND the new
`transition_rpm` free to retrim -- `solve.run()`'s own thrust-equality
constraint solves for the RPM that hits 527.75 N at Transition3 (45 deg,
21.5 m/s), through the real optimizer (SLSQP), not a naive forward eval.

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pinned_solve.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pinned_solve.log 2>&1

One pinned SLSQP solve (2 free DVs: cruise_rpm, transition_rpm) -- budget a
few minutes given this case's maxiter=400 acoustic-off headroom; likely much
faster since only 2 scalar DVs are free.
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

from BladeAD.optimisation.case import load_case_dict
from BladeAD.optimisation import solve

GEOM_PKL = "stage1_opt-hover_elec_grid-cosine32_ac_eps-cruise_noise=72.4090.pkl"

if __name__ == "__main__":
    case = load_case_dict(_HERE)
    case["transition"] = {
        "tilt_deg": 45.0,
        "airspeed_m_s": 21.5,
        "thrust_n": 527.75,   # 4222 N vehicle / 8 rotors
        "num_azimuthal": 8,
    }
    print(f"transition config: {case['transition']}")

    t0 = time.time()
    result = solve.run(
        case, _HERE, seed_dict=None,
        pin_geometry_from=os.path.join(_HERE, GEOM_PKL),
        with_acoustics=False,   # not needed for this check -- oblique-flow mechanism only
    )
    print(f"=== done in {time.time() - t0:.1f} s ===")
    print(f"transition_rpm={result['transition_rpm']:.2f}  "
          f"transition_thrust={result['transition_thrust']:.2f} N "
          f"(target {result['transition_thrust_target']:.2f})")
    print(f"transition_power={result['transition_power']:.1f} W  "
          f"transition_electrical_power={result['transition_electrical_power']:.1f} W")
    print(f"transition_torque_margin={result['transition_torque_margin']}")
    _cl = result['transition_max_sectional_cl']
    print(f"transition_max_sectional_cl={_cl:.4f}" if _cl is not None
          else "transition_max_sectional_cl=None (known PittPeters gap)")
    print(f"constraint_checks = {result['constraint_checks']}")
