"""Step 3 of `notes/2026-09-15-transition-point-scoping.md` -- confirm CSDL
can differentiate through `PittPetersModel` before it's ever used in a real
optimization. Computes d(total_thrust)/d(rpm) and d(total_power)/d(rpm) via
`csdl.derivative` (reverse-mode AD) at the same Transition3 (45 deg) point as
`transition_pitt_peters_smoke.py`, then cross-checks against a central finite
difference (3 independent forward evals, since inline CSDL builds/executes
the graph fresh each time -- there is no single persistent graph to perturb
in place).

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pitt_peters_gradient_check.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pitt_peters_gradient_check.log 2>&1

4 forward evals (1 AD + 3 FD) at ~0.1 s each per the smoke test -- expect well
under a minute.
"""
import os
import pickle
import sys
import time

import numpy as np

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

import csdl_alpha as csdl
from BladeAD.core.pitt_peters.pitt_peters_model import PittPetersModel
from BladeAD.utils.var_groups import RotorMeshParameters, RotorAnalysisInputs

from BladeAD.optimisation.case import load_case_dict
from BladeAD.optimisation.solve import _isa_atmos, _spec_from_case, _airfoil_model, _radial_grid

TILT_DEG = 45.0
FLIGHT_SPEED = 21.5
NUM_AZIMUTHAL = 8
GEOM_PKL = "stage1_opt-hover_elec_grid-cosine32_ac_eps-cruise_noise=72.4090.pkl"

with open(os.path.join(_HERE, GEOM_PKL), "rb") as f:
    _GEOM = pickle.load(f)
_CHORD = np.asarray(_GEOM["chord_profile"], dtype=float)
_TWIST = np.asarray(_GEOM["twist_profile"], dtype=float)
_RPM0 = float(_GEOM["cruise_rpm"])
_CASE = load_case_dict(_HERE)
_SPEC = _spec_from_case(_CASE)
_NUM_RADIAL, _NORM_STATIONS = _radial_grid(_SPEC)
_AIRFOIL = _airfoil_model(_CASE, clamp_reynolds=True)
_TILT_RAD = np.deg2rad(TILT_DEG)


def _eval_at(rpm_value, want_ad_grad=False):
    """One forward eval (fresh recorder each call -- inline CSDL). Returns
    (thrust, power) as floats, and optionally the AD d(thrust)/d(rpm),
    d(power)/d(rpm) if `want_ad_grad`."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    thrust_vector = csdl.Variable(value=np.array([np.cos(_TILT_RAD), 0.0, np.sin(_TILT_RAD)]))
    thrust_origin = csdl.Variable(value=np.array([0.0, 0.0, 0.0]))
    chord_v = csdl.Variable(value=_CHORD)
    twist_v = csdl.Variable(value=_TWIST)
    rpm_v = csdl.Variable(value=np.array([rpm_value]))

    mesh = RotorMeshParameters(
        thrust_vector=thrust_vector, thrust_origin=thrust_origin,
        chord_profile=chord_v, twist_profile=twist_v,
        radius=csdl.Variable(value=_SPEC.radius),
        num_radial=_SPEC.num_radial, num_azimuthal=NUM_AZIMUTHAL,
        num_blades=_SPEC.n_blades, norm_hub_radius=_SPEC.hub_radius / _SPEC.radius,
        norm_radial_stations=_NORM_STATIONS,
    )
    inputs = RotorAnalysisInputs(
        rpm=rpm_v,
        mesh_velocity=csdl.Variable(value=np.array([[FLIGHT_SPEED, 0.0, 0.0]])),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = _isa_atmos(500.0)

    out = PittPetersModel(num_nodes=1, airfoil_model=_AIRFOIL,
                          integration_scheme="trapezoidal").evaluate(inputs=inputs)

    thrust = float(out.total_thrust.value[0])
    power = float(out.total_power.value[0])

    ad_grads = None
    if want_ad_grad:
        derivs = csdl.derivative([out.total_thrust, out.total_power], rpm_v, mode="reverse")
        d_thrust_d_rpm = float(np.asarray(derivs[out.total_thrust].value).reshape(-1)[0])
        d_power_d_rpm = float(np.asarray(derivs[out.total_power].value).reshape(-1)[0])
        ad_grads = (d_thrust_d_rpm, d_power_d_rpm)

    recorder.stop()
    return thrust, power, ad_grads


def main():
    print(f"tilt={TILT_DEG} deg  V={FLIGHT_SPEED} m/s  rpm0={_RPM0:.2f}")

    t0 = time.time()
    thrust0, power0, ad_grads = _eval_at(_RPM0, want_ad_grad=True)
    ad_dthrust, ad_dpower = ad_grads
    print(f"AD eval done in {time.time() - t0:.2f} s")
    print(f"  thrust0={thrust0:.3f} N  power0={power0:.1f} W")
    print(f"  AD d(thrust)/d(rpm) = {ad_dthrust:.6f} N/rpm")
    print(f"  AD d(power)/d(rpm)  = {ad_dpower:.4f} W/rpm")

    h = 1.0  # rpm step for central FD
    t1 = time.time()
    thrust_p, power_p, _ = _eval_at(_RPM0 + h)
    thrust_m, power_m, _ = _eval_at(_RPM0 - h)
    print(f"FD evals done in {time.time() - t1:.2f} s (h={h} rpm)")

    fd_dthrust = (thrust_p - thrust_m) / (2 * h)
    fd_dpower = (power_p - power_m) / (2 * h)
    print(f"  FD d(thrust)/d(rpm) = {fd_dthrust:.6f} N/rpm")
    print(f"  FD d(power)/d(rpm)  = {fd_dpower:.4f} W/rpm")

    def _rel_err(ad, fd):
        return abs(ad - fd) / max(abs(fd), 1e-12)

    err_thrust = _rel_err(ad_dthrust, fd_dthrust)
    err_power = _rel_err(ad_dpower, fd_dpower)
    print(f"relative error: thrust {err_thrust:.2e}   power {err_power:.2e}")

    ok = err_thrust < 1e-3 and err_power < 1e-3
    print(f"=== {'PASS' if ok else 'FAIL'} (AD matches FD to <1e-3 relative: {ok}) ===")


if __name__ == "__main__":
    main()
