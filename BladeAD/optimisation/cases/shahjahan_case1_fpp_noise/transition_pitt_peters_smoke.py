"""Standalone smoke test -- ONE oblique-flow evaluation at Transition3 (45 deg
tilt, Table 6 of Shahjahan 2024), using PittPetersModel instead of BEMModel.
No pipeline wiring, no optimizer -- a single forward eval on a pinned, known
blade geometry to check (a) it runs in this env and (b) the numbers are
physically sane. Step 2 of `notes/2026-09-15-transition-point-scoping.md`.

Geometry pinned: the known cruise-noise front's quiet-end point (72.41 dB,
`stage1_opt-hover_elec_grid-cosine32.pkl`'s sibling -- reusing
`stage1_opt-hover_elec_grid-cosine32_ac_eps-cruise_noise=72.4090.pkl`'s dense
chord_profile/twist_profile arrays directly, no CP reconstruction needed).

Convention (PI, 2026-09-15): tilt 0deg=cruise, 90deg=hover. Table 6
Transition3: tilt=45deg, flight speed=21.5 m/s, vehicle thrust=4222 N ->
per-rotor 527.75 N (n_rotors=8). thrust_vector tilted by `tilt` from
horizontal ([cos(tilt), 0, sin(tilt)]); mesh_velocity stays the flight speed
along the vehicle x-axis ([V, 0, 0]) -- `compute_local_frame_velocities`
(shared preprocessing, confirmed generic) decomposes it into axial/in-plane
components relative to the tilted thrust_vector automatically.

Run (NOT `conda run`, NOT `| tail`):

    conda activate rotor_design
    cd .../999-software/bladead_repo
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
      python -u BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pitt_peters_smoke.py \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_noise/transition_pitt_peters_smoke.log 2>&1

Single forward eval, no optimizer -- expect well under a minute.
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
FLIGHT_SPEED = 21.5      # m/s, Table 6 Transition3
VEHICLE_THRUST_N = 4222.0
N_ROTORS = 8
PER_ROTOR_THRUST_N = VEHICLE_THRUST_N / N_ROTORS
NUM_AZIMUTHAL = 8         # matches NUM_AZIMUTHAL_ACOUSTIC (verified elsewhere)

GEOM_PKL = "stage1_opt-hover_elec_grid-cosine32_ac_eps-cruise_noise=72.4090.pkl"


def main():
    print(f"tilt={TILT_DEG} deg  V={FLIGHT_SPEED} m/s  "
          f"per-rotor thrust target={PER_ROTOR_THRUST_N:.2f} N (vehicle {VEHICLE_THRUST_N} / {N_ROTORS})")

    with open(os.path.join(_HERE, GEOM_PKL), "rb") as f:
        geom = pickle.load(f)
    print(f"pinned geometry: {GEOM_PKL}")
    print(f"  rpm(hover)={geom['rpm']:.1f}  cruise_rpm={geom['cruise_rpm']:.1f}  "
          f"taper={geom['taper']:.3f}  washout={geom['twist_washout_deg']:.2f} deg")

    case = load_case_dict(_HERE)
    spec = _spec_from_case(case)
    num_radial, norm_stations = _radial_grid(spec)
    airfoil_model = _airfoil_model(case, clamp_reynolds=True)

    chord_profile = np.asarray(geom["chord_profile"], dtype=float)
    twist_profile = np.asarray(geom["twist_profile"], dtype=float)
    assert chord_profile.shape[0] == num_radial, (chord_profile.shape, num_radial)

    tilt_rad = np.deg2rad(TILT_DEG)

    # RPM: only free variable at this point (FPP -- pitch shared, not interpolated).
    # Seed with the cruise rpm of the pinned geometry as a starting guess.
    rpm_guess = float(geom["cruise_rpm"])

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    thrust_vector = csdl.Variable(value=np.array([np.cos(tilt_rad), 0.0, np.sin(tilt_rad)]))
    thrust_origin = csdl.Variable(value=np.array([0.0, 0.0, 0.0]))
    chord_v = csdl.Variable(value=chord_profile)
    twist_v = csdl.Variable(value=twist_profile)
    rpm_v = csdl.Variable(value=np.array([rpm_guess]))

    mesh = RotorMeshParameters(
        thrust_vector=thrust_vector,
        thrust_origin=thrust_origin,
        chord_profile=chord_v,
        twist_profile=twist_v,
        radius=csdl.Variable(value=spec.radius),
        num_radial=spec.num_radial,
        num_azimuthal=NUM_AZIMUTHAL,
        num_blades=spec.n_blades,
        norm_hub_radius=spec.hub_radius / spec.radius,
        norm_radial_stations=norm_stations,
    )
    inputs = RotorAnalysisInputs(
        rpm=rpm_v,
        mesh_velocity=csdl.Variable(value=np.array([[FLIGHT_SPEED, 0.0, 0.0]])),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = _isa_atmos(500.0)   # 500 m, matches this case's cruise altitude

    t0 = time.time()
    out = PittPetersModel(num_nodes=1, airfoil_model=airfoil_model,
                          integration_scheme="trapezoidal").evaluate(inputs=inputs)
    dt = time.time() - t0

    thrust = float(out.total_thrust.value[0])
    power = float(out.total_power.value[0])
    torque = float(out.total_torque.value[0])
    print(f"=== PittPeters forward eval done in {dt:.1f} s ===")
    print(f"total_thrust = {thrust:.2f} N   (target {PER_ROTOR_THRUST_N:.2f} N, "
          f"ratio {thrust / PER_ROTOR_THRUST_N:.3f})")
    print(f"total_power  = {power:.1f} W")
    print(f"total_torque = {torque:.2f} N.m")

    # sanity flags -- not asserted, just surfaced
    if not np.isfinite(thrust) or not np.isfinite(power):
        print("!! NON-FINITE OUTPUT -- model did not evaluate cleanly")
    if thrust <= 0:
        print("!! non-positive thrust -- check tilt/rpm sign conventions")


if __name__ == "__main__":
    main()
