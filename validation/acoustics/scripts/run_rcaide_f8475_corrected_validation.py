from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "f8745_d4"
REPORTS = ROOT / "reports"


def configure_imports(rcaide_root):
    sys.path.insert(0, str(rcaide_root))
    sys.path.insert(0, str(rcaide_root / "VnV" / "Vehicles" / "Rotors"))


def atmosphere(temperature_k):
    pressure = 101325.0
    gas_constant = 287.05287
    gamma = 1.4
    density = pressure / (gas_constant * temperature_k)
    speed_of_sound = np.sqrt(gamma * gas_constant * temperature_k)
    viscosity = (
        1.716e-5
        * (temperature_k / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature_k + 110.4)
    )
    return density, speed_of_sound, viscosity


def build_case(rotor, published):
    from RCAIDE.Framework.Mission.Common import Results
    from RCAIDE.Framework.Mission.Segments.Segment import Segment

    segment = Segment()
    conditions = Results()
    temperature = float(published["temperature_k"])
    density, speed_of_sound, viscosity = atmosphere(temperature)
    velocity = float(published["axial_velocity_m_per_s"])
    conditions.aeroacoustics.relative_microphone_locations = np.zeros((1, 2, 3))
    conditions.aerodynamics.angles.alpha = np.zeros((1, 1))
    conditions.freestream.density = np.array([[density]])
    conditions.freestream.dynamic_viscosity = np.array([[viscosity]])
    conditions.freestream.speed_of_sound = np.array([[speed_of_sound]])
    conditions.freestream.temperature = np.array([[temperature]])
    conditions.frames.inertial.velocity_vector = np.array([[velocity, 0.0, 0.0]])
    conditions.energy.throttle = np.ones((1, 1))
    conditions.freestream.mach_number = np.array([[velocity / speed_of_sound]])
    conditions.frames.planet.true_course = np.eye(3)[None, :, :]
    conditions.frames.wind.transform_to_inertial = np.eye(3)[None, :, :]
    conditions.frames.body.transform_to_inertial = np.eye(3)[None, :, :]
    segment.state.conditions = conditions
    rotor.number_azimuthal_stations = 16
    rotor.use_2d_analysis = True
    rotor.append_operating_conditions(
        segment, conditions.energy, conditions.aeroacoustics
    )
    segment.state.conditions.expand_rows(1)
    rotor_conditions = conditions.energy.converters[rotor.tag]
    rotor_conditions.omega[:, 0] = float(published["rpm"]) * 2.0 * np.pi / 60.0
    return segment, conditions


def set_blade_angle(rotor, base_twist, blade_angle_deg):
    radial_fraction = rotor.radius_distribution / rotor.tip_radius
    reference_angle = np.interp(0.75, radial_fraction, np.rad2deg(base_twist))
    rotor.twist_distribution = base_twist + np.deg2rad(
        blade_angle_deg - reference_angle
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcaide-root", type=Path, required=True)
    args = parser.parse_args()
    rcaide_root = args.rcaide_root.resolve()
    configure_imports(rcaide_root)

    import RCAIDE
    from F8745_D4_Propeller import F8745_D4_Propeller
    from RCAIDE.Library.Methods.Aeroacoustics.Physics_Based_Frequency_Domain.Rotor import (
        compute_rotor_noise,
    )
    from RCAIDE.Library.Methods.Powertrain.Converters.Rotor.compute_rotor_performance import (
        compute_rotor_performance,
    )

    published_conditions = np.genfromtxt(
        FIXTURE / "published_operating_conditions.csv", delimiter=",", names=True
    )
    rows = []
    aerodynamic_rows = []
    for case_index, published in enumerate(published_conditions, start=1):
        rotor = F8745_D4_Propeller()
        rotor.tag = "F8475_D4_Propeller"
        base_twist = rotor.twist_distribution.copy()
        segment, conditions = build_case(rotor, published)
        rotor_conditions = conditions.energy.converters[rotor.tag]
        target_power_kw = float(published["shaft_power_kw"])

        def residual(blade_angle_deg):
            set_blade_angle(rotor, base_twist, blade_angle_deg)
            current = conditions.energy.converters[rotor.tag]
            current["throttle"] = np.ones((1, 1))
            current["omega"] = np.array(
                [[float(published["rpm"]) * 2.0 * np.pi / 60.0]]
            )
            compute_rotor_performance(rotor, conditions)
            current = conditions.energy.converters[rotor.tag]
            power_kw = float(
                current.torque[0, 0] * current.omega[0, 0] / 1000.0
            )
            return power_kw - target_power_kw

        scan_angles = np.linspace(-10.0, 50.0, 25)
        scan_residuals = [residual(angle) for angle in scan_angles]
        brackets = [
            (scan_angles[index], scan_angles[index + 1])
            for index in range(len(scan_angles) - 1)
            if scan_residuals[index] * scan_residuals[index + 1] <= 0.0
        ]
        if not brackets:
            raise RuntimeError(
                f"Case {case_index} power target is not bracketed; residual range "
                f"{min(scan_residuals):.3f} to {max(scan_residuals):.3f} kW"
            )
        reference_angle = float(
            published["computed_three_quarter_radius_blade_angle_deg"]
        )
        lower, upper = min(
            brackets, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - reference_angle)
        )
        blade_angle_deg = brentq(residual, lower, upper, xtol=1e-6)
        set_blade_angle(rotor, base_twist, blade_angle_deg)
        current = conditions.energy.converters[rotor.tag]
        current["throttle"] = np.ones((1, 1))
        current["omega"] = np.array(
            [[float(published["rpm"]) * 2.0 * np.pi / 60.0]]
        )
        compute_rotor_performance(rotor, conditions)
        rotor_conditions = conditions.energy.converters[rotor.tag]
        power_kw = float(
            rotor_conditions.torque[0, 0] * rotor_conditions.omega[0, 0] / 1000.0
        )
        thrust_n = float(rotor_conditions.thrust[0, 0])
        aerodynamic_rows.append(
            {
                "case": case_index,
                "matched_blade_angle_deg": blade_angle_deg,
                "measured_power_kw": target_power_kw,
                "rcaide_power_kw": power_kw,
                "measured_thrust_n": float(published["measured_thrust_n"]),
                "rcaide_thrust_n": thrust_n,
                "thrust_error_percent": 100.0
                * (thrust_n - float(published["measured_thrust_n"]))
                / float(published["measured_thrust_n"]),
            }
        )

        angles = np.deg2rad([60.0, 90.0])
        microphones = np.column_stack(
            (-4.0 * np.cos(angles), 4.0 * np.sin(angles), np.zeros(2))
        )
        conditions.aeroacoustics.relative_microphone_locations = microphones[None, :, :]
        conditions.aeroacoustics.number_of_microphones = 2
        analysis = RCAIDE.Framework.Analyses.Aeroacoustics.Physics_Based_Frequency_Domain()
        settings = analysis.settings
        settings.fidelity = "plane_source"
        settings.use_plane_loading_surrogate = False
        settings.wing_wake_interactional_dB_adjustment = 0.0
        compute_rotor_noise(microphones, rotor, segment, settings)
        spectrum = conditions.aeroacoustics.converters[
            rotor.tag
        ].SPL_harmonic_bpf_spectrum[0, :, :18]
        for observer, angle in enumerate((60, 90)):
            for harmonic, spl_db in enumerate(spectrum[observer], start=1):
                rows.append(
                    {
                        "case": case_index,
                        "observer_angle_deg": angle,
                        "harmonic": harmonic,
                        "rcaide_plane_source_spl_db": spl_db,
                    }
                )

    for path, data in (
        (REPORTS / "rcaide_f8475_corrected_aerodynamics.csv", aerodynamic_rows),
        (REPORTS / "rcaide_f8475_corrected_plane_source.csv", rows),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
