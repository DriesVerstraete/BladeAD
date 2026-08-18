from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def observer_positions(angles_deg, distance_m, y_sign):
    rows = []
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        if angle_rad < np.pi / 2:
            x = -distance_m * np.cos(angle_rad)
            y = y_sign * distance_m * np.sin(angle_rad)
        else:
            x = distance_m * np.sin(angle_rad - np.pi / 2)
            y = y_sign * distance_m * np.cos(angle_rad - np.pi / 2)
        rows.append(
            {
                "angle_deg": angle_deg,
                "x_m": x,
                "y_m": y,
                "z_m": 0.0,
            }
        )
    return rows


def extract_f8745(validation, rotor_module, output_root: Path):
    case_dir = output_root / "f8745_d4"
    rotor = rotor_module.F8745_D4_Propeller()
    beta = np.asarray(rotor.twist_distribution)
    beta = beta + np.deg2rad(21.0) - beta[round(len(beta) * 0.75)]
    geometry_rows = []
    for index in range(len(rotor.radius_distribution)):
        geometry_rows.append(
            {
                "station": index,
                "radius_m": rotor.radius_distribution[index],
                "radius_over_tip_radius": rotor.radius_distribution[index] / rotor.tip_radius,
                "chord_m": rotor.chord_distribution[index],
                "twist_deg": np.rad2deg(beta[index]),
                "thickness_to_chord": rotor.thickness_to_chord[index],
            }
        )
    write_rows(
        case_dir / "geometry.csv",
        geometry_rows[0].keys(),
        geometry_rows,
    )

    rpms = (2390.0, 2710.0, 2630.0)
    velocities = (77.2, 77.0, 77.2)
    condition_rows = [
        {
            "case": index + 1,
            "rpm": rpm,
            "axial_velocity_m_per_s": velocity,
            "density_kg_per_m3": 1.2250,
            "dynamic_viscosity_pa_s": 1.81e-5,
            "speed_of_sound_m_per_s": 343.376,
            "temperature_k": 288.16889478,
            "angle_of_attack_deg": 0.0,
            "three_quarter_radius_twist_deg": 21.0,
            "num_azimuthal_stations": 16,
        }
        for index, (rpm, velocity) in enumerate(zip(rpms, velocities))
    ]
    write_rows(
        case_dir / "operating_conditions.csv",
        condition_rows[0].keys(),
        condition_rows,
    )

    published_rows = [
        {
            "case": 1,
            "rpm": 2400.0,
            "axial_velocity_m_per_s": 77.2,
            "shaft_power_kw": 73.6,
            "measured_thrust_n": 642.0,
            "shaft_angle_deg": 0.0,
            "temperature_k": 290.3,
            "measured_three_quarter_radius_blade_angle_deg": 20.8,
            "computed_three_quarter_radius_blade_angle_deg": 22.3,
        },
        {
            "case": 2,
            "rpm": 2700.0,
            "axial_velocity_m_per_s": 77.0,
            "shaft_power_kw": 184.6,
            "measured_thrust_n": 1907.0,
            "shaft_angle_deg": 0.0,
            "temperature_k": 289.4,
            "measured_three_quarter_radius_blade_angle_deg": 20.8,
            "computed_three_quarter_radius_blade_angle_deg": 22.0,
        },
        {
            "case": 3,
            "rpm": 2700.0,
            "axial_velocity_m_per_s": 77.2,
            "shaft_power_kw": 152.1,
            "measured_thrust_n": 1500.0,
            "shaft_angle_deg": 0.0,
            "temperature_k": 287.0,
            "measured_three_quarter_radius_blade_angle_deg": 19.9,
            "computed_three_quarter_radius_blade_angle_deg": 21.2,
        },
    ]
    write_rows(
        case_dir / "published_operating_conditions.csv",
        published_rows[0].keys(),
        published_rows,
    )

    angles = (1.0, 10.0, 20.0, 30.1, 40.0, 50.0, 59.9, 70.0, 80.0, 89.9,
              100.0, 110.0, 120.1, 130.0, 140.0, 150.1, 160.0, 170.0, 179.0)
    write_rows(
        case_dir / "observers.csv",
        ("angle_deg", "x_m", "y_m", "z_m"),
        observer_positions(angles, 20.0, 1.0),
    )

    plot_parameters = validation.plot_parameters()
    data = validation.Hararmonic_Noise_Validation_Data(plot_parameters)[0]
    experimental_rows = []
    for case in range(1, 4):
        for angle in (60, 90):
            values = getattr(data, f"Exp_Test_Case_{case}_{angle}deg")
            for harmonic, spl_db in zip(data.harmonics, values):
                experimental_rows.append(
                    {
                        "case": case,
                        "observer_angle_reported_deg": angle,
                        "harmonic": int(harmonic),
                        "spl_db": spl_db,
                    }
                )
    write_rows(
        case_dir / "experimental_harmonics.csv",
        experimental_rows[0].keys(),
        experimental_rows,
    )


def extract_apc(validation, rotor_module, output_root: Path):
    case_dir = output_root / "apc_11x4"
    rotor = rotor_module.APC_11x4_Propeller()
    geometry_rows = []
    for index in range(len(rotor.radius_distribution)):
        geometry_rows.append(
            {
                "station": index,
                "radius_m": rotor.radius_distribution[index],
                "radius_over_tip_radius": rotor.radius_distribution[index] / rotor.tip_radius,
                "chord_m": rotor.chord_distribution[index],
                "twist_deg": np.rad2deg(rotor.twist_distribution[index]),
                "thickness_to_chord": rotor.thickness_to_chord[index],
            }
        )
    write_rows(case_dir / "geometry.csv", geometry_rows[0].keys(), geometry_rows)

    rpms = np.array((3600.0, 4200.0, 4800.0))
    tip_radius = float(rotor.tip_radius)
    velocities = 0.08 * rpms * 2.0 * np.pi / 60.0 * tip_radius
    condition_rows = [
        {
            "case": index + 1,
            "rpm": rpm,
            "inflow_ratio": 0.08,
            "axial_velocity_m_per_s": velocity,
            "density_kg_per_m3": 1.225,
            "dynamic_viscosity_pa_s": 1.78899787e-5,
            "speed_of_sound_m_per_s": 343.0,
            "temperature_k": 286.16889478,
            "angle_of_attack_deg": 0.0,
            "num_azimuthal_stations": 16,
        }
        for index, (rpm, velocity) in enumerate(zip(rpms, velocities))
    ]
    write_rows(
        case_dir / "operating_conditions.csv",
        condition_rows[0].keys(),
        condition_rows,
    )

    angles = (45.0, 67.5, 90.001, 112.5, 135.0)
    write_rows(
        case_dir / "observers.csv",
        ("angle_deg", "x_m", "y_m", "z_m"),
        observer_positions(angles, 1.905, -1.0),
    )

    plot_parameters = validation.plot_parameters()
    data = validation.Broadband_Noise_Validation_Data(plot_parameters)[-1]
    frequencies = np.asarray(data.Exp_APC_SF_freqency_spectrum)
    total_rows = []
    for rpm, values in zip(rpms, np.asarray(data.Exp_APC_SF_1_3_Spectrum)):
        for frequency, spl_db in zip(frequencies, values[: len(frequencies)]):
            total_rows.append(
                {
                    "rpm": rpm,
                    "one_third_octave_center_hz": frequency,
                    "total_spl_db": spl_db,
                }
            )
    write_rows(
        case_dir / "experimental_total_spectrum.csv",
        total_rows[0].keys(),
        total_rows,
    )

    broadband_rows = []
    reported_angles = (45.0, 22.5)
    for reported_angle, values in zip(reported_angles, np.asarray(data.Exp_broadband_APC)):
        for frequency, spl_db in zip(frequencies, values):
            broadband_rows.append(
                {
                    "rpm": 4200.0,
                    "observer_angle_reported_deg": reported_angle,
                    "one_third_octave_center_hz": frequency,
                    "broadband_spl_db": spl_db,
                }
            )
    write_rows(
        case_dir / "experimental_broadband_spectrum.csv",
        broadband_rows[0].keys(),
        broadband_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcaide-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures",
    )
    args = parser.parse_args()
    rcaide_root = args.rcaide_root.resolve()
    sys.path.insert(0, str(rcaide_root))

    vnv = rcaide_root / "VnV"
    validation = load_module(
        "frequency_domain_test",
        vnv / "Verification" / "analysis_aeroacoustics" / "frequency_domain_test.py",
    )
    rotor_dir = vnv / "Vehicles" / "Rotors"
    f8745 = load_module("f8745_geometry", rotor_dir / "F8745_D4_Propeller.py")
    apc = load_module("apc_geometry", rotor_dir / "APC_11x4_Propeller.py")
    extract_f8745(validation, f8745, args.output_root)
    extract_apc(validation, apc, args.output_root)


if __name__ == "__main__":
    main()
