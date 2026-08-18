from __future__ import annotations

import csv
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from scipy.optimize import brentq

from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    evaluate_rotor_acoustics,
)
from BladeAD.core.airfoil.zero_d_airfoil_model import (
    ZeroDAirfoilModel,
    ZeroDAirfoilPolarParameters,
)
from BladeAD.utils.var_groups import AtmosStates, RotorAnalysisInputs, RotorMeshParameters


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "f8745_d4"
REPORTS = ROOT / "reports"
RADIUS_M = 1.015
NUM_BLADES = 2
REFERENCE_PRESSURE_PA = 101325.0
GAS_CONSTANT_AIR = 287.05287
GAMMA_AIR = 1.4


def _variable(value):
    return csdl.Variable(value=np.asarray(value, dtype=float))


def clark_y_model():
    return ZeroDAirfoilModel(
        ZeroDAirfoilPolarParameters(
            alpha_stall_minus=-14.25,
            alpha_stall_plus=14.75,
            Cl_stall_minus=-1.0371,
            Cl_stall_plus=1.5240,
            Cd_stall_minus=0.03284,
            Cd_stall_plus=0.04222,
            Cl_0=0.3918,
            Cd_0=0.00645,
            Cl_alpha=6.26611449,
        )
    )


def atmosphere_from_temperature(temperature_k):
    density = REFERENCE_PRESSURE_PA / (GAS_CONSTANT_AIR * temperature_k)
    speed_of_sound = np.sqrt(GAMMA_AIR * GAS_CONSTANT_AIR * temperature_k)
    dynamic_viscosity = (
        1.716e-5
        * (temperature_k / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature_k + 110.4)
    )
    return density, speed_of_sound, dynamic_viscosity


def thickness_shape(num_chordwise=100):
    with np.load(
        FIXTURE / "rcaide_line_source_baseline.npz", allow_pickle=False
    ) as archive:
        y_upper = archive["rotor.airfoils.airfoil.geometry.y_upper_surface"]
        y_lower = archive["rotor.airfoils.airfoil.geometry.y_lower_surface"]
        x_upper = archive["rotor.airfoils.airfoil.geometry.x_upper_surface"]
    locations = np.linspace(0.0, 1.0, num_chordwise)
    full_thickness = np.interp(locations, x_upper, y_upper - y_lower)
    weights = np.full(num_chordwise, locations[1] - locations[0])
    weights[[0, -1]] *= 0.5
    return full_thickness / np.max(full_thickness), locations - 0.5, weights


def evaluate_case(
    case,
    blade_angle_deg,
    evaluate_acoustics=False,
    num_radial=30,
    num_chordwise=100,
    tonal_model="hanson_line",
):
    source_geometry = np.genfromtxt(FIXTURE / "geometry.csv", delimiter=",", names=True)
    radial_fraction = np.linspace(0.2, 0.98, num_radial)
    geometry = {
        name: np.interp(
            radial_fraction,
            source_geometry["radius_over_tip_radius"],
            source_geometry[name],
        )
        for name in ("chord_m", "twist_deg", "thickness_to_chord")
    }
    published = np.genfromtxt(
        FIXTURE / "published_operating_conditions.csv", delimiter=",", names=True
    )[case - 1]
    reference_blade_angle = np.interp(0.75, radial_fraction, geometry["twist_deg"])
    blade_angle_offset = blade_angle_deg - reference_blade_angle
    temperature = float(published["temperature_k"])
    density, speed_of_sound, viscosity = atmosphere_from_temperature(temperature)
    normalized_thickness, chordwise_locations, chordwise_weights = thickness_shape(
        num_chordwise
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    mesh = RotorMeshParameters(
        thrust_vector=_variable([1.0, 0.0, 0.0]),
        thrust_origin=_variable([0.0, 0.0, 0.0]),
        chord_profile=_variable(geometry["chord_m"]),
        twist_profile=_variable(
            np.deg2rad(geometry["twist_deg"] + blade_angle_offset)
        ),
        radius=_variable([RADIUS_M]),
        num_radial=num_radial,
        num_azimuthal=16,
        num_blades=NUM_BLADES,
        norm_hub_radius=0.2,
        thickness_to_chord=_variable(geometry["thickness_to_chord"]),
        normalized_thickness_shape=_variable(normalized_thickness),
        thickness_shape_chordwise_locations=_variable(chordwise_locations),
        thickness_shape_chordwise_weights=_variable(chordwise_weights),
    )
    inputs = RotorAnalysisInputs(
        rpm=_variable([published["rpm"]]),
        mesh_velocity=_variable([[published["axial_velocity_m_per_s"], 0.0, 0.0]]),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = AtmosStates(
        density=_variable([density]),
        speed_of_sound=_variable([speed_of_sound]),
        temperature=_variable([temperature]),
        dynamic_viscosity=_variable([viscosity]),
    )
    bem = BEMModel(
        num_nodes=1,
        airfoil_model=clark_y_model(),
        integration_scheme="trapezoidal",
    ).evaluate(inputs)
    result = {
        "power_kw": float(bem.total_power.value[0] / 1000.0),
        "thrust_n": float(bem.total_thrust.value[0]),
        "torque_nm": float(bem.total_torque.value[0]),
    }
    if evaluate_acoustics:
        angles = np.deg2rad([60.0, 90.0])
        observer_positions = np.column_stack(
            (-4.0 * np.cos(angles), 4.0 * np.sin(angles), np.zeros(2))
        )
        acoustic = evaluate_rotor_acoustics(
            inputs,
            bem,
            AcousticObserverData(positions=_variable(observer_positions)),
            RotorAcousticSettings(
                modes=tuple(range(1, 19)),
                load_harmonics=(0,),
                tonal_model=tonal_model,
                tonal_enabled=True,
                thickness_enabled=True,
                a_weighting_enabled=False,
            ),
        )
        result["tonal_mode_spl"] = acoustic.tonal_mode_spl.value.copy()[0]
        result["loading_mode_spl"] = acoustic.loading_mode_spl.value.copy()[0]
        result["thickness_mode_spl"] = acoustic.thickness_mode_spl.value.copy()[0]
        if tonal_model == "hanson_line":
            result["loading_cosine_pressure"] = (
                acoustic.loading_cosine_pressure.value.copy()[0]
            )
            result["loading_sine_pressure"] = (
                acoustic.loading_sine_pressure.value.copy()[0]
            )
            result["thickness_cosine_pressure"] = (
                acoustic.thickness_cosine_pressure.value.copy()[0]
            )
            result["thickness_sine_pressure"] = (
                acoustic.thickness_sine_pressure.value.copy()[0]
            )
    recorder.stop()
    return result


def solve_case(case):
    published = np.genfromtxt(
        FIXTURE / "published_operating_conditions.csv", delimiter=",", names=True
    )[case - 1]
    target_power = float(published["shaft_power_kw"])

    def residual(blade_angle_deg):
        return evaluate_case(case, blade_angle_deg)["power_kw"] - target_power

    blade_angle_deg = brentq(residual, 10.0, 35.0, xtol=1e-6)
    result = evaluate_case(case, blade_angle_deg, evaluate_acoustics=True)
    result.update(
        {
            "case": case,
            "rpm": float(published["rpm"]),
            "axial_velocity_m_per_s": float(published["axial_velocity_m_per_s"]),
            "matched_blade_angle_deg": blade_angle_deg,
            "paper_computed_blade_angle_deg": float(
                published["computed_three_quarter_radius_blade_angle_deg"]
            ),
            "paper_measured_blade_angle_deg": float(
                published["measured_three_quarter_radius_blade_angle_deg"]
            ),
            "measured_power_kw": target_power,
            "measured_thrust_n": float(published["measured_thrust_n"]),
        }
    )
    return result


def energetic_spl(values_db):
    return 10.0 * np.log10(np.sum(10.0 ** (np.asarray(values_db) / 10.0)))


def main():
    experimental = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True
    )
    results = [solve_case(case) for case in range(1, 4)]
    aerodynamic_rows = []
    acoustic_rows = []
    for result in results:
        aerodynamic_rows.append(
            {
                "case": result["case"],
                "rpm": result["rpm"],
                "axial_velocity_m_per_s": result["axial_velocity_m_per_s"],
                "matched_blade_angle_deg": result["matched_blade_angle_deg"],
                "paper_computed_blade_angle_deg": result[
                    "paper_computed_blade_angle_deg"
                ],
                "paper_measured_blade_angle_deg": result[
                    "paper_measured_blade_angle_deg"
                ],
                "measured_power_kw": result["measured_power_kw"],
                "bladead_power_kw": result["power_kw"],
                "measured_thrust_n": result["measured_thrust_n"],
                "bladead_thrust_n": result["thrust_n"],
                "thrust_error_percent": 100.0
                * (result["thrust_n"] - result["measured_thrust_n"])
                / result["measured_thrust_n"],
            }
        )
        for observer, angle in enumerate((60, 90)):
            selected = experimental[
                (experimental["case"] == result["case"])
                & (experimental["observer_angle_reported_deg"] == angle)
            ]
            prediction = result["tonal_mode_spl"][observer]
            measured = selected["spl_db"]
            error = prediction - measured
            for harmonic, measured_value, predicted_value, error_value in zip(
                selected["harmonic"], measured, prediction, error
            ):
                acoustic_rows.append(
                    {
                        "case": result["case"],
                        "observer_angle_deg": angle,
                        "harmonic": int(harmonic),
                        "experimental_spl_db": measured_value,
                        "bladead_spl_db": predicted_value,
                        "signed_error_db": error_value,
                    }
                )

    for path, rows in (
        (REPORTS / "bladead_f8475_bem_aerodynamics.csv", aerodynamic_rows),
        (REPORTS / "bladead_f8475_bem_hanson_detailed.csv", acoustic_rows),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# F8475 D-4 BladeAD BEM-to-Hanson validation",
        "",
        "This path uses the fixture geometry, a Clark-Y ZeroD polar fitted to the RCAIDE",
        "Re=1,000,000 XFOIL table, BladeAD BEM loads, and BladeAD Hanson acoustics. No RCAIDE",
        "aerodynamic loads enter the calculation. Blade angle is independently adjusted to",
        "match each Table 4 shaft power, following Weir and Powers.",
        "",
        "Pressure was not reported in Table 4; density is inferred using 101325 Pa and the",
        "reported temperature. Clark-Y Reynolds-number variation is not represented.",
        "",
        "## Aerodynamics",
        "",
        "| Case | BladeAD blade angle | Paper computed angle | Power (kW) | Thrust measured/predicted (N) | Thrust error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aerodynamic_rows:
        lines.append(
            f"| {row['case']} | {row['matched_blade_angle_deg']:.3f}° | "
            f"{row['paper_computed_blade_angle_deg']:.3f}° | "
            f"{row['bladead_power_kw']:.3f} | {row['measured_thrust_n']:.1f} / "
            f"{row['bladead_thrust_n']:.1f} | {row['thrust_error_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Hanson tonal comparison at 4 m",
            "",
            "| Case | Angle | Harmonic MAE | Overall error |",
            "|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        for observer, angle in enumerate((60, 90)):
            selected = experimental[
                (experimental["case"] == result["case"])
                & (experimental["observer_angle_reported_deg"] == angle)
            ]
            prediction = result["tonal_mode_spl"][observer]
            measured = selected["spl_db"]
            lines.append(
                f"| {result['case']} | {angle}° | "
                f"{np.mean(np.abs(prediction - measured)):.3f} dB | "
                f"{energetic_spl(prediction) - energetic_spl(measured):+.3f} dB |"
            )
    lines.extend(
        [
            "",
            "## Source-component energetic levels",
            "",
            "| Case | Angle | Loading | Thickness | Combined |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        for observer, angle in enumerate((60, 90)):
            lines.append(
                f"| {result['case']} | {angle}° | "
                f"{energetic_spl(result['loading_mode_spl'][observer]):.3f} dB | "
                f"{energetic_spl(result['thickness_mode_spl'][observer]):.3f} dB | "
                f"{energetic_spl(result['tonal_mode_spl'][observer]):.3f} dB |"
            )
    lines.extend(
        [
            "",
            "The observer convention is audited in `f8475_directivity_audit.md`. The",
            "paper-to-BladeAD angle mapping is correct; at 90 degrees the axial-loading term",
            "vanishes and thickness is the largest isolated component. The residual in-plane",
            "deficit is therefore not corrected by mirroring or relabeling the observer.",
        ]
    )
    (REPORTS / "f8475_bladead_bem_hanson_validation.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
