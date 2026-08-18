from __future__ import annotations

import csv
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    evaluate_rotor_acoustics,
)
from BladeAD.utils.var_groups import AtmosStates, RotorAnalysisInputs, RotorMeshParameters
from run_bladead_f8475_bem_hanson_validation import clark_y_model


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "hartzell_f9684"
REPORTS = ROOT / "reports"
RADIUS_M = 1.015
NUM_BLADES = 2
GAMMA_AIR = 1.4
GAS_CONSTANT_AIR = 287.05287


def _variable(value):
    return csdl.Variable(value=np.asarray(value, dtype=float))


def energetic_spl(values_db):
    return 10.0 * np.log10(np.sum(10.0 ** (np.asarray(values_db) / 10.0)))


def _atmosphere(density, speed_of_sound):
    temperature = speed_of_sound**2 / (GAMMA_AIR * GAS_CONSTANT_AIR)
    viscosity = (
        1.716e-5
        * (temperature / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature + 110.4)
    )
    return temperature, viscosity


def evaluate_case(
    condition, tonal_model, max_harmonic, num_radial=40, num_azimuthal=16
):
    source_geometry = np.genfromtxt(FIXTURE / "geometry.csv", delimiter=",", names=True)
    radial_fraction = np.linspace(0.2, 0.99, num_radial)
    chord = RADIUS_M * np.interp(
        radial_fraction,
        source_geometry["radius_over_tip_radius"],
        source_geometry["chord_over_tip_radius"],
    )
    twist = np.interp(
        radial_fraction,
        source_geometry["radius_over_tip_radius"],
        source_geometry["twist_deg"],
    )
    thickness_to_chord = np.interp(
        radial_fraction,
        source_geometry["radius_over_tip_radius"],
        source_geometry["thickness_to_chord"],
    )
    twist += float(condition["three_quarter_radius_blade_angle_deg"]) - np.interp(
        0.75, radial_fraction, twist
    )

    density = float(condition["density_kg_per_m3"])
    speed_of_sound = float(condition["speed_of_sound_m_per_s"])
    temperature, viscosity = _atmosphere(density, speed_of_sound)
    rpm = (
        float(condition["tip_mach"])
        * speed_of_sound
        / (2.0 * np.pi * RADIUS_M)
        * 60.0
    )
    axial_velocity = float(condition["freestream_mach"]) * speed_of_sound
    shaft_frequency = rpm / 60.0
    diameter = 2.0 * RADIUS_M
    target_thrust = (
        float(condition["thrust_coefficient"])
        * density
        * shaft_frequency**2
        * diameter**4
    )
    target_power = (
        float(condition["power_coefficient"])
        * density
        * shaft_frequency**3
        * diameter**5
    )

    chordwise = np.linspace(-0.5, 0.5, 101)
    normalized_thickness = 4.0 * (chordwise + 0.5) * (0.5 - chordwise)
    weights = np.full(chordwise.size, chordwise[1] - chordwise[0])
    weights[[0, -1]] *= 0.5

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    mesh = RotorMeshParameters(
        thrust_vector=_variable([1.0, 0.0, 0.0]),
        thrust_origin=_variable([0.0, 0.0, 0.0]),
        chord_profile=_variable(chord),
        twist_profile=_variable(np.deg2rad(twist)),
        radius=_variable([RADIUS_M]),
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=NUM_BLADES,
        norm_hub_radius=0.2,
        thickness_to_chord=_variable(thickness_to_chord),
        normalized_thickness_shape=_variable(normalized_thickness),
        thickness_shape_chordwise_locations=_variable(chordwise),
        thickness_shape_chordwise_weights=_variable(weights),
    )
    inputs = RotorAnalysisInputs(
        rpm=_variable([rpm]),
        mesh_velocity=_variable([[axial_velocity, 0.0, 0.0]]),
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
    bem_thrust = float(bem.total_thrust.value[0])
    bem_power = float(bem.total_power.value[0])
    thrust_scale = target_thrust / bem_thrust
    drag_scale = target_power / bem_power
    bem.sectional_thrust = bem.sectional_thrust * thrust_scale
    bem.sectional_drag = bem.sectional_drag * drag_scale

    acoustic = evaluate_rotor_acoustics(
        inputs,
        bem,
        AcousticObserverData(positions=_variable([[0.0, 4.0, 0.0]])),
        RotorAcousticSettings(
            modes=tuple(range(1, max_harmonic + 1)),
            load_harmonics=(0,),
            tonal_model=tonal_model,
            tonal_enabled=True,
            thickness_enabled=True,
            a_weighting_enabled=False,
        ),
    )
    result = {
        "rpm": rpm,
        "axial_velocity_m_per_s": axial_velocity,
        "target_thrust_n": target_thrust,
        "target_power_kw": target_power / 1000.0,
        "bem_thrust_n": bem_thrust,
        "bem_power_kw": bem_power / 1000.0,
        "thrust_scale": thrust_scale,
        "drag_scale": drag_scale,
        "total_spl": acoustic.tonal_mode_spl.value.copy()[0, 0],
        "loading_spl": acoustic.loading_mode_spl.value.copy()[0, 0],
        "thickness_spl": acoustic.thickness_mode_spl.value.copy()[0, 0],
    }
    recorder.stop()
    return result


def main():
    conditions = np.genfromtxt(
        FIXTURE / "operating_conditions.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    experimental = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    detailed = []
    summary = []
    aerodynamic = []
    for condition in conditions:
        case = condition["case"]
        selected = experimental[experimental["case"] == case]
        harmonics = selected["bpf_harmonic"]
        measured = selected["spl_db"]
        max_harmonic = int(np.max(harmonics))
        for model in ("lowson", "hanson_line"):
            result = evaluate_case(condition, model, max_harmonic)
            predicted = result["total_spl"]
            for band, count in (("bpf1_6", 6), ("all_available", len(measured))):
                band_measured = measured[:count]
                band_predicted = predicted[:count]
                error = band_predicted - band_measured
                overall_error = energetic_spl(band_predicted) - energetic_spl(
                    band_measured
                )
                summary.append(
                    {
                        "case": case,
                        "model": model,
                        "evaluation_band": band,
                        "last_harmonic": count,
                        "harmonic_mae_db": np.mean(np.abs(error)),
                        "maximum_absolute_error_db": np.max(np.abs(error)),
                        "overall_error_db": overall_error,
                        "passes_frozen_gate": bool(
                            np.mean(np.abs(error)) <= 3.0
                            and abs(overall_error) <= 3.0
                        ),
                    }
                )
            for harmonic, measured_value, predicted_value, loading, thickness in zip(
                harmonics,
                measured,
                predicted,
                result["loading_spl"],
                result["thickness_spl"],
            ):
                detailed.append(
                    {
                        "case": case,
                        "model": model,
                        "harmonic": int(harmonic),
                        "experimental_spl_db": measured_value,
                        "bladead_loading_spl_db": loading,
                        "bladead_thickness_spl_db": thickness,
                        "bladead_total_spl_db": predicted_value,
                        "signed_error_db": predicted_value - measured_value,
                    }
                )
            if model == "lowson":
                aerodynamic.append(
                    {
                        "case": case,
                        **{
                            key: value
                            for key, value in result.items()
                            if not key.endswith("_spl")
                        },
                    }
                )

    for path, rows in (
        (REPORTS / "bladead_f9684_summary.csv", summary),
        (REPORTS / "bladead_f9684_detailed.csv", detailed),
        (REPORTS / "bladead_f9684_aerodynamics.csv", aerodynamic),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# BladeAD Hartzell F-9684-14 tonal validation",
        "",
        "The frozen gate uses the first six measured BPF harmonics at the DNW reference",
        "microphone in the propeller plane, 4 m from the axis. BEM supplies radial load shape;",
        "sectional thrust and drag are independently scaled so their integrals reproduce measured",
        "`C_T` and `C_P`. Lowson and Hanson use identical aerodynamic sources and geometry.",
        "",
        "| Case | Model | Band | Harmonic MAE | Maximum error | Overall error | Gate |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | {row['model']} | {row['evaluation_band']} | "
            f"{row['harmonic_mae_db']:.2f} dB | "
            f"{row['maximum_absolute_error_db']:.2f} dB | {row['overall_error_db']:+.2f} dB | "
            f"{'pass' if row['passes_frozen_gate'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The frozen gate remains BPF1--6 harmonic MAE <= 3 dB and absolute energetic overall",
            "error <= 3 dB. The all-available rows diagnose higher-harmonic roll-off and do not",
            "retroactively change that gate. Figure-digitization uncertainty and the absence of",
            "measured sectional loads remain explicit source limitations.",
            "",
            "Lowson retains the measured higher-harmonic roll-off: its all-available MAE remains",
            "1.72 dB through BC-4 BPF13 and 2.32 dB through AC-2 BPF24. Hanson instead falls",
            "progressively below the data, qualitatively resembling the excessive post-BPF6",
            "roll-off reported for the compact Shahjahan model. The higher harmonics barely alter",
            "the energetic totals because the first few tones dominate them.",
        ]
    )
    (REPORTS / "bladead_f9684_validation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
