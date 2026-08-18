from __future__ import annotations

import csv
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    evaluate_rotor_acoustics,
)
from BladeAD.utils.var_groups import (
    AtmosStates,
    RotorAnalysisInputs,
    RotorAnalysisOutputs,
    RotorMeshParameters,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "f8745_d4"
REPORTS = ROOT / "reports"
MODEL_NAME = "bladead_lowson_bm_rcaide_aero"


def _variable(value):
    return csdl.Variable(value=np.asarray(value, dtype=float))


def _energetic_spl(values_db):
    values_db = np.asarray(values_db, dtype=float)
    return 10.0 * np.log10(np.sum(10.0 ** (values_db / 10.0)))


def evaluate_f8745(
    source_velocity_scale=1.0,
    return_components=False,
    tonal_model="lowson",
    hanson_legacy_adapter=False,
    load_scale=1.0,
    radial_redistribution=0.0,
):
    geometry = np.genfromtxt(FIXTURE / "geometry.csv", delimiter=",", names=True)
    observers = np.genfromtxt(FIXTURE / "observers.csv", delimiter=",", names=True)
    conditions = np.genfromtxt(
        FIXTURE / "operating_conditions.csv", delimiter=",", names=True
    )
    archive_path = FIXTURE / "rcaide_line_source_baseline.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        thrust = archive[
            "energy.converters.F8745_D4_Propeller.disc_thrust_distribution"
        ]
        torque = archive[
            "energy.converters.F8745_D4_Propeller.disc_torque_distribution"
        ]
        azimuth = archive[
            "energy.converters.F8745_D4_Propeller.disc_azimuthal_distribution"
        ]
        radius = archive["energy.converters.F8745_D4_Propeller.disc_radial_distribution"]
        y_upper = archive["rotor.airfoils.airfoil.geometry.y_upper_surface"]
        y_lower = archive["rotor.airfoils.airfoil.geometry.y_lower_surface"]
        x_upper = archive["rotor.airfoils.airfoil.geometry.x_upper_surface"]

    radial_shape = 2.0 * (radius / 1.015) - 1.0
    redistribution = 1.0 + radial_redistribution * radial_shape

    def redistribute_fixed_total(load):
        shifted = load * redistribution
        return shifted * (
            np.sum(load, axis=1, keepdims=True)
            / np.sum(shifted, axis=1, keepdims=True)
        )

    thrust = load_scale * redistribute_fixed_total(thrust)
    torque = load_scale * redistribute_fixed_total(torque)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    shape = thrust.shape
    num_cases, num_radial, num_azimuthal = shape
    num_blades = 2
    radial_width = np.broadcast_to(
        np.gradient(geometry["radius_m"])[None, :, None], shape
    ).copy()
    if tonal_model == "hanson_line":
        radial_width[:, (0, -1), :] *= 0.5
    adapter_scale = np.ones(shape)
    if hanson_legacy_adapter:
        adapter_scale = num_azimuthal * radial_width / 1.015
    zero_field = _variable(np.zeros(shape))
    zero_case = _variable(np.zeros(num_cases))
    full_thickness = y_upper - y_lower
    normalized_thickness_shape = full_thickness / np.max(full_thickness)
    chordwise_locations = x_upper - 0.5
    chordwise_weights = np.full(len(x_upper), x_upper[1] - x_upper[0])
    chordwise_weights[[0, -1]] *= 0.5
    mesh = RotorMeshParameters(
        thrust_vector=_variable(np.array([1.0, 0.0, 0.0])),
        thrust_origin=_variable(np.zeros(3)),
        chord_profile=_variable(geometry["chord_m"]),
        twist_profile=_variable(np.deg2rad(geometry["twist_deg"])),
        radius=_variable(np.full(num_cases, 1.015)),
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        norm_hub_radius=0.2,
        thickness_to_chord=_variable(geometry["thickness_to_chord"]),
        normalized_thickness_shape=_variable(normalized_thickness_shape),
        thickness_shape_chordwise_locations=_variable(chordwise_locations),
        thickness_shape_chordwise_weights=_variable(chordwise_weights),
    )
    atmosphere = AtmosStates(
        density=_variable(conditions["density_kg_per_m3"]),
        speed_of_sound=_variable(conditions["speed_of_sound_m_per_s"]),
        temperature=_variable(conditions["temperature_k"]),
        dynamic_viscosity=_variable(conditions["dynamic_viscosity_pa_s"]),
    )
    inputs = RotorAnalysisInputs(
        rpm=_variable(conditions["rpm"]),
        mesh_velocity=_variable(
            np.column_stack(
                (
                    source_velocity_scale * conditions["axial_velocity_m_per_s"],
                    np.zeros((num_cases, 2)),
                )
            )
        ),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = atmosphere
    outputs = RotorAnalysisOutputs(
        axial_induced_velocity=zero_field,
        tangential_induced_velocity=zero_field,
        sectional_thrust=_variable(num_blades * thrust * adapter_scale),
        sectional_torque=_variable(num_blades * torque * adapter_scale),
        sectional_drag=_variable(num_blades * torque / radius * adapter_scale),
        total_thrust=zero_case,
        total_torque=zero_case,
        total_power=zero_case,
        efficiency=zero_case,
        figure_of_merit=zero_case,
        thrust_coefficient=zero_case,
        torque_coefficient=zero_case,
        power_coefficient=zero_case,
        radial_stations=_variable(radius),
        radial_element_width=_variable(radial_width),
        radial_integration_weights=_variable(np.ones(shape)),
        azimuth_angle=_variable(azimuth),
        sectional_loads_include_all_blades=True,
    )
    observer_indices = (6, 9)
    observer_positions = np.column_stack(
        (
            observers["x_m"][list(observer_indices)],
            observers["y_m"][list(observer_indices)],
            observers["z_m"][list(observer_indices)],
        )
    )
    acoustics = evaluate_rotor_acoustics(
        inputs,
        outputs,
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
    prediction = acoustics.tonal_mode_spl.value.copy()
    if return_components:
        prediction = {
            "combined": prediction,
            "loading": acoustics.loading_mode_spl.value.copy(),
            "thickness": acoustics.thickness_mode_spl.value.copy(),
        }
        if tonal_model == "hanson_line":
            prediction.update(
                {
                    "loading_cosine_pressure": acoustics.loading_cosine_pressure.value.copy(),
                    "loading_sine_pressure": acoustics.loading_sine_pressure.value.copy(),
                    "thickness_cosine_pressure": acoustics.thickness_cosine_pressure.value.copy(),
                    "thickness_sine_pressure": acoustics.thickness_sine_pressure.value.copy(),
                }
            )
    recorder.stop()
    return prediction


def compare_to_experiment(prediction, model_name=MODEL_NAME):
    experimental_rows = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True
    )
    summary = []
    detail = []
    for case in range(1, 4):
        for observer_index, angle in enumerate((60, 90)):
            selected = experimental_rows[
                (experimental_rows["case"] == case)
                & (experimental_rows["observer_angle_reported_deg"] == angle)
            ]
            experiment = selected["spl_db"]
            predicted = prediction[case - 1, observer_index, :]
            signed_error = predicted - experiment
            overall_experiment = _energetic_spl(experiment)
            overall_prediction = _energetic_spl(predicted)
            mean_absolute_error = np.mean(np.abs(signed_error))
            overall_error = overall_prediction - overall_experiment
            summary.append(
                {
                    "case": f"F8745-D4-{case}",
                    "model": model_name,
                    "component": "tonal_harmonics",
                    "reported_observer_angle_deg": angle,
                    "points": len(experiment),
                    "mean_signed_error_db": np.mean(signed_error),
                    "mean_absolute_error_db": mean_absolute_error,
                    "maximum_absolute_error_db": np.max(np.abs(signed_error)),
                    "experimental_energetic_overall_db": overall_experiment,
                    "prediction_energetic_overall_db": overall_prediction,
                    "overall_error_db": overall_error,
                    "passes_frozen_gate": bool(
                        mean_absolute_error <= 3.0 and abs(overall_error) <= 3.0
                    ),
                }
            )
            for harmonic, measured, predicted_value, error in zip(
                selected["harmonic"], experiment, predicted, signed_error
            ):
                detail.append(
                    {
                        "case": f"F8745-D4-{case}",
                        "model": model_name,
                        "reported_observer_angle_deg": angle,
                        "harmonic": int(harmonic),
                        "experimental_spl_db": measured,
                        "prediction_spl_db": predicted_value,
                        "signed_error_db": error,
                    }
                )
    return summary, detail


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, summary):
    lines = [
        "# F8745-D4 BladeAD tonal validation",
        "",
        "This comparison evaluates BladeAD Lowson loading plus Barry–Magliozzi thickness noise",
        "using the frozen RCAIDE line-source run's aerodynamic disk loads. BladeAD BEM is not",
        "used. The result measures the complete load-adapter, propagation, and acoustic-model",
        "chain; the accompanying interface audit separates those contributions where possible.",
        "",
        "No calibration or fixture-specific correction is applied.",
        "",
        "| Case | Angle (deg) | MAE (dB) | Max (dB) | Overall error (dB) | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | {row['reported_observer_angle_deg']} | "
            f"{row['mean_absolute_error_db']:.3f} | "
            f"{row['maximum_absolute_error_db']:.3f} | "
            f"{row['overall_error_db']:.3f} | FAIL |"
        )
    lines.extend(
        [
            "",
            "The frozen gate requires both absolute overall error and mean per-harmonic absolute",
            "error to be no greater than 3 dB. Every case fails both criteria. BladeAD",
            "systematically underpredicts the measured harmonics, so the current Lowson model",
            "must not yet be used as experimental design authority for this forward-flight case.",
            "",
            "The failure does not invalidate the HG equation/reference verification. It shows that",
            "the current BladeAD acoustic chain does not reproduce the mechanisms or source",
            "representation captured by the F8745-D4 measurements and RCAIDE Hanson models. The",
            "audit rules out BEM and basic load conversion, but retains propagation convention as",
            "an unresolved contributor.",
            "",
            "Detailed harmonic results are in `bladead_f8745_detailed.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary, detail = compare_to_experiment(evaluate_f8745())
    _write_csv(REPORTS / "bladead_f8745_summary.csv", summary)
    _write_csv(REPORTS / "bladead_f8745_detailed.csv", detail)
    _write_report(REPORTS / "f8745_bladead_validation.md", summary)


if __name__ == "__main__":
    main()
