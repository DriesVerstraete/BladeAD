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
from BladeAD.core.airfoil.zero_d_airfoil_model import (
    ZeroDAirfoilModel,
    ZeroDAirfoilPolarParameters,
)
from BladeAD.utils.var_groups import AtmosStates, RotorAnalysisInputs, RotorMeshParameters


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "dji_9443"
REPORTS = ROOT / "reports"


def _variable(value):
    return csdl.Variable(value=np.asarray(value, dtype=float))


def fallback_airfoil_model():
    return ZeroDAirfoilModel(
        ZeroDAirfoilPolarParameters(
            alpha_stall_minus=-10.0,
            alpha_stall_plus=15.0,
            Cl_stall_minus=-1.0,
            Cl_stall_plus=1.5,
            Cd_stall_minus=0.02,
            Cd_stall_plus=0.06,
            Cl_0=0.5,
            Cd_0=0.008,
            Cl_alpha=5.1566,
        )
    )


def energetic_spl(values_db):
    return 10.0 * np.log10(np.sum(10.0 ** (np.asarray(values_db) / 10.0)))


def evaluate_model(tonal_model, num_radial=40, num_azimuthal=16):
    condition = np.genfromtxt(FIXTURE / "operating_conditions.csv", delimiter=",", names=True)
    chord_source = np.genfromtxt(FIXTURE / "chord_distribution.csv", delimiter=",", names=True)
    twist_source = np.genfromtxt(FIXTURE / "twist_distribution.csv", delimiter=",", names=True)
    observers = np.genfromtxt(FIXTURE / "observers.csv", delimiter=",", names=True)
    radius = float(condition["tip_radius_m"])
    hub_fraction = float(condition["hub_radius_m"] / radius)
    radial_fraction = np.linspace(hub_fraction, 0.99, num_radial)
    chord = radius * np.interp(
        radial_fraction,
        chord_source["radius_over_tip_radius"],
        chord_source["chord_over_tip_radius"],
    )
    twist = np.deg2rad(
        np.interp(
            radial_fraction,
            twist_source["radius_over_tip_radius"],
            twist_source["twist_deg"],
        )
    )
    angles = np.deg2rad(observers["reported_angle_from_rotor_plane_deg"])
    observer_radius = observers["radius_m"]
    observer_positions = np.column_stack(
        (observer_radius * np.sin(angles), observer_radius * np.cos(angles), np.zeros(5))
    )
    axial_velocity = (
        float(condition["advance_ratio"])
        * float(condition["rpm"])
        / 60.0
        * 2.0
        * radius
    )

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    mesh = RotorMeshParameters(
        thrust_vector=_variable([1.0, 0.0, 0.0]),
        thrust_origin=_variable([0.0, 0.0, 0.0]),
        chord_profile=_variable(chord),
        twist_profile=_variable(twist),
        radius=_variable([radius]),
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=int(condition["number_of_blades"]),
        norm_hub_radius=hub_fraction,
    )
    inputs = RotorAnalysisInputs(
        rpm=_variable([condition["rpm"]]),
        mesh_velocity=_variable([[axial_velocity, 0.0, 0.0]]),
        mesh_parameters=mesh,
    )
    inputs.atmos_states = AtmosStates(
        density=_variable([condition["density_kg_per_m3"]]),
        speed_of_sound=_variable([condition["speed_of_sound_m_per_s"]]),
        temperature=_variable([293.15]),
        dynamic_viscosity=_variable([condition["dynamic_viscosity_pa_s"]]),
    )
    bem = BEMModel(
        num_nodes=1,
        airfoil_model=fallback_airfoil_model(),
        integration_scheme="trapezoidal",
    ).evaluate(inputs)
    acoustic = evaluate_rotor_acoustics(
        inputs,
        bem,
        AcousticObserverData(positions=_variable(observer_positions)),
        RotorAcousticSettings(
            modes=(1, 2),
            load_harmonics=(0,),
            tonal_model=tonal_model,
            tonal_enabled=True,
            thickness_enabled=False,
            a_weighting_enabled=False,
        ),
    )
    n = float(condition["rpm"]) / 60.0
    diameter = 2.0 * radius
    thrust = float(bem.total_thrust.value[0])
    result = {
        "model": tonal_model,
        "thrust_n": thrust,
        "thrust_coefficient": thrust
        / (float(condition["density_kg_per_m3"]) * n**2 * diameter**4),
        "tonal_mode_spl": acoustic.tonal_mode_spl.value.copy()[0],
    }
    recorder.stop()
    return result


def main():
    experimental = np.genfromtxt(FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True)
    angles = np.array([-45.0, -22.5, 0.0, 22.5, 45.0])
    detailed_rows = []
    summary_rows = []
    for model in ("lowson", "hanson_line"):
        result = evaluate_model(model)
        errors = []
        for observer_index, angle in enumerate(angles):
            selected = experimental[experimental["observer_angle_reported_deg"] == angle]
            measured = selected["spl_db"]
            predicted = result["tonal_mode_spl"][observer_index]
            error = predicted - measured
            errors.extend(error)
            for harmonic, frequency, measured_value, predicted_value, error_value in zip(
                selected["harmonic"],
                selected["frequency_hz"],
                measured,
                predicted,
                error,
            ):
                detailed_rows.append(
                    {
                        "model": model,
                        "observer_angle_from_rotor_plane_deg": angle,
                        "harmonic": int(harmonic),
                        "frequency_hz": frequency,
                        "experimental_spl_db": measured_value,
                        "bladead_spl_db": predicted_value,
                        "signed_error_db": error_value,
                    }
                )
        measured_all = experimental["spl_db"].reshape(2, 5).T
        predicted_all = result["tonal_mode_spl"]
        summary_rows.append(
            {
                "model": model,
                "source_model": "generic_zero_d_polar",
                "thickness_enabled": False,
                "measured_thrust_coefficient": 0.072,
                "bladead_thrust_coefficient": result["thrust_coefficient"],
                "harmonic_mae_db": np.mean(np.abs(errors)),
                "maximum_absolute_error_db": np.max(np.abs(errors)),
                "two_harmonic_overall_error_db": energetic_spl(predicted_all.ravel())
                - energetic_spl(measured_all.ravel()),
            }
        )
    for path, rows in (
        (REPORTS / "bladead_dji9443_detailed.csv", detailed_rows),
        (REPORTS / "bladead_dji9443_summary.csv", summary_rows),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=rows[0].keys(), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
