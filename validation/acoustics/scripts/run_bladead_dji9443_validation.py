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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "dji_9443"
REPORTS = ROOT / "reports"


def _variable(value):
    return csdl.Variable(value=np.asarray(value, dtype=float))


class RadiallyInterpolatedPolarOperation(csdl.CustomExplicitOperation):
    def __init__(self, radial_span, section_span, polars):
        self.radial_span = radial_span
        self.section_span = section_span
        self.polars = polars
        super().__init__()

    @staticmethod
    def _interpolate_polar(alpha, polar, coefficient):
        alpha_table = np.deg2rad(polar["Alpha"])
        values = polar[coefficient]
        prediction = np.interp(alpha, alpha_table, values)
        intervals = np.searchsorted(alpha_table, alpha, side="right") - 1
        intervals = np.clip(intervals, 0, len(alpha_table) - 2)
        slopes = np.diff(values) / np.diff(alpha_table)
        derivative = slopes[intervals]
        derivative = np.where(
            (alpha < alpha_table[0]) | (alpha > alpha_table[-1]), 0.0, derivative
        )
        return prediction, derivative

    def _predict(self, alpha):
        cl = np.zeros_like(alpha)
        cd = np.zeros_like(alpha)
        dcl = np.zeros_like(alpha)
        dcd = np.zeros_like(alpha)
        for radial_index, span in enumerate(self.radial_span):
            upper = np.searchsorted(self.section_span, span, side="right")
            upper = np.clip(upper, 1, len(self.section_span) - 1)
            lower = upper - 1
            denominator = self.section_span[upper] - self.section_span[lower]
            weight = (span - self.section_span[lower]) / denominator
            section_alpha = alpha[:, radial_index, :]
            cl_lower, dcl_lower = self._interpolate_polar(
                section_alpha, self.polars[lower], "Cl"
            )
            cl_upper, dcl_upper = self._interpolate_polar(
                section_alpha, self.polars[upper], "Cl"
            )
            cd_lower, dcd_lower = self._interpolate_polar(
                section_alpha, self.polars[lower], "Cd"
            )
            cd_upper, dcd_upper = self._interpolate_polar(
                section_alpha, self.polars[upper], "Cd"
            )
            cl[:, radial_index, :] = (1.0 - weight) * cl_lower + weight * cl_upper
            cd[:, radial_index, :] = (1.0 - weight) * cd_lower + weight * cd_upper
            dcl[:, radial_index, :] = (1.0 - weight) * dcl_lower + weight * dcl_upper
            dcd[:, radial_index, :] = (1.0 - weight) * dcd_lower + weight * dcd_upper
        return cl, cd, dcl, dcd

    def evaluate(self, alpha):
        self.declare_input("alpha", alpha)
        indices = np.arange(np.prod(alpha.shape))
        cl = self.create_output("Cl", alpha.shape)
        cd = self.create_output("Cd", alpha.shape)
        self.declare_derivative_parameters("Cl", "alpha", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cd", "alpha", rows=indices, cols=indices)
        return cl, cd

    def compute(self, input_vals, output_vals):
        cl, cd, _, _ = self._predict(input_vals["alpha"])
        output_vals["Cl"] = cl
        output_vals["Cd"] = cd

    def compute_derivatives(self, inputs, outputs, derivatives):
        _, _, dcl, dcd = self._predict(inputs["alpha"])
        derivatives["Cl", "alpha"] = dcl.ravel()
        derivatives["Cd", "alpha"] = dcd.ravel()


class DJI9443TabulatedAirfoilModel:
    def __init__(self, radial_fraction, hub_fraction):
        sections = np.genfromtxt(
            FIXTURE / "airfoil_sections.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
        self.radial_span = (radial_fraction - hub_fraction) / (1.0 - hub_fraction)
        self.section_span = sections["normalized_blade_span"]
        self.polars = [
            np.genfromtxt(
                FIXTURE / "airfoil_polars" / polar_file, delimiter=",", names=True
            )
            for polar_file in sections["polar_file"]
        ]

    def evaluate(self, alpha, Re, Ma):
        return RadiallyInterpolatedPolarOperation(
            self.radial_span, self.section_span, self.polars
        ).evaluate(alpha)


def energetic_spl(values_db):
    return 10.0 * np.log10(np.sum(10.0 ** (np.asarray(values_db) / 10.0)))


def evaluate_model(tonal_model, num_radial=40, num_azimuthal=16):
    condition = np.genfromtxt(
        FIXTURE / "operating_conditions.csv", delimiter=",", names=True
    )
    chord_source = np.genfromtxt(
        FIXTURE / "chord_distribution.csv", delimiter=",", names=True
    )
    twist_source = np.genfromtxt(
        FIXTURE / "twist_distribution.csv", delimiter=",", names=True
    )
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
        (
            observer_radius * np.sin(angles),
            observer_radius * np.cos(angles),
            np.zeros(5),
        )
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
        airfoil_model=DJI9443TabulatedAirfoilModel(radial_fraction, hub_fraction),
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
    experimental = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True
    )
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
                "source_model": "flowunsteady_section_polars",
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
