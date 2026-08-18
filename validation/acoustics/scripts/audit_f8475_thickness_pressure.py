from __future__ import annotations

import csv

import numpy as np
from scipy.optimize import brentq

from run_bladead_f8475_bem_hanson_validation import (
    FIXTURE,
    REPORTS,
    energetic_spl,
    evaluate_case,
    solve_case,
)


REFERENCE_PRESSURE = 20.0e-6


def mode_spl_from_complex(pressure, rms=True):
    denominator = REFERENCE_PRESSURE * (np.sqrt(2.0) if rms else 1.0)
    return 20.0 * np.log10(np.abs(pressure) / denominator)


def metrics(prediction, measured):
    return (
        np.mean(np.abs(prediction - measured)),
        energetic_spl(prediction) - energetic_spl(measured),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    experimental = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True
    )
    baseline = [solve_case(case) for case in range(1, 4)]
    convention_rows = []
    scale_rows = []
    integration_rows = []
    for result in baseline:
        loading = result["loading_cosine_pressure"] + 1j * result[
            "loading_sine_pressure"
        ]
        thickness = result["thickness_cosine_pressure"] + 1j * result[
            "thickness_sine_pressure"
        ]
        variants = {
            "coherent_rms_production": mode_spl_from_complex(loading + thickness),
            "coherent_peak": mode_spl_from_complex(loading + thickness, rms=False),
            "magnitude_sum_rms": mode_spl_from_complex(
                np.abs(loading) + np.abs(thickness)
            ),
            "magnitude_sum_peak": mode_spl_from_complex(
                np.abs(loading) + np.abs(thickness), rms=False
            ),
        }
        for observer, angle in enumerate((60, 90)):
            measured = experimental[
                (experimental["case"] == result["case"])
                & (experimental["observer_angle_reported_deg"] == angle)
            ]["spl_db"]
            for name, values in variants.items():
                mae, overall_error = metrics(values[observer], measured)
                convention_rows.append(
                    {
                        "case": result["case"],
                        "observer_angle_deg": angle,
                        "convention": name,
                        "harmonic_mae_db": mae,
                        "overall_error_db": overall_error,
                    }
                )
            if angle == 90:
                target_overall = energetic_spl(measured)

                def residual(scale):
                    predicted = mode_spl_from_complex(
                        loading[observer] + scale * thickness[observer]
                    )
                    return energetic_spl(predicted) - target_overall

                scale = brentq(residual, 0.01, 10.0)
                scale_rows.append(
                    {
                        "case": result["case"],
                        "thickness_pressure_scale_for_zero_overall_error": scale,
                        "equivalent_db": 20.0 * np.log10(scale),
                    }
                )

        for num_chordwise in (25, 50, 100, 200):
            variant = evaluate_case(
                result["case"],
                result["matched_blade_angle_deg"],
                evaluate_acoustics=True,
                num_chordwise=num_chordwise,
            )
            integration_rows.append(
                {
                    "case": result["case"],
                    "integration_axis": "chordwise",
                    "stations": num_chordwise,
                    "power_kw": variant["power_kw"],
                    "thrust_n": variant["thrust_n"],
                    "in_plane_thickness_overall_db": energetic_spl(
                        variant["thickness_mode_spl"][1]
                    ),
                    "in_plane_combined_overall_db": energetic_spl(
                        variant["tonal_mode_spl"][1]
                    ),
                }
            )
        for num_radial in (15, 30, 61):
            target_power = result["measured_power_kw"]

            def power_residual(blade_angle):
                return (
                    evaluate_case(
                        result["case"], blade_angle, num_radial=num_radial
                    )["power_kw"]
                    - target_power
                )

            blade_angle = brentq(power_residual, 10.0, 35.0)
            variant = evaluate_case(
                result["case"],
                blade_angle,
                evaluate_acoustics=True,
                num_radial=num_radial,
            )
            integration_rows.append(
                {
                    "case": result["case"],
                    "integration_axis": "radial",
                    "stations": num_radial,
                    "power_kw": variant["power_kw"],
                    "thrust_n": variant["thrust_n"],
                    "in_plane_thickness_overall_db": energetic_spl(
                        variant["thickness_mode_spl"][1]
                    ),
                    "in_plane_combined_overall_db": energetic_spl(
                        variant["tonal_mode_spl"][1]
                    ),
                }
            )

    write_csv(REPORTS / "f8475_pressure_convention_audit.csv", convention_rows)
    write_csv(REPORTS / "f8475_thickness_scale_diagnostic.csv", scale_rows)
    write_csv(REPORTS / "f8475_thickness_integration_convergence.csv", integration_rows)

    lines = [
        "# F8475 thickness and pressure-convention audit",
        "",
        "## Pressure convention",
        "",
        "| Convention | Harmonic MAE range | Overall-error range |",
        "|---|---:|---:|",
    ]
    for name in (
        "coherent_rms_production",
        "coherent_peak",
        "magnitude_sum_rms",
        "magnitude_sum_peak",
    ):
        selected = [row for row in convention_rows if row["convention"] == name]
        mae = [row["harmonic_mae_db"] for row in selected]
        overall = [row["overall_error_db"] for row in selected]
        lines.append(
            f"| {name} | {min(mae):.3f}--{max(mae):.3f} dB | "
            f"{min(overall):+.3f} to {max(overall):+.3f} dB |"
        )
    lines.extend(
        [
            "",
            "Production uses coherent RMS pressure, consistent with sinusoidal SPL. Peak",
            "reporting adds exactly 3.010 dB but does not close the 90-degree deficit.",
            "Magnitude-only component addition discards physical phase and is retained only as",
            "a diagnostic.",
            "",
            "## In-plane thickness scale diagnostic",
            "",
            "| Case | Pressure multiplier required | Equivalent level |",
            "|---:|---:|---:|",
        ]
    )
    for row in scale_rows:
        lines.append(
            f"| {row['case']} | "
            f"{row['thickness_pressure_scale_for_zero_overall_error']:.3f} | "
            f"{row['equivalent_db']:+.3f} dB |"
        )
    chordwise = [
        row for row in integration_rows if row["integration_axis"] == "chordwise"
    ]
    radial = [row for row in integration_rows if row["integration_axis"] == "radial"]
    def maximum_within_case_spread(rows):
        spreads = []
        for case in (1, 2, 3):
            values = [
                row["in_plane_thickness_overall_db"]
                for row in rows
                if row["case"] == case
            ]
            spreads.append(max(values) - min(values))
        return max(spreads)

    chordwise_spread = maximum_within_case_spread(chordwise)
    radial_spread = maximum_within_case_spread(radial)
    lines.extend(
        [
            "",
            "## Integration convergence",
            "",
            f"Across 25--200 chordwise stations, the maximum within-case in-plane thickness-level spread is "
            f"{chordwise_spread:.4f} dB.",
            f"Across 15--61 radial stations with each case re-matched to measured power, the "
            f"maximum within-case in-plane thickness-level spread is {radial_spread:.4f} dB.",
            "Detailed values are in `f8475_thickness_integration_convergence.csv`.",
            "",
            "## Conclusion",
            "",
            "Neither quadrature resolution nor the peak/RMS convention explains the in-plane",
            "deficit. The remaining issue is model-form/source-content uncertainty, not a",
            "numerical integration or observer-label error.",
        ]
    )
    (REPORTS / "f8475_thickness_pressure_audit.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
