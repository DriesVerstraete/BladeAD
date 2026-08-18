from __future__ import annotations

import csv

import numpy as np

from run_bladead_f8475_bem_hanson_validation import (
    FIXTURE,
    REPORTS,
    energetic_spl,
    evaluate_case,
    solve_case,
)


def main():
    experimental = np.genfromtxt(
        FIXTURE / "experimental_harmonics.csv", delimiter=",", names=True
    )
    rows = []
    summary = []
    for case in range(1, 4):
        matched = solve_case(case)
        result = evaluate_case(
            case,
            matched["matched_blade_angle_deg"],
            evaluate_acoustics=True,
            tonal_model="lowson",
        )
        for observer, angle in enumerate((60, 90)):
            selected = experimental[
                (experimental["case"] == case)
                & (experimental["observer_angle_reported_deg"] == angle)
            ]
            measured = selected["spl_db"]
            prediction = result["tonal_mode_spl"][observer]
            error = prediction - measured
            summary.append(
                {
                    "case": case,
                    "observer_angle_deg": angle,
                    "harmonic_mae_db": np.mean(np.abs(error)),
                    "overall_error_db": energetic_spl(prediction)
                    - energetic_spl(measured),
                }
            )
            for harmonic, measured_value, predicted_value, error_value in zip(
                selected["harmonic"], measured, prediction, error
            ):
                rows.append(
                    {
                        "case": case,
                        "observer_angle_deg": angle,
                        "harmonic": int(harmonic),
                        "experimental_spl_db": measured_value,
                        "bladead_lowson_spl_db": predicted_value,
                        "signed_error_db": error_value,
                    }
                )
    for path, data in (
        (REPORTS / "bladead_f8475_bem_lowson_detailed.csv", rows),
        (REPORTS / "bladead_f8475_bem_lowson_summary.csv", summary),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
