from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "f8745_d4"
REPORTS = ROOT / "reports"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def energetic_spl(values):
    values = np.asarray(values, dtype=float)
    return 10.0 * np.log10(np.sum(10.0 ** (values / 10.0)))


def main():
    experiment = {
        (int(row["case"]), int(float(row["observer_angle_reported_deg"])), int(row["harmonic"])): float(row["spl_db"])
        for row in read_rows(FIXTURE / "experimental_harmonics.csv")
    }
    sources = {
        "bladead_lowson": (
            REPORTS / "bladead_f8475_bem_lowson_detailed.csv",
            "bladead_lowson_spl_db",
        ),
        "bladead_hanson": (
            REPORTS / "bladead_f8475_bem_hanson_detailed.csv",
            "bladead_spl_db",
        ),
        "rcaide_plane_source": (
            REPORTS / "rcaide_f8475_corrected_plane_source.csv",
            "rcaide_plane_source_spl_db",
        ),
    }
    predictions = {}
    for model, (path, value_field) in sources.items():
        for row in read_rows(path):
            angle_field = (
                "observer_angle_reported_deg"
                if "observer_angle_reported_deg" in row
                else "observer_angle_deg"
            )
            key = (
                int(row["case"]),
                int(float(row[angle_field])),
                int(row["harmonic"]),
            )
            predictions[(model,) + key] = float(row[value_field])

    detail = []
    grouped = defaultdict(list)
    for model in sources:
        for key, measured in sorted(experiment.items()):
            predicted = predictions[(model,) + key]
            error = predicted - measured
            case, angle, harmonic = key
            grouped[(model, case, angle)].append((measured, predicted, error))
            detail.append(
                {
                    "model": model,
                    "case": case,
                    "observer_angle_deg": angle,
                    "harmonic": harmonic,
                    "experimental_spl_db": measured,
                    "prediction_spl_db": predicted,
                    "signed_error_db": error,
                }
            )
    summary = []
    for (model, case, angle), values in sorted(grouped.items()):
        measured = np.array([value[0] for value in values])
        predicted = np.array([value[1] for value in values])
        error = np.array([value[2] for value in values])
        summary.append(
            {
                "model": model,
                "case": case,
                "observer_angle_deg": angle,
                "harmonic_mae_db": np.mean(np.abs(error)),
                "maximum_absolute_error_db": np.max(np.abs(error)),
                "overall_error_db": energetic_spl(predicted) - energetic_spl(measured),
            }
        )
    for path, rows in (
        (REPORTS / "f8475_corrected_three_model_detailed.csv", detail),
        (REPORTS / "f8475_corrected_three_model_summary.csv", summary),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# F8475 corrected-condition three-model comparison",
        "",
        "All models use the published 2400/2700/2700 RPM cases, reported temperatures,",
        "power-matched blade angle at physical r/R=0.75, nominal 4 m observers, and the",
        "published 60/90-degree polar angles. BladeAD Lowson and Hanson share BladeAD BEM",
        "loads. RCAIDE independently power-matches its own BEM and uses plane-source fidelity",
        "with its legacy +15 dB adjustment disabled.",
        "",
        "| Model | Case | Angle | Harmonic MAE | Maximum error | Overall error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['case']} | {row['observer_angle_deg']}° | "
            f"{row['harmonic_mae_db']:.3f} dB | "
            f"{row['maximum_absolute_error_db']:.3f} dB | "
            f"{row['overall_error_db']:+.3f} dB |"
        )
    lines.extend(
        [
            "",
            "The frozen acceptance gate is harmonic MAE <=3 dB and absolute overall error",
            "<=3 dB. Passing this gate does not remove the aerodynamic-source caveat: both",
            "BladeAD and RCAIDE overpredict measured thrust after matching measured power.",
            "",
            "## Interpretation",
            "",
            "- BladeAD Lowson passes the overall gate for all six comparisons and the harmonic",
            "  MAE gate for four of six; the two misses are Case 1 at 60 and 90 degrees",
            "  (3.147 and 3.166 dB).",
            "- BladeAD Hanson passes the overall gate at 60 degrees but fails harmonic MAE in",
            "  every comparison and underpredicts all three 90-degree overall levels by 8--9 dB.",
            "- RCAIDE plane-source passes both gates at 90 degrees, but overpredicts 60-degree",
            "  overall level by 9.9--11.7 dB and fails harmonic MAE there.",
            "",
            "Lowson is therefore the strongest current absolute model across both observer",
            "angles. The opposing Hanson and RCAIDE directivity biases rule out treating either",
            "as validation truth.",
        ]
    )
    (REPORTS / "f8475_corrected_three_model_comparison.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
