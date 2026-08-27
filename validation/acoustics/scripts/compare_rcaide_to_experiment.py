from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def energetic_spl(values_db):
    values_db = np.asarray(values_db, dtype=float)
    return 10.0 * np.log10(np.sum(10.0 ** (values_db / 10.0)))


def append_comparison(summary, detail, case, model, component, angle, x_name, x, experiment, prediction):
    experiment = np.asarray(experiment, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    signed_error = prediction - experiment
    overall_experiment = energetic_spl(experiment)
    overall_prediction = energetic_spl(prediction)
    summary.append(
        {
            "case": case,
            "model": model,
            "component": component,
            "reported_observer_angle_deg": angle,
            "points": len(experiment),
            "mean_signed_error_db": np.mean(signed_error),
            "mean_absolute_error_db": np.mean(np.abs(signed_error)),
            "maximum_absolute_error_db": np.max(np.abs(signed_error)),
            "experimental_energetic_overall_db": overall_experiment,
            "prediction_energetic_overall_db": overall_prediction,
            "overall_error_db": overall_prediction - overall_experiment,
        }
    )
    for coordinate, measured, predicted, error in zip(x, experiment, prediction, signed_error):
        detail.append(
            {
                "case": case,
                "model": model,
                "component": component,
                "reported_observer_angle_deg": angle,
                "coordinate_name": x_name,
                "coordinate": coordinate,
                "experimental_spl_db": measured,
                "prediction_spl_db": predicted,
                "signed_error_db": error,
            }
        )


def compare_f8745(summary, detail):
    rows = read_csv(FIXTURES / "f8745_d4" / "experimental_harmonics.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["case"]), int(row["observer_angle_reported_deg"]))].append(row)
    observer_index = {60: 6, 90: 9}
    archives = {
        "rcaide_line_source": "rcaide_line_source_baseline.npz",
        "rcaide_plane_source": "rcaide_plane_source_baseline.npz",
    }
    key = "acoustics.converters.F8745_D4_Propeller.SPL_harmonic_bpf_spectrum"
    for model, filename in archives.items():
        with np.load(FIXTURES / "f8745_d4" / filename, allow_pickle=False) as baseline:
            spectrum = baseline[key]
            for (case, angle), experimental_rows in sorted(grouped.items()):
                harmonics = [int(row["harmonic"]) for row in experimental_rows]
                experimental = [float(row["spl_db"]) for row in experimental_rows]
                prediction = spectrum[case - 1, observer_index[angle], : len(harmonics)]
                append_comparison(
                    summary,
                    detail,
                    f"F8745-D4-{case}",
                    model,
                    "tonal_harmonics",
                    angle,
                    "harmonic",
                    harmonics,
                    experimental,
                    prediction,
                )


def compare_apc(summary, detail):
    total_rows = read_csv(FIXTURES / "apc_11x4" / "experimental_total_spectrum.csv")
    broadband_rows = read_csv(
        FIXTURES / "apc_11x4" / "experimental_broadband_spectrum.csv"
    )
    archive = FIXTURES / "apc_11x4" / "rcaide_plane_source_baseline.npz"
    total_key = "acoustics.converters.APC_11x4_Propeller.SPL_1_3_spectrum"
    broadband_key = "acoustics.converters.APC_11x4_Propeller.SPL_broadband_1_3_spectrum"
    total_grouped = defaultdict(list)
    for row in total_rows:
        total_grouped[int(float(row["rpm"]))].append(row)
    broadband_grouped = defaultdict(list)
    for row in broadband_rows:
        broadband_grouped[float(row["observer_angle_reported_deg"])].append(row)

    with np.load(archive, allow_pickle=False) as baseline:
        total_spectrum = baseline[total_key]
        broadband_spectrum = baseline[broadband_key]
        rpm_case = {3600: 0, 4200: 1, 4800: 2}
        for rpm, experimental_rows in sorted(total_grouped.items()):
            frequencies = [float(row["one_third_octave_center_hz"]) for row in experimental_rows]
            experimental = [float(row["total_spl_db"]) for row in experimental_rows]
            prediction = total_spectrum[rpm_case[rpm], 0, 8:29]
            append_comparison(
                summary,
                detail,
                f"APC-11x4-{rpm}-RPM",
                "rcaide_plane_source",
                "total_one_third_octave",
                45.0,
                "frequency_hz",
                frequencies,
                experimental,
                prediction,
            )

        resolved_geometry_mapping = {45.0: 4, 22.5: 3}
        for angle, experimental_rows in sorted(broadband_grouped.items()):
            frequencies = [float(row["one_third_octave_center_hz"]) for row in experimental_rows]
            experimental = [float(row["broadband_spl_db"]) for row in experimental_rows]
            prediction = broadband_spectrum[1, resolved_geometry_mapping[angle], 8:29]
            append_comparison(
                summary,
                detail,
                "APC-11x4-4200-RPM",
                "rcaide_plane_source_resolved_geometry_mapping",
                "broadband_one_third_octave",
                angle,
                "frequency_hz",
                frequencies,
                experimental,
                prediction,
            )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary):
    lines = [
        "# Acoustic validation matrix",
        "",
        "This matrix was frozen before any BladeAD tonal or broadband model prediction existed.",
        "RCAIDE is a comparison point, not validation truth.",
        "",
        "## Acceptance criteria for BladeAD",
        "",
        "- Tonal: absolute overall error <= 3 dB and mean per-harmonic absolute error <= 3 dB.",
        "- Broadband: absolute overall error <= 3 dB and mean band error <= 5 dB over bands",
        "  materially above the measurement/background floor.",
        "- Trends: no unexplained systematic observer-angle, RPM, or harmonic/frequency trend.",
        "- Derivatives: relative error <= 1e-5 when well scaled; absolute error <= 1e-7 near zero,",
        "  with convergence over at least three finite-difference step sizes.",
        "",
        "These thresholds are not applied to force RCAIDE to pass and must not be relaxed after",
        "seeing BladeAD results. A failed model is narrowed, extended, or rejected rather than tuned",
        "to these fixtures.",
        "",
        "## RCAIDE comparison results",
        "",
        "| Case | Model | Component | Angle (deg) | N | MAE (dB) | Max (dB) | Overall error (dB) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | {row['model']} | {row['component']} | "
            f"{float(row['reported_observer_angle_deg']):.1f} | {row['points']} | "
            f"{float(row['mean_absolute_error_db']):.3f} | "
            f"{float(row['maximum_absolute_error_db']):.3f} | "
            f"{float(row['overall_error_db']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Detailed signed errors are in `rcaide_vs_experiment_detailed.csv`; unrounded summary",
            "metrics are in `rcaide_vs_experiment_summary.csv`.",
            "",
            "## Resolved observer mapping",
            "",
            "The experimental labels are downstream angles from the rotor plane. RCAIDE observer",
            "indices 4 and 3 have driver parameters 135 and 112.5 degrees, but their Cartesian",
            "positions resolve to 45 and 22.5 degrees from the rotor plane. The rows labelled",
            "`rcaide_plane_source_resolved_geometry_mapping` use that physical equivalence.",
        ]
    )
    marker = "## Physical-validation readiness"
    if path.exists():
        existing = path.read_text()
        if marker in existing:
            lines.extend(["", existing[existing.index(marker) :].rstrip()])
    path.write_text("\n".join(lines) + "\n")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = []
    detail = []
    compare_f8745(summary, detail)
    compare_apc(summary, detail)
    write_csv(REPORTS / "rcaide_vs_experiment_summary.csv", summary)
    write_csv(REPORTS / "rcaide_vs_experiment_detailed.csv", detail)
    write_markdown(REPORTS / "validation_matrix.md", summary)


if __name__ == "__main__":
    main()
