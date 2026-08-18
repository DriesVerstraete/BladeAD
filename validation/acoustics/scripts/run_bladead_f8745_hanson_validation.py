from __future__ import annotations

import csv

import numpy as np

from run_bladead_f8745_validation import (
    FIXTURE,
    REPORTS,
    compare_to_experiment,
    evaluate_f8745,
)


MODEL_NAME = "bladead_hanson_line_physical_adapter"


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _rcaide_comparison(prediction):
    with np.load(FIXTURE / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
        rcaide = archive[
            "acoustics.converters.F8745_D4_Propeller.SPL_harmonic_bpf_spectrum"
        ][:, (6, 9), 1:19]
        adjustment = float(archive["settings.wing_wake_interactional_dB_adjustment"])
    rows = []
    for case in range(3):
        for observer, angle in enumerate((60, 90)):
            difference = prediction[case, observer] - rcaide[case, observer]
            unadjusted_difference = prediction[case, observer] - (
                rcaide[case, observer] - adjustment
            )
            rows.append(
                {
                    "case": f"F8745-D4-{case + 1}",
                    "reported_observer_angle_deg": angle,
                    "mean_signed_difference_db": np.mean(difference),
                    "mean_absolute_difference_db": np.mean(np.abs(difference)),
                    "maximum_absolute_difference_db": np.max(np.abs(difference)),
                    "mean_difference_to_rcaide_without_wing_wake_adjustment_db": np.mean(
                        unadjusted_difference
                    ),
                    "mean_absolute_difference_to_rcaide_without_wing_wake_adjustment_db": np.mean(
                        np.abs(unadjusted_difference)
                    ),
                }
            )
    return rows


def _adapter_comparison(physical, legacy):
    rows = []
    for case in range(3):
        for observer, angle in enumerate((60, 90)):
            difference = legacy[case, observer] - physical[case, observer]
            rows.append(
                {
                    "case": f"F8745-D4-{case + 1}",
                    "reported_observer_angle_deg": angle,
                    "legacy_minus_physical_mean_db": np.mean(difference),
                    "legacy_minus_physical_minimum_db": np.min(difference),
                    "legacy_minus_physical_maximum_db": np.max(difference),
                }
            )
    return rows


def _write_report(summary, parity, adapter):
    lines = [
        "# F8745-D4 BladeAD Hanson line-source validation",
        "",
        "This comparison evaluates aligned-inflow BladeAD Hanson loading plus helicoidal-surface",
        "thickness noise using the frozen RCAIDE aerodynamic disk loads and archived F8745",
        "airfoil thickness shape. BladeAD BEM, transverse inflow, and fixture-specific",
        "calibration are not used.",
        "",
        "The BladeAD production adapter uses per-blade N/m loads, normalized Fourier",
        "coefficients, and one nondimensional radial integration. RCAIDE's archived prediction",
        "uses its original unnormalized FFT and element-force radial convention, so code-to-code",
        "agreement is diagnostic rather than an implementation acceptance criterion.",
        "",
        "## Experimental comparison",
        "",
        "| Case | Angle (deg) | MAE (dB) | Max (dB) | Overall error (dB) | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        gate = "PASS" if row["passes_frozen_gate"] else "FAIL"
        lines.append(
            f"| {row['case']} | {row['reported_observer_angle_deg']} | "
            f"{row['mean_absolute_error_db']:.3f} | "
            f"{row['maximum_absolute_error_db']:.3f} | "
            f"{row['overall_error_db']:.3f} | {gate} |"
        )
    lines.extend(
        [
            "",
        "## RCAIDE combined line-source comparison",
            "",
        "| Case | Angle | Mean vs archive | MA vs archive | Mean vs RCAIDE −15 dB | MA vs RCAIDE −15 dB |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in parity:
        lines.append(
            f"| {row['case']} | {row['reported_observer_angle_deg']} | "
            f"{row['mean_signed_difference_db']:.3f} | "
            f"{row['mean_absolute_difference_db']:.3f} | "
            f"{row['mean_difference_to_rcaide_without_wing_wake_adjustment_db']:.3f} | "
            f"{row['mean_absolute_difference_to_rcaide_without_wing_wake_adjustment_db']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Both models now contain Hanson loading and helicoidal-surface thickness sources.",
            "The archived RCAIDE configuration adds 15 dB to every harmonic for wing-wake",
            "interaction. BladeAD has no corresponding empirical adjustment; the table therefore",
            "shows comparisons both with and without that uplift.",
            "The remaining difference includes source-shape normalization, complex versus",
            "magnitude-only component summation, peak/RMS convention, propagation geometry, and",
            "RCAIDE's legacy radial/Fourier scaling. The separate term audit shows that changing",
            "only BladeAD's reporting from coherent RMS to coherent peak reduces mean absolute",
            "code difference to 1.37–2.61 dB. Production retains physically labelled RMS pressure.",
            "",
            "## RCAIDE legacy loading-adapter scaling with BladeAD thickness unchanged",
            "",
            "| Case | Angle (deg) | Mean (dB) | Minimum (dB) | Maximum (dB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in adapter:
        lines.append(
            f"| {row['case']} | {row['reported_observer_angle_deg']} | "
            f"{row['legacy_minus_physical_mean_db']:.3f} | "
            f"{row['legacy_minus_physical_minimum_db']:.3f} | "
            f"{row['legacy_minus_physical_maximum_db']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Detailed experimental harmonic errors are in",
            "`bladead_f8745_hanson_detailed.csv`; RCAIDE summary differences are in",
            "`bladead_f8745_hanson_rcaide_comparison.csv`; isolated adapter effects are in",
            "`bladead_f8745_hanson_adapter_comparison.csv`.",
        ]
    )
    (REPORTS / "f8745_bladead_hanson_validation.md").write_text("\n".join(lines) + "\n")


def main():
    prediction = evaluate_f8745(tonal_model="hanson_line")
    legacy_prediction = evaluate_f8745(
        tonal_model="hanson_line", hanson_legacy_adapter=True
    )
    summary, detail = compare_to_experiment(prediction, MODEL_NAME)
    parity = _rcaide_comparison(prediction)
    adapter = _adapter_comparison(prediction, legacy_prediction)
    _write_csv(REPORTS / "bladead_f8745_hanson_summary.csv", summary)
    _write_csv(REPORTS / "bladead_f8745_hanson_detailed.csv", detail)
    _write_csv(REPORTS / "bladead_f8745_hanson_rcaide_comparison.csv", parity)
    _write_csv(REPORTS / "bladead_f8745_hanson_adapter_comparison.csv", adapter)
    _write_report(summary, parity, adapter)


if __name__ == "__main__":
    main()
