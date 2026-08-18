from __future__ import annotations

import csv

import numpy as np

from run_bladead_f8745_validation import (
    FIXTURE,
    REPORTS,
    compare_to_experiment,
    evaluate_f8745,
)


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _archived_totals():
    with np.load(FIXTURE / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
        return (
            archive["energy.converters.F8745_D4_Propeller.thrust"][:, 0],
            archive["energy.converters.F8745_D4_Propeller.torque"][:, 0],
        )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    total_thrust, total_torque = _archived_totals()
    rows = []
    cases = [
        ("magnitude_minus_10pct", 0.9, 0.0),
        ("baseline", 1.0, 0.0),
        ("magnitude_plus_10pct", 1.1, 0.0),
        ("root_shift_fixed_total", 1.0, -0.2),
        ("tip_shift_fixed_total", 1.0, 0.2),
    ]
    for model in ("lowson", "hanson_line"):
        for label, load_scale, radial_redistribution in cases:
            prediction = evaluate_f8745(
                tonal_model=model,
                load_scale=load_scale,
                radial_redistribution=radial_redistribution,
            )
            summary, _ = compare_to_experiment(
                prediction,
                model_name=f"bladead_{model}_load_sensitivity",
            )
            for index, row in enumerate(summary):
                case_index = index // 2
                rows.append(
                    {
                        "tonal_model": model,
                        "perturbation": label,
                        "load_scale": load_scale,
                        "radial_redistribution": radial_redistribution,
                        "case": row["case"],
                        "reported_observer_angle_deg": row[
                            "reported_observer_angle_deg"
                        ],
                        "archived_total_thrust_n": total_thrust[case_index],
                        "perturbed_total_thrust_n": load_scale
                        * total_thrust[case_index],
                        "archived_total_torque_nm": total_torque[case_index],
                        "perturbed_total_torque_nm": load_scale
                        * total_torque[case_index],
                        "harmonic_mae_db": row["mean_absolute_error_db"],
                        "overall_error_db": row["overall_error_db"],
                    }
                )
    _write_csv(REPORTS / "f8745_load_sensitivity.csv", rows)

    baseline = {
        (row["tonal_model"], row["case"], row["reported_observer_angle_deg"]): row
        for row in rows
        if row["perturbation"] == "baseline"
    }
    lines = [
        "# F8745-D4 aerodynamic-source sensitivity",
        "",
        "The experimental source located for this fixture is Weir and Powers, AIAA Paper",
        "87-0527, *Comparisons of Predicted Propeller Noise with Windtunnel and Flyover Data*.",
        "The accessible primary-source metadata identifies the paper and test, but measured thrust,",
        "torque/power, aerodynamic uncertainty, and sectional loading remain **not reported** in the",
        "material recovered for this audit. The perturbations below therefore bound sensitivity;",
        "they are not experimental uncertainty intervals.",
        "",
        "Magnitude cases scale signed thrust and torque distributions together by ±10%. Radial",
        "cases multiply each signed distribution by `1 + s(2r/R-1)` with `s=±0.2`, then normalize",
        "each case and azimuth back to its original thrust and torque. Thus radial cases preserve",
        "the archived integral loads exactly while shifting loading rootward or tipward.",
        "",
        "| Model | Perturbation | Max |Δ overall error| (dB) | Max |Δ harmonic MAE| (dB) |",
        "|---|---|---:|---:|",
    ]
    for model in ("lowson", "hanson_line"):
        for label, _, _ in cases:
            if label == "baseline":
                continue
            selected = [
                row for row in rows if row["tonal_model"] == model and row["perturbation"] == label
            ]
            overall_delta = []
            mae_delta = []
            for row in selected:
                reference = baseline[
                    (model, row["case"], row["reported_observer_angle_deg"])
                ]
                overall_delta.append(abs(row["overall_error_db"] - reference["overall_error_db"]))
                mae_delta.append(abs(row["harmonic_mae_db"] - reference["harmonic_mae_db"]))
            lines.append(
                f"| {model} | {label} | {max(overall_delta):.3f} | {max(mae_delta):.3f} |"
            )
    lines.extend(
        [
            "",
            "These bounded perturbations do not close the experimental discrepancy. They quantify",
            "how strongly the present conclusions depend on plausible but unverified source-load",
            "changes; they do not validate either acoustic model or the RCAIDE aerodynamic loads.",
            "Detailed results are in `f8745_load_sensitivity.csv`.",
        ]
    )
    (REPORTS / "f8745_load_sensitivity.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
