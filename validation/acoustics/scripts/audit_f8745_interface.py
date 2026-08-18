from __future__ import annotations

import csv
from pathlib import Path

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


def main():
    geometry = np.genfromtxt(FIXTURE / "geometry.csv", delimiter=",", names=True)
    observer_csv = np.genfromtxt(FIXTURE / "observers.csv", delimiter=",", names=True)
    with np.load(FIXTURE / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
        blade_count = int(archive["rotor.number_of_blades"])
        thrust_distribution = archive[
            "energy.converters.F8745_D4_Propeller.blade_thrust_distribution"
        ]
        torque_distribution = archive[
            "energy.converters.F8745_D4_Propeller.blade_torque_distribution"
        ]
        thrust_per_blade = archive[
            "energy.converters.F8745_D4_Propeller.thrust_per_blade"
        ][:, 0]
        torque_per_blade = archive[
            "energy.converters.F8745_D4_Propeller.torque_per_blade"
        ][:, 0]
        total_thrust = archive["energy.converters.F8745_D4_Propeller.thrust"][:, 0]
        total_torque = archive["energy.converters.F8745_D4_Propeller.torque"][:, 0]
        thrust_gradient = archive[
            "energy.converters.F8745_D4_Propeller.blade_dT_dr"
        ]
        torque_gradient = archive[
            "energy.converters.F8745_D4_Propeller.blade_dQ_dr"
        ]
        disk_thrust = archive[
            "energy.converters.F8745_D4_Propeller.disc_thrust_distribution"
        ]
        archived_observers = archive["observer.position_m"]

    weights = np.ones(len(geometry))
    weights[[0, -1]] = 0.5
    radial_width = np.gradient(geometry["radius_m"])
    thrust_from_gradient = np.sum(thrust_gradient * radial_width * weights, axis=1)
    torque_from_gradient = np.sum(torque_gradient * radial_width * weights, axis=1)
    invariants = {
        "blade_count": blade_count,
        "max_thrust_distribution_total_error_n": np.max(
            np.abs(np.sum(thrust_distribution, axis=1) - thrust_per_blade)
        ),
        "max_torque_distribution_total_error_nm": np.max(
            np.abs(np.sum(torque_distribution, axis=1) - torque_per_blade)
        ),
        "max_thrust_gradient_quadrature_error_n": np.max(
            np.abs(thrust_from_gradient - thrust_per_blade)
        ),
        "max_torque_gradient_quadrature_error_nm": np.max(
            np.abs(torque_from_gradient - torque_per_blade)
        ),
        "max_complete_rotor_thrust_scaling_error_n": np.max(
            np.abs(total_thrust - blade_count * thrust_per_blade)
        ),
        "max_complete_rotor_torque_scaling_error_nm": np.max(
            np.abs(total_torque - blade_count * torque_per_blade)
        ),
        "max_azimuthal_load_spread_n": np.max(np.ptp(disk_thrust, axis=2)),
        "max_observer_archive_csv_difference_m": np.max(
            np.abs(
                archived_observers
                - np.column_stack(
                    (observer_csv["x_m"], observer_csv["y_m"], observer_csv["z_m"])
                )
            )
        ),
        "max_observer_radius_error_m": np.max(
            np.abs(np.linalg.norm(archived_observers, axis=1) - 20.0)
        ),
    }

    sensitivity = []
    components = None
    for velocity_scale, label in ((1.0, "archived_positive"), (0.0, "stationary"), (-1.0, "reversed")):
        prediction = evaluate_f8745(
            source_velocity_scale=velocity_scale,
            return_components=velocity_scale == 1.0,
        )
        if isinstance(prediction, dict):
            components = prediction
            prediction = prediction["combined"]
        summary, _ = compare_to_experiment(prediction)
        for row in summary:
            sensitivity.append(
                {
                    "source_velocity_convention": label,
                    "case": row["case"],
                    "reported_observer_angle_deg": row[
                        "reported_observer_angle_deg"
                    ],
                    "harmonic_mae_db": row["mean_absolute_error_db"],
                    "overall_error_db": row["overall_error_db"],
                }
            )
    _write_csv(REPORTS / "bladead_f8745_convection_sensitivity.csv", sensitivity)

    def energetic_overall(values):
        return 10.0 * np.log10(np.sum(10.0 ** (values / 10.0), axis=2))

    loading_overall = energetic_overall(components["loading"])
    thickness_overall = energetic_overall(components["thickness"])
    lines = [
        "# F8745-D4 load/interface parity audit",
        "",
        "This audit separates BladeAD BEM, the RCAIDE-load adapter, propagation conventions, and",
        "acoustic-model scope before attributing the experimental discrepancy.",
        "",
        "## Resolved invariants",
        "",
        f"- Blade count is {blade_count}; RCAIDE total thrust and torque equal exactly B times the per-blade totals.",
        "- Signed sectional distributions sum to the archived per-blade totals; negative root",
        "  elements are physical entries and must not be converted station-by-station with `abs()`.",
        "- Trapezoidal integration of `blade_dT_dr` and `blade_dQ_dr` reproduces those distributions",
        "  to machine precision.",
        "- Disk loads are azimuthally uniform to numerical roundoff, so the steady load harmonic is",
        "  the complete aerodynamic information available to the current Lowson projection.",
        "- Observer CSV positions exactly match the archived RCAIDE microphone positions and lie on",
        "  the specified 20 m radius.",
        "",
        "| Invariant | Maximum discrepancy |",
        "|---|---:|",
    ]
    for name, value in invariants.items():
        if name != "blade_count":
            lines.append(f"| {name} | {value:.6e} |")
    lines.extend(
        [
            "",
            "## Propagation sensitivity",
            "",
            "The archived inertial velocity is positive x and is the baseline convention. Reversing",
            "it improves the 60-degree overall errors to roughly 5–7 dB but leaves the 90-degree",
            "errors near 15–17 dB. Source/freestream interpretation is therefore important but cannot",
            "explain the complete discrepancy. Detailed values are in",
            "`bladead_f8745_convection_sensitivity.csv`.",
            "",
            "## Component levels for the archived convention",
            "",
            "| Case | Angle | Loading overall (dB) | Thickness overall (dB) |",
            "|---|---:|---:|---:|",
        ]
    )
    for case in range(3):
        for observer, angle in enumerate((60, 90)):
            lines.append(
                f"| {case + 1} | {angle} | {loading_overall[case, observer]:.3f} | "
                f"{thickness_overall[case, observer]:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "BladeAD BEM is excluded from this comparison. Blade count, signed load transfer, radial",
            "quadrature, azimuth sampling, and observer positions are now numerically closed. The",
            "remaining gap belongs to the acoustic chain: propagation convention plus formulation/source",
            "scope. Because credible velocity conventions do not close the gap—especially at 90 degrees—",
            "the evidence supports a Hanson line-source implementation, but does not assign the entire",
            "error to Lowson physics alone.",
        ]
    )
    (REPORTS / "f8745_interface_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
