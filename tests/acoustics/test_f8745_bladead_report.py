from pathlib import Path

import numpy as np


REPORTS = Path(__file__).parents[2] / "validation" / "acoustics" / "reports"


def test_f8745_bladead_validation_result_is_frozen_and_fails_gate():
    summary = np.genfromtxt(
        REPORTS / "bladead_f8745_summary.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    detail = np.genfromtxt(
        REPORTS / "bladead_f8745_detailed.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    assert summary.shape == (6,)
    assert detail.shape == (108,)
    np.testing.assert_array_equal(summary["points"], 18)
    assert not np.any(summary["passes_frozen_gate"])
    np.testing.assert_allclose(
        summary["mean_absolute_error_db"],
        [14.23572387, 13.43431218, 17.84851325, 13.90859881, 22.15927403, 17.16005805],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        summary["overall_error_db"],
        [-12.39891546, -14.76286478, -14.39423592, -15.18512075, -15.24708982, -16.87213733],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_f8745_signed_load_adapter_preserves_archived_invariants():
    fixture = REPORTS.parent / "fixtures" / "f8745_d4"
    geometry = np.genfromtxt(fixture / "geometry.csv", delimiter=",", names=True)
    observers = np.genfromtxt(fixture / "observers.csv", delimiter=",", names=True)
    with np.load(fixture / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
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
        disk_thrust = archive[
            "energy.converters.F8745_D4_Propeller.disc_thrust_distribution"
        ]
        archived_observers = archive["observer.position_m"]
    np.testing.assert_allclose(np.sum(thrust_distribution, axis=1), thrust_per_blade)
    np.testing.assert_allclose(np.sum(torque_distribution, axis=1), torque_per_blade)
    np.testing.assert_allclose(total_thrust, blade_count * thrust_per_blade)
    assert np.min(thrust_distribution) < 0.0
    assert np.max(np.ptp(disk_thrust, axis=2)) < 5.0e-12
    np.testing.assert_allclose(
        archived_observers,
        np.column_stack((observers["x_m"], observers["y_m"], observers["z_m"])),
    )
    np.testing.assert_allclose(np.diff(geometry["radius_m"]), 0.0273)


def test_f8745_hanson_loading_only_result_and_adapter_audit_are_frozen():
    summary = np.genfromtxt(
        REPORTS / "bladead_f8745_hanson_summary.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    adapter = np.genfromtxt(
        REPORTS / "bladead_f8745_hanson_adapter_comparison.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    assert summary.shape == (6,)
    assert adapter.shape == (6,)
    assert not np.any(summary["passes_frozen_gate"])
    np.testing.assert_allclose(
        summary["mean_absolute_error_db"],
        [20.71458206, 24.07985894, 23.91298880, 24.78959933, 28.21500182, 27.99245112],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        adapter["legacy_minus_physical_mean_db"],
        [-3.29007044, -0.33508259, -3.80308285, -0.32007332, -3.78305332, -0.37297470],
        rtol=0.0,
        atol=1.0e-7,
    )


def test_f8745_rcaide_term_reproduction_exposes_uniform_adjustment():
    audit = np.genfromtxt(
        REPORTS / "f8745_hanson_rcaide_term_audit.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    assert audit.shape == (6,)
    assert np.max(audit["maximum_archive_reproduction_error_db"]) < 3.0
    fixture = REPORTS.parent / "fixtures" / "f8745_d4"
    with np.load(fixture / "rcaide_line_source_baseline.npz", allow_pickle=False) as archive:
        assert float(archive["settings.wing_wake_interactional_dB_adjustment"]) == 15.0


def test_f8745_pressure_convention_audit_is_frozen():
    audit = np.genfromtxt(
        REPORTS / "f8745_hanson_pressure_convention_audit.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    assert audit.shape == (18,)
    coherent_peak = audit[audit["convention"] == "coherent_peak"]
    magnitude_peak = audit[audit["convention"] == "magnitude_sum_peak"]
    assert np.max(coherent_peak["mean_absolute_difference_db"]) < 2.7
    assert np.max(magnitude_peak["mean_absolute_difference_db"]) > 5.5


def test_f8745_load_sensitivity_bounds_do_not_close_validation_gap():
    sensitivity = np.genfromtxt(
        REPORTS / "f8745_load_sensitivity.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    assert sensitivity.shape == (60,)
    baseline = sensitivity[sensitivity["perturbation"] == "baseline"]
    perturbed = sensitivity[sensitivity["perturbation"] != "baseline"]
    assert baseline.shape == (12,)
    assert np.max(perturbed["overall_error_db"]) < -11.0
    radial = perturbed[
        np.isin(
            perturbed["perturbation"],
            ["root_shift_fixed_total", "tip_shift_fixed_total"],
        )
    ]
    np.testing.assert_allclose(
        radial["archived_total_thrust_n"], radial["perturbed_total_thrust_n"]
    )
    np.testing.assert_allclose(
        radial["archived_total_torque_nm"], radial["perturbed_total_torque_nm"]
    )
