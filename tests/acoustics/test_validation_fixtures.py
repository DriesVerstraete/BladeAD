import csv
import hashlib
from pathlib import Path

import numpy as np


FIXTURES = Path(__file__).resolve().parents[2] / "validation" / "acoustics" / "fixtures"
REPORTS = FIXTURES.parent / "reports"


def read_csv(case, filename):
    with (FIXTURES / case / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_f8745_fixture_dimensions_and_reference_values():
    geometry = read_csv("f8745_d4", "geometry.csv")
    conditions = read_csv("f8745_d4", "operating_conditions.csv")
    published_conditions = read_csv("f8745_d4", "published_operating_conditions.csv")
    observers = read_csv("f8745_d4", "observers.csv")
    harmonics = read_csv("f8745_d4", "experimental_harmonics.csv")

    assert len(geometry) == 30
    assert len(conditions) == 3
    assert len(observers) == 19
    assert len(harmonics) == 3 * 2 * 18
    np.testing.assert_allclose([float(row["rpm"]) for row in conditions], [2390, 2710, 2630])
    np.testing.assert_allclose(
        [float(row["rpm"]) for row in published_conditions], [2400, 2700, 2700]
    )
    np.testing.assert_allclose(
        [float(row["measured_thrust_n"]) for row in published_conditions],
        [642, 1907, 1500],
    )
    np.testing.assert_allclose(
        [float(row["shaft_power_kw"]) for row in published_conditions],
        [73.6, 184.6, 152.1],
    )
    assert harmonics[0] == {
        "case": "1",
        "observer_angle_reported_deg": "60",
        "harmonic": "1",
        "spl_db": "103.23",
    }
    assert harmonics[-1]["spl_db"] == "98.0404"


def test_apc_fixture_dimensions_and_reference_values():
    geometry = read_csv("apc_11x4", "geometry.csv")
    conditions = read_csv("apc_11x4", "operating_conditions.csv")
    observers = read_csv("apc_11x4", "observers.csv")
    total = read_csv("apc_11x4", "experimental_total_spectrum.csv")
    broadband = read_csv("apc_11x4", "experimental_broadband_spectrum.csv")

    assert len(geometry) == 18
    assert len(conditions) == 3
    assert len(observers) == 5
    assert len(total) == 3 * 21
    assert len(broadband) == 2 * 21
    np.testing.assert_allclose([float(row["rpm"]) for row in conditions], [3600, 4200, 4800])
    assert total[0]["total_spl_db"] == "22.149"
    assert total[-1]["total_spl_db"] == "50.186"
    assert broadband[0]["broadband_spl_db"] == "24.8571428"
    assert broadband[-1]["broadband_spl_db"] == "42.57142"


def test_dji_9443_acoustic_fixture_is_frozen():
    conditions = read_csv("dji_9443", "operating_conditions.csv")
    chord = read_csv("dji_9443", "chord_distribution.csv")
    twist = read_csv("dji_9443", "twist_distribution.csv")
    airfoil_sections = read_csv("dji_9443", "airfoil_sections.csv")
    observers = read_csv("dji_9443", "observers.csv")
    harmonics = read_csv("dji_9443", "experimental_harmonics.csv")
    narrowband = read_csv("dji_9443", "experimental_narrowband_spectrum.csv")
    one_third_octave = read_csv("dji_9443", "experimental_one_third_octave_spectrum.csv")
    oaspl = read_csv("dji_9443", "experimental_oaspl.csv")

    assert len(conditions) == 1
    assert len(chord) == 26
    assert len(twist) == 42
    assert len(airfoil_sections) == 7
    assert len(observers) == 5
    assert len(harmonics) == 10
    assert len(narrowband) == 296
    assert len(one_third_octave) == 26
    assert len(oaspl) == 5
    assert float(conditions[0]["rpm"]) == 5400.0
    assert float(conditions[0]["observer_radius_m"]) == 1.905
    assert float(conditions[0]["measured_thrust_coefficient"]) == 0.072
    assert float(chord[-1]["radius_over_tip_radius"]) == 1.0
    assert float(twist[-1]["radius_over_tip_radius"]) == 1.0
    np.testing.assert_allclose(
        [float(row["reynolds_number"]) for row in airfoil_sections],
        [3317, 13131, 28404, 41039, 44913, 42526, 22978],
    )
    np.testing.assert_allclose(
        [float(row["normalized_blade_span"]) for row in airfoil_sections],
        [0.0, 0.0857143, 0.185714, 0.371429, 0.714286, 0.942857, 1.0],
    )
    np.testing.assert_allclose(
        [float(row["reported_angle_from_rotor_plane_deg"]) for row in observers],
        [-45.0, -22.5, 0.0, 22.5, 45.0],
    )
    assert harmonics[0]["spl_db"] == "47.6377952755906"
    assert harmonics[-1]["spl_db"] == "22.50000000000003"


def test_dji_9443_section_polars_match_pinned_flowunsteady_sources():
    fixture = FIXTURES / "dji_9443"
    sections = read_csv("dji_9443", "airfoil_sections.csv")
    expected_hashes = [
        "831fadfdf98315d07b47f2f70077ab7c64ae60c4b0723023a2e463aa38adec97",
        "8a75dc4a31cd9daf0a16925e89c8bd40da41b55851f507263952fa524a9b3d8b",
        "5cc924814c04302cfabbf1b539b51a6b7aad61b6a4d55a31ee3d41befca96a62",
        "ad6f314a308a85939dad81ba77a63d467aa34d1cca6c95382911e28b9341a273",
        "624b38db8cb2b7af25dc4f8c89c74c00ea038a9c3a76552bb4f754bfed72ce1d",
        "7b2391b57f5ddd560fca732c8e514f09fcb40780e21a632279ce18e729a7f3ca",
        "57ed68f0b2a9252fe0257aad5d95ed7e4b30ad08d8114f379c308cb58d9ca74a",
    ]

    for section, expected_hash in zip(sections, expected_hashes):
        path = fixture / "airfoil_polars" / section["polar_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        assert set(rows[0]) == {"Alpha", "Cl", "Cd", "Cm"}


def test_dji_9443_generated_model_comparison_is_complete():
    with (REPORTS / "bladead_dji9443_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (REPORTS / "bladead_dji9443_detailed.csv").open(newline="") as stream:
        detail = list(csv.DictReader(stream))

    assert [row["model"] for row in summary] == ["lowson", "hanson_line"]
    assert len(detail) == 20
    assert {int(row["harmonic"]) for row in detail} == {1, 2}
    np.testing.assert_allclose(
        [float(row["harmonic_mae_db"]) for row in summary],
        [6.463037598364946, 12.483681616887882],
    )
    assert all(
        row["source_model"] == "flowunsteady_section_polars" for row in summary
    )


def test_rcaide_baseline_archives_are_complete_and_pinned():
    expected = {
        "f8745_d4/rcaide_line_source_baseline.npz": (3, 19, 29),
        "f8745_d4/rcaide_plane_source_baseline.npz": (3, 19, 29),
        "apc_11x4/rcaide_plane_source_baseline.npz": (3, 5, 29),
    }
    source_commit = "c88217f3fd0ef9740e86cfc4241bb4362bb7a766"
    for relative_path, spectrum_shape in expected.items():
        path = FIXTURES / relative_path
        with np.load(path, allow_pickle=False) as baseline:
            assert str(baseline["source_commit"]) == source_commit
            assert bool(baseline["runtime_numpy_trapezoid_compatibility_alias"])
            spectrum_keys = [
                key for key in baseline.files if key.endswith("SPL_1_3_spectrum")
            ]
            assert len(spectrum_keys) == 1
            spectrum = baseline[spectrum_keys[0]]
            assert spectrum.shape == spectrum_shape
            assert np.isfinite(spectrum).all()
            assert len(baseline.files) >= 280


def test_validation_matrix_is_frozen_and_complete():
    with (REPORTS / "rcaide_vs_experiment_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (REPORTS / "rcaide_vs_experiment_detailed.csv").open(newline="") as stream:
        detail = list(csv.DictReader(stream))

    assert len(summary) == 17
    assert len(detail) == 321
    first = summary[0]
    assert first["case"] == "F8745-D4-1"
    assert first["model"] == "rcaide_line_source"
    np.testing.assert_allclose(float(first["mean_absolute_error_db"]), 1.309866604925333)
    assert "must not be relaxed" in (REPORTS / "validation_matrix.md").read_text()
