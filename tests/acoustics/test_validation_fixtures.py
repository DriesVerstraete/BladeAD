import csv
from pathlib import Path

import numpy as np


FIXTURES = Path(__file__).resolve().parents[2] / "validation" / "acoustics" / "fixtures"


def read_csv(case, filename):
    with (FIXTURES / case / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_f8745_fixture_dimensions_and_reference_values():
    geometry = read_csv("f8745_d4", "geometry.csv")
    conditions = read_csv("f8745_d4", "operating_conditions.csv")
    observers = read_csv("f8745_d4", "observers.csv")
    harmonics = read_csv("f8745_d4", "experimental_harmonics.csv")

    assert len(geometry) == 30
    assert len(conditions) == 3
    assert len(observers) == 19
    assert len(harmonics) == 3 * 2 * 18
    np.testing.assert_allclose([float(row["rpm"]) for row in conditions], [2390, 2710, 2630])
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
