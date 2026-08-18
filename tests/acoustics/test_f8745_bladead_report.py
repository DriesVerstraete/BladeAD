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
        [14.15054535, 13.31977709, 17.83730060, 13.89151626, 22.14021376, 17.13173215],
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        summary["overall_error_db"],
        [-11.48833819, -14.16773977, -14.28836899, -15.14517602, -15.05601134, -16.77938343],
        rtol=0.0,
        atol=1.0e-7,
    )
