from pathlib import Path

import numpy as np


FIXTURE = Path(__file__).parents[2] / "validation" / "acoustics" / "fixtures" / "lowson_hg_matlab"


def test_hg_matlab_fixture_preserves_pinned_source_values():
    hg = np.genfromtxt(FIXTURE / "hg_matlab_total_spl.csv", delimiter=",", names=True)
    experimental = np.genfromtxt(FIXTURE / "experimental_spl.csv", delimiter=",", names=True)
    radial = np.genfromtxt(FIXTURE / "radial_inputs.csv", delimiter=",", names=True)
    assert hg.shape == (37,)
    assert experimental.shape == (16,)
    assert radial.shape == (40,)
    np.testing.assert_allclose(hg["source_angle_deg"], np.linspace(90.0, -90.0, 37))
    np.testing.assert_allclose(
        hg["total_spl_db"][[0, 17, 18, 36]],
        [56.09030376, 46.37119990, 46.72683781, 56.09030376],
    )
    np.testing.assert_allclose(
        experimental["spl_db"][[0, 10, 15]],
        [57.40552099, 48.04448109, 50.42903289],
    )
    np.testing.assert_allclose(radial["radius_over_R"], np.linspace(0.21, 0.99, 40))
