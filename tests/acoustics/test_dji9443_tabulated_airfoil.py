import numpy as np

from validation.acoustics.scripts.run_bladead_dji9443_validation import (
    DJI9443TabulatedAirfoilModel,
    FIXTURE,
    RadiallyInterpolatedPolarOperation,
    thickness_geometry,
)


def test_dji9443_polar_model_recovers_each_section_table_at_five_degrees():
    sections = np.genfromtxt(
        FIXTURE / "airfoil_sections.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    hub_fraction = 0.00624 / 0.12
    radial_fraction = hub_fraction + sections["normalized_blade_span"] * (
        1.0 - hub_fraction
    )
    model = DJI9443TabulatedAirfoilModel(radial_fraction, hub_fraction)
    operation = RadiallyInterpolatedPolarOperation(
        model.radial_span, model.section_span, model.polars
    )
    alpha = np.full((1, 7, 1), np.deg2rad(5.0))

    cl, cd, _, _ = operation._predict(alpha)

    expected_cl = [
        np.interp(5.0, polar["Alpha"], polar["Cl"]) for polar in model.polars
    ]
    expected_cd = [
        np.interp(5.0, polar["Alpha"], polar["Cd"]) for polar in model.polars
    ]
    np.testing.assert_allclose(cl.ravel(), expected_cl)
    np.testing.assert_allclose(cd.ravel(), expected_cd)


def test_dji9443_polar_model_alpha_derivatives_match_central_difference():
    hub_fraction = 0.00624 / 0.12
    radial_fraction = np.linspace(hub_fraction, 0.99, 40)
    model = DJI9443TabulatedAirfoilModel(radial_fraction, hub_fraction)
    operation = RadiallyInterpolatedPolarOperation(
        model.radial_span, model.section_span, model.polars
    )
    alpha = np.full((1, 40, 1), np.deg2rad(5.25))
    step = 1e-7

    _, _, dcl, dcd = operation._predict(alpha)
    cl_plus, cd_plus, _, _ = operation._predict(alpha + step)
    cl_minus, cd_minus, _, _ = operation._predict(alpha - step)

    np.testing.assert_allclose(dcl, (cl_plus - cl_minus) / (2.0 * step), rtol=1e-7)
    np.testing.assert_allclose(dcd, (cd_plus - cd_minus) / (2.0 * step), rtol=1e-7)


def test_dji9443_thickness_geometry_recovers_mapped_section_contours():
    sections = np.genfromtxt(
        FIXTURE / "airfoil_sections.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    hub_fraction = 0.00624 / 0.12
    radial_fraction = hub_fraction + sections["normalized_blade_span"] * (
        1.0 - hub_fraction
    )

    thickness, shape, locations, weights = thickness_geometry(
        radial_fraction, hub_fraction
    )

    np.testing.assert_allclose(
        thickness,
        [
            0.21937464622660874,
            0.21937464622660874,
            0.0839752240287128,
            0.06508004899171618,
            0.06916227378309095,
            0.072963629631095,
            0.072963629631095,
        ],
        rtol=1e-7,
    )
    assert shape.shape == (7, 101)
    np.testing.assert_allclose(np.max(shape, axis=1), 1.0)
    np.testing.assert_allclose(locations[[0, -1]], [-0.5, 0.5])
    np.testing.assert_allclose(np.sum(weights), 1.0)
