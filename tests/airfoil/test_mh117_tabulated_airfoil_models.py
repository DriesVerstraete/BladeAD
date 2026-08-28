import numpy as np
import pytest
import csdl_alpha as csdl

from BladeAD.core.airfoil.smooth_neural_airfoil_model import MH117SmoothNeuralAirfoilModel
from BladeAD.core.airfoil.tabulated_airfoil_model import (
    MH117BSplineAirfoilModel,
    MH117PchipAirfoilModel,
)


@pytest.mark.parametrize("model_class", [MH117PchipAirfoilModel, MH117BSplineAirfoilModel])
def test_mh117_model_recovers_table_nodes(model_class):
    model = model_class()
    alpha = np.deg2rad(np.array([-14.0, -4.0, 0.0, 10.0, 45.0, 90.0]))
    reynolds = np.full(alpha.shape, 1.0e6)
    cl, cd = model.predict(alpha, reynolds)
    row = np.where(model.surface.reynolds == 1.0e6)[0][0]
    columns = [np.where(model.surface.alpha_deg == angle)[0][0] for angle in [-14, -4, 0, 10, 45, 90]]
    np.testing.assert_allclose(cl, model.surface.cl[row, columns], atol=1e-12)
    np.testing.assert_allclose(cd, model.surface.cd[row, columns], atol=1e-12)


@pytest.mark.parametrize("model_class", [MH117PchipAirfoilModel, MH117BSplineAirfoilModel])
def test_mh117_model_derivatives_match_central_difference(model_class):
    model = model_class()
    alpha = np.deg2rad(np.array([2.25, 8.25, 16.25]))
    reynolds = np.array([2.5e5, 8.75e5, 4.25e6])
    _, _, dcl_da, dcd_da, dcl_dre, dcd_dre = model.surface.predict(alpha, reynolds)
    alpha_step = 1.0e-7
    re_step = 10.0
    cl_plus, cd_plus = model.predict(alpha + alpha_step, reynolds)
    cl_minus, cd_minus = model.predict(alpha - alpha_step, reynolds)
    np.testing.assert_allclose(dcl_da, (cl_plus - cl_minus) / (2 * alpha_step), rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(dcd_da, (cd_plus - cd_minus) / (2 * alpha_step), rtol=2e-5, atol=1e-8)
    cl_plus, cd_plus = model.predict(alpha, reynolds + re_step)
    cl_minus, cd_minus = model.predict(alpha, reynolds - re_step)
    np.testing.assert_allclose(dcl_dre, (cl_plus - cl_minus) / (2 * re_step), rtol=2e-5, atol=1e-12)
    np.testing.assert_allclose(dcd_dre, (cd_plus - cd_minus) / (2 * re_step), rtol=2e-5, atol=1e-12)


def test_mh117_model_rejects_extrapolation():
    model = MH117PchipAirfoilModel()
    with pytest.raises(ValueError, match="Re is outside"):
        model.predict(np.array([0.0]), np.array([1.0e5]))


@pytest.mark.parametrize(
    "model_class",
    [MH117PchipAirfoilModel, MH117BSplineAirfoilModel, MH117SmoothNeuralAirfoilModel],
)
def test_mh117_negative_alpha_solver_continuation_is_constant(model_class):
    model = model_class()
    reynolds = np.array([1.0e6])
    boundary = model.predict(np.deg2rad(np.array([-14.0])), reynolds, derivatives=True)
    continued = model.predict(np.deg2rad(np.array([-55.0])), reynolds, derivatives=True)
    np.testing.assert_allclose(continued[:2], boundary[:2])
    np.testing.assert_allclose(continued[2:4], 0.0)
    np.testing.assert_allclose(continued[4:], boundary[4:])


@pytest.mark.parametrize("model_class", [MH117PchipAirfoilModel, MH117BSplineAirfoilModel])
def test_mh117_tabulated_model_complete_graph_derivatives(model_class):
    model = model_class()
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    alpha = csdl.Variable(name="alpha", value=np.deg2rad(np.array([2.25, 8.25, 16.25])))
    reynolds = csdl.Variable(name="Re", value=np.array([2.5e5, 8.75e5, 4.25e6]))
    cl, cd = model.evaluate(alpha, reynolds, csdl.Variable(value=np.zeros(3)))
    errors = csdl.derivative_utils.verify_derivatives(
        [cl, cd], [alpha, reynolds], 1e-7, print_results=False, raise_on_error=True
    )
    assert errors is not None
    recorder.stop()


def test_mh117_smooth_neural_model_has_positive_drag_and_derivatives():
    model = MH117SmoothNeuralAirfoilModel()
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    alpha = csdl.Variable(name="alpha", value=np.deg2rad(np.array([2.25, 8.25, 16.25])))
    reynolds = csdl.Variable(name="Re", value=np.array([2.5e5, 8.75e5, 4.25e6]))
    mach = csdl.Variable(value=np.zeros(3))
    cl, cd = model.evaluate(alpha, reynolds, mach)
    assert np.all(cd.value > 0)
    errors = csdl.derivative_utils.verify_derivatives(
        [cl, cd], [alpha, reynolds], 1e-6, print_results=False, raise_on_error=True
    )
    assert errors is not None
    recorder.stop()
