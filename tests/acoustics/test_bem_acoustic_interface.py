import csdl_alpha as csdl
import numpy as np

from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.airfoil.zero_d_airfoil_model import (
    ZeroDAirfoilModel,
    ZeroDAirfoilPolarParameters,
)
from BladeAD.core.preprocessing.preprocess_variables import preprocess_input_variables
from BladeAD.utils.integration_schemes import integrate_quantity
from BladeAD.utils.var_groups import AtmosStates, RotorAnalysisInputs, RotorMeshParameters


def test_preprocessing_uses_declared_hub_ratio_and_exposes_mesh_quantities():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    shape = (1, 5, 4)
    outputs = preprocess_input_variables(
        shape=shape,
        radius=csdl.Variable(value=np.ones(shape)),
        chord_profile=csdl.Variable(value=np.full(5, 0.1)),
        twist_profile=csdl.Variable(value=np.zeros(5)),
        rpm=csdl.Variable(value=np.array([1200.0])),
        norm_hub_radius=0.3,
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        origin_velocity=csdl.Variable(value=np.zeros((1, 3))),
        atmos_states=AtmosStates(),
        num_blades=2,
    )

    np.testing.assert_allclose(outputs.element_width.value, np.full(shape, 0.175))
    np.testing.assert_allclose(
        outputs.radius_vector_exp.value[0, :, 0], [0.37, 0.51, 0.65, 0.79, 0.93]
    )
    np.testing.assert_allclose(
        outputs.azimuth_angle_exp.value[0, 0, :], [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    )
    recorder.stop()


def test_bem_exposes_lowson_mesh_and_complete_rotor_sectional_loads():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 15
    num_azimuthal = 4
    num_blades = 2
    polar = ZeroDAirfoilPolarParameters(
        alpha_stall_minus=-10.0,
        alpha_stall_plus=15.0,
        Cl_stall_minus=-1.0,
        Cl_stall_plus=1.5,
        Cd_stall_minus=0.02,
        Cd_stall_plus=0.06,
        Cl_0=0.5,
        Cd_0=0.008,
        Cl_alpha=5.1566,
    )
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        chord_profile=csdl.Variable(value=np.linspace(0.25, 0.05, num_radial)),
        twist_profile=csdl.Variable(
            value=np.linspace(np.deg2rad(50.0), np.deg2rad(20.0), num_radial)
        ),
        radius=csdl.Variable(value=1.2),
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        norm_hub_radius=0.3,
    )
    inputs = RotorAnalysisInputs(
        rpm=csdl.Variable(value=np.array([2000.0])),
        mesh_velocity=csdl.Variable(value=np.array([[50.0, 0.0, 0.0]])),
        mesh_parameters=mesh,
    )
    outputs = BEMModel(
        num_nodes=1,
        airfoil_model=ZeroDAirfoilModel(polar_parameters=polar),
        integration_scheme="trapezoidal",
    ).evaluate(inputs)

    expected_shape = (1, num_radial, num_azimuthal)
    assert outputs.radial_stations.shape == expected_shape
    assert outputs.radial_element_width.shape == expected_shape
    assert outputs.azimuth_angle.shape == expected_shape
    assert outputs.sectional_loads_include_all_blades is True
    np.testing.assert_allclose(
        outputs.radial_element_width.value,
        np.full(expected_shape, (1.2 - 0.3 * 1.2) / (num_radial - 1)),
    )

    integrated_sectional_thrust = integrate_quantity(
        outputs.sectional_thrust, "trapezoidal"
    )
    integrated_sectional_torque = integrate_quantity(
        outputs.sectional_torque, "trapezoidal"
    )
    np.testing.assert_allclose(
        integrated_sectional_thrust.value, outputs.total_thrust.value, rtol=5e-3
    )
    np.testing.assert_allclose(
        integrated_sectional_torque.value, outputs.total_torque.value, rtol=5e-3
    )
    recorder.stop()
