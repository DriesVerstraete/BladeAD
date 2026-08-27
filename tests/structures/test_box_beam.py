from types import SimpleNamespace

import csdl_alpha as csdl
import numpy as np
import pytest

from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.airfoil.zero_d_airfoil_model import (
    ZeroDAirfoilModel,
    ZeroDAirfoilPolarParameters,
)
from BladeAD.core.structures import (
    IsotropicMaterial,
    compute_box_section_properties,
    evaluate_box_beam_structure,
)
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters


MATERIAL = IsotropicMaterial(
    density=1600.0,
    youngs_modulus=70.0e9,
    shear_modulus=5.0e9,
    tensile_allowable=600.0e6,
    compressive_allowable=450.0e6,
    shear_allowable=70.0e6,
)


def _profile(value, num_radial):
    return csdl.Variable(value=np.full(num_radial, value))


def _rotor_outputs(
    num_radial,
    num_blades,
    span,
    hub_radius,
    normal_line_load,
    tangential_line_load,
):
    element_width = span / num_radial
    radial = hub_radius + (np.arange(num_radial) + 0.5) * element_width
    shape = (1, num_radial, 1)
    return SimpleNamespace(
        sectional_thrust=csdl.Variable(
            value=np.full(shape, num_blades * normal_line_load * element_width)
        ),
        sectional_drag=csdl.Variable(
            value=np.full(shape, num_blades * tangential_line_load * element_width)
        ),
        radial_stations=csdl.Variable(value=radial.reshape(shape)),
        radial_element_width=csdl.Variable(
            value=np.full(shape, element_width)
        ),
        radial_integration_weights=csdl.Variable(value=np.ones(shape)),
        sectional_loads_include_all_blades=True,
    )


def test_rectangular_box_properties_match_closed_form():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    width = csdl.Variable(value=np.array([0.20]))
    height = csdl.Variable(value=np.array([0.10]))
    cap = csdl.Variable(value=np.array([0.005]))
    web = csdl.Variable(value=np.array([0.004]))
    section = compute_box_section_properties(width, height, cap, web, MATERIAL)

    inner_width = 0.20 - 2.0 * 0.004
    inner_height = 0.10 - 2.0 * 0.005
    expected_area = 0.20 * 0.10 - inner_width * inner_height
    expected_ix = (0.20 * 0.10**3 - inner_width * inner_height**3) / 12.0
    expected_iz = (0.10 * 0.20**3 - inner_height * inner_width**3) / 12.0
    np.testing.assert_allclose(section.inner_width.value, inner_width)
    np.testing.assert_allclose(section.inner_height.value, inner_height)
    np.testing.assert_allclose(section.area.value, expected_area)
    np.testing.assert_allclose(section.second_moment_chordwise.value, expected_ix)
    np.testing.assert_allclose(section.second_moment_thickness.value, expected_iz)
    recorder.stop()


def test_uniform_cantilever_and_centrifugal_loads_match_closed_form():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 8
    num_blades = 3
    span = 4.0
    hub_radius = 1.0
    normal_line_load = 100.0
    tangential_line_load = 30.0
    rpm_value = 1200.0
    outputs = evaluate_box_beam_structure(
        _rotor_outputs(
            num_radial,
            num_blades,
            span,
            hub_radius,
            normal_line_load,
            tangential_line_load,
        ),
        csdl.Variable(value=np.array([rpm_value])),
        num_blades,
        _profile(0.20, num_radial),
        _profile(0.10, num_radial),
        _profile(0.005, num_radial),
        _profile(0.004, num_radial),
        MATERIAL,
    )

    area = outputs.section.area.value[0, 0, 0]
    omega = rpm_value * 2.0 * np.pi / 60.0
    tip_radius = hub_radius + span
    expected_blade_mass = MATERIAL.density * area * span
    expected_axial = (
        MATERIAL.density
        * area
        * omega**2
        * (tip_radius**2 - hub_radius**2)
        / 2.0
    )
    np.testing.assert_allclose(outputs.blade_mass.value, expected_blade_mass)
    np.testing.assert_allclose(outputs.total_blade_mass.value, num_blades * expected_blade_mass)
    np.testing.assert_allclose(outputs.axial_force.value[0, 0, 0], expected_axial)
    np.testing.assert_allclose(outputs.normal_shear_force.value[0, 0, 0], normal_line_load * span)
    np.testing.assert_allclose(outputs.tangential_shear_force.value[0, 0, 0], tangential_line_load * span)
    np.testing.assert_allclose(
        outputs.flapwise_bending_moment.value[0, 0, 0],
        normal_line_load * span**2 / 2.0,
    )
    np.testing.assert_allclose(
        outputs.edgewise_bending_moment.value[0, 0, 0],
        tangential_line_load * span**2 / 2.0,
    )
    recorder.stop()


def _linear_load_root_moment(num_radial):
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    span = 3.0
    element_width = span / num_radial
    radial_from_hub = (np.arange(num_radial) + 0.5) * element_width
    line_load = 20.0 + 15.0 * radial_from_hub
    rotor_outputs = _rotor_outputs(num_radial, 2, span, 0.0, 0.0, 0.0)
    rotor_outputs.sectional_thrust = csdl.Variable(
        value=(2 * line_load * element_width).reshape(1, num_radial, 1)
    )
    outputs = evaluate_box_beam_structure(
        rotor_outputs,
        csdl.Variable(value=np.array([0.0])),
        2,
        _profile(0.20, num_radial),
        _profile(0.10, num_radial),
        _profile(0.005, num_radial),
        _profile(0.004, num_radial),
        MATERIAL,
    )
    result = outputs.flapwise_bending_moment.value[0, 0, 0]
    recorder.stop()
    return result


def test_load_integration_converges_for_linear_distributed_load():
    exact = 20.0 * 3.0**2 / 2.0 + 15.0 * 3.0**3 / 3.0
    coarse_error = abs(_linear_load_root_moment(6) - exact)
    fine_error = abs(_linear_load_root_moment(24) - exact)
    assert fine_error < coarse_error / 10.0


def test_structure_outputs_have_verified_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 5
    normal_load = csdl.Variable(
        name="normal_load",
        value=np.linspace(120.0, 40.0, num_radial).reshape(1, num_radial, 1),
    )
    rpm = csdl.Variable(name="rpm", value=np.array([900.0]))
    width = csdl.Variable(name="spar_width", value=np.full(num_radial, 0.18))
    cap = csdl.Variable(name="cap_thickness", value=np.full(num_radial, 0.006))
    rotor_outputs = _rotor_outputs(num_radial, 1, 2.0, 0.3, 0.0, 20.0)
    rotor_outputs.sectional_thrust = normal_load
    outputs = evaluate_box_beam_structure(
        rotor_outputs,
        rpm,
        1,
        width,
        _profile(0.09, num_radial),
        cap,
        _profile(0.004, num_radial),
        MATERIAL,
        aggregation_rho=30.0,
    )
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.blade_mass, outputs.maximum_utilization],
        [normal_load, rpm, width, cap],
        1.0e-8,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()


def test_structure_input_validation():
    with pytest.raises(ValueError):
        IsotropicMaterial(0.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    rotor_outputs = _rotor_outputs(3, 2, 1.0, 0.2, 10.0, 2.0)
    rotor_outputs.sectional_loads_include_all_blades = False
    with pytest.raises(ValueError):
        evaluate_box_beam_structure(
            rotor_outputs,
            csdl.Variable(value=np.array([1000.0])),
            2,
            _profile(0.2, 3),
            _profile(0.1, 3),
            _profile(0.005, 3),
            _profile(0.004, 3),
            MATERIAL,
        )
    recorder.stop()


def test_real_bem_outputs_connect_to_box_beam_model():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 9
    num_azimuthal = 4
    num_blades = 2
    chord = csdl.Variable(value=np.linspace(0.22, 0.08, num_radial))
    rpm = csdl.Variable(value=np.array([1800.0]))
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        chord_profile=chord,
        twist_profile=csdl.Variable(
            value=np.linspace(np.deg2rad(45.0), np.deg2rad(18.0), num_radial)
        ),
        radius=csdl.Variable(value=1.1),
        num_radial=num_radial,
        num_azimuthal=num_azimuthal,
        num_blades=num_blades,
        norm_hub_radius=0.25,
    )
    inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=csdl.Variable(value=np.array([[35.0, 0.0, 0.0]])),
        mesh_parameters=mesh,
    )
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
    rotor_outputs = BEMModel(
        num_nodes=1,
        airfoil_model=ZeroDAirfoilModel(polar_parameters=polar),
        integration_scheme="trapezoidal",
    ).evaluate(inputs)
    outputs = evaluate_box_beam_structure(
        rotor_outputs,
        rpm,
        num_blades,
        0.45 * chord,
        0.12 * chord,
        csdl.Variable(value=np.full(num_radial, 0.0035)),
        csdl.Variable(value=np.full(num_radial, 0.0030)),
        MATERIAL,
    )

    assert outputs.axial_force.shape == (1, num_radial, num_azimuthal)
    assert outputs.corner_normal_stress.shape == (1, num_radial, num_azimuthal, 4)
    assert np.all(np.isfinite(outputs.maximum_utilization.value))
    assert np.all(outputs.blade_mass.value > 0.0)
    assert np.all(outputs.section.inner_width.value > 0.0)
    assert np.all(outputs.section.inner_height.value > 0.0)
    recorder.stop()
