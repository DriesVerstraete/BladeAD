import csdl_alpha as csdl
import numpy as np

from BladeAD.core.BEM.bem_model import BEMModel
from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    evaluate_rotor_acoustics,
)
from BladeAD.core.airfoil.zero_d_airfoil_model import (
    ZeroDAirfoilModel,
    ZeroDAirfoilPolarParameters,
)
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters


def test_real_bem_to_lowson_tonal_api_and_observer_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 15
    root_chord = csdl.Variable(name="root_chord", value=np.array([0.22]))
    tip_chord = csdl.Variable(name="tip_chord", value=np.array([0.06]))
    root_twist = csdl.Variable(name="root_twist", value=np.array([np.deg2rad(45.0)]))
    tip_twist = csdl.Variable(name="tip_twist", value=np.array([np.deg2rad(18.0)]))
    chord_profile = csdl.linear_combination(root_chord, tip_chord, num_radial).flatten()
    twist_profile = csdl.linear_combination(root_twist, tip_twist, num_radial).flatten()
    rpm = csdl.Variable(name="rpm", value=np.array([1800.0]))
    collective_pitch = csdl.Variable(
        name="collective_pitch", value=np.array([np.deg2rad(2.0)])
    )
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        chord_profile=chord_profile,
        twist_profile=twist_profile + collective_pitch,
        radius=csdl.Variable(value=1.0),
        num_radial=num_radial,
        num_azimuthal=8,
        num_blades=2,
        norm_hub_radius=0.25,
        thickness_to_chord=csdl.Variable(value=np.full(num_radial, 0.12)),
        normalized_thickness_shape=csdl.Variable(value=np.array([0.0, 1.0, 0.6, 0.0])),
        thickness_shape_chordwise_locations=csdl.Variable(
            value=np.array([-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5])
        ),
        thickness_shape_chordwise_weights=csdl.Variable(
            value=np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
        ),
    )
    inputs = RotorAnalysisInputs(
        rpm=rpm,
        mesh_velocity=csdl.Variable(value=np.array([[15.0, 0.0, 0.0]])),
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
    bem_outputs = BEMModel(
        num_nodes=1,
        airfoil_model=ZeroDAirfoilModel(polar_parameters=polar),
        integration_scheme="trapezoidal",
    ).evaluate(inputs)
    observer_position = csdl.Variable(
        name="observer_position", value=np.array([[20.0, 15.0, 5.0]])
    )
    acoustic_outputs = evaluate_rotor_acoustics(
        inputs,
        bem_outputs,
        AcousticObserverData(positions=observer_position),
        RotorAcousticSettings(
            modes=(1, 2),
            # Sears harmonics are analytic and are not Nyquist-limited by the
            # BEM azimuth grid (eight samples in this regression).
            load_harmonics=tuple(range(11)),
            tonal_enabled=True,
            thickness_enabled=True,
            sears_enabled=True,
            broadband_enabled=False,
        ),
    )

    assert acoustic_outputs.tonal_mode_spl.shape == (1, 1, 2)
    assert acoustic_outputs.tonal_spl.shape == (1, 1)
    assert acoustic_outputs.total_spl_a_weighted.shape == (1, 1)
    assert acoustic_outputs.thickness_spl.shape == (1, 1)
    assert acoustic_outputs.loading_spl.shape == (1, 1)
    assert acoustic_outputs.loading_mode_spl.shape == (1, 1, 2)
    assert np.all(np.isfinite(acoustic_outputs.tonal_spl.value))
    np.testing.assert_allclose(
        acoustic_outputs.total_pressure_squared.value,
        acoustic_outputs.tonal_pressure_squared.value,
    )
    assert np.all(
        acoustic_outputs.total_pressure_squared.value
        > acoustic_outputs.loading_pressure_squared.value
    )
    assert np.all(
        acoustic_outputs.tonal_mode_spl.value
        > acoustic_outputs.loading_mode_spl.value
    )
    observer_errors = csdl.derivative_utils.verify_derivatives(
        [acoustic_outputs.tonal_spl],
        [observer_position],
        1e-5,
        print_results=False,
        raise_on_error=True,
    )
    assert observer_errors is not None
    design_variables = [
        rpm,
        root_chord,
        tip_chord,
        root_twist,
        tip_twist,
        collective_pitch,
    ]
    design_errors = csdl.derivative_utils.verify_derivatives(
        [acoustic_outputs.tonal_spl],
        design_variables,
        1.0e-5,
        print_results=False,
        raise_on_error=False,
    )
    for variable in design_variables:
        result = design_errors[(acoustic_outputs.tonal_spl, variable)]
        assert np.linalg.norm(result["value"]) > 1.0e-8
        assert result["rel_error"] < 5.0e-4

    hanson_outputs = evaluate_rotor_acoustics(
        inputs,
        bem_outputs,
        AcousticObserverData(positions=observer_position),
        RotorAcousticSettings(
            modes=(1, 2),
            load_harmonics=(0,),
            tonal_model="hanson_line",
            tonal_enabled=True,
            thickness_enabled=True,
            broadband_enabled=False,
        ),
    )
    assert hanson_outputs.tonal_mode_spl.shape == (1, 1, 2)
    assert hanson_outputs.thickness_mode_spl.shape == (1, 1, 2)
    assert hanson_outputs.loading_cosine_pressure.shape == (1, 1, 2)
    assert hanson_outputs.thickness_cosine_pressure.shape == (1, 1, 2)
    assert np.all(
        np.abs(hanson_outputs.tonal_cosine_pressure.value)
        > 0.0
    )
    assert np.all(np.isfinite(hanson_outputs.tonal_spl.value))
    hanson_errors = csdl.derivative_utils.verify_derivatives(
        [hanson_outputs.tonal_spl],
        [rpm, observer_position],
        1.0e-5,
        print_results=False,
        raise_on_error=True,
    )
    assert hanson_errors is not None

    broadband_outputs = evaluate_rotor_acoustics(
        inputs,
        bem_outputs,
        AcousticObserverData(positions=observer_position),
        RotorAcousticSettings(
            tonal_enabled=False,
            thickness_enabled=False,
            broadband_enabled=True,
            broadband_center_frequencies=(100.0, 1000.0, 10000.0),
        ),
    )
    assert broadband_outputs.broadband_one_third_octave_spl.shape == (1, 1, 3)
    assert broadband_outputs.broadband_spl.shape == (1, 1)
    np.testing.assert_allclose(
        broadband_outputs.total_pressure_squared.value,
        broadband_outputs.broadband_pressure_squared.value,
    )
    assert np.all(np.isfinite(broadband_outputs.total_spl.value))

    broadband_errors = csdl.derivative_utils.verify_derivatives(
        [broadband_outputs.total_spl],
        [rpm, root_chord, observer_position],
        1.0e-5,
        print_results=False,
        raise_on_error=False,
    )
    for variable in (rpm, root_chord, observer_position):
        result = broadband_errors[(broadband_outputs.total_spl, variable)]
        assert np.linalg.norm(result["value"]) > 1.0e-8
        assert result["rel_error"] < 5.0e-4

    combined_outputs = evaluate_rotor_acoustics(
        inputs,
        bem_outputs,
        AcousticObserverData(positions=observer_position),
        RotorAcousticSettings(
            modes=(1, 2),
            load_harmonics=tuple(range(11)),
            tonal_enabled=True,
            thickness_enabled=True,
            sears_enabled=True,
            broadband_enabled=True,
            broadband_center_frequencies=(100.0, 1000.0, 10000.0),
        ),
    )
    np.testing.assert_allclose(
        combined_outputs.total_pressure_squared.value,
        combined_outputs.tonal_pressure_squared.value
        + combined_outputs.broadband_pressure_squared.value,
    )
    assert np.all(combined_outputs.total_spl.value > combined_outputs.tonal_spl.value)
    recorder.stop()
