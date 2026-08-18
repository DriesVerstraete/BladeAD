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
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        chord_profile=csdl.Variable(value=np.linspace(0.22, 0.06, num_radial)),
        twist_profile=csdl.Variable(
            value=np.linspace(np.deg2rad(45.0), np.deg2rad(18.0), num_radial)
        ),
        radius=csdl.Variable(value=1.0),
        num_radial=num_radial,
        num_azimuthal=8,
        num_blades=2,
        norm_hub_radius=0.25,
        thickness_to_chord=csdl.Variable(value=np.full(num_radial, 0.12)),
    )
    inputs = RotorAnalysisInputs(
        rpm=csdl.Variable(value=np.array([1800.0])),
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
            load_harmonics=(0, 1, 2, 3),
            tonal_enabled=True,
            thickness_enabled=True,
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
    errors = csdl.derivative_utils.verify_derivatives(
        [acoustic_outputs.tonal_spl],
        [observer_position],
        1e-5,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
