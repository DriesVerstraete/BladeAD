import argparse
import json
from pathlib import Path

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


VARIABLE_NAMES = (
    "rpm",
    "root_chord",
    "tip_chord",
    "root_twist",
    "tip_twist",
    "collective_pitch",
)
OUTPUT_NAMES = ("tonal_spl", "sectional_thrust", "sectional_drag")


def build_case():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 15
    variables = {
        "root_chord": csdl.Variable(name="root_chord", value=np.array([0.22])),
        "tip_chord": csdl.Variable(name="tip_chord", value=np.array([0.06])),
        "root_twist": csdl.Variable(
            name="root_twist", value=np.array([np.deg2rad(45.0)])
        ),
        "tip_twist": csdl.Variable(
            name="tip_twist", value=np.array([np.deg2rad(18.0)])
        ),
        "rpm": csdl.Variable(name="rpm", value=np.array([1800.0])),
        "collective_pitch": csdl.Variable(
            name="collective_pitch", value=np.array([np.deg2rad(2.0)])
        ),
    }
    chord_profile = csdl.linear_combination(
        variables["root_chord"], variables["tip_chord"], num_radial
    ).flatten()
    twist_profile = csdl.linear_combination(
        variables["root_twist"], variables["tip_twist"], num_radial
    ).flatten()
    mesh = RotorMeshParameters(
        thrust_vector=csdl.Variable(value=np.array([1.0, 0.0, 0.0])),
        thrust_origin=csdl.Variable(value=np.zeros(3)),
        chord_profile=chord_profile,
        twist_profile=twist_profile + variables["collective_pitch"],
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
        rpm=variables["rpm"],
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
    acoustic_outputs = evaluate_rotor_acoustics(
        inputs,
        bem_outputs,
        AcousticObserverData(
            positions=csdl.Variable(value=np.array([[20.0, 15.0, 5.0]]))
        ),
        RotorAcousticSettings(
            modes=(1, 2),
            load_harmonics=tuple(range(11)),
            tonal_enabled=True,
            thickness_enabled=True,
            sears_enabled=True,
            broadband_enabled=False,
        ),
    )
    outputs = {
        "tonal_spl": acoustic_outputs.tonal_spl,
        "sectional_thrust": bem_outputs.sectional_thrust,
        "sectional_drag": bem_outputs.sectional_drag,
    }
    return recorder, variables, outputs


def run_step(step):
    recorder, variables, outputs = build_case()
    try:
        output_variables = [outputs[name] for name in OUTPUT_NAMES]
        design_variables = [variables[name] for name in VARIABLE_NAMES]
        finite_differences = {}
        actual_steps = {}
        for variable_name in VARIABLE_NAMES:
            variable = variables[variable_name]
            original_value = variable.value.copy()
            actual_step = step * max(float(np.linalg.norm(original_value)), 1.0)
            actual_steps[variable_name] = actual_step
            variable.value = original_value + actual_step
            recorder.active_graph.execute_inline()
            plus_values = {name: outputs[name].value.copy() for name in OUTPUT_NAMES}
            variable.value = original_value - actual_step
            recorder.active_graph.execute_inline()
            minus_values = {name: outputs[name].value.copy() for name in OUTPUT_NAMES}
            variable.value = original_value
            recorder.active_graph.execute_inline()
            for output_name in OUTPUT_NAMES:
                output = outputs[output_name]
                finite_differences[output_name, variable_name] = (
                    (plus_values[output_name] - minus_values[output_name])
                    / (2.0 * actual_step)
                ).reshape(output.size, variable.size)
        derivatives = csdl.derivative(ofs=output_variables, wrts=design_variables)
        recorder.active_graph.execute_inline()
        results = {}
        for output_name in OUTPUT_NAMES:
            for variable_name in VARIABLE_NAMES:
                output = outputs[output_name]
                variable = variables[variable_name]
                fd_value = finite_differences[output_name, variable_name]
                csdl_value = derivatives[output, variable].value.copy()
                absolute_error = np.linalg.norm(csdl_value - fd_value)
                reference_norm = np.linalg.norm(fd_value)
                results[f"{output_name}:{variable_name}"] = {
                    "step": actual_steps[variable_name],
                    "csdl_norm": float(np.linalg.norm(csdl_value)),
                    "fd_norm": float(reference_norm),
                    "absolute_error": float(absolute_error),
                    "relative_error": float(absolute_error / reference_norm),
                }
        return results
    finally:
        recorder.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", nargs="+", type=float, default=(1.0e-4, 1.0e-5, 1.0e-6)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {f"{step:.0e}": run_step(step) for step in args.steps}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
