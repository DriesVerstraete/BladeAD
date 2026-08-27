import argparse
import json
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem

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
CASE_DATA = {
    "forward_flight": {
        "mesh_velocity": np.array([[15.0, 0.0, 0.0]]),
        "observer_position": np.array([[20.0, 15.0, 5.0]]),
    },
    "hover_like": {
        "mesh_velocity": np.array([[0.5, 0.0, 0.0]]),
        "observer_position": np.array([[0.0, 20.0, 5.0]]),
    },
}


def build_problem(case_name):
    case = CASE_DATA[case_name]
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    num_radial = 15
    variables = {
        "rpm": csdl.Variable(name="rpm", value=np.array([1800.0])),
        "root_chord": csdl.Variable(name="root_chord", value=np.array([0.22])),
        "tip_chord": csdl.Variable(name="tip_chord", value=np.array([0.06])),
        "root_twist": csdl.Variable(
            name="root_twist", value=np.array([np.deg2rad(45.0)])
        ),
        "tip_twist": csdl.Variable(
            name="tip_twist", value=np.array([np.deg2rad(18.0)])
        ),
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
        normalized_thickness_shape=csdl.Variable(
            value=np.array([0.0, 1.0, 0.6, 0.0])
        ),
        thickness_shape_chordwise_locations=csdl.Variable(
            value=np.array([-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5])
        ),
        thickness_shape_chordwise_weights=csdl.Variable(
            value=np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
        ),
    )
    inputs = RotorAnalysisInputs(
        rpm=variables["rpm"],
        mesh_velocity=csdl.Variable(value=case["mesh_velocity"]),
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
            positions=csdl.Variable(value=case["observer_position"])
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
    for variable in variables.values():
        variable.set_as_design_variable()
    acoustic_outputs.tonal_spl.set_as_objective()
    simulator = csdl.experimental.PySimulator(recorder=recorder)
    problem = CSDLAlphaProblem(
        problem_name=f"tonal_production_graph_{case_name}", simulator=simulator
    )
    return recorder, problem, variables, acoustic_outputs.tonal_spl


def verify_case(case_name, relative_step):
    recorder, problem, variables, tonal_spl = build_problem(case_name)
    try:
        x0 = problem.x0.copy()
        problem_primal = float(problem._compute_objective(x0, force_rerun=True))
        direct_primal = float(tonal_spl.value.reshape(-1)[0])
        problem_gradient = problem._compute_objective_gradient(
            x0, force_rerun=True
        ).copy()
        direct_derivatives = csdl.derivative(
            ofs=[tonal_spl], wrts=[variables[name] for name in VARIABLE_NAMES]
        )
        recorder.active_graph.execute_inline()
        direct_gradient = np.array(
            [
                float(direct_derivatives[tonal_spl, variables[name]].value[0, 0])
                for name in VARIABLE_NAMES
            ]
        )
        finite_difference = np.zeros_like(problem_gradient)
        steps = np.zeros_like(problem_gradient)
        for index, value in enumerate(x0):
            step = relative_step * max(abs(value), 1.0)
            steps[index] = step
            plus = x0.copy()
            minus = x0.copy()
            plus[index] += step
            minus[index] -= step
            finite_difference[index] = (
                problem._compute_objective(plus, force_rerun=True)
                - problem._compute_objective(minus, force_rerun=True)
            ) / (2.0 * step)
        problem._compute_objective(x0, force_rerun=True)
        fd_error = np.linalg.norm(problem_gradient - finite_difference)
        direct_error = np.linalg.norm(problem_gradient - direct_gradient)
        return {
            "primal": {
                "problem": problem_primal,
                "direct": direct_primal,
                "absolute_error": abs(problem_primal - direct_primal),
            },
            "gradient": {
                "variable_order": VARIABLE_NAMES,
                "steps": steps.tolist(),
                "problem": problem_gradient.tolist(),
                "direct_csdl": direct_gradient.tolist(),
                "central_difference": finite_difference.tolist(),
                "problem_vs_direct_absolute_error": float(direct_error),
                "problem_vs_direct_relative_error": float(
                    direct_error / np.linalg.norm(direct_gradient)
                ),
                "problem_vs_fd_absolute_error": float(fd_error),
                "problem_vs_fd_relative_error": float(
                    fd_error / np.linalg.norm(finite_difference)
                ),
            },
        }
    finally:
        recorder.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        case_name: verify_case(case_name, args.step) for case_name in CASE_DATA
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
