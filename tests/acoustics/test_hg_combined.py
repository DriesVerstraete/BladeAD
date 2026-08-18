from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.aggregation import pressure_squared_to_spl
from BladeAD.core.acoustics.tonal import (
    compute_barry_magliozzi_thickness_noise,
    compute_lowson_loading_pressure,
    compute_sears_load_harmonics,
    synthesize_lowson_rotor_pressure,
)


FIXTURE = Path(__file__).parents[2] / "validation" / "acoustics" / "fixtures" / "lowson_hg_matlab"


def test_combined_lowson_hg_hover_directivity():
    """Reproduce the pinned LSDO Lowson/Sears/Barry-Magliozzi comparison case."""
    radial = np.genfromtxt(FIXTURE / "radial_inputs.csv", delimiter=",", names=True)
    reference = np.genfromtxt(
        FIXTURE / "hg_matlab_total_spl.csv", delimiter=",", names=True
    )["total_spl_db"][::-1]

    blade_count = 3
    rotor_radius = 0.3556
    density = 1.225
    speed_of_sound = 340.3
    angular_speed = np.array([1500.0 * 2.0 * np.pi / 60.0])
    element_width = 0.02 * rotor_radius
    radius = radial["radius_over_R"] * rotor_radius
    chord = radial["chord_m"]
    inflow = radial["lambda_i"] / radial["radius_over_R"]
    lift_per_length = (
        0.5 * density * chord * (rotor_radius * angular_speed[0]) ** 2 * radial["CL"]
    )
    weights = np.ones(radius.size)
    weights[[0, -1]] = 0.5
    # The source supplies per-blade force per unit radius. Sears expects
    # complete-rotor steady elemental loads and divides by B exactly once.
    steady_thrust = blade_count * lift_per_length * np.cos(inflow) * element_width * weights
    steady_drag = blade_count * lift_per_length * np.sin(inflow) * element_width * weights

    theta = np.linspace(0.0, np.pi, 37)
    observer_axial = 1.5 * np.cos(theta)[None, :]
    observer_in_plane = 1.5 * np.sin(theta)[None, :]
    observer_distance = np.full((1, theta.size), 1.5)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    coefficients = compute_sears_load_harmonics(
        csdl.Variable(value=steady_thrust[None, :]),
        csdl.Variable(value=steady_drag[None, :]),
        csdl.Variable(value=radius[None, :]),
        csdl.Variable(value=np.full((1, radius.size), element_width)),
        csdl.Variable(value=weights[None, :]),
        csdl.Variable(value=chord[None, :]),
        csdl.Variable(value=inflow[None, :]),
        csdl.Variable(value=angular_speed),
        csdl.Variable(value=np.array([density])),
        blade_count,
        harmonics=tuple(range(11)),
    )
    loading_pressure = compute_lowson_loading_pressure(
        coefficients,
        csdl.Variable(value=radius[None, :]),
        csdl.Variable(value=angular_speed),
        csdl.Variable(value=observer_axial),
        csdl.Variable(value=observer_in_plane),
        csdl.Variable(value=observer_distance),
        csdl.Variable(value=np.array([speed_of_sound])),
        blade_count,
        modes=(1,),
        convected_distance=csdl.Variable(value=observer_distance),
    )
    loading = synthesize_lowson_rotor_pressure(
        loading_pressure.cosine_pressure,
        loading_pressure.sine_pressure,
        blade_count,
    )
    thickness = compute_barry_magliozzi_thickness_noise(
        csdl.Variable(value=radius[None, :]),
        csdl.Variable(value=np.full((1, radius.size), element_width)),
        csdl.Variable(value=chord[None, :]),
        csdl.Variable(value=np.full((1, radius.size), 0.12)),
        csdl.Variable(value=angular_speed),
        csdl.Variable(value=observer_axial),
        csdl.Variable(value=observer_in_plane),
        csdl.Variable(value=observer_distance),
        csdl.Variable(value=np.array([density])),
        csdl.Variable(value=np.array([speed_of_sound])),
        csdl.Variable(value=np.array([0.0])),
        blade_count,
        modes=(1,),
    )
    total_spl = pressure_squared_to_spl(
        loading.mode_pressure_squared + thickness.mode_pressure_squared
    ).value.reshape(-1)
    recorder.stop()

    error = total_spl - reference
    # Characterisation gate: preserve level and directivity without calibration.
    assert abs(np.mean(error)) < 0.5
    assert np.max(np.abs(error - np.mean(error))) < 1.8
    assert np.max(np.abs(error)) < 2.2
