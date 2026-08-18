from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.tonal import compute_barry_magliozzi_thickness_noise


FIXTURE = (
    Path(__file__).parents[2]
    / "validation"
    / "acoustics"
    / "fixtures"
    / "lowson_hg_matlab"
    / "bm_thickness_reference.csv"
)


def _evaluate_thickness(angular_speed=None):
    reference = np.genfromtxt(FIXTURE, delimiter=",", names=True)
    theta = np.deg2rad(reference["angle_from_axis_deg"])
    observer_radius = np.sqrt(1.8594751405**2 + 1.3020185105**2)
    axial = (observer_radius * np.cos(theta))[None, :]
    in_plane = (observer_radius * np.sin(theta))[None, :]
    distance = np.full_like(axial, observer_radius)
    num_radial = 40
    rotor_radius = 0.1588
    radial_stations = (
        rotor_radius * np.linspace(0.21, 0.99, num_radial)
    )[None, :]
    width = np.full(
        (1, num_radial), (1.0 - 0.2) * rotor_radius / (num_radial - 1)
    )
    if angular_speed is None:
        angular_speed = csdl.Variable(value=np.array([5500.0 * 2.0 * np.pi / 60.0]))
    outputs = compute_barry_magliozzi_thickness_noise(
        csdl.Variable(value=radial_stations),
        csdl.Variable(value=width),
        csdl.Variable(value=np.full((1, num_radial), 0.03176)),
        csdl.Variable(value=np.full((1, num_radial), 0.12)),
        angular_speed,
        csdl.Variable(value=axial),
        csdl.Variable(value=in_plane),
        csdl.Variable(value=distance),
        csdl.Variable(value=np.array([1.225])),
        csdl.Variable(value=np.array([343.0])),
        csdl.Variable(value=np.array([0.0])),
        num_blades=4,
        modes=(1,),
        pressure_squared_floor=1e-30,
    )
    return reference, outputs


def test_barry_magliozzi_thickness_agrees_with_hj_reference_directivity():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference, outputs = _evaluate_thickness()
    error = outputs.total_spl.value[0] - reference["thickness_spl_db"]
    assert np.max(np.abs(error)) < 1.0
    assert np.ptp(error) < 0.01
    recorder.stop()


def test_barry_magliozzi_thickness_derivative_wrt_angular_speed():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    angular_speed = csdl.Variable(
        name="angular_speed", value=np.array([5500.0 * 2.0 * np.pi / 60.0])
    )
    _, outputs = _evaluate_thickness(angular_speed)
    errors = csdl.derivative_utils.verify_derivatives(
        [outputs.total_spl],
        [angular_speed],
        1e-5,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
    recorder.stop()
