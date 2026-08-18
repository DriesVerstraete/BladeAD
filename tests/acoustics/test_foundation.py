import csdl_alpha as csdl
import numpy as np
import pytest

from BladeAD.core.acoustics import (
    AcousticObserverData,
    RotorAcousticSettings,
    a_weighting_db,
    energetic_sum,
    evaluate_observer_geometry,
    evaluate_rotor_acoustics,
    pressure_squared_to_spl,
)
from BladeAD.utils.var_groups import RotorAnalysisInputs, RotorMeshParameters


@pytest.fixture
def recorder():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    yield recorder
    recorder.stop()


def test_observer_geometry_and_directivity(recorder):
    observers = AcousticObserverData(
        positions=np.array([[10.0, 0.0, 0.0], [0.0, 3.0, 4.0]]),
        names=("axis", "sideline"),
    )
    distance, direction, axis_cosine = evaluate_observer_geometry(
        observers, np.zeros(3), np.array([2.0, 0.0, 0.0])
    )

    np.testing.assert_allclose(distance.value, [10.0, 5.0])
    np.testing.assert_allclose(
        direction.value, [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]
    )
    np.testing.assert_allclose(axis_cosine.value, [1.0, 0.0], atol=1.0e-14)


def test_pressure_squared_aggregation_and_spl(recorder):
    reference_pressure = 20.0e-6
    first = csdl.Variable(value=np.array([reference_pressure**2, 2.0e-8]))
    second = csdl.Variable(value=np.array([0.0, 3.0e-8]))
    total = energetic_sum(first, second)
    spl = pressure_squared_to_spl(
        total, reference_pressure=reference_pressure, pressure_squared_floor=1.0e-30
    )

    np.testing.assert_allclose(total.value, [reference_pressure**2, 5.0e-8])
    np.testing.assert_allclose(
        spl.value, 10.0 * np.log10(total.value / reference_pressure**2)
    )


def test_a_weighting_reference_values(recorder):
    frequencies = csdl.Variable(value=np.array([100.0, 1000.0, 10000.0]))
    weighting = a_weighting_db(frequencies)

    np.testing.assert_allclose(
        weighting.value, [-19.144954, 0.0001415, -2.491569], atol=1e-6
    )


def test_foundation_api_blade_passing_frequencies(recorder):
    mesh = RotorMeshParameters(
        thrust_vector=np.array([1.0, 0.0, 0.0]),
        thrust_origin=np.zeros(3),
        chord_profile=np.ones(5),
        twist_profile=np.zeros(5),
        radius=1.0,
        num_radial=5,
        num_azimuthal=1,
        num_blades=3,
    )
    inputs = RotorAnalysisInputs(
        rpm=csdl.Variable(value=np.array([1200.0, 1800.0])),
        mesh_velocity=csdl.Variable(value=np.zeros((2, 3))),
        mesh_parameters=mesh,
    )
    outputs = evaluate_rotor_acoustics(
        rotor_inputs=inputs,
        rotor_outputs=None,
        observers=AcousticObserverData(positions=np.array([[10.0, 0.0, 0.0]])),
        settings=RotorAcousticSettings(modes=(1, 2, 3)),
    )

    np.testing.assert_allclose(
        outputs.blade_passing_frequencies.value,
        [[60.0, 120.0, 180.0], [90.0, 180.0, 270.0]],
    )


def test_observer_distance_derivative(recorder):
    observer_position = csdl.Variable(
        name="observer_position", value=np.array([[3.0, 4.0, 1.0]])
    )
    distance, _, _ = evaluate_observer_geometry(
        AcousticObserverData(positions=observer_position),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
    )

    errors = csdl.derivative_utils.verify_derivatives(
        [distance],
        [observer_position],
        1.0e-6,
        print_results=False,
        raise_on_error=True,
    )
    assert errors is not None
