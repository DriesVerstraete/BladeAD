from __future__ import annotations

import csdl_alpha as csdl
import numpy as np

from BladeAD.core.acoustics.aggregation import pressure_squared_to_spl
from BladeAD.core.acoustics.convection import compute_convected_distance
from BladeAD.core.acoustics.observers import (
    evaluate_observer_geometry,
    evaluate_observer_geometry_nodes,
)
from BladeAD.core.acoustics.tonal import (
    compute_hanson_line_load_harmonics,
    compute_hanson_line_source_loading,
    compute_hanson_retarded_geometry,
    compute_hanson_thickness_noise,
    compute_load_harmonics,
    compute_barry_magliozzi_thickness_noise,
    compute_lowson_loading_pressure,
    compute_sears_load_harmonics,
    synthesize_lowson_rotor_pressure,
)
from BladeAD.core.acoustics.weighting import a_weighting_db
from BladeAD.core.acoustics.var_groups import (
    AcousticObserverData,
    RotorAcousticOutputs,
    RotorAcousticSettings,
)


def evaluate_rotor_acoustics(
    rotor_inputs,
    rotor_outputs,
    observers: AcousticObserverData,
    settings: RotorAcousticSettings | None = None,
):
    settings = settings or RotorAcousticSettings()
    if settings.tonal_model not in {"lowson", "hanson_line"}:
        raise ValueError("Tonal model must be 'lowson' or 'hanson_line'.")
    if settings.thickness_enabled and not settings.tonal_enabled:
        raise ValueError("Thickness noise requires tonal acoustics to be enabled.")
    if settings.tonal_model == "hanson_line" and settings.sears_enabled:
        raise ValueError("Sears loading is available only with the Lowson model.")
    if settings.broadband_enabled:
        raise NotImplementedError(
            "Broadband models are not available in the acoustic foundation."
        )
    if not settings.modes or any(int(mode) != mode or mode <= 0 for mode in settings.modes):
        raise ValueError("Acoustic modes must be positive integers.")

    mesh = rotor_inputs.mesh_parameters
    rpm = rotor_inputs.rpm
    if not isinstance(rpm, csdl.Variable):
        rpm = csdl.Variable(value=np.asarray(rpm, dtype=float))
    if rpm.shape == ():
        rpm = rpm.reshape((1,))
    if len(rpm.shape) != 1:
        raise ValueError("RPM must be scalar or one-dimensional.")

    modes = csdl.Variable(value=np.asarray(settings.modes, dtype=float))
    fundamental = rpm * mesh.num_blades / 60.0
    fundamental_expanded = csdl.expand(
        fundamental, (rpm.shape[0], modes.shape[0]), "i->ij"
    )
    modes_expanded = csdl.expand(modes, fundamental_expanded.shape, "j->ij")
    blade_passing_frequencies = fundamental_expanded * modes_expanded

    if not settings.tonal_enabled:
        distance, direction, axis_cosine = evaluate_observer_geometry(
            observers=observers,
            source_origin=mesh.thrust_origin,
            thrust_axis=mesh.thrust_vector,
        )
        return RotorAcousticOutputs(
            observer_distance=distance,
            observer_direction=direction,
            observer_axis_cosine=axis_cosine,
            blade_passing_frequencies=blade_passing_frequencies,
        )

    if rotor_outputs is None:
        raise ValueError("Tonal acoustics require rotor analysis outputs.")
    required = (
        "sectional_thrust",
        "sectional_drag",
        "radial_stations",
        "radial_element_width",
        "radial_integration_weights",
        "azimuth_angle",
    )
    if any(getattr(rotor_outputs, name, None) is None for name in required):
        raise ValueError("Rotor outputs do not expose the required tonal sectional data.")
    if rotor_outputs.sectional_loads_include_all_blades is not True:
        raise ValueError("Tonal integration requires complete-rotor sectional loads.")
    if any(int(value) != value for value in settings.load_harmonics):
        raise ValueError("Load harmonics must be integers.")
    load_harmonics = tuple(int(value) for value in settings.load_harmonics)
    num_azimuthal = rotor_outputs.sectional_thrust.shape[2]
    if not load_harmonics or any(value < 0 for value in load_harmonics):
        raise ValueError("Load harmonics must be non-negative integers.")
    if len(set(load_harmonics)) != len(load_harmonics):
        raise ValueError("Load harmonics must be unique.")
    if settings.tonal_model == "hanson_line" and load_harmonics != (0,):
        raise ValueError("The initial Hanson API supports only the steady load harmonic (0,).")
    if not settings.sears_enabled and max(load_harmonics) > (num_azimuthal - 1) // 2:
        raise ValueError("Azimuth resolution is insufficient for requested load harmonics.")

    num_nodes = rpm.shape[0]
    def node_vectors(value):
        variable = value if isinstance(value, csdl.Variable) else csdl.Variable(value=np.asarray(value))
        if variable.shape == (3,):
            return csdl.expand(variable, (num_nodes, 3), "j->ij")
        if variable.shape != (num_nodes, 3):
            raise ValueError("Rotor origin and thrust axis must have shape (3,) or (node, 3).")
        return variable

    origins = node_vectors(mesh.thrust_origin)
    axes = node_vectors(mesh.thrust_vector)
    distance, direction, axial_distance, in_plane_distance = evaluate_observer_geometry_nodes(
        observers, origins, axes
    )
    source_velocity = rotor_inputs.mesh_velocity
    if source_velocity.shape != (num_nodes, 3):
        raise ValueError("Mesh velocity must have shape (node, 3).")
    sound_speed = rotor_inputs.atmos_states.speed_of_sound
    if not isinstance(sound_speed, csdl.Variable):
        sound_speed = csdl.Variable(value=np.asarray(sound_speed, dtype=float).reshape(-1))
    if sound_speed.shape == (1,) and num_nodes > 1:
        sound_speed = csdl.expand(sound_speed, (num_nodes,))
    if sound_speed.shape != (num_nodes,):
        raise ValueError("Speed of sound must be scalar or have shape (node,).")
    convected_distance = compute_convected_distance(
        distance, direction, source_velocity, sound_speed
    )
    def node_radial(value, name):
        variable = value if isinstance(value, csdl.Variable) else csdl.Variable(value=np.asarray(value))
        if variable.shape == (mesh.num_radial,):
            return csdl.expand(variable, (num_nodes, mesh.num_radial), "r->ir")
        if variable.shape != (num_nodes, mesh.num_radial):
            raise ValueError(f"{name} must have shape (radial,) or (node, radial).")
        return variable

    def node_scalar(value, name):
        variable = value if isinstance(value, csdl.Variable) else csdl.Variable(
            value=np.asarray(value, dtype=float).reshape(-1)
        )
        if variable.shape == (1,) and num_nodes > 1:
            variable = csdl.expand(variable, (num_nodes,))
        if variable.shape != (num_nodes,):
            raise ValueError(f"{name} must be scalar or have shape (node,).")
        return variable

    density = node_scalar(rotor_inputs.atmos_states.density, "Density")
    if settings.tonal_model == "hanson_line":
        tip_radius = node_scalar(mesh.radius, "Tip radius")
        line_loads = compute_hanson_line_load_harmonics(
            rotor_outputs.sectional_thrust,
            rotor_outputs.sectional_drag,
            rotor_outputs.radial_element_width,
            rotor_outputs.radial_integration_weights,
            rotor_outputs.azimuth_angle,
            tip_radius,
            mesh.num_blades,
            load_harmonics,
        )
        unit_axes = axes / csdl.expand(
            csdl.sqrt(csdl.sum(axes**2, axes=(1,))), axes.shape, "i->ij"
        )
        axial_mach = csdl.sum(source_velocity * unit_axes, axes=(1,)) / sound_speed
        polar_angle = csdl.arccos(axial_distance / distance)
        observer_azimuth = csdl.Variable(value=np.zeros(distance.shape))
        retarded_geometry = compute_hanson_retarded_geometry(
            distance, polar_angle, axial_mach
        )
        loading_pressure = compute_hanson_line_source_loading(
            line_loads.axial_real,
            line_loads.axial_imaginary,
            line_loads.circumferential_real,
            line_loads.circumferential_imaginary,
            rotor_outputs.radial_stations[:, :, 0]
            / csdl.expand(tip_radius, (num_nodes, mesh.num_radial), "i->ir"),
            line_loads.nondimensional_radial_weights,
            rpm * 2.0 * np.pi / 60.0,
            tip_radius,
            sound_speed,
            axial_mach,
            retarded_geometry.distance,
            retarded_geometry.polar_angle,
            observer_azimuth,
            mesh.num_blades,
            settings.modes,
            load_harmonics,
        )
        tonal = synthesize_lowson_rotor_pressure(
            loading_pressure.cosine_pressure,
            loading_pressure.sine_pressure,
            1,
            settings.reference_pressure,
            settings.pressure_squared_floor,
        )
    elif settings.sears_enabled:
        if getattr(rotor_outputs, "sectional_inflow_angle", None) is None:
            raise ValueError("Sears loading requires sectional inflow angle.")
        coefficients = compute_sears_load_harmonics(
            rotor_outputs.sectional_thrust[:, :, 0]
            * rotor_outputs.radial_integration_weights[:, :, 0],
            rotor_outputs.sectional_drag[:, :, 0]
            * rotor_outputs.radial_integration_weights[:, :, 0],
            rotor_outputs.radial_stations[:, :, 0],
            rotor_outputs.radial_element_width[:, :, 0],
            rotor_outputs.radial_integration_weights[:, :, 0],
            node_radial(mesh.chord_profile, "Chord profile"),
            rotor_outputs.sectional_inflow_angle[:, :, 0],
            rpm * 2.0 * np.pi / 60.0,
            density,
            mesh.num_blades,
            load_harmonics,
            settings.sears_gust_amplification,
        )
    else:
        coefficients = compute_load_harmonics(
            rotor_outputs.sectional_thrust * rotor_outputs.radial_integration_weights,
            rotor_outputs.sectional_drag * rotor_outputs.radial_integration_weights,
            rotor_outputs.azimuth_angle,
            mesh.num_blades,
            load_harmonics,
        )
    if settings.tonal_model == "lowson":
        loading_pressure = compute_lowson_loading_pressure(
            coefficients,
            rotor_outputs.radial_stations[:, :, 0],
            rpm * 2.0 * np.pi / 60.0,
            axial_distance,
            in_plane_distance,
            distance,
            sound_speed,
            mesh.num_blades,
            settings.modes,
            convected_distance,
        )
        tonal = synthesize_lowson_rotor_pressure(
            loading_pressure.cosine_pressure,
            loading_pressure.sine_pressure,
            mesh.num_blades,
            settings.reference_pressure,
            settings.pressure_squared_floor,
        )
    combined_mode_squared = tonal.mode_pressure_squared
    combined_cosine_pressure = tonal.rotor_cosine_pressure
    combined_sine_pressure = tonal.rotor_sine_pressure
    thickness = None
    if settings.thickness_enabled:
        if mesh.thickness_to_chord is None:
            raise ValueError("Thickness noise requires mesh thickness_to_chord data.")

        if settings.tonal_model == "hanson_line":
            shape_inputs = (
                mesh.normalized_thickness_shape,
                mesh.thickness_shape_chordwise_locations,
                mesh.thickness_shape_chordwise_weights,
            )
            if any(value is None for value in shape_inputs):
                raise ValueError("Hanson thickness noise requires chordwise thickness-shape data.")
            thickness_pressure = compute_hanson_thickness_noise(
                rotor_outputs.radial_stations[:, :, 0]
                / csdl.expand(tip_radius, (num_nodes, mesh.num_radial), "i->ir"),
                line_loads.nondimensional_radial_weights,
                node_radial(mesh.chord_profile, "Chord profile"),
                node_radial(mesh.thickness_to_chord, "Thickness-to-chord profile"),
                mesh.normalized_thickness_shape,
                mesh.thickness_shape_chordwise_locations,
                mesh.thickness_shape_chordwise_weights,
                rpm * 2.0 * np.pi / 60.0,
                tip_radius,
                density,
                sound_speed,
                axial_mach,
                retarded_geometry.distance,
                retarded_geometry.polar_angle,
                observer_azimuth,
                mesh.num_blades,
                settings.modes,
            )
            thickness = synthesize_lowson_rotor_pressure(
                thickness_pressure.cosine_pressure,
                thickness_pressure.sine_pressure,
                1,
                settings.reference_pressure,
                settings.pressure_squared_floor,
            )
            combined_cosine_pressure = (
                tonal.rotor_cosine_pressure + thickness.rotor_cosine_pressure
            )
            combined_sine_pressure = tonal.rotor_sine_pressure + thickness.rotor_sine_pressure
            combined_mode_squared = 0.5 * (
                combined_cosine_pressure**2 + combined_sine_pressure**2
            )
        else:
            velocity_squared = csdl.sum(source_velocity**2, axes=(1,))
            mach_number = csdl.sqrt(velocity_squared + 1.0e-24) / sound_speed
            thickness = compute_barry_magliozzi_thickness_noise(
                rotor_outputs.radial_stations[:, :, 0],
                rotor_outputs.radial_element_width[:, :, 0],
                node_radial(mesh.chord_profile, "Chord profile"),
                node_radial(mesh.thickness_to_chord, "Thickness-to-chord profile"),
                rpm * 2.0 * np.pi / 60.0,
                axial_distance,
                in_plane_distance,
                distance,
                density,
                sound_speed,
                mach_number,
                mesh.num_blades,
                settings.modes,
                settings.reference_pressure,
                settings.pressure_squared_floor,
            )
            combined_mode_squared = tonal.mode_pressure_squared + thickness.mode_pressure_squared

    combined_total_squared = csdl.reshape(
        csdl.sum(combined_mode_squared, axes=(2,)), distance.shape
    )
    combined_mode_spl = pressure_squared_to_spl(
        combined_mode_squared,
        settings.reference_pressure,
        settings.pressure_squared_floor,
    )
    combined_total_spl = csdl.reshape(
        pressure_squared_to_spl(
            combined_total_squared,
            settings.reference_pressure,
            settings.pressure_squared_floor,
        ),
        distance.shape,
    )
    total_spl_a_weighted = None
    if settings.a_weighting_enabled:
        weighting = a_weighting_db(blade_passing_frequencies)
        weighting = csdl.expand(
            weighting,
            (num_nodes, distance.shape[1], len(settings.modes)),
            "im->iom",
        )
        weighted_mode_squared = combined_mode_squared * csdl.exp(
            np.log(10.0) / 10.0 * weighting
        )
        weighted_total = csdl.reshape(
            csdl.sum(weighted_mode_squared, axes=(2,)), distance.shape
        )
        total_spl_a_weighted = csdl.reshape(
            pressure_squared_to_spl(
                weighted_total, settings.reference_pressure, settings.pressure_squared_floor
            ),
            distance.shape,
        )

    return RotorAcousticOutputs(
        observer_distance=distance,
        observer_direction=direction,
        observer_axis_cosine=csdl.reshape(axial_distance / distance, distance.shape),
        blade_passing_frequencies=blade_passing_frequencies,
        tonal_pressure_squared=combined_total_squared,
        total_pressure_squared=combined_total_squared,
        tonal_spl=combined_total_spl,
        loading_pressure_squared=tonal.total_pressure_squared,
        loading_spl=tonal.total_spl,
        loading_mode_spl=tonal.mode_spl,
        loading_cosine_pressure=tonal.rotor_cosine_pressure,
        loading_sine_pressure=tonal.rotor_sine_pressure,
        thickness_pressure_squared=None if thickness is None else thickness.total_pressure_squared,
        thickness_spl=None if thickness is None else thickness.total_spl,
        thickness_mode_spl=None if thickness is None else thickness.mode_spl,
        thickness_cosine_pressure=(
            None
            if thickness is None or settings.tonal_model != "hanson_line"
            else thickness.rotor_cosine_pressure
        ),
        thickness_sine_pressure=(
            None
            if thickness is None or settings.tonal_model != "hanson_line"
            else thickness.rotor_sine_pressure
        ),
        tonal_mode_spl=combined_mode_spl,
        tonal_cosine_pressure=combined_cosine_pressure,
        tonal_sine_pressure=combined_sine_pressure,
        total_spl=combined_total_spl,
        total_spl_a_weighted=total_spl_a_weighted,
    )
