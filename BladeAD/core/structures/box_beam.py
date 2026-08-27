from __future__ import annotations

from dataclasses import dataclass
import math

import csdl_alpha as csdl
import numpy as np


@dataclass(frozen=True)
class IsotropicMaterial:
    density: float
    youngs_modulus: float
    shear_modulus: float
    tensile_allowable: float
    compressive_allowable: float
    shear_allowable: float

    def __post_init__(self) -> None:
        for name in (
            "density",
            "youngs_modulus",
            "shear_modulus",
            "tensile_allowable",
            "compressive_allowable",
            "shear_allowable",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class BoxSectionProperties(csdl.VariableGroup):
    inner_width: csdl.Variable
    inner_height: csdl.Variable
    area: csdl.Variable
    second_moment_chordwise: csdl.Variable
    second_moment_thickness: csdl.Variable
    axial_stiffness: csdl.Variable
    bending_stiffness_chordwise: csdl.Variable
    bending_stiffness_thickness: csdl.Variable


@dataclass
class BoxBeamOutputs(csdl.VariableGroup):
    section: BoxSectionProperties
    blade_mass: csdl.Variable
    total_blade_mass: csdl.Variable
    axial_force: csdl.Variable
    normal_shear_force: csdl.Variable
    tangential_shear_force: csdl.Variable
    flapwise_bending_moment: csdl.Variable
    edgewise_bending_moment: csdl.Variable
    corner_normal_stress: csdl.Variable
    web_shear_stress: csdl.Variable
    cap_shear_stress: csdl.Variable
    tensile_utilization: csdl.Variable
    compressive_utilization: csdl.Variable
    shear_utilization: csdl.Variable
    maximum_utilization: csdl.Variable


def compute_box_section_properties(
    outer_width: csdl.Variable,
    outer_height: csdl.Variable,
    cap_thickness: csdl.Variable,
    web_thickness: csdl.Variable,
    material: IsotropicMaterial,
) -> BoxSectionProperties:
    """Compute thin-walled rectangular box properties from outer dimensions."""

    inner_width = outer_width - 2.0 * web_thickness
    inner_height = outer_height - 2.0 * cap_thickness
    area = outer_width * outer_height - inner_width * inner_height
    second_moment_chordwise = (
        outer_width * outer_height**3 - inner_width * inner_height**3
    ) / 12.0
    second_moment_thickness = (
        outer_height * outer_width**3 - inner_height * inner_width**3
    ) / 12.0
    return BoxSectionProperties(
        inner_width=inner_width,
        inner_height=inner_height,
        area=area,
        second_moment_chordwise=second_moment_chordwise,
        second_moment_thickness=second_moment_thickness,
        axial_stiffness=material.youngs_modulus * area,
        bending_stiffness_chordwise=(
            material.youngs_modulus * second_moment_chordwise
        ),
        bending_stiffness_thickness=(
            material.youngs_modulus * second_moment_thickness
        ),
    )


def _expand_profile(profile: csdl.Variable, shape: tuple[int, int, int]) -> csdl.Variable:
    num_nodes, num_radial, _ = shape
    if profile.shape == shape:
        return profile
    if profile.shape == (num_radial,):
        return csdl.expand(profile, shape, "j->ijk")
    if profile.shape == (num_nodes, num_radial):
        return csdl.expand(profile, shape, "ij->ijk")
    raise ValueError(
        f"Structural profiles must have shape {(num_radial,)}, "
        f"{(num_nodes, num_radial)}, or {shape}; received {profile.shape}."
    )


def _validate_rotor_outputs(rotor_outputs, num_blades: int) -> tuple[int, int, int]:
    if not isinstance(num_blades, int) or num_blades < 1:
        raise ValueError("num_blades must be a positive integer")
    required = (
        "sectional_thrust",
        "sectional_drag",
        "radial_stations",
        "radial_element_width",
        "radial_integration_weights",
    )
    for name in required:
        if getattr(rotor_outputs, name, None) is None:
            raise ValueError(f"Rotor outputs must provide {name}")
    shape = rotor_outputs.sectional_thrust.shape
    if len(shape) != 3:
        raise ValueError("Sectional loads must have shape (node, radial, azimuth)")
    for name in required[1:]:
        if getattr(rotor_outputs, name).shape != shape:
            raise ValueError(f"{name} must have the sectional-load shape {shape}")
    if not rotor_outputs.sectional_loads_include_all_blades:
        raise ValueError("Structural coupling requires complete-rotor sectional loads")
    return shape


def evaluate_box_beam_structure(
    rotor_outputs,
    rpm: csdl.Variable,
    num_blades: int,
    outer_width: csdl.Variable,
    outer_height: csdl.Variable,
    cap_thickness: csdl.Variable,
    web_thickness: csdl.Variable,
    material: IsotropicMaterial,
    aggregation_rho: float = 50.0,
) -> BoxBeamOutputs:
    """Evaluate one rotating blade with an isotropic rectangular box spar.

    Sectional aerodynamic loads must use BladeAD's complete-rotor convention. Internal loads are
    evaluated at element inboard faces, and each azimuth is retained as a separate load case.
    """

    if not math.isfinite(aggregation_rho) or aggregation_rho <= 0.0:
        raise ValueError("aggregation_rho must be finite and positive")
    shape = _validate_rotor_outputs(rotor_outputs, num_blades)
    num_nodes, num_radial, num_azimuthal = shape
    if rpm.shape != (num_nodes,):
        raise ValueError(f"rpm must have shape {(num_nodes,)}; received {rpm.shape}")

    outer_width = _expand_profile(outer_width, shape)
    outer_height = _expand_profile(outer_height, shape)
    cap_thickness = _expand_profile(cap_thickness, shape)
    web_thickness = _expand_profile(web_thickness, shape)
    section = compute_box_section_properties(
        outer_width,
        outer_height,
        cap_thickness,
        web_thickness,
        material,
    )

    weights = rotor_outputs.radial_integration_weights
    element_width = rotor_outputs.radial_element_width
    radial_stations = rotor_outputs.radial_stations
    normal_element_force = rotor_outputs.sectional_thrust * weights / num_blades
    tangential_element_force = rotor_outputs.sectional_drag * weights / num_blades
    element_mass = material.density * section.area * element_width * weights
    angular_speed = csdl.expand(rpm, shape, "i->ijk") * (2.0 * np.pi / 60.0)
    centrifugal_element_force = element_mass * angular_speed**2 * radial_stations

    axial_force = csdl.Variable(shape=shape, value=0.0)
    normal_shear_force = csdl.Variable(shape=shape, value=0.0)
    tangential_shear_force = csdl.Variable(shape=shape, value=0.0)
    flapwise_bending_moment = csdl.Variable(shape=shape, value=0.0)
    edgewise_bending_moment = csdl.Variable(shape=shape, value=0.0)
    for radial_index in range(num_radial):
        inboard_face = (
            radial_stations[:, radial_index, :]
            - 0.5 * element_width[:, radial_index, :]
        )
        outboard_shape = (num_nodes, num_radial - radial_index, num_azimuthal)
        lever_arm = radial_stations[:, radial_index:, :] - csdl.expand(
            inboard_face, outboard_shape, "ik->ijk"
        )
        axial_force = axial_force.set(
            csdl.slice[:, radial_index, :],
            csdl.sum(centrifugal_element_force[:, radial_index:, :], axes=(1,)),
        )
        normal_shear_force = normal_shear_force.set(
            csdl.slice[:, radial_index, :],
            csdl.sum(normal_element_force[:, radial_index:, :], axes=(1,)),
        )
        tangential_shear_force = tangential_shear_force.set(
            csdl.slice[:, radial_index, :],
            csdl.sum(tangential_element_force[:, radial_index:, :], axes=(1,)),
        )
        flapwise_bending_moment = flapwise_bending_moment.set(
            csdl.slice[:, radial_index, :],
            csdl.sum(
                normal_element_force[:, radial_index:, :] * lever_arm,
                axes=(1,),
            ),
        )
        edgewise_bending_moment = edgewise_bending_moment.set(
            csdl.slice[:, radial_index, :],
            csdl.sum(
                tangential_element_force[:, radial_index:, :] * lever_arm,
                axes=(1,),
            ),
        )

    axial_stress = axial_force / section.area
    flapwise_bending_stress = (
        flapwise_bending_moment
        * outer_height
        / (2.0 * section.second_moment_chordwise)
    )
    edgewise_bending_stress = (
        edgewise_bending_moment
        * outer_width
        / (2.0 * section.second_moment_thickness)
    )
    corner_normal_stress = csdl.Variable(
        shape=shape + (4,), value=0.0
    )
    corner_signs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    for corner, (flapwise_sign, edgewise_sign) in enumerate(corner_signs):
        corner_normal_stress = corner_normal_stress.set(
            csdl.slice[:, :, :, corner],
            axial_stress
            + flapwise_sign * flapwise_bending_stress
            + edgewise_sign * edgewise_bending_stress,
        )

    inner_height = outer_height - 2.0 * cap_thickness
    inner_width = outer_width - 2.0 * web_thickness
    web_shear_stress = normal_shear_force / (2.0 * web_thickness * inner_height)
    cap_shear_stress = tangential_shear_force / (2.0 * cap_thickness * inner_width)
    absolute_web_shear = csdl.sqrt(web_shear_stress**2 + 1.0e-24)
    absolute_cap_shear = csdl.sqrt(cap_shear_stress**2 + 1.0e-24)

    tensile_utilization = csdl.maximum(
        corner_normal_stress / material.tensile_allowable,
        axes=(1, 2, 3),
        rho=aggregation_rho,
    )
    compressive_utilization = csdl.maximum(
        -corner_normal_stress / material.compressive_allowable,
        axes=(1, 2, 3),
        rho=aggregation_rho,
    )
    shear_field = csdl.Variable(shape=shape + (2,), value=0.0)
    shear_field = shear_field.set(csdl.slice[:, :, :, 0], absolute_web_shear)
    shear_field = shear_field.set(csdl.slice[:, :, :, 1], absolute_cap_shear)
    shear_utilization = csdl.maximum(
        shear_field / material.shear_allowable,
        axes=(1, 2, 3),
        rho=aggregation_rho,
    )
    maximum_utilization = csdl.maximum(
        tensile_utilization,
        compressive_utilization,
        shear_utilization,
        rho=aggregation_rho,
    )
    blade_mass = csdl.sum(element_mass[:, :, 0], axes=(1,))

    return BoxBeamOutputs(
        section=section,
        blade_mass=blade_mass,
        total_blade_mass=num_blades * blade_mass,
        axial_force=axial_force,
        normal_shear_force=normal_shear_force,
        tangential_shear_force=tangential_shear_force,
        flapwise_bending_moment=flapwise_bending_moment,
        edgewise_bending_moment=edgewise_bending_moment,
        corner_normal_stress=corner_normal_stress,
        web_shear_stress=web_shear_stress,
        cap_shear_stress=cap_shear_stress,
        tensile_utilization=tensile_utilization,
        compressive_utilization=compressive_utilization,
        shear_utilization=shear_utilization,
        maximum_utilization=maximum_utilization,
    )
