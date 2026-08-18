# BladeAD BEM conventions for acoustics

**Audit date:** 2026-08-18

**Audited branch:** `spl-develop`

**Purpose:** freeze the aerodynamic-to-acoustic interface before implementing Lowson noise.

## Array axes

The BEM working shape is `(num_nodes, num_radial, num_azimuthal)`. Radius is the second axis and
azimuth is the third. Azimuth samples are uniformly placed from zero through
`2*pi - 2*pi/num_azimuthal`; blade lag, if used, shifts these values.

The radial coordinate is stored internally at cell-centre-like locations generated from
`0.5/num_radial` through `1 - 0.5/num_radial`, mapped between hub and tip. This radius vector,
the azimuth array, and the element width are not currently exposed on `RotorAnalysisOutputs`.

## Sectional load meaning

`sectional_thrust` is BladeAD's blade-element `dT2`:

`B * Cx * 0.5 * rho * V_rel^2 * chord * dr`

`sectional_torque` is the corresponding complete-rotor element torque and `sectional_drag` is
`sectional_torque / radius`. The factor `B` is already included. Therefore these fields represent
complete-rotor sectional forces at each radius/azimuth sample, not one-blade loads. A per-blade
Lowson source must divide them by `num_blades` exactly once.

The azimuth axis is retained in sectional outputs. Total thrust/torque use `integrate_quantity`,
which radially integrates and averages uniformly over azimuth; it does not sum azimuth samples as
simultaneous blades.

## Total versus sectional formulation

Total thrust and torque are integrated from the momentum expressions `dT` and `dQ`, while exposed
sectional thrust/torque use the nominally equivalent blade-element expressions `dT2` and `dQ2`.
Acoustic consistency tests must compare the integrated exposed sectional loads against totals and
quantify any numerical mismatch rather than assume exact identity.

## Radial-spacing limitation

`preprocess_input_variables` computes hub radius using `norm_hub_radius`, but computes
`dr = (radius - 0.2 * radius) / (num_radial - 1)`. The `0.2` is hardcoded. Any mesh using a
non-default hub ratio is internally inconsistent. Acoustic development must not silently inherit
or conceal this issue; either correct it with aerodynamic regression tests or restrict the first
validated acoustic interface to `norm_hub_radius == 0.2`.

## Force directions and frames

Sectional thrust is axial in the rotor-local thrust-axis direction. `sectional_drag` is the
tangential in-plane force magnitude inferred from torque. BladeAD decomposes mesh velocity relative
to `thrust_vector`; it does not provide a vehicle-frame acoustic convention. Observer transforms
must therefore use the explicit source origin and thrust axis already established by the acoustic
observer interface.

## Required interface changes before Lowson coupling

- Expose the in-graph dimensional radius stations, element width, and azimuth angles.
- Document on `RotorAnalysisOutputs` that sectional forces include all blades.
- Add a regression test integrating sectional thrust/torque and comparing them with totals.
- Decide the hardcoded hub-ratio issue before allowing non-default acoustic meshes.
- Use per-blade loads for Lowson by dividing complete-rotor sectional loads by blade count.

These are interface requirements, not permission to change aerodynamic behaviour without separate
regression evidence.
