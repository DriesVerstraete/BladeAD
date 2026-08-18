# F8475 D-4 data dictionary

`f8745_d4` is the legacy RCAIDE identifier. The paper's designation is F8475 D-4.

## Coordinate and index conventions

- Geometry uses SI units and follows the RCAIDE rotor's local radial definition.
- Observer Cartesian coordinates are relative to the rotor origin in the convention constructed
  by RCAIDE's validation driver; `angle_deg` retains the driver's angle parameter.
- In legacy-driver files, case index 1--3 maps to 2390, 2710, and 2630 RPM.
- In the published Table 4 file, case index 1--3 maps to 2400, 2700, and 2700 RPM.
- Baseline acoustic arrays use axes `(case, observer)` or `(case, observer, spectral_bin)`.
- Baseline disk-load arrays use axes `(case, radial_station, azimuth_station)`.

## CSV files

- `geometry.csv`: station index, dimensional/nondimensional radius, chord, twist, and thickness
  ratio. `twist_deg` includes the validation driver's shift to 21 degrees at its selected
  three-quarter-radius station.
- `operating_conditions.csv`: conditions hard-coded by RCAIDE's legacy validation driver: RPM,
  axial speed, uniform atmospheric state, angle of attack, 21-degree blade-angle target, and
  azimuthal resolution. This file is retained for exact baseline reproducibility.
- `published_operating_conditions.csv`: Weir and Powers Table 4 values: RPM, tunnel speed, shaft
  power, measured thrust, shaft angle, temperature, measured blade angle, and computed blade
  angle. The paper's "pitch angle at 3/4 radius" is labelled here as blade angle to avoid confusing
  it with vehicle attitude.
- `observers.csv`: driver angle and Cartesian microphone position in metres.
- `experimental_harmonics.csv`: measured unweighted SPL in dB by case, reported observer angle,
  and blade-passing harmonic 1--18.

## RCAIDE baseline files

- `rcaide_line_source_baseline.npz`: untouched RCAIDE line-source prediction.
- `rcaide_plane_source_baseline.npz`: untouched RCAIDE plane-source prediction.
- Matching `.manifest.json` files list every stored key, shape, dtype, source commit, driver hash,
  fidelity, and compatibility intervention.

NPZ keys retain their RCAIDE hierarchy with dots, for example
`acoustics.converters.F8745_D4_Propeller.SPL_harmonic_bpf_spectrum`. The archive includes all
serializable numeric/string fields reached from aeroacoustics, aerodynamics, energy, freestream,
frames, rotor, and settings containers—not only the spectra currently used by tests. Important
families include total/harmonic/broadband SPL and dBA spectra, frequency bands, sectional and disk
loads, induced velocities, coefficients, RPM, thrust, torque, geometry, atmosphere, frames,
observers, propagation settings, and runtime/source metadata.

The 29-bin RCAIDE spectra exceed the 18 experimental harmonics. Comparisons must select bins by
their meaning/frequency, not truncate solely by array shape without checking the source convention.

## BladeAD load-adapter convention

The BladeAD comparison uses the signed `disc_thrust_distribution` and
`disc_torque_distribution`. These are per-blade, radially integrated distributions repeated over
the azimuth axis. They contain legitimate negative root-region elements but sum exactly to
`thrust_per_blade` and `torque_per_blade`; applying an elementwise absolute value is incorrect.
BladeAD's complete-rotor sectional convention is obtained by multiplying each signed distribution
by `number_of_blades` exactly once. Radial integration weights are then unity because the archived
distributions already contain trapezoidal element widths and endpoint weights.

The baseline acoustic propagation uses the archived positive-x inertial velocity. Stationary and
reversed-velocity variants are retained only as convention sensitivity checks, not alternative
accepted baselines.
