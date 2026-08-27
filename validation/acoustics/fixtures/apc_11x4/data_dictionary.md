# APC 11x4 data dictionary

## Coordinate and index conventions

- Geometry uses SI units and follows the RCAIDE rotor's local radial definition.
- Observer Cartesian coordinates are relative to the rotor origin in the convention constructed
  by RCAIDE's validation driver; `angle_deg` retains the driver's angle parameter.
- Case index 1--3 maps in order to 3600, 4200, and 4800 RPM.
- Baseline acoustic arrays use axes `(case, observer)` or `(case, observer, spectral_bin)`.
- Baseline disk-load arrays use axes `(case, radial_station, azimuth_station)`.

## CSV files

- `geometry.csv`: station index, dimensional/nondimensional radius, chord, twist, and thickness
  ratio.
- `operating_conditions.csv`: RPM, inflow ratio, axial speed, atmosphere, angle of attack, and
  azimuthal resolution.
- `observers.csv`: driver angle and Cartesian microphone position in metres.
- `experimental_total_spectrum.csv`: measured total unweighted SPL by RPM and one-third-octave
  centre frequency.
- `experimental_broadband_spectrum.csv`: measured broadband unweighted SPL at 4200 RPM by reported
  observer angle and one-third-octave centre frequency.

The experimental broadband angles are measured from the rotor plane on the downstream side.
RCAIDE's semicircle driver parameter is measured from the upstream rotor axis. Consequently,
observer index 3 (`112.5°`) is `22.5°` downstream from the rotor plane and observer index 4
(`135°`) is `45°` downstream from the rotor plane. The Cartesian coordinates independently give
the same angles through `atan2(x, |y|)`. The source's `60deg`/`90deg` verification-variable names
are stale and contradict its plotted `22.5°`/`45°` labels and coordinates.

## RCAIDE baseline files

- `rcaide_plane_source_baseline.npz`: untouched RCAIDE plane-source tonal plus broadband result.
- The matching `.manifest.json` lists every key, shape, dtype, source commit, driver hash,
  fidelity, and compatibility intervention.

NPZ keys retain their RCAIDE hierarchy with dots, for example
`acoustics.converters.APC_11x4_Propeller.SPL_broadband_1_3_spectrum`. The archive includes all
serializable numeric/string fields reached from aeroacoustics, aerodynamics, energy, freestream,
frames, rotor, and settings containers. Important families include total/harmonic/broadband SPL
and dBA spectra, one-third-octave frequencies, sectional and disk loads, induced velocities,
coefficients, RPM, thrust, torque, geometry, atmosphere, frames, observers, model settings, and
runtime/source metadata.

RCAIDE stores 29 one-third-octave bins; the experimental files contain 21 bands from 100 to
10,000 Hz. Comparisons must match centre frequencies explicitly.
