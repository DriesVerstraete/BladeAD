# APC 11x4 broadband validation provenance

## Intended use

Validate total and broadband one-third-octave spectra for the differentiable BladeAD broadband
model.

## Located source material

- RCAIDE SPL fork commit: `c88217f3fd0ef9740e86cfc4241bb4362bb7a766`.
- RCAIDE validation driver:
  `VnV/Verification/analysis_aeroacoustics/frequency_domain_test.py`.
- RCAIDE geometry constructor: `VnV/Vehicles/Rotors/APC_11x4_Propeller.py`.
- The RCAIDE driver contains digitised experimental spectra but does not report a complete
  bibliographic citation beside the arrays; the primary measurement source must be resolved
  before the experimental CSV is accepted.
- RCAIDE repository licence: GNU AGPLv3.

## Extraction status

- Geometry: extracted to `geometry.csv`; 18 stations.
- Operating conditions: extracted to `operating_conditions.csv`; three RPM cases at inflow ratio
  0.08.
- Observers: extracted to `observers.csv`; five positions at 1.905 m.
- Experimental total spectrum: extracted to `experimental_total_spectrum.csv`; 63 values covering
  three RPM cases and 21 one-third-octave bands.
- Experimental broadband spectrum: extracted to `experimental_broadband_spectrum.csv`; 42 values
  covering two reported angles and 21 bands at 4200 RPM.
- RCAIDE prediction: generated for plane-source fidelity, including tonal and broadband outputs,
  full numeric containers, and a manifest. See `data_dictionary.md` and the baseline-generation
  report.

The RCAIDE plots label the two experimental broadband spectra as 45 and 22.5 degrees from the
rotor plane. The corresponding downstream Cartesian observers are indices 4 and 3: their driver
parameters are 135 and 112.5 degrees, respectively. The geometry independently resolves the
experimental angles through `atan2(x, |y|)`; see `reports/apc_observer_mapping_audit.md`.

Extraction is reproducible through `scripts/extract_rcaide_fixtures.py`. Fixture dimensions and
boundary values are asserted in `tests/acoustics/test_validation_fixtures.py`. No RCAIDE
production code is copied into BladeAD.
