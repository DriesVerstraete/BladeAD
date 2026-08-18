# F8745-D4 tonal validation provenance

## Intended use

Validate harmonic spectra and directivity for the differentiable BladeAD tonal model.

## Located source material

- RCAIDE SPL fork commit: `c88217f3fd0ef9740e86cfc4241bb4362bb7a766`.
- RCAIDE validation driver:
  `VnV/Verification/analysis_aeroacoustics/frequency_domain_test.py`.
- RCAIDE geometry constructor: `VnV/Vehicles/Rotors/F8745_D4_Propeller.py`.
- The RCAIDE driver attributes its experimental data to Weir and Powers, *Comparisons of
  predicted propeller noise with windtunnel ...*; complete bibliographic details are not reported
  in that file and must be resolved before the experimental CSV is accepted.
- RCAIDE repository licence: GNU AGPLv3.

## Extraction status

- Geometry: extracted to `geometry.csv`; 30 interpolated stations, including the validation
  driver's shift to 21 degrees twist at its selected three-quarter-radius index.
- Operating conditions: extracted to `operating_conditions.csv`; three cases.
- Observers: extracted to `observers.csv`; 19 positions at 20 m.
- Experimental harmonic spectra: extracted to `experimental_harmonics.csv`; 108 values covering
  three cases, two reported angles, and 18 harmonics.
- RCAIDE predictions: generated for line-source and plane-source fidelity, with full numeric
  containers and manifests. See `data_dictionary.md` and the baseline-generation report.

Extraction is reproducible through `scripts/extract_rcaide_fixtures.py`. Fixture dimensions and
boundary values are asserted in `tests/acoustics/test_validation_fixtures.py`. No RCAIDE
production code is copied into BladeAD.
