# F8475 D-4 tonal validation provenance

The directory and RCAIDE symbols retain the legacy `f8745_d4` spelling for compatibility.
Weir and Powers identify the test propeller as **Hartzell F8475 D-4**.

## Intended use

Validate harmonic spectra and directivity for the differentiable BladeAD tonal model.

## Located source material

- RCAIDE SPL fork commit: `c88217f3fd0ef9740e86cfc4241bb4362bb7a766`.
- RCAIDE validation driver:
  `VnV/Verification/analysis_aeroacoustics/frequency_domain_test.py`.
- RCAIDE geometry constructor: `VnV/Vehicles/Rotors/F8745_D4_Propeller.py`.
- The RCAIDE driver attributes its experimental data to D. S. Weir and J. O. Powers,
  *Comparisons of Predicted Propeller Noise with Windtunnel and Flyover Data*, AIAA Paper
  87-0527, 25th AIAA Aerospace Sciences Meeting, Reno, Nevada, January 1987.
- D. S. Weir and J. O. Powers, *Comparisons of Predicted Propeller Noise with Windtunnel and
  Flyover Data*, AIAA Paper 87-0527, January 1987, DOI `10.2514/6.1987-527`. A private local copy
  was inspected on 2026-08-19 and is intentionally excluded from this repository.
- Table 4 reports RPM, tunnel speed, shaft power, thrust, shaft angle, temperature, measured
  three-quarter-radius blade angle, and the blade angle used by the prediction. These values are
  transcribed in `published_operating_conditions.csv`.
- The paper reports direct strain-gauge thrust and torque, a spinner tare correction to thrust,
  and shaft power derived from torque and RPM. Measurement uncertainty and measured sectional
  loading are **not reported**.
- The prediction iterated blade angle to match measured power coefficient; thrust was a comparison
  output, not the matched aerodynamic quantity.
- RCAIDE repository licence: GNU AGPLv3.

## Extraction status

- Geometry: extracted to `geometry.csv`; 30 interpolated stations, including the validation
  driver's shift to 21 degrees twist at its selected three-quarter-radius index.
- Legacy RCAIDE-driver conditions: extracted to `operating_conditions.csv`; three cases. These do
  not reproduce Table 4 exactly and must not be labelled as the published test conditions.
- Published conditions: transcribed from Weir and Powers Table 4 to
  `published_operating_conditions.csv`; three cases.
- Observers: extracted to `observers.csv`; 19 positions at 20 m.
- Experimental harmonic spectra: extracted to `experimental_harmonics.csv`; 108 values covering
  three cases, two reported angles, and 18 harmonics.
- RCAIDE predictions: generated for line-source and plane-source fidelity, with full numeric
  containers and manifests. See `data_dictionary.md` and the baseline-generation report.

RCAIDE extraction and the explicit Table 4 transcription are reproducible through
`scripts/extract_rcaide_fixtures.py`. Fixture dimensions and boundary values are asserted in
`tests/acoustics/test_validation_fixtures.py`. No RCAIDE production code is copied into BladeAD.
