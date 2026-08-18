# BladeAD DJI 9443 tonal validation

## Frozen case

- DJI 9443 two-bladed rotor at 5400 RPM in hover.
- Published chord and twist distributions, 0.12 m tip radius, and 0.00624 m hub radius.
- Corrected current FLOWUnsteady/Zawodny observer convention: five microphones from -45 to
  +45 degrees at 1.905 m, with angle measured from the rotor plane.
- Experimental targets are the digitized BPF1 (180 Hz) and BPF2 (360 Hz) directivity levels.
- The unweighted experimental OASPL is retained in the fixture but is not used as a tonal gate
  because it includes broadband rotor and motor noise.

## Reproducible calculation

`run_bladead_dji9443_validation.py` evaluates the same BladeAD BEM loads and observer grid with
Lowson and Hanson loading-noise models. Thickness is disabled in both calculations to isolate a
like-for-like loading-source comparison. The current run uses the project's generic ZeroD polar;
the seven source-rotor low-Reynolds-number section polars are not yet represented in BladeAD.

BladeAD predicts `C_T = 0.08023` against the measured `C_T = 0.072`, an 11.44% overprediction.
This is therefore a coupled aeroacoustic result rather than isolated acoustic validation.

## Results

| Model | BPF1 MAE (dB) | BPF2 MAE (dB) | Combined MAE (dB) | Maximum error (dB) | Two-harmonic energetic error (dB) |
|---|---:|---:|---:|---:|---:|
| Lowson | 5.85 | 15.02 | 10.43 | 41.70 | -3.51 |
| Hanson | 11.87 | 21.04 | 16.45 | 47.72 | -9.53 |

Neither model passes the frozen 3 dB overall and harmonic-MAE criteria. Lowson is consistently
closer and remains the primary model, but its BPF2 directivity collapses toward the +45-degree
observer. Hanson is approximately 6.02 dB below Lowson at every point, consistent with a fixed
pressure-amplitude convention difference in this loading-only axisymmetric case; it does not fix
the directivity trend.

The experiment identifies motor noise as material. Its contribution at the blade-passing tones,
particularly near a rotor-loading directivity null, is not separately reported in the recovered
data and is therefore an unresolved alternative explanation for part of the measured level.

The next diagnostic is source-model closure: represent or fit the seven published DJI sectional
polars, re-run BEM, and determine how much of the tonal error remains after the measured thrust
coefficient is reproduced. No acceptance threshold should be changed.
