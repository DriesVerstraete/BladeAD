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
like-for-like loading-source comparison. The current run linearly interpolates the seven
FLOWUnsteady low-Reynolds-number section polars in angle of attack and normalized blade span,
matching FLOWUnsteady's executable hub-to-tip mapping convention.

BladeAD predicts `C_T = 0.07533` against the measured `C_T = 0.072`, a 4.63% overprediction.
This is therefore a coupled aeroacoustic result rather than isolated acoustic validation.

## Results

| Model | BPF1 MAE (dB) | BPF2 MAE (dB) | Combined MAE (dB) | Maximum error (dB) | Two-harmonic energetic error (dB) |
|---|---:|---:|---:|---:|---:|
| Lowson | 3.20 | 9.73 | 6.46 | 22.77 | -2.10 |
| Hanson | 9.22 | 15.75 | 12.48 | 28.79 | -8.12 |

Neither model passes both frozen criteria. Lowson passes the energetic criterion at -2.10 dB and
nearly passes BPF1 at 3.20 dB MAE, but its BPF2 directivity still collapses toward the +45-degree
observer. Hanson is approximately 6.02 dB below Lowson at every point and does not fix the
directivity trend.

The experiment identifies motor noise as material. Its contribution at the blade-passing tones,
particularly near a rotor-loading directivity null, is not separately reported in the recovered
data and is therefore an unresolved alternative explanation for part of the measured level.

Using the real section polars reduced the thrust-coefficient error from 11.44% to 4.63%, Lowson's
combined MAE from 10.43 to 6.46 dB, and its energetic error from -3.51 to -2.10 dB. The next
diagnostic is therefore the remaining radiation/directivity formulation and possible motor-tone
contamination, not further tuning of the frozen acceptance threshold.
