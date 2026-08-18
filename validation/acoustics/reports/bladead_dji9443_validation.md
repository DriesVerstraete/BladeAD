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
Lowson and Hanson loading-plus-thickness models. The current run linearly interpolates the seven
FLOWUnsteady low-Reynolds-number section polars in angle of attack and normalized blade span. It
also reconstructs radially varying thickness from the mapped source contours; Hanson uses the
full radial-by-chordwise shapes, while Lowson Barry--Magliozzi uses their radial `t/c` maxima.

BladeAD predicts `C_T = 0.07533` against the measured `C_T = 0.072`, a 4.63% overprediction.
This is therefore a coupled aeroacoustic result rather than isolated acoustic validation.

## Results

| Source case | Model | BPF1 MAE (dB) | BPF2 MAE (dB) | Combined MAE (dB) | Maximum error (dB) | Two-harmonic energetic error (dB) |
|---|---|---:|---:|---:|---:|---:|
| Geometry-driven | Lowson | 1.98 | 7.22 | 4.60 | 14.88 | -1.42 |
| Geometry-driven | Hanson | 8.42 | 13.43 | 10.92 | 22.01 | -7.74 |
| Measured-`C_T` load-scaled | Lowson | 2.28 | 7.34 | 4.81 | 14.94 | -1.75 |
| Measured-`C_T` load-scaled | Hanson | 8.74 | 13.67 | 11.21 | 22.08 | -8.09 |

Neither model passes both frozen criteria. Lowson passes the energetic criterion and BPF1 harmonic
criterion, but BPF2 still collapses toward the +45-degree observer. Hanson remains substantially
low and does not fix the directivity trend.

The experiment identifies motor noise as material. Its contribution at the blade-passing tones,
particularly near a rotor-loading directivity null, is not separately reported in the recovered
data and is therefore an unresolved alternative explanation for part of the measured level.

Adding the real thickness geometry further reduces Lowson BPF1/BPF2 MAE from 3.20/9.73 dB to
1.98/7.22 dB. Thickness is therefore material but does not explain the experimental +45-degree
BPF2 level.

## Figure 14 computational-reference comparison

At BPF2, geometry-driven Lowson agrees with the digitized PAS computational curves over their
overlapping reported angle domains:

| Component | Compared angles | BladeAD minus PAS range (dB) |
|---|---|---:|
| Loading | -45, -22.5, 0, +22.5 degrees | +0.11 to +1.52 |
| Thickness | -22.5, 0, +22.5 degrees | +0.53 to +0.69 |
| Total | -45, -22.5, 0, +22.5 degrees | -0.03 to +0.94 |

PAS does not report loading beyond +28.7 degrees or total beyond +43.1 degrees. Both its total
curve and BladeAD trend down toward the experimental +45-degree point, which remains about 12--15
dB higher. Zawodny et al. likewise report considerable BPF2 discrepancies between experiment and
both deterministic prediction methods, controlled by thickness/loading phase. BladeAD Lowson's
Barry--Magliozzi combination is energetic, whereas Hanson combines complex pressures coherently;
neither matches the experimental endpoint. This establishes that the main BPF2 discrepancy is
shared with the published deterministic models rather than unique to BladeAD.

## Measured-thrust diagnostic

The measured-`C_T` diagnostic multiplies both sectional thrust and drag by
`0.072/0.0753316 = 0.955775`, preserving RPM, geometry, radial loading shape, and observers. Since
measured torque is not reported, equal thrust/drag scaling is an explicit diagnostic assumption,
not a reconstructed experimental source state. It lowers every loading component by
`20 log10(0.955775) = -0.393 dB` while leaving thickness unchanged. The remaining error therefore
cannot be explained by the 4.63% geometry-driven thrust overprediction.
