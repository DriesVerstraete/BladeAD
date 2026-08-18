# F8475 corrected-condition three-model comparison

All models use the published 2400/2700/2700 RPM cases, reported temperatures,
power-matched blade angle at physical r/R=0.75, nominal 4 m observers, and the
published 60/90-degree polar angles. BladeAD Lowson and Hanson share BladeAD BEM
loads. RCAIDE independently power-matches its own BEM and uses plane-source fidelity
with its legacy +15 dB adjustment disabled.

| Model | Case | Angle | Harmonic MAE | Maximum error | Overall error |
|---|---:|---:|---:|---:|---:|
| bladead_hanson | 1 | 60° | 4.623 dB | 5.828 dB | -2.104 dB |
| bladead_hanson | 1 | 90° | 7.740 dB | 9.374 dB | -8.200 dB |
| bladead_hanson | 2 | 60° | 7.174 dB | 16.301 dB | -2.060 dB |
| bladead_hanson | 2 | 90° | 9.162 dB | 10.570 dB | -9.034 dB |
| bladead_hanson | 3 | 60° | 7.679 dB | 14.211 dB | -2.753 dB |
| bladead_hanson | 3 | 90° | 8.366 dB | 10.054 dB | -8.426 dB |
| bladead_lowson | 1 | 60° | 3.147 dB | 8.888 dB | +1.994 dB |
| bladead_lowson | 1 | 90° | 3.166 dB | 7.955 dB | +0.307 dB |
| bladead_lowson | 2 | 60° | 2.515 dB | 6.025 dB | +1.461 dB |
| bladead_lowson | 2 | 90° | 2.274 dB | 7.363 dB | +0.636 dB |
| bladead_lowson | 3 | 60° | 2.462 dB | 5.661 dB | +0.932 dB |
| bladead_lowson | 3 | 90° | 2.816 dB | 8.030 dB | +1.207 dB |
| rcaide_plane_source | 1 | 60° | 9.094 dB | 11.788 dB | +9.871 dB |
| rcaide_plane_source | 1 | 90° | 1.591 dB | 3.589 dB | +1.333 dB |
| rcaide_plane_source | 2 | 60° | 6.423 dB | 13.867 dB | +11.452 dB |
| rcaide_plane_source | 2 | 90° | 1.842 dB | 4.180 dB | +1.039 dB |
| rcaide_plane_source | 3 | 60° | 7.644 dB | 14.417 dB | +11.734 dB |
| rcaide_plane_source | 3 | 90° | 1.265 dB | 5.029 dB | +1.857 dB |

The frozen acceptance gate is harmonic MAE <=3 dB and absolute overall error
<=3 dB. Passing this gate does not remove the aerodynamic-source caveat: both
BladeAD and RCAIDE overpredict measured thrust after matching measured power.

## Interpretation

- BladeAD Lowson passes the overall gate for all six comparisons and the harmonic
  MAE gate for four of six; the two misses are Case 1 at 60 and 90 degrees
  (3.147 and 3.166 dB).
- BladeAD Hanson passes the overall gate at 60 degrees but fails harmonic MAE in
  every comparison and underpredicts all three 90-degree overall levels by 8--9 dB.
- RCAIDE plane-source passes both gates at 90 degrees, but overpredicts 60-degree
  overall level by 9.9--11.7 dB and fails harmonic MAE there.

Lowson is therefore the strongest current absolute model across both observer
angles. The opposing Hanson and RCAIDE directivity biases rule out treating either
as validation truth.
