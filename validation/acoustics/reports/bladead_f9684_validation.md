# BladeAD Hartzell F-9684-14 tonal validation

The frozen comparison uses the first six measured BPF harmonics at the DNW reference
microphone in the propeller plane, 4 m from the axis. BEM supplies radial load shape;
sectional thrust and drag are independently scaled so their integrals reproduce measured
`C_T` and `C_P`. Lowson and Hanson use identical aerodynamic sources and geometry.

| Case | Model | Harmonic MAE | Maximum error | Overall error | Gate |
|---|---|---:|---:|---:|---|
| BC-4 | lowson | 2.41 dB | 2.82 dB | +1.94 dB | pass |
| BC-4 | hanson_line | 5.43 dB | 5.98 dB | -5.73 dB | fail |
| AC-2 | lowson | 3.12 dB | 4.10 dB | +3.64 dB | fail |
| AC-2 | hanson_line | 5.00 dB | 5.80 dB | -4.48 dB | fail |

The frozen gate remains harmonic MAE <= 3 dB and absolute energetic overall error
<= 3 dB. Figure-digitization uncertainty and the absence of measured sectional loads
remain explicit source limitations; they are not reasons to relax the gate.
