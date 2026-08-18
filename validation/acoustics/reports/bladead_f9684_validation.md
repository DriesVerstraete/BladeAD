# BladeAD Hartzell F-9684-14 tonal validation

The frozen gate uses the first six measured BPF harmonics at the DNW reference
microphone in the propeller plane, 4 m from the axis. BEM supplies radial load shape;
sectional thrust and drag are independently scaled so their integrals reproduce measured
`C_T` and `C_P`. Lowson and Hanson use identical aerodynamic sources and geometry.

| Case | Model | Band | Harmonic MAE | Maximum error | Overall error | Gate |
|---|---|---|---:|---:|---:|---|
| BC-4 | lowson | bpf1_6 | 1.57 dB | 2.94 dB | -0.61 dB | pass |
| BC-4 | lowson | all_available | 1.72 dB | 3.77 dB | -0.62 dB | pass |
| BC-4 | hanson_line | bpf1_6 | 9.31 dB | 11.09 dB | -8.28 dB | fail |
| BC-4 | hanson_line | all_available | 10.38 dB | 13.20 dB | -8.29 dB | fail |
| AC-2 | lowson | bpf1_6 | 1.65 dB | 2.17 dB | -0.26 dB | pass |
| AC-2 | lowson | all_available | 2.32 dB | 4.21 dB | -0.51 dB | pass |
| AC-2 | hanson_line | bpf1_6 | 9.11 dB | 10.30 dB | -8.38 dB | fail |
| AC-2 | hanson_line | all_available | 12.04 dB | 15.85 dB | -8.67 dB | fail |

The frozen gate remains BPF1--6 harmonic MAE <= 3 dB and absolute energetic overall
error <= 3 dB. The all-available rows diagnose higher-harmonic roll-off and do not
retroactively change that gate. Figure-digitization uncertainty and the absence of
measured sectional loads remain explicit source limitations.

Lowson retains the measured higher-harmonic roll-off: its all-available MAE remains
1.72 dB through BC-4 BPF13 and 2.32 dB through AC-2 BPF24. Hanson instead falls
progressively below the data, qualitatively resembling the excessive post-BPF6
roll-off reported for the compact Shahjahan model. The higher harmonics barely alter
the energetic totals because the first few tones dominate them.
