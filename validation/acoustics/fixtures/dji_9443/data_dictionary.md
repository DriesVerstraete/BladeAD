# DJI 9443 fixture data dictionary

- `operating_conditions.csv`: frozen 5400 RPM hover condition and measured thrust coefficient.
- `chord_distribution.csv`: raw nondimensional chord stations, `c/R` versus `r/R`.
- `twist_distribution.csv`: raw blade twist in degrees versus `r/R`.
- `airfoil_sections.csv`: FLOWUnsteady normalized hub-to-tip span mapping from blade stations to
  contour and polar files, with the filename-encoded Reynolds number made explicit. The upstream
  header says `r/R`, but its implementation maps position `p` to `(Rhub + p*(Rtip-Rhub))/Rtip`.
- `airfoil_polars/`: seven unchanged FLOWUnsteady section tables containing angle of attack,
  lift coefficient, drag coefficient, and pitching-moment coefficient.
- `airfoil_contours/`: six unchanged FLOWUnsteady section contours used to reconstruct the
  radially varying thickness-to-chord and normalized chordwise thickness shapes.
- `LICENSE_FLOWUNSTEADY.txt`: upstream MIT licence and copyright notice covering the imported
  FLOWUnsteady data.
- `observers.csv`: nominal and digitized microphone angles measured from the rotor plane.
- `experimental_harmonics.csv`: digitized experimental BPF1 and BPF2 directivity levels.
- `experimental_narrowband_spectrum.csv`: digitized experimental spectrum at -45 degrees.
- `experimental_one_third_octave_spectrum.csv`: digitized experimental one-third-octave spectrum
  at -45 degrees.
- `experimental_oaspl.csv`: digitized experimental unweighted OASPL directivity. This includes
  rotor broadband and motor noise and is not a like-for-like tonal-model acceptance target.
- `flowunsteady_reference_bpf2_*.csv`: digitized OF2-PSW and PAS thickness, loading, and total
  predictions from Figure 14. These are source-labelled computational references, not experiment.
