# Hartzell F-9684-14 fixture data dictionary

- `geometry.csv`: square-tip F-9684-14 radial geometry. Values at `r/R = 0.6`, `0.75`, and
  `1.0` use the tabulated/labelled values in Delfs et al.; inboard points are digitized from
  their Figure 6 and carry plot-resolution uncertainty.
- `operating_conditions.csv`: measured nondimensional conditions from Delfs et al. Table 1.
- `observers.csv`: the unambiguous reference microphone in the propeller plane at 4 m.
- `experimental_harmonics.csv`: measured blade-passing harmonics digitized from the uncalibrated
  square-tip panel of Delfs et al. Figure 7. Its abscissa is `nB`; with two blades, plotted
  positions 2, 4, ... map to BPF harmonics 1, 2, .... AC-2 remains above the 60 dB plot floor
  through BPF24; BC-4 remains above it through BPF13.

All SPL values are unweighted dB re 20 micropascals. The fixture is intended for a tonal
loading-plus-thickness comparison, not a broadband or directivity validation.
