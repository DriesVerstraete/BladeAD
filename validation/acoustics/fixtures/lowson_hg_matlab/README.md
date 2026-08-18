# Lowson HG-MATLAB hover directivity case

This fixture preserves the comparison data embedded in
`lsdo_acoustics/core/models/tonal/Lowson/tests/test_Lowson_hover_verif.py` at pinned commit
`7c76e0d01a71d59582d9ec3d62493dd7d37bdd69`.

## Scope warning

The 37-point `hg_matlab_total_spl.csv` curve is **not** a loading-only Lowson reference. The
source test sets `toggle_thickness_noise=True`, combines its hover Sears loading branch with the
Barry–Magliozzi thickness model, and labels the plot “Lowson + thickness noise”. It therefore
must not be used as an acceptance gate for BladeAD's current loading-only implementation.

The source case uses three blades, 1500 rpm, radius 0.3556 m, density 1.225 kg/m3, speed of sound
340.3 m/s, observer radius 1.5 m, 40 radial stations from `r/R=0.21` to `0.99`, and the first
blade-passing mode. The source maps its observer and plotted angle arrays inconsistently; that
mapping must be resolved when the combined model is reproduced.

`experimental_spl.csv` preserves the experimental points plotted by the same source file. The
source does not report measurement uncertainty, spectral bandwidth, microphone calibration, or
a primary citation for these transcribed values.

## Acceptance status

- Loading-only equation (10): independently verified in `tests/acoustics/test_loading.py`.
- HG combined curve: fixture frozen, **not yet an active acceptance test**.
- Required before activation: Barry–Magliozzi thickness pressure, Sears hover-loading convention,
  exact observer-angle mapping, and separate component output comparisons.
