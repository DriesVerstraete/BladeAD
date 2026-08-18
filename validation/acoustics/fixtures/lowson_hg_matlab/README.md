# Lowson HG-MATLAB hover directivity case

This fixture preserves the comparison data embedded in
`lsdo_acoustics/core/models/tonal/Lowson/tests/test_Lowson_hover_verif.py` at pinned commit
`7c76e0d01a71d59582d9ec3d62493dd7d37bdd69`.

## Scope warning

The 37-point `hg_matlab_total_spl.csv` curve is **not** a loading-only Lowson reference. The
source test sets `toggle_thickness_noise=True`, combines its hover Sears loading branch with the
Barry–Magliozzi thickness model, and labels the plot “Lowson + thickness noise”. It therefore
must not be used as an acceptance gate for BladeAD's current loading-only implementation.

`radial_inputs.csv` preserves every radial input array present in the pinned test: nondimensional
radius, adjusted chord, thrust and drag coefficient distributions, inflow ratio, and lift
coefficient. The pinned acoustic call derives its supplied steady thrust and drag distributions
from chord, lift coefficient, and inflow angle; `dCT_dr` and `dCD_dr` are retained as independent
aerodynamic provenance even though that call does not consume them.

The source case uses three blades, 1500 rpm, radius 0.3556 m, density 1.225 kg/m3, speed of sound
340.3 m/s, observer radius 1.5 m, 40 radial stations from `r/R=0.21` to `0.99`, and the first
blade-passing mode. The source maps its observer and plotted angle arrays inconsistently; that
mapping must be resolved when the combined model is reproduced.

`experimental_spl.csv` preserves the experimental points plotted by the same source file. The
source does not report measurement uncertainty, spectral bandwidth, microphone calibration, or
a primary citation for these transcribed values.

`bm_thickness_reference.csv` preserves the separate 33-point HJ Barry–Magliozzi thickness-only
reference used by the pinned `test_BM_thickness.py`: four blades, 5500 rpm, radius 0.1588 m,
constant 0.03176 m chord, thickness ratio 0.12, observer radius 2.2703 m, and angles 10–170
degrees from the rotor axis.

The audited BladeAD port is approximately 0.64 dB above this HJ curve at every angle. The pinned
source test also treats the curve as a plotted comparison rather than an exact assertion. The
frozen acceptance criteria are therefore less than 1 dB maximum absolute error and less than
0.01 dB peak-to-peak variation after removing the common level offset; no calibration factor is
applied.

## Acceptance status

- Loading-only equation (10): independently verified in `tests/acoustics/test_loading.py`.
- HG combined curve: active uncalibrated characterisation test in
  `tests/acoustics/test_hg_combined.py`. With harmonics 0–10, the BladeAD result has a -0.384 dB
  mean offset, 2.059 dB maximum absolute error, and 1.675 dB maximum residual after removing the
  common offset relative to the frozen HG-MATLAB curve.
- The source observer order is retained and compared with the reversed HG array, exactly as in the
  pinned source test. Separate loading and thickness outputs remain available for diagnosis.
