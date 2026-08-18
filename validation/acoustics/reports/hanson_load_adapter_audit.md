# Hanson load-adapter audit

**Date:** 2026-08-18

## BladeAD production convention

The adapter converts BladeAD complete-rotor elemental forces to:

- per-blade axial and circumferential line loading in N/m;
- normalized complex coefficients `F_k = mean(F(psi) exp(-i k psi))`; and
- nondimensional radial weights `w_r dr/R_tip`.

For the steady harmonic, integrating the returned line loading with the dimensional BladeAD
quadrature exactly recovers the original per-blade force. Synthetic first-cosine and second-sine
loads recover the expected half-amplitude positive-frequency complex coefficients.

## RCAIDE line-source convention found during parity tracing

The audited RCAIDE routine applies `rfft` directly to
`disc_thrust_distribution / R_tip`. In the F8745 fixture,
`disc_thrust_distribution` is an elemental force containing dimensional `dr`, not `dT/dr`.
The routine then trapezoidally integrates that result over nondimensional radius.

The `rfft` is unnormalized. For the 16-point azimuth grid and uniform interior spacing in the
F8745 fixture, the RCAIDE steady line-load input relative to the physical `dT/dr` coefficient is:

```text
N_azimuth * dr / R_tip = 16 * 0.0273 / 1.015 = 0.4303448276
```

At the first and last stations RCAIDE's half-width element gives half this ratio. Thus its
unnormalized azimuth transform and second radial integration partly cancel; copying either factor
alone would not reproduce RCAIDE and copying both would preserve a dimensionally inconsistent
adapter.

This audit records a code-to-code discrepancy, not a correction to RCAIDE and not an experimental
accuracy conclusion. BladeAD uses the explicit physical convention above. Any RCAIDE parity driver
must label the legacy scaling and keep it outside the production CSDL graph.

## Verification

`tests/acoustics/test_hanson_loads.py` checks:

- normalized complex Fourier coefficients;
- radial reconstruction of steady per-blade force;
- direct adapter derivatives; and
- derivatives through the complete adapter-to-Hanson-pressure chain.
