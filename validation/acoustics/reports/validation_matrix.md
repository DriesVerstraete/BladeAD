# Acoustic validation matrix

This matrix was frozen before any BladeAD tonal or broadband model prediction existed.
RCAIDE is a comparison point, not validation truth.

## Acceptance criteria for BladeAD

- Tonal: absolute overall error <= 3 dB and mean per-harmonic absolute error <= 3 dB.
- Broadband: absolute overall error <= 3 dB and mean band error <= 5 dB over bands
  materially above the measurement/background floor.
- Trends: no unexplained systematic observer-angle, RPM, or harmonic/frequency trend.
- Derivatives: relative error <= 1e-5 when well scaled; absolute error <= 1e-7 near zero,
  with convergence over at least three finite-difference step sizes.

These thresholds are not applied to force RCAIDE to pass and must not be relaxed after
seeing BladeAD results. A failed model is narrowed, extended, or rejected rather than tuned
to these fixtures.

## RCAIDE comparison results

| Case | Model | Component | Angle (deg) | N | MAE (dB) | Max (dB) | Overall error (dB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| F8745-D4-1 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 1.310 | 2.332 | -1.332 |
| F8745-D4-1 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 3.253 | 7.217 | -4.958 |
| F8745-D4-2 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 4.369 | 10.436 | -3.343 |
| F8745-D4-2 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 4.516 | 9.312 | -5.550 |
| F8745-D4-3 | rcaide_line_source | tonal_harmonics | 60.0 | 18 | 8.599 | 15.230 | -4.361 |
| F8745-D4-3 | rcaide_line_source | tonal_harmonics | 90.0 | 18 | 7.570 | 9.094 | -7.245 |
| F8745-D4-1 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 6.177 | 7.584 | -4.393 |
| F8745-D4-1 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 13.309 | 14.731 | -12.792 |
| F8745-D4-2 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 7.471 | 15.780 | -3.321 |
| F8745-D4-2 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 14.141 | 16.136 | -13.275 |
| F8745-D4-3 | rcaide_plane_source | tonal_harmonics | 60.0 | 18 | 12.351 | 21.130 | -5.171 |
| F8745-D4-3 | rcaide_plane_source | tonal_harmonics | 90.0 | 18 | 17.263 | 20.646 | -14.789 |
| APC-11x4-3600-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 7.150 | 14.958 | -2.418 |
| APC-11x4-4200-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 6.674 | 17.919 | -2.278 |
| APC-11x4-4800-RPM | rcaide_plane_source | total_one_third_octave | 45.0 | 21 | 6.800 | 13.213 | -2.898 |
| APC-11x4-4200-RPM | rcaide_plane_source_source_driver_angle_mapping | broadband_one_third_octave | 22.5 | 21 | 9.453 | 22.765 | -3.654 |
| APC-11x4-4200-RPM | rcaide_plane_source_source_driver_angle_mapping | broadband_one_third_octave | 45.0 | 21 | 5.296 | 16.370 | 1.262 |

Detailed signed errors are in `rcaide_vs_experiment_detailed.csv`; unrounded summary
metrics are in `rcaide_vs_experiment_summary.csv`.

## Known mapping limitation

For APC broadband data, RCAIDE labels experimental curves as 45 and 22.5 degrees but
compares them to simulated observer indices 4 and 3, whose driver angle parameters are
135 and 112.5 degrees. The rows labelled
`rcaide_plane_source_source_driver_angle_mapping` reproduce that source behaviour and
must not be interpreted as a resolved physical-angle equivalence.

## Physical-validation readiness

| Fixture | Aerodynamic evidence | Acoustic evidence | Current use |
|---|---|---|---|
| F8475 D-4 (`F8745-D4` legacy ID) | Table 4 measured thrust and shaft power; prediction matched power coefficient by adjusting blade angle; RCAIDE-generated sectional loads; measurement uncertainty and measured sectional loading not reported | 18 harmonics at 60° and 90° for three cases | Coupled code/experiment diagnostic; legacy RCAIDE conditions differ from Table 4 and must be corrected before physical validation |
| DJI 9443, 5400 RPM hover | Measured `C_T=0.072`; open chord/twist geometry and seven section polars; BladeAD gives `C_T=0.07533` | Digitized BPF1/BPF2 and OASPL directivity on the corrected 1.905 m observer plane | Coupled loading-noise validation; directivity diagnosis pending |
| APC 11x4 | RCAIDE-generated source state | Total and broadband one-third-octave spectra | Broadband fixture; physical angle mapping remains unresolved |

The F8745 load-sensitivity report shows that ±10% common thrust/torque scaling changes overall
error by at most 0.85 dB, while the tested fixed-integral radial redistributions change it by at
most 0.15 dB. These bounds do not explain the observed tonal underprediction.

## F8475 condition audit

Weir and Powers Table 4 reports 2400/2700/2700 RPM and measured blade angles of
20.8/20.8/19.9 degrees at three-quarter radius. RCAIDE's validation driver instead uses
2390/2710/2630 RPM, forces all three cases to 21 degrees, applies one atmospheric state, and places
observers at 20 m rather than the paper's nominal 4 m in-flow array. Existing results above remain
valid only as reproduction of that legacy driver, not as a like-for-like paper validation.

The archived legacy line-source state gives shaft powers of 72.323, 148.424, and 135.773 kW and
thrusts of 823.186, 1749.349, and 1583.665 N. Relative to Table 4, the power errors are -1.73%,
-19.60%, and -10.73%, while thrust errors are +28.22%, -8.27%, and +5.58%. Case 1's near power
match therefore does not imply an aerodynamic match: its thrust is 28% high. A corrected physical
comparison requires solving each published case at its own RPM, temperature, and blade angle,
matching measured shaft power before recomputing the acoustic prediction.

## BladeAD F8475 geometry-to-noise result

The RCAIDE-independent BladeAD path uses the F8475 geometry, a Clark-Y ZeroD polar fitted to the
available Re=1,000,000 XFOIL table, and each published operating condition. Matching Table 4 shaft
power gives physical r/R=0.75 blade angles of 21.240, 21.458, and 20.580 degrees, 0.54--1.06
degrees below the paper's computed angles. Predicted thrust is high
by 23.21%, 7.32%, and 12.19%.

At the paper's nominal 4 m microphone distance, Hanson overall errors are -2.10 to -2.75 dB at
60 degrees and -8.20 to -9.03 dB at 90 degrees. Harmonic MAE is 4.62--7.68 dB at 60 degrees and
7.74--9.16 dB at 90 degrees. This is a substantial improvement over the legacy 20 m/frozen-load
comparison, but it does not pass the frozen harmonic gate and retains a strong observer-angle
trend. Pressure was not reported and is assumed to be 101325 Pa; Reynolds-number variation in the
Clark-Y polar is not represented.

The observer audit confirms that the paper's 30-degree out-of-plane and 0-degree in-plane
microphones map to polar directivity angles 60 and 90 degrees. The 90-degree result is invariant to
upstream/downstream mirroring; its deficit is dominated by the in-plane source formulation, where
Hanson's axial-loading term vanishes and thickness is the largest isolated component. The legacy
RCAIDE driver's 20 m observer radius and +15 dB adjustment nearly cancel because 20-to-4 m
spherical spreading is +13.98 dB.

The thickness/convention audit finds maximum within-case thickness-level changes of 0.0473 dB
over 25--200 chordwise stations and 1.3578 dB over 15--61 radial stations with power re-matched.
Peak rather than RMS reporting adds exactly 3.010 dB but leaves the 90-degree overall errors at
-6.02 to -5.19 dB. Matching measured in-plane overall level through thickness pressure alone would
require nonphysical diagnostic multipliers of 2.90--3.31 (+9.26 to +10.41 dB). Numerical
quadrature and peak/RMS convention therefore do not explain the deficit.

## Corrected-condition model comparison

The published-condition, 4 m, power-matched comparison has now been run for BladeAD Lowson,
BladeAD Hanson, and an independently power-matched RCAIDE plane-source calculation with its legacy
+15 dB adjustment disabled. BladeAD Lowson is the best-balanced result: all six overall errors are
within +0.31 to +1.99 dB, and four of six harmonic MAEs pass the 3 dB gate; the two misses are
3.147 and 3.166 dB for Case 1. Hanson passes overall level at 60 degrees but is 8--9 dB low at
90 degrees. RCAIDE plane-source passes both gates at 90 degrees but is 9.9--11.7 dB high overall
at 60 degrees. Neither Hanson nor RCAIDE should be treated as validation truth.

After matching measured shaft power, RCAIDE predicts thrust high by 30.64%, 12.82%, and 19.26%;
BladeAD predicts it high by 23.21%, 7.32%, and 12.19%. Acoustic agreement therefore remains a
coupled aeroacoustic result rather than isolated validation of acoustic radiation alone.

## DJI 9443 corrected-observer model comparison

At 5400 RPM, BladeAD with the seven FLOWUnsteady section polars predicts `C_T=0.07533`, 4.63%
above the measured `C_T=0.072`. Against five digitized observer angles and two blade-passing
harmonics, Lowson gives a 6.46 dB combined harmonic MAE and -2.10 dB energetic error. Hanson gives
12.48 dB and -8.12 dB, respectively. Lowson therefore passes the energetic gate but not the
harmonic gate; BPF1 is close at 3.20 dB MAE, while BPF2 remains the dominant miss at 9.73 dB.
These loading-only results use the same BEM source; thickness is disabled and measured OASPL is
excluded because it includes broadband and motor noise. The remaining work is a directivity and
source-contamination diagnosis rather than aerodynamic-polar closure.

Scaling both sectional thrust and drag to match measured `C_T` exactly changes every loading-tone
prediction by -0.393 dB. Lowson's combined MAE becomes 6.86 dB and its energetic error -2.50 dB;
Hanson becomes 12.88 dB and -8.52 dB. The equal thrust/drag scale is only a diagnostic because
measured torque is not reported. This bound rules out the residual 4.63% thrust mismatch as the
cause of the BPF2/directivity failure.
