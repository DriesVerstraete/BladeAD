# F8475 D-4 observer/directivity audit

## Finding

The 90-degree deficit is not caused by mapping the paper's microphone labels to BladeAD polar
angles.

- Weir and Powers Figure 2 explicitly identifies the out-of-plane microphone as 30 degrees from
  the propeller disk plane and polar directivity angle theta = 60 degrees.
- The same figure identifies the in-plane microphone as 0 degrees from the disk plane and theta =
  90 degrees.
- The BladeAD validation positions are `(-4 cos(theta), 4 sin(theta), 0)` relative to a positive-X
  rotor axis. BladeAD computes `arccos(axial_distance / distance)`, giving internal angles of 120
  and 90 degrees. The 120-degree value represents the paper's upstream 60-degree ray; its sine and
  axial cosine have the expected signs.
- Reversing the rotor axis or mirroring the upstream ray leaves the in-plane angle exactly 90
  degrees. It therefore cannot correct the 90-degree result.

## Source decomposition

| Case | Angle | Loading overall | Thickness overall | Combined overall |
|---:|---:|---:|---:|---:|
| 1 | 60 deg | 103.928 dB | 95.309 dB | 104.328 dB |
| 1 | 90 deg | 101.033 dB | 103.731 dB | 105.228 dB |
| 2 | 60 deg | 113.967 dB | 102.188 dB | 114.099 dB |
| 2 | 90 deg | 110.625 dB | 113.234 dB | 114.580 dB |
| 3 | 60 deg | 112.574 dB | 102.471 dB | 112.801 dB |
| 3 | 90 deg | 109.115 dB | 113.664 dB | 114.447 dB |

At exactly 90 degrees, Hanson's axial-loading factor is proportional to `cos(theta)` and vanishes.
The remaining loading contribution is circumferential, while thickness is the largest isolated
component in all three cases. The 90-degree combined underprediction is therefore primarily an
in-plane source-formulation problem, not an observer-coordinate problem.

The paper reports the in-flow array only as nominally 4 m from the propeller and does not provide
exact Cartesian coordinates for these two microphones. Closing an 8--9 dB deficit by spherical
spreading alone would require a distance of roughly 1.4--1.6 m instead of 4 m, inconsistent with
the paper's statement that the microphones were approximately two propeller diameters away.

## RCAIDE-driver interaction

The legacy RCAIDE driver places microphones at 20 m and applies a uniform +15 dB wing-wake
adjustment. Moving from 20 m to 4 m changes spherical spreading by +13.98 dB, so those two legacy
choices nearly cancel numerically. Their apparent agreement must not be used as evidence that
either choice represents the experiment.

## Conclusion

Keep the published 60/90-degree mapping and 4 m nominal distance. The next model audit should
target missing source content or model-form differences between the Hanson line-source
approximation and the paper's full-surface Farassat implementation. The follow-on thickness audit
found converged chordwise quadrature, modest radial sensitivity, and only the expected +3.010 dB
peak/RMS shift. Do not tune observer coordinates or pressure conventions to reduce the error.
