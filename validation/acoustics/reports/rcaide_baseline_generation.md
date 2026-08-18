# RCAIDE acoustic baseline generation

**Source commit:** `c88217f3fd0ef9740e86cfc4241bb4362bb7a766`  
**Source driver SHA-256:** `e642ef193f6290b31cf87d18f909baf77af3d3eec322ad2bda49515b005ff2f9`  
**Runtime:** Python 3.12.13, NumPy 1.26.4, macOS arm64  
**Generated models:** F8745-D4 line source, F8745-D4 plane source, APC 11x4 plane source

RCAIDE's equations, settings, geometries, operating cases, and validation-driver assertions were
not changed. NumPy 1.26.4 lacks the `np.trapezoid` name used by the pinned RCAIDE source, so the
runner explicitly aliased it to the mathematically equivalent legacy `np.trapz`. This intervention
is recorded in every archive and manifest.

The generator captured all serializable fields from the RCAIDE aeroacoustics, aerodynamics,
energy, freestream, frames, rotor, and settings containers. This preserves spectra as well as
intermediate aerodynamic loads, coefficients, geometry, state, and configuration needed to
diagnose later disagreement.

RCAIDE's own driver completed all three model runs and its permissive regression assertions. Its
reported selected total-SPL results were:

| Model | 60-degree value (dB) | 90-degree value (dB) |
| --- | ---: | ---: |
| F8745-D4 line source | 105.0772 | 108.4291 |
| F8745-D4 plane source | 102.0184 | 100.6013 |

The driver's printed ratios are percentage-of-dB-style regression checks and are not adopted as
physical validation metrics. BladeAD validation will instead compute signed dB errors, MAE,
maximum absolute error, and overall SPL error directly against the experimental fixtures.

The APC driver completed with maximum relative-to-dB-array differences of 0.9228 and 0.5624 for
its two selected broadband comparisons. These large values and the unresolved observer-angle
mapping require explicit investigation; the baseline is evidence of current RCAIDE behaviour,
not evidence that the model is experimentally accurate.
