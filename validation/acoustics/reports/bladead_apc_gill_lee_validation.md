# BladeAD Gill–Lee APC 11×4 validation

**Condition:** 4200 RPM, inflow ratio 0.08, 1.905 m observer radius

BladeAD BEM rotorcraft thrust coefficient: `0.007478`.
Frozen RCAIDE load converted to the same convention: `0.006986`.
The experimental source does not report measured thrust or torque for this condition.

## Model provenance and convention

The equations are ported from the official `lsdo_acoustics` Gill--Lee implementation
at commit `7c76e0d01a71d59582d9ec3d62493dd7d37bdd69` (MIT licence).
That source model fixes the inner planform-integration radius at `0.2R`; BladeAD
retains that value as the default Gill--Lee convention even though the APC BEM mesh
starts at `0.15R`. This is source fidelity, not a fitted validation parameter.
Gill--Lee is an empirical rotor broadband correlation rather than RCAIDE's BPM
boundary-layer model. Exact numerical training-envelope bounds are not stated in the
available implementation, so extrapolation risk must be assessed per application.

| Source load | Angle from rotor plane | Band MAE (dB) | Maximum error (dB) | Overall error (dB) | Gate |
|---|---:|---:|---:|---:|---|
| geometry_driven | 22.5° | 3.815 | 7.680 | 2.824 | PASS |
| geometry_driven | 45.0° | 3.109 | 7.891 | 1.013 | PASS |
| rcaide_source_load | 22.5° | 3.809 | 7.748 | 2.738 | PASS |
| rcaide_source_load | 45.0° | 3.151 | 8.025 | 0.876 | PASS |

The frozen gate requires broadband overall SPL within 3 dB and mean band error
within 5 dB over the 100–10,000 Hz measured bands. This is a coupled comparison
because measured integrated and sectional aerodynamic loads are not reported.
Both the geometry-driven BladeAD load and the frozen RCAIDE integrated-load
sensitivity case pass at both resolved observers; the broadband adoption gate is
therefore cleared subject to the stated empirical-model and load-data limitations.
