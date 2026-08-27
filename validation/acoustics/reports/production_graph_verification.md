# Production-graph tonal verification

**Date:** 2026-08-26
**Path:** BladeAD BEM → Lowson loading and Barry–Magliozzi thickness → tonal SPL →
`PySimulator` → `CSDLAlphaProblem`

`validation/acoustics/scripts/run_production_graph_verification.py` exercises the complete
optimisation-facing graph for forward-flight and hover-like inflow. Both cases use two BPF modes,
Sears load harmonics 0–10, one off-axis observer, and six scalar design variables: RPM, root/tip
chord, root/tip twist, and collective blade-pitch. These are coupled numerical-verification cases,
not additional experimental validation datasets.

| Case | Tonal SPL (dB) | Problem/direct primal error | Problem/direct gradient error | Problem/central-FD relative error |
|---|---:|---:|---:|---:|
| Forward flight, 15 m/s | 75.475862 | 0.0 | 0.0 | 2.09e-8 |
| Hover-like, 0.5 m/s | 89.129537 | 0.0 | 0.0 | 7.63e-9 |

Central differences use a relative step of `1e-5`, scaled by `max(|x|, 1)`. All twelve reported
gradient components are finite and nonzero. The regression
`tests/acoustics/test_production_graph.py` freezes the forward-flight primal and gradient checks.

The complete acoustics regression passed on 2026-08-26: 58 tests. The first scratch-directory
launch exposed an existing test-discovery dependency on the repository root; the successful run
set the BladeAD checkout explicitly on `PYTHONPATH` and made no scientific or implementation
change.

## Gate disposition

The tonal production-graph prerequisite is cleared. The broadband physical-validation gate and
the later production observer/objective definition remain open, so this result does not authorize
an acoustic objective in rotor-design decisions.
