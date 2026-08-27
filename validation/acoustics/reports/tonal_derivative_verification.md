# Coupled tonal derivative verification

**Date:** 2026-08-26
**Path:** BladeAD BEM → Sears loading → Lowson radiation → Barry–Magliozzi thickness → total SPL

The active regression in `tests/acoustics/test_tonal_api.py` compares CSDL reverse-mode
derivatives with finite differences for a two-bladed rotor, two BPF modes, Sears harmonics 0–10,
and one off-axis observer. Chord and twist use root/tip control variables. The perturbation is
`1e-5` in each variable's native units.

| Variable | CSDL derivative norm | FD derivative norm | Relative error |
|---|---:|---:|---:|
| RPM | 0.00918094 | 0.00918178 | 9.22e-5 |
| Root chord | 31.13292 | 31.13268 | 7.70e-6 |
| Tip chord | 58.44457 | 58.44445 | 2.10e-6 |
| Root twist | 2.64908 | 2.64926 | 6.98e-5 |
| Tip twist | 3.67755 | 3.67789 | 9.39e-5 |
| Collective pitch | 6.32663 | 6.32756 | 1.47e-4 |

Those values used CSDL-alpha's one-sided finite-difference helper and are retained as the original
baseline. They are superseded for Gate D by the explicit central-difference convergence study
below.

During this gate, BEM was found to omit the declared collective/cyclic pitch and lag inputs when
calling preprocessing. The forwarding was restored and the collective-pitch derivative is now
explicitly required to be nonzero.

## Three-step central-difference convergence study

`validation/acoustics/scripts/run_tonal_derivative_convergence.py` rebuilds a fresh CSDL recorder
graph at every step and compares reverse-mode derivatives with explicit central differences. The
three relative step factors are `1e-4`, `1e-5`, and `1e-6`. Each perturbation is scaled by
`max(||x||, 1)`, so the 1800 RPM input uses steps of 0.18, 0.018, and 0.0018 RPM; variables below
one in native units use the listed factors directly.

The study covers tonal SPL and the complete radial-by-azimuthal sectional-thrust and
sectional-drag arrays with respect to RPM, root/tip chord, root/tip twist, and collective
blade-pitch. All 18 derivative blocks are nonzero.

| Relative step factor | Maximum tonal-SPL error | Maximum sectional-thrust error | Maximum sectional-drag error |
|---:|---:|---:|---:|
| `1e-4` | 1.97e-6 | 1.39e-5 | 1.78e-5 |
| `1e-5` | 4.28e-7 | 1.39e-7 | 1.78e-7 |
| `1e-6` | 5.86e-7 | 7.90e-7 | 1.08e-6 |

At the middle step, every derivative block clears the provisional `1e-5` relative-error target.
The error decreases strongly from `1e-4` to `1e-5`; the modest rise at `1e-6` is the expected
roundoff/implicit-solve floor and remains below the gate. Tighter BEM convergence is therefore not
required for this derivative-consistency gate.

## Gate D disposition

The coupled Lowson derivative gate is cleared for RPM, chord, twist, collective blade-pitch,
sectional thrust, and sectional drag. This does not clear the separate full-production-graph or
broadband gates.
