# Coupled tonal derivative verification

**Date:** 2026-08-18  
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

All derivatives are nonzero and remain below the current coupled-path gate of `5e-4` relative
error. The brief's provisional `1e-5` target is met for both chord controls but not yet for RPM,
twist, or collective pitch. The limiting comparison is finite differencing through BEM's
bracketed implicit inflow solve; this is recorded as conditioning evidence, not hidden by a looser
claim.

During this gate, BEM was found to omit the declared collective/cyclic pitch and lag inputs when
calling preprocessing. The forwarding was restored and the collective-pitch derivative is now
explicitly required to be nonzero.

## Remaining Gate D work

- Run isolated three-step finite-difference convergence studies without repeatedly extending one
  CSDL recorder graph.
- Add direct sectional-thrust and sectional-drag sensitivity gates.
- Decide whether tighter BEM solve convergence is necessary to demonstrate the provisional
  `1e-5` coupled-path target reliably.
