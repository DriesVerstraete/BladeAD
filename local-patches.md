# Local patches to `BladeAD`

Fork of `github.com/LSDOlab/BladeAD`, `origin` = `github.com/DriesVerstraete/BladeAD`,
`upstream` = `LSDOlab/BladeAD`. Working branch `spl-develop` (set as the fork's GitHub default
branch). This file logs every local modification, matching the pattern used for the RCAIDE fork
(`999-software/rcaide/local-patches.md`) — check here before diffing against upstream or
investigating "unexpected" BladeAD behavior.

## 2026-08-10 — Expose sectional Cl/Cd as `RotorAnalysisOutputs` fields (BEM only)

**Files:** `BladeAD/core/BEM/compute_quantities_of_interest.py`,
`BladeAD/utils/var_groups.py`

**What:** added `sectional_lift_coefficient`/`sectional_drag_coefficient` fields to
`RotorAnalysisOutputs` (both optional, default `None` — `Pitt-Peters`/`Peters-He` don't populate
them), and set them from the already-computed `Cl`/`Cd` local variables inside
`compute_quantities_of_interest()` right before it returns `bem_outputs`.

**Why:** `Cl`/`Cd` are already real, in-graph `csdl.Variable`s inside `BEMModel.evaluate()`
(passed into `compute_quantities_of_interest()` to build `Cx`/`Ct`), but were never attached to
the returned output object — no way to use them as an optimization constraint (e.g. max
sectional Cl, matching RCAIDE's `design_electric_rotor()`'s own `max_sectional_cl_hov`
constraint) without this. No new computation added — this only exposes existing graph values, so
gradients flow through automatically (`modopt`'s SLSQP needs a real AD path for every
constraint; a value computed outside the graph, e.g. from `.value` post-hoc, cannot be used as a
gradient-based constraint at all).

**Verification:** not just checked for a populated field — a real `max_sectional_cl` constraint
built from `outputs.sectional_lift_coefficient` was added to a working SLSQP optimization run
and confirmed to **actually bind** (converged with `max_cl == 0.8`, the constraint's upper
bound, `Success: True`), proving gradients genuinely flow through the new field and shape the
optimizer's search, not just that it's populated with plausible-looking numbers.

**Consequence if not patched**: no way to add an AD-compatible max-sectional-Cl (or similar
Cl-derived) constraint to any BladeAD optimization problem — would have to either modify BladeAD
per-run (this patch, done properly instead) or drop the constraint entirely.

Full context: `01-programs/program-evtol-long-range-delivery-drone/03-projects/
06-rotor-optimisation/decisions/` (BladeAD redesign work, O39).

## 2026-08-26 — Differentiable Gill--Lee broadband acoustics

**Files:** `BladeAD/core/acoustics/broadband/`, `BladeAD/core/acoustics/api.py`,
`BladeAD/core/acoustics/var_groups.py`, and acoustic validation/tests.

**What:** ported the official `lsdo_acoustics` Gill--Lee empirical one-third-octave model into
the CSDL-alpha graph, exposed broadband-only and energetically combined tonal+broadband API
paths, and added equation, derivative, integration, and APC 11x4 validation coverage.

**Why:** the rotor optimiser requires a differentiable broadband contribution before defining
its production acoustic objective. The upstream model's fixed `0.2R` planform-integration inner
radius is retained as an explicit setting and default; it is not tuned to the APC data.

**Provenance:** equations ported from official `lsdo_acoustics` commit
`7c76e0d01a71d59582d9ec3d62493dd7d37bdd69` under its MIT licence.

## 2026-08-26 — Smooth multi-observer SPL aggregation

**Files:** `BladeAD/core/acoustics/aggregation.py`, package exports, and foundation tests.

**What:** added a differentiable log-sum-exp maximum over observer SPL with a user-specified
worst-case upper-bias bound in dB. Raw per-observer outputs remain unchanged.

**Why:** Gate E requires a smooth multi-observer objective. The parameterisation makes the
departure from the true maximum explicit and testable rather than hiding an arbitrary KS factor.

## 2026-08-27 — Selectable differentiable motor models

**Files:** `BladeAD/core/motor/`, `tests/motor/`, and `validation/motor/README.md`

**What:** added two parameter-driven CSDL-alpha motor models behind one selector: `mcdonald`,
McDonald's positive-polynomial speed--torque loss model, and `three_constant`, the standard
`Kv`--resistance--no-load-current equivalent circuit. Both return shaft power, electrical power,
loss, and efficiency; the three-constant model additionally returns current and voltage.

**Why:** the standalone rotor study will initially use Shahjahan's multi-kW McDonald calibration
and later examine smaller propellers for which explicitly supplied three-constant motor data are
useful. Coefficients remain caller inputs so no motor-size calibration is silently embedded in
BladeAD. Reference-value, input-validation, and derivative tests cover both models.
