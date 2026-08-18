# Local patches to `BladeAD`

Fork of `github.com/LSDOlab/BladeAD`, `origin` = `github.com/DriesVerstraete/BladeAD`,
`upstream` = `LSDOlab/BladeAD`. Working branch `spl-develop` (set as the fork's GitHub default
branch). This file logs every local modification, matching the pattern used for the RCAIDE fork
(`999-software/rcaide/local-patches.md`) — check here before diffing against upstream or
investigating "unexpected" BladeAD behavior.

## 2026-08-18 — Make BladeAD authoritative for acoustic validation assets

**Files:** `validation/acoustics/`

**What:** established the version-controlled layout and provenance rules for acoustic fixtures,
source-tool baselines, validation drivers, and compact reports. Experimental measurements and
RCAIDE predictions are required to remain separate. Added reproducible extraction and integrity
tests for F8745-D4 and APC 11x4 geometry, conditions, observers, experimental spectra, and pinned
RCAIDE source-tool baselines. Baseline archives preserve complete serializable numeric containers,
manifests, data dictionaries, runtime/source metadata, and the required NumPy compatibility alias.
Added a reproducible RCAIDE-versus-experiment validation matrix with detailed signed errors and
acceptance criteria frozen before BladeAD acoustic-model predictions. Added a source-grounded
audit of BEM sectional-load, radial, azimuthal, integration, and frame conventions.

**Why:** keep the validation evidence beside the reusable acoustic implementation and tests so a
fresh checkout can locate and reproduce the model's evidence without project-local knowledge.

## 2026-08-18 — Add differentiable rotor-acoustics foundation

**Files:** `BladeAD/core/acoustics/`, `tests/acoustics/test_foundation.py`

**What:** added the opt-in rotor-acoustics API boundary, observer geometry and directivity,
blade-passing frequencies, pressure-squared aggregation, SPL conversion, IEC-style A-weighting,
and foundation tests including an observer-position derivative check. Tonal and broadband models
remain explicitly disabled until their independent validation fixtures are frozen.

**Why:** establish the shared CSDL-alpha graph and numerically tested conventions required by the
project's accepted propeller-acoustics integration brief without changing existing aerodynamic
execution or results.

**Verification:** `rotor_design` ran `pytest tests/acoustics/test_foundation.py`: 5 passed,
including the finite-difference comparison of the CSDL observer-position derivative.

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
