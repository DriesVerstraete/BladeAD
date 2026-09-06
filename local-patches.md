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

## 2026-08-27 — Differentiable rotating-blade box beam

**Files:** `BladeAD/core/structures/`, `tests/structures/`, and
`validation/structures/README.md`

**What:** added a parameter-driven CSDL-alpha rectangular box-spar model. It couples directly to
BladeAD's complete-rotor sectional thrust/drag convention, retains azimuthal load cases, integrates
centrifugal and two-axis bending loads at element inboard faces, and returns mass, section
properties, raw stress fields, and smooth allowable utilizations.

**Why:** the standalone multidisciplinary rotor optimisation needs structural mass and explicit
failure constraints without breaking the AD graph. The first model is deliberately a static
equivalent-isotropic fidelity level; its documented exclusions prevent it being mistaken for a
composite laminate or aeroelastic analysis.

## 2026-09-06 — Non-uniform (custom) radial-station grid for the BEM

**Branch:** `nonuniform-radial-grid` → merged to `neuralfoil-csdl-pass1` (both local-only; not
yet pushed to `origin`).

**Files:** `BladeAD/utils/var_groups.py`, `BladeAD/core/preprocessing/preprocess_variables.py`,
`BladeAD/core/BEM/bem_model.py`, `BladeAD/utils/parameterization.py`,
`BladeAD/utils/integration_schemes.py`.

**What:** `RotorMeshParameters` gained an optional `norm_radial_stations` field — a
strictly-increasing array of normalized station locations in the open interval (0, 1), length
`num_radial`. When supplied, `preprocess_input_variables` builds `norm_radius_exp` from it and a
**per-node element-width vector** from edge midpoints (outer edges pinned at hub and tip, so the
widths sum to `radius - r_hub` exactly), and `BEMModel.evaluate` routes
`compute_quantities_of_interest` through a plain **Riemann** sum (edge-midpoint rule) instead of
the uniform-spacing trapezoidal/Simpson weights. `BsplineParameterization` /
`get_bspline_mtx` gained an optional `sample_locations` arg so chord/twist control points can be
sampled at the true normalized station positions (fixed span fractions regardless of spacing).
Also fixed a pre-existing typo in `integration_schemes.py`'s 3-D Riemann branch
(`valu=` → `value=`, previously unreachable).

**All new behaviour is gated behind `norm_radial_stations=None` / `sample_locations=None` =
byte-identical legacy behaviour.** Verified: `pytest tests/` 86/86 pass; a hover-only Shahjahan
optimisation run on the `None` path is bit-identical (max abs diff 0.0) to the pre-patch code.

**Why:** BladeAD's BEM otherwise offers only `num_radial` (a count) with hard-coded uniform
`np.linspace` spacing and a scalar element width — no way to use a cosine (Chebyshev) aero grid.
The Shahjahan CSDL optimiser's grid-convergence study
(`06-rotor-optimisation/shahjahan-rotor/findings/findings-aero-structural-grid-decoupling.md`)
settled on cosine-48 (≈ uniform-96 on dT/dr shape convergence, at half the stations). Full
context: `06-rotor-optimisation/decisions/2026-09-06-shahjahan-proprotor-csdl-optimizer.md`
§ "Step 2".

**Consequence if not patched:** the CSDL/BladeAD rotor optimiser is stuck with a uniform grid
and must run ~n=96 for converged tip-region loading where a cosine n≈48 would do.

## 2026-09-06 — Opt-in Reynolds clamp for tabulated airfoil polars

**Branch:** `neuralfoil-csdl-pass1`.

**File:** `BladeAD/core/airfoil/tabulated_airfoil_model.py`.

**What:** `_PolarSurface` gained a class attribute ``clamp_reynolds`` (default ``False``). When
left False, ``_prepare`` raises on out-of-range Re exactly as before. When an instance sets it
True, out-of-range Re is clamped into ``[reynolds[0], reynolds[-1]]`` with a one-shot
``RuntimeWarning`` instead of raising -- symmetric with the low-end alpha clamp
``_prepare`` already applies unconditionally.

**Why:** a gradient-based optimiser transiently probes Re below the NeuralFoil table floor
(30 000) during the SLSQP search -- most sharply with a hover-noise objective, which drives tip
speed down. A hard raise there aborts the entire solve. The converged design is steered back
in-range by the thrust / Cl / noise constraints, and its outboard sectional Re is verified
in-range separately (the tip-envelope diagnostic in ``optimize_shahjahan_stage1.py``).
``MachCorrectedBSplineAirfoilModel`` (the Shahjahan optimiser's airfoil model) sets
``self.surface.clamp_reynolds = True``; nothing else does, so validation / accuracy uses keep
the strict behaviour.

**Verification:** pytest 86/86 (default path unchanged); the Shahjahan ``acoustic-only`` and
``hover-only --with-acoustics`` solves both converge with all constraints satisfied and
converged-design outboard Re well inside the table (5e5--2.5e6).

**Consequence if not patched:** any BladeAD gradient optimisation with a tabulated-polar airfoil
model can die mid-solve the first time SLSQP steps outside the Re table, with no recovery.
