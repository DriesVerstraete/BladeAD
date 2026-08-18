# Local patches to `BladeAD`

Fork of `github.com/LSDOlab/BladeAD`, `origin` = `github.com/DriesVerstraete/BladeAD`,
`upstream` = `LSDOlab/BladeAD`. Working branch `spl-develop` (set as the fork's GitHub default
branch). This file logs every local modification, matching the pattern used for the RCAIDE fork
(`999-software/rcaide/local-patches.md`) — check here before diffing against upstream or
investigating "unexpected" BladeAD behavior.

## 2026-08-18 — Add F8745-D4 experimental tonal validation

**Files:** `validation/acoustics/scripts/run_bladead_f8745_validation.py`,
`validation/acoustics/reports/`, `tests/acoustics/test_f8745_bladead_report.py`

**What:** added a reproducible BladeAD Lowson plus Barry–Magliozzi comparison against all 108
F8745-D4 measured harmonics, using frozen RCAIDE disk loads to isolate acoustic-radiation error.

**Why:** apply the pre-frozen experimental gate before allowing the tonal model to influence an
optimised rotor design, while keeping RCAIDE predictions separate from experimental truth.

**Verification:** every case fails the frozen 3 dB MAE/overall thresholds, with 13.320–22.140 dB
harmonic MAE and -11.488 to -16.779 dB overall error. The uncalibrated failure is preserved as an
active regression and the model is not accepted as forward-flight experimental design authority.

## 2026-08-18 — Restore BEM pitch and lag input forwarding

**Files:** `BladeAD/core/BEM/bem_model.py`, `tests/acoustics/test_tonal_api.py`

**What:** forwarded the declared collective/cyclic pitch and lag variables from
`RotorAnalysisInputs` into BEM preprocessing. Added a complete BEM-to-tonal-SPL derivative gate
for RPM, root/tip chord and twist controls, and collective pitch.

**Why:** BEM accepted these public inputs but omitted them at the preprocessing call, making pitch
and lag ineffective and their aerodynamic/acoustic sensitivities identically zero.

**Verification:** all six design sensitivities are nonzero and agree with finite differences to
less than `5e-4` relative error; detailed results are preserved in
`validation/acoustics/reports/tonal_derivative_verification.md`.

## 2026-08-18 — Expose Sears loading through the tonal API

**Files:** `BladeAD/core/acoustics/api.py`, `BladeAD/core/acoustics/var_groups.py`,
`tests/acoustics/test_tonal_api.py`

**What:** added opt-in Sears load harmonics and configurable gust amplification to the real BEM
tonal chain, using BEM inflow angle and the same radial quadrature convention as steady loads.

**Why:** allow the HG hover-loading branch to be selected without replacing the default direct
Fourier projection used for azimuthally resolved BEM loads.

**Verification:** the real BEM end-to-end test now exercises Sears loading, thickness noise,
combined SPL, A-weighting, and observer-position derivatives through the complete graph.

## 2026-08-18 — Add differentiable Sears hover-loading harmonics

**Files:** `BladeAD/core/acoustics/tonal/sears.py`, `tests/acoustics/test_sears.py`,
`BladeAD/core/BEM/compute_quantities_of_interest.py`

**What:** added Sears gust-response load harmonics using the audited 0.06 gust-amplification
convention, while retaining complete-rotor steady loads and producing per-blade elemental
coefficients for Lowson equation (10). Exposed and applied BEM's actual radial quadrature weights
so trapezoidal/Simpson endpoints are not overcounted acoustically.

**Why:** reproduce the hover-loading branch used in the HG case without inheriting its duplicated
blade axis or mixing force-per-length and elemental-force conventions.

**Verification:** independent SciPy evaluation of all four Sears load coefficients, CSDL derivative
verification with respect to angular speed, and BEM interface coverage for radial quadrature
weights.

## 2026-08-18 — Integrate thickness noise into the tonal API

**Files:** `BladeAD/core/acoustics/api.py`, `BladeAD/core/acoustics/var_groups.py`,
`BladeAD/utils/var_groups.py`, `tests/acoustics/test_tonal_api.py`

**What:** added opt-in thickness-to-chord geometry and connected Barry–Magliozzi thickness noise
to the real BEM tonal API. Loading and thickness component pressure/SPL remain separately exposed;
the combined output follows the audited HG implementation's energetic per-mode addition.

**Why:** support the validated thickness source without hiding it inside a combined SPL and make
the current non-phase-coherent combination assumption explicit.

**Verification:** the real BEM end-to-end test now includes thickness, checks separate and combined
outputs, confirms combined mean-square pressure exceeds loading alone, and verifies observer-position
derivatives through the full combined chain.

## 2026-08-18 — Add differentiable Barry–Magliozzi thickness noise

**Files:** `BladeAD/core/acoustics/tonal/thickness.py`,
`tests/acoustics/test_thickness.py`, `validation/acoustics/fixtures/lowson_hg_matlab/`

**What:** added per-mode RMS pressure, mean-square pressure, SPL, and total SPL for the
Barry–Magliozzi thickness formulation used by the pinned HG validation implementation.

**Why:** thickness noise is a distinct coherent tonal source and must be validated independently
before combining it with Lowson loading pressure in the 37-point HG directivity case.

**Verification:** direct comparison with all 33 HJ thickness-only reference angles (less than
1 dB absolute error and less than 0.01 dB shape-residual range, with the observed common offset
left uncalibrated) plus CSDL derivative verification with respect to angular speed.

## 2026-08-18 — Freeze the HG-MATLAB combined directivity fixture

**Files:** `validation/acoustics/fixtures/lowson_hg_matlab/`,
`tests/acoustics/test_hg_fixture.py`

**What:** preserved all 37 HG-MATLAB directivity values and 16 separately labelled experimental
points from the pinned `lsdo_acoustics` hover verification source, with conditions, provenance,
known missing metadata, and an explicit scope warning.

**Why:** the upstream curve combines Sears hover loading and Barry–Magliozzi thickness noise. It
cannot honestly validate a loading-only Lowson kernel; freezing it now prevents later component
mixing or silent transcription changes.

**Verification:** fixture integrity test checks row counts, the complete angle grid, and selected
source anchors. The combined curve remains inactive as a model-acceptance test until thickness,
Sears, and observer-angle conventions are reproduced separately.

## 2026-08-18 — Connect the real BEM-to-Lowson tonal chain

**Files:** `BladeAD/core/acoustics/api.py`, `BladeAD/core/acoustics/observers.py`,
`BladeAD/core/acoustics/var_groups.py`, `tests/acoustics/test_tonal_api.py`

**What:** enabled the opt-in tonal API to connect real BEM sectional loads and mesh data through
Fourier projection, node-dependent observer geometry, moving-source convection, arbitrary-harmonic
Lowson pressure, coherent rotor synthesis, per-mode/total SPL, and A-weighted total SPL.

**Why:** expose one differentiable production path instead of requiring users to assemble tested
acoustic primitives manually. The API rejects aliased load-harmonic requests and rotor outputs
whose load convention is not explicitly complete-rotor.

**Verification:** a real BEM solve feeds the complete tonal API for two acoustic modes and four
load harmonics; output shapes and finite values are checked, and CSDL derivatives of tonal SPL
with respect to the observer position are verified by finite difference through the full chain.

## 2026-08-18 — Add coherent complete-rotor tonal synthesis

**Files:** `BladeAD/core/acoustics/tonal/synthesis.py`,
`tests/acoustics/test_synthesis.py`

**What:** added coherent blade-amplitude synthesis at acoustic orders `n = mode * B`, per-mode
mean-square pressure and SPL, and energetic summation across distinct tonal frequencies.

**Why:** the paragraph following equation (11) of Lowson and Ollerhead states that blade-passing
harmonics add, which requires a factor of `B` in pressure amplitude (and therefore `B^2` in
mean-square pressure), rather than energetic addition of independent blade SPLs.

**Verification:** analytic pressure and SPL comparisons, an explicit `20 log10(B)` coherent-gain
test, and CSDL derivative verification from total SPL to per-blade pressure.

## 2026-08-18 — Add arbitrary-harmonic Lowson loading pressure

**Files:** `BladeAD/core/acoustics/tonal/loading.py`,
`BladeAD/core/acoustics/tonal/load_harmonics.py`, `tests/acoustics/`

**What:** implemented the axial and circumferential-force terms of Lowson and Ollerhead (1969)
equation (10) for arbitrary non-negative loading harmonics. Real CSDL graphs represent the
published complex phase exactly, including negative Bessel orders when `lambda > n`. Corrected
the Fourier projection to use the equation (9) convention: the zero harmonic is the mean,
non-zero cosine/sine coefficients are twice the mean projection, and the sine coefficient has
the positive sign associated with BladeAD's increasing azimuth phase.

**Why:** avoid the opaque parity/sign matrices and normalization inherited from `lsdo_acoustics`;
the direct complex-equation reduction is easier to audit and differentiates cleanly.

**Verification:** independent SciPy/Python-complex evaluation of equation (10), including
`lambda > n`; exact reduction to the separately audited steady kernel for `lambda=0`; CSDL
derivative verification with respect to unsteady thrust coefficients; full Fourier coefficient
value and derivative tests.

## 2026-08-18 — Add Lowson moving-source convection distance

**Files:** `BladeAD/core/acoustics/convection.py`,
`BladeAD/core/acoustics/tonal/lowson.py`, `tests/acoustics/`

**What:** added the differentiable retarded-position correction
`S * (1 - source_velocity dot observer_direction / speed_of_sound)` and allowed the Lowson
steady-loading kernel to use that convected distance consistently in its Bessel and pressure
terms.

**Why:** represent the forward/aft amplification specified immediately after equation (11) of
Lowson and Ollerhead (1969), while keeping observer kinematics separate from the pressure kernel.

**Verification:** analytic forward/aft/cross-stream values, CSDL derivative verification with
respect to all source-velocity components, and an integration test confirming that the supplied
convected distance replaces the stationary radiation distance in the Lowson kernel.

## 2026-08-18 — Add Lowson steady-loading pressure kernel

**Files:** `BladeAD/core/acoustics/tonal/lowson.py`,
`tests/acoustics/test_lowson.py`

**What:** added the differentiable, stationary-source Lowson pressure terms for steady loading
(`lambda=0`) at requested blade-passing harmonics. The kernel evaluates Bessel arguments and
radiation/near-field directivity, preserves radial pressure contributions, and returns separate
per-blade cosine and sine components. Moving-source convection, coherent blade synthesis, and SPL
aggregation are intentionally deferred.

**Why:** validate signs, Bessel orders, dimensional scaling, and gradients before higher-level
aggregation can conceal implementation errors.

**Verification:** independent SciPy/NumPy pressure reference for two observers and two modes,
explicit odd/even acoustic-order routing coverage, and CSDL derivative verification with respect
to angular speed. The steady-load signs were reduced directly from a reproduced version of the
published complex-amplitude equation: the audit found and avoided an odd-order sign regression in
the pinned vectorized `lsdo_acoustics` implementation; its older loop implementation retains the
published signs.

## 2026-08-18 — Start Lowson with differentiable load harmonics

**Files:** `BladeAD/core/acoustics/tonal/`, `tests/acoustics/test_load_harmonics.py`

**What:** added real-valued azimuthal cosine/sine coefficients for thrust and drag using BladeAD's
actual azimuth grid. Complete-rotor sectional loads are divided by blade count exactly once.

**Why:** isolate and verify the load convention and Fourier projection before adding Lowson's
Bessel/directivity pressure terms.

**Verification:** independent NumPy discrete-Fourier comparison plus CSDL derivative verification
with respect to complete-rotor sectional thrust.

## 2026-08-18 — Expose BEM mesh/load conventions for Lowson acoustics

**Files:** `BladeAD/core/BEM/`, `BladeAD/core/preprocessing/preprocess_variables.py`,
`BladeAD/utils/var_groups.py`, `tests/acoustics/test_bem_acoustic_interface.py`

**What:** exposed dimensional radial stations, radial element width, and azimuth angles on BEM
outputs; explicitly marked sectional loads as already including all blades; corrected radial
element width to use the declared normalized hub radius instead of a hardcoded 0.2.

**Why:** Lowson requires per-blade loads at known source locations and phases. These values must
remain inside the CSDL graph and non-default hub ratios must not silently use inconsistent spacing.

**Verification:** a real BEM solve at `norm_hub_radius=0.3` verifies exposed mesh shapes and values,
then integrates sectional thrust and torque back to the corresponding totals within 0.5%.

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
