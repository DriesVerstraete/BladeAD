# BladeAD lessons learned

Fork of `github.com/LSDOlab/BladeAD` (`origin` = `github.com/DriesVerstraete/BladeAD`, branch
`spl-develop`). Generic gotchas found while using this fork — not local modifications (see
`local-patches.md` for those). Check here before debugging "unexpected" BladeAD behavior.

## 1. `CompositeAirfoilModel`'s built-in transition smoothing can crash on ordinary geometries

**What:** `CompositeAirfoilModel(sections=..., airfoil_models=..., smoothing=True)` (the
default) calls `smooth_quantities_spanwise()` at each section boundary, which averages a
window of `2 * transition_window` (default 3) radial stations centered on the boundary index.
On a real front-prop-rotor geometry (20 radial stations, a 2-airfoil 50/50 split), this raised
`ValueError: cannot reshape array of size 0 into shape (...)` inside `getindex.py` — an
out-of-bounds/empty slice, from an index-arithmetic mismatch between the section boundary
(computed from `num_radial` inferred off the `alpha` array's shape inside `evaluate()`) and
BEM's actual internal radial grid.

**Why:** not an edge case requiring an unusually small `num_radial` or a boundary near 0/1 —
it reproduced on a normal 20-station rotor with a boundary at the span midpoint. Root cause not
fully isolated (not worth the time for a mechanism this project doesn't rely on — see below).

**How to apply:** pass `smoothing=False` when constructing `CompositeAirfoilModel`, unless you
specifically need its built-in transition blending and have verified it doesn't crash on your
geometry. This project's own approach (linearly blending Cl/Cd directly between adjacent
airfoils at fixed Reynolds number, per Bronz's PhD thesis — see
`06-rotor-optimisation/results/2026-09-05-shahjahan-airfoil-blend-mechanism-check.md`) is an
external, deliberate replacement for this mechanism anyway, so disabling it is not a loss of
capability for this project's multi-airfoil work.

**Confirmed:** 2026-09-05, front prop_rotor front-point-0 geometry, 20 radial stations, 2- and
4-airfoil `CompositeAirfoilModel` composites, both crashed with `smoothing=True` (default) and
ran cleanly with `smoothing=False`.

## 2. `optimisation/` sweeps run with acoustics ON by default -- ~8× the BEM cost

**What:** `optimisation/sweep_common.py` hard-codes `BASE_OPTS = {"with_acoustics": True}` for
*every* subprocess solve, regardless of whether the case has an acoustic objective. Acoustics
active forces `NUM_AZIMUTHAL_ACOUSTIC = 8` on the hover + cruise BEM points (8 azimuthal
stations vs 1), so a full `run_case` sweep is roughly 8× slower than an acoustics-off solve of
the same case.

**Why it matters:** a `with_acoustics=False` acceptance run of one solve is a bad predictor of
the sweep's per-node time. The only acoustic thing a no-noise-objective case (Case 1) gets from
this is a cruise-vehicle-noise <= 62 dB constraint, which does not match Shahjahan's *hover*
noise constraint anyway.

**How to apply:** for a case with no noise objective, patch it off in the launcher —
`import BladeAD.optimisation.sweep_common as sc; sc.BASE_OPTS = {"with_acoustics": False}`
before `run_case` (BASE_OPTS is expanded in the parent process before each subprocess spawn, so
the patch propagates). Cuts a laptop Case-1 explore run from ~10-20 h to ~2 h.
**Confirmed:** 2026-09-07, Case 1 VPP + FPP explore runs.

## 3. FPP (fixed-pitch) anchors are seed-sensitive -- hover-biased local optima

**What:** with `rotor.fixed_pitch = True` (one collective for hover + cruise + OEI), the
`hover_elec` anchor runs first and pulls the shared geometry hover-ward. A cold or VPP-hover
seed then lands SLSQP in a hover-optimal corner where the cruise blade tip sits at negative AoA
(windmilling), cruise propulsive efficiency collapses to ~18%, and `cruise_rpm` rails at its
bound. The optimiser reports "converged" but the design is physically absurd.

**How to apply:** seed FPP by inverse design -- `case["seed"] = "inverse-design"` with
`fixed_pitch` set picks `seed.fpp_blend_blade()` (hover-ideal chord + hover/cruise twist
blend). And use the radius-dependent section-Cl_max stall constraint (`constraints.stall_margin`
+ `airfoil.clmax_ref_reynolds`) -- the flat `cl_max = 0.8` is over-constrained for FPP and forces
the ~4× cruise penalty on its own.
**Confirmed:** 2026-09-07, `shahjahan_case1_fpp_explore` first run.

## 4. A per-station constraint on a BEM sectional output must reshape the numpy array to `cl.shape` -- `csdl_alpha` division does not numpy-broadcast

**What:** `bem_out.sectional_lift_coefficient` (and every sectional QoI) is a `csdl.Variable` of
shape `(num_nodes, num_radial, num_azimuthal)` -- 3-D even for a single hover point
(`(1, num_radial, 1)`). Dividing it by a flat `(num_radial,)` numpy array via
`cl / csdl.Variable(value=prof)` raises `ValueError: Shapes do not match` in `csdl_alpha`'s
`division.div` -- it does NOT broadcast `(n,)` against `(nn, nr, naz)` the way numpy would. Cost
2026-09-07: the section-Cl_max stall constraint (`solve._stall_ratio`) crashed the FPP explore
run on its first anchor, 1 s in; a latent copy of the same bug sat in the OEI stall path.

**How to apply:** build the per-station numpy profile, then `prof.reshape(cl.shape)` before
wrapping it in a `csdl.Variable` for the elementwise op. `np.repeat(profile, num_azimuthal)` gives
the correct C-order `(num_radial, num_azimuthal)` layout, so a plain `.reshape(cl.shape)` lines
the stations up. Applies to any elementwise op (`/`, `*`, `-`) between a csdl sectional Variable
and a numpy per-station array.
**Confirmed:** 2026-09-07, `solve.py::_stall_ratio` fix (2 hunks, `shahjahan_case1_fpp_explore`).

## 5. Warm-starting a near-unconstrained single-objective extreme diverges -- run the non-proxy anchor COLD, seed only the proxy anchor

**What:** for a pairwise Pareto edge you need both endpoints. The sweep-slot's own
single-objective extreme (`direct_solve(ctx, name)` -- e.g. min `hover_elec`, a `cap`
objective, with only the thrust equalities + stall/taper/torque active) is a well-posed
problem that SLSQP solves cleanly **cold**. Warm-starting it from a far geometry (an NSGA-II
basin point at the other end of the edge) makes SLSQP diverge -- it "converges" and returns
a point with `hover_thrust` ~160× the target. This is exactly what `sweep_common.objective_anchor`'s
"anchors do NOT warm-start" comment warns about. The **proxy** extreme (unconstrained
`min_power_solve(ctx, {})`) is the opposite: it's the one a cold anchor historically truncated
(the original FPP `hover_elec__cruise_elec` ε front, stuck in a high-rpm basin) -- seed *that*
one from the NSGA basin.

**How to apply:** `sweep_common.seeded_anchor` encodes the split -- proxy role → warm from the
seed pkl; non-proxy → `warm=False` (cold). Don't "helpfully" seed both. Cost 2026-09-08:
Pareto-tracer edge smoke #1, the seeded cold anchor returned garbage.
**Confirmed:** 2026-09-08, `pareto_tracer.trace` / `seeded_anchor`.
