# BladeAD.optimisation — deferred code items

Workflow-maintenance items only. Research direction (the fidelity ladder, the
N>=4 method decision, the benchmark protocol, transition averaging, GXBeam) lives
in the SPL rotor-optimisation project `roadmap.md`, not here.

## Open

- **`forward.py` duplicates `solve.py`'s graph build** (2026-09-08). `forward.evaluate`
  (the NSGA-II / population-method eval kernel) rebuilds the same BEM + motor +
  stall-ratio graph as `solve.run()` lines ~362-540, minus the modopt plumbing.
  Kept separate deliberately so the ε-constraint workflow carries zero risk from
  the NSGA-II work. Reconcile by extracting a shared `build_graph(case, dvs,
  *, as_design_vars)` that both call. Decision:
  `06-rotor-optimisation/decisions/2026-09-08-nsga2-rotor-setup.md`.
  **Elevated 2026-09-09 (PI):** `forward.py` computes **no acoustic quantity**, so
  `cruise_noise` / `hover_noise` cannot be NSGA-II scout objectives (`nsga2_runner`
  raises `ValueError` on an acoustic scout slot). For the rotor noise case this
  means the scout can miss a separate low-noise basin. Port
  `add_spl_hover_cruise_acoustics` into `forward.evaluate` (naturally falls out of
  the shared-`build_graph` refactor) + expose `cruise_noise` in `_RESULT_KEY_MAP`.
  Then run the acoustic scout **parallel** (`nsga2_runner.run(n_workers=...)` ->
  `optimisation_framework` `multiprocess.Pool`; today `n_workers=1`) so the ~10-100x
  acoustic-eval cost stays tens-of-minutes, not hours. Interim workaround
  (Option D) is a post-hoc acoustic sweep of the BEM scout's final population --
  see `06-rotor-optimisation/briefs/noise-objective-deblock-steps.md`.

- **Section Cl_max, live-Re v2** -- `solve._section_clmax_profile` uses a fixed
  reference Reynolds per airfoil (`airfoil.clmax_ref_reynolds`). A v2 would
  interpolate Cl_max(Re) against the live per-station Re instead. Only worth it
  if a case shows the fixed reference is materially wrong. Derivation of the
  current values + when to revisit them: decision
  `06-rotor-optimisation/decisions/2026-09-07-section-clmax-stall-constraint.md`.

- **`grid._node` first-stair retry has no cap.** When a column's uncapped natural
  value is far from any feasible cap (e.g. the max-FM blade: natural noise ~80.6,
  no feasible cap at all), the retry loop grinds through every `n_stair` cap at
  `maxiter=60` (~4 min each) before ending. Give up after 2-3 consecutive
  infeasible caps when none has been feasible.
- **`nobj.pick_next_target_n` gap re-pick / region fixation** — root cause is the
  shared picker in `_sweep.py` (which was AircraftDesign's; now vendored). A fix
  there (stop re-scoring a gap whose midpoint solve doesn't improve hypervolume)
  would let `nobj.py` drop its whole local guard block (consumed-pair set,
  `tried` targets, near-duplicate skip, `it_cap`). Until then the guards stay.
- **`grid.py` `f_stair` sweep for a `floor` stair slot** is coded but untested
  ("toward its extreme" = loosen up) -- exercise it on the first (floor, floor)
  3-objective case.
- **`nobj.py` free-non-proxy legs** always take a loose bound from the front's
  current extreme now (the interpolated-tight-floor path is gone). Revisit if a
  case wants tighter interior spacing; consider `maxiter=30` for the N-obj sweep
  so a thrash fails in ~2 min not ~5.
- **`_proxy_anchor_impl` probe step sizing** uses `x_max <= 2.0` to pick
  absolute-vs-scaled step, and geometric doubling overshoots the feasibility wall
  by one failed solve. Fine for FM / efficiency / thrust-margin / power; revisit
  for an odd-scale slot.
- **`nobj.py` checkpointing** — `grid.py` is naturally resumable (per-node pkl
  reuse). If `nobj.py` becomes a primary method, add a `_nobj_checkpoint.pkl`
  written after every added point, seeded on start, deleted at the end.
- **2-objective cases** end to end — the orchestrator runs the one active edge and
  skips `nobj`; exercised by construction (`cases/shahjahan_test_2obj`), not yet
  run start-to-finish.
- **Inverse-design hover seed RPM** is taken from the case seed / rpm-bound
  midpoint; a target-tip-Mach pick would be more principled. Cold-start polish
  only (the min-power direction removed the need for a clever anchor seed).
- **Front-prop overlay** (`optimize_front_prop_rotor`, outside this package) still
  uses `plotting/pareto_overlay_plot.py`'s legacy hardcoded `keys=` path and
  duplicates the acoustic wrapper. Migrate onto the spec-driven path eventually.
- **`code/optimize_shahjahan_proprotor/mach_corrected_airfoil.py`** (frozen SPL
  scripts) is now superseded by
  `BladeAD/core/airfoil/mach_corrected_bspline_airfoil_model.py`. Point the frozen
  scripts at core, or leave them frozen.

- **Pareto tracer Step 2b: CSDL-alpha `jac` for the BFGS predictor** (2026-09-08).
  The generic predictor-corrector loop
  (`optimisation_framework/gradient/pareto_tracer.py`) is backend-agnostic and
  done; the BladeAD adapter (`BladeAD/optimisation/pareto_tracer.py`) currently
  walks the front with the `secant` predictor (no derivatives). Step 2b wires
  CSDL-alpha reverse-mode Jacobians into the bordered-KKT tangent so the
  predictor becomes `bfgs`, landing each station close enough that the corrector
  converges in 1-2 SLSQP iters instead of 4-5 (~2x trace speedup). Scope:
  first-order Jacobians only (already extracted by `modopt.CSDLAlphaProblem`) --
  NOT exact Hessians. ~0.5 d, low risk. **Trigger: do it only when a case shows
  the corrector routinely taking 4-5+ SLSQP iters/station, or a trace visibly
  stalls / step-shrinks repeatedly.** So far secant reproduces the corrected
  epsilon front to ~1%. Decision:
  `06-rotor-optimisation/decisions/2026-09-08-pareto-tracer-and-hybrid-pipeline.md`
  ("Revisit exact Hessians only if tracing visibly stalls" -- same logic).

- **Codex sandbox can't run the `rotor_design` workload** (2026-09-08, both hybrid
  passes). Codex reports every dry run BLOCKED by: (1) `~/.matplotlib` / fontconfig
  cache not writable, (2) an MPI `bind() ... Operation not permitted` on
  `mca_btl_tcp_component_create_listen`. Claude works around (1) with
  `MPLCONFIGDIR=/tmp/mpl-... MPLBACKEND=Agg`; (2) appears to be a harmless warning
  (`mpi4py` import in the CSDL stack trying to open a listen socket) but the run
  never gets far enough in the sandbox to confirm. **Check:** whether
  `OMPI_MCA_btl=^tcp` / `OMPI_MCA_plm=isolated` (or dropping the `mpi4py` import
  path entirely for a single-process solve) lets a Codex dry run complete, and
  whether the MPL env vars belong in the repo `AGENTS.md` / a Codex profile so
  every future pass has them. Until then Codex verifies by `py_compile` + unit
  tests only and Claude runs the real dry run.

## Migration follow-ups (2026-09-07)

- OneDrive `06-rotor-optimisation/code/rotor_pareto_slots/`, `code/rotor_pareto/`,
  `code/rotor_acoustics/`, `airfoils/neuralfoil_tables/` are retained as a
  fallback -- delete after a full `run_case` from this package confirms parity.
- Append the objective-slots acceptance result (~88 feasible pts) to
  `06-rotor-optimisation/briefs/rotor-pareto-objective-slots-refactor-REPORT.md`.
- Run the STAGE variants (`run_acceptance.py` reordered / twoobj / strlist) or
  judge from structure + the anchor smoke.
