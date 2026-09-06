# BladeAD.optimisation — deferred code items

Workflow-maintenance items only. Research direction (the fidelity ladder, the
N>=4 method decision, the benchmark protocol, transition averaging, GXBeam) lives
in the SPL rotor-optimisation project `roadmap.md`, not here.

## Open

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

## Migration follow-ups (2026-09-07)

- OneDrive `06-rotor-optimisation/code/rotor_pareto_slots/`, `code/rotor_pareto/`,
  `code/rotor_acoustics/`, `airfoils/neuralfoil_tables/` are retained as a
  fallback -- delete after a full `run_case` from this package confirms parity.
- Append the objective-slots acceptance result (~88 feasible pts) to
  `06-rotor-optimisation/briefs/rotor-pareto-objective-slots-refactor-REPORT.md`.
- Run the STAGE variants (`run_acceptance.py` reordered / twoobj / strlist) or
  judge from structure + the anchor smoke.
