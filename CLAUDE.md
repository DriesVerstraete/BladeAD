# BladeAD fork — context

**Before any work here:** read `001-dashboard/capabilities/bladead-optimiser.md` — current capability frontier, next step, and consumers. This file covers repo conventions only.

Fork of `github.com/LSDOlab/BladeAD` (`origin` = `github.com/DriesVerstraete/BladeAD`, working branch `spl-develop`). Lab-internal shared infrastructure — not a published library.

## Where things live

- **Optimisation capability** — `BladeAD/optimisation/` (case-driven ε-constraint Pareto workflow). Entry point `from BladeAD.optimisation.orchestrator import run_case`.
- **Code cleanups / tech debt** — `BladeAD/optimisation/todos.md` (research direction does NOT go there — it's in the rotor-optimisation project).
- **Code gotchas** — `lessons-learned.md` (read before debugging unexpected BladeAD behaviour).
- **Local modifications to upstream** — `local-patches.md`.
- **Env setup / install gotchas** — `999-software/rotor_design/CLAUDE.md`.

## Running case / optimisation scripts

Hand the user the command to run themselves (multi-minute jobs) — never `conda run` and never a `| tail` pipe. `conda run` captures the child's stdout and only flushes it when the process exits, so `run.log` stays empty for the whole run even with `python -u`. Give the user:

1. `conda activate rotor_design`
2. `python -u <path>/run.py > <path>/run.log 2>&1`

so they can `tail -f` the log live. Per-generation / per-iteration progress must be visible from the first line.

## Consumers

- `01-programs/.../03-projects/06-rotor-optimisation` (research) — the Shahjahan validation cases.

Status, plan, and next capability step are tracked in `001-dashboard/capabilities/bladead-optimiser.md` and its `-log.md`, not in this repo.
