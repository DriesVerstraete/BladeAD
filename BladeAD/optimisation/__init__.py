"""Reusable, case-driven rotor multi-objective optimisation workflow for BladeAD.

Point ``orchestrator.run_case(case_dir)`` at a directory holding a ``case.py``
(with a module-level ``CASE = {...}`` dict) and it runs the whole pipeline --
pre-flight feasibility gate, cold-start seed, single-objective anchors, the
pairwise 2-objective edge fronts, the N-objective interior front, and the
overlay plot -- writing every pkl / log / plot into that directory.

Modules
-------
case          -- load + validate a case directory's ``CASE`` dict; ISA/grid helpers
seed          -- deterministic inverse-design cold-start seed (pure NumPy)
solve         -- one BEM + acoustic optimisation for a ``run(**opts)``
sweep_common  -- min_power_solve / proxy anchor / per-case anchor registry / warm pool
edges         -- the 3 pairwise 2-objective edge sweeps
grid          -- structured epsilon-grid staircase 3-objective front (primary for N==3)
nobj          -- picker-based N-objective front (N > 3; needs ``moocore``)
plot          -- overlay figure (wraps ``plotting.pareto_overlay_plot``)
orchestrator  -- run_case(case_dir)
acoustics     -- SPL hover/cruise acoustic wiring on BladeAD's acoustic core
_sweep        -- vendored epsilon-constraint gap-picker (was AircraftDesign's)
_hypervolume  -- UHVI scorer for the N>3 picker (lazy ``moocore``)

Design invariants
-----------------
* **No argparse.** Config is the case module; run switches are function kwargs.
* **min cruise power s.t. FM >= floor is the ONLY working scalarisation
  direction** for the coupled multi-point trade. The swept objective is an
  inequality; the proxy (power) is minimised.
* **The seed is pure NumPy, never routed through the CSDL airfoil model** --
  structurally incapable of the divergence modes it prevents.

Objective identity lives only in ``case.py`` (the ``ObjectiveSpec`` list +
``_CANONICAL_SPECS``) and ``solve.py`` (the BEM-output reads). Adding an
objective (OEI, transition, electrical power) is a spec edit.
"""
