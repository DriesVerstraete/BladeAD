"""THROWAWAY: FPP explore with the hover stall constraint effectively removed.

`stall_margin` is hard-capped to (0, 1] by `case.py::_validate`, so the section-
Cl_max path cannot express "no margin". This variant drops `stall_margin` and
falls back to the legacy flat cap on the raw peak sectional Cl, set at
`cl_max = 3.0` -- far above any section's Cl_max (~1.9), so the hover stall
constraint is inert. Sections are then free to run deep into the asb 360-degree
Viterna/flat-plate blend (alpha ~30-40 deg) where NeuralFoil's own
analysis_confidence is ~0.

Read against `_sm100` (sm 1.00) and `_sm099`: the sm 0.99 -> 1.00 step moved the
knee only ~1%, with hover_stall_ratio pinned at 0.993, so this run pins down the
far end of the curve for the writeup -- expect ~9.0-9.1 kW cruise at the knee.
Compute the post-hoc stall ratio from `hover_max_sectional_cl` /
`section_clmax_profile` (or the airfoil tables at the design Re).

EDGE FRONTS ONLY (run.py stages 0-2). Acoustics forced OFF by run.py.
Delete this dir once the sensitivity is read.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["constraints"].pop("stall_margin", None)
CASE["constraints"]["cl_max"] = 3.0
