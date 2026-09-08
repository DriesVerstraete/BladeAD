"""THROWAWAY: FPP explore with the hover stall constraint relaxed 0.90 -> 0.99.

`shahjahan_case1_fpp_explore` pinned the hover blade at stall ratio 0.894 against
the 0.90 cap -- fine pitch, so the fixed-pitch blade cruises badly (eta ~0.55,
cruise electrical power ~10.7 kW vs the VPP ~6.3 kW). This variant lets the
hover sections reach 0.99 of their section Cl_max (incipient stall, still inside
the NeuralFoil table's fitted alpha range -- no post-stall extrapolation) to see
whether the extra hover headroom lets the optimiser pick a coarser blade that
cruises better.

EDGE FRONTS ONLY: run.py runs stages 0-2 (pre-flight + anchors + the 3 pairwise
2-objective fronts) and skips the interior grid -- the edges are enough to read
how far the front has shifted. Acoustics forced OFF by run.py.

Delete this dir once the sensitivity is read.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["constraints"]["stall_margin"] = 0.99
