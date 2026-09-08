"""THROWAWAY: FPP explore, hover stall constraint at the edge of confidence.

Follow-on to `_sm099`. That run's knee still bound the hover blade against its
stall cap (hover_stall_ratio 0.983 / 0.99) with the binding station mid-blade
(MH113/MH115, Re > 5e5 -- inside NeuralFoil's confident range). This variant
lets sections reach exactly their section Cl_max (stall_margin = 1.0), i.e. the
stall angle itself, where NeuralFoil's analysis_confidence is still ~0.9 and the
polar has not yet handed over to the asb 360-degree Viterna blend.

Pair with `_smNONE` (stall_margin 1.5 -- effectively unconstrained, deep
post-stall allowed) to see whether the cruise optimum wants to sit near
cl/clmax = 1.0 or dive into the Viterna region.

EDGE FRONTS ONLY (run.py stages 0-2). Acoustics forced OFF by run.py.
Delete this dir once the sensitivity is read.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_explore.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["constraints"]["stall_margin"] = 1.0
