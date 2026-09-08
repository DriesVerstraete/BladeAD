"""NSGA-II, Shahjahan Case 1 FPP, 2-objective -- initial population seeded from
the CORRECTED (re-seeded) epsilon front PLUS the original local front's
min-hover-power geometry (a different DV basin).

Run B of the A/B pair (2026-09-08). Same as `..._fseedA` but with one extra
seed from the original `shahjahan_case1_fpp_explore_sm099` front
(He ~19121, hover rpm ~2000, twist0 ~31deg -- distinct from the corrected
front's basin). Tests whether NSGA-II keeps both basins or collapses to the
dominant one.
"""
import copy
import os

from BladeAD.optimisation.cases.shahjahan_case1_fpp_nsga2.case import CASE as _BASE

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEED_DIR = os.path.abspath(os.path.join(_HERE, "..", "shahjahan_case1_fpp_reseed"))
_LOCAL_MIN_HE = os.path.abspath(os.path.join(
    _HERE, "..", "shahjahan_case1_fpp_explore_sm099",
    "stage1_opt-hover_elec_grid-cosine32.pkl"))     # the original front's min-hover-power point

CASE = copy.deepcopy(_BASE)
CASE["nsga2"] = dict(CASE["nsga2"],
                     save_tag="fseedB",
                     hot_start=False,
                     front_seed_from=[_RESEED_DIR, _LOCAL_MIN_HE])
