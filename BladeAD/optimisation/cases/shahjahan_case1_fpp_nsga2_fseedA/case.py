"""NSGA-II, Shahjahan Case 1 FPP, 2-objective -- initial population seeded from
the CORRECTED (re-seeded) epsilon front geometries only (`shahjahan_case1_fpp_reseed`).

Run A of the A/B pair (2026-09-08). Tests whether a population method can *hold*
the corrected front it cannot *build* -- all seed geometries in one DV basin
(hover rpm ~1330-1950). Run B adds the original local front's min-hover-power
geometry (a different basin) for a cross-basin diversity test.

Same rotor / constraints / penalty / pop / gen as `shahjahan_case1_fpp_nsga2`
(run1); only the initial population differs.
"""
import copy
import os

from BladeAD.optimisation.cases.shahjahan_case1_fpp_nsga2.case import CASE as _BASE

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEED_DIR = os.path.abspath(os.path.join(_HERE, "..", "shahjahan_case1_fpp_reseed"))

CASE = copy.deepcopy(_BASE)
CASE["nsga2"] = dict(CASE["nsga2"],
                     save_tag="fseedA",
                     hot_start=False,
                     front_seed_from=[_RESEED_DIR])   # globs stage1_*.pkl at run time
