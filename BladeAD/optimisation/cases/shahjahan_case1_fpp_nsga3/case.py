"""NSGA-III, Shahjahan Case 1 FPP, 2-objective (hover vs cruise electrical
power) -- the population-method-comparison counterpart to the first NSGA-II run
(`shahjahan_case1_fpp_nsga2` run1, 2026-09-08).

Identical to run1 in every respect EXCEPT the algorithm: same pop (65), gen
(100), RNG seed (1), operators (SBX eta 15 / poly-mut eta 20), penalty
(overshoot_mu 10, ks_cap 1.0 -- run1's values, NOT the softened fseed values),
and the same initial population (inverse-design FPP blend + hover-rpm prescale +
Gaussian clones + LHS fill -- NO front seeding). Only NSGA-III's
reference-direction survival replaces NSGA-II's crowding distance.

`n_ref_partitions` defaults to pop_size - 1 -> 64 partitions -> 65 reference
directions for 2 objectives (pop ~ n_ref_dirs, the standard NSGA-III setup).

Question: does reference-direction niching hold a spread front where crowding
distance collapsed to the utopia corner (run1) / eroded a handed front
(fseedA/B)?
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_nsga2.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["nsga2"] = dict(
    CASE["nsga2"],
    algorithm="nsga3",
    save_tag="nsga3run1",
    hot_start=False,
    overshoot_mu=10.0,        # run1's penalty, not the softened fseed 5.0
    overshoot_ks_cap=1.0,
    # n_ref_partitions=None -> pop_size - 1
)
