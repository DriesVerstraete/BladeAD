"""MOEA/D-CDP, Shahjahan Case 1 FPP, 2-objective (hover vs cruise electrical
power) -- the decomposition-method entry in the population-method comparison
(2026-09-08).

Identical to the first NSGA-II run (`shahjahan_case1_fpp_nsga2` run1) in every
respect EXCEPT the algorithm: same pop (65), gen (100), RNG seed (1), operators
(SBX eta 15 / poly-mut eta 20), penalty (overshoot_mu 10, ks_cap 1.0), and the
same initial population (inverse-design FPP blend + prescale + clones + LHS, NO
front seeding).

MOEA/D decomposes the front into 65 scalarised subproblems (one per reference
direction, Tchebycheff for 2 objectives) and updates each locally against its
neighbourhood -- spread is structurally enforced, not a post-hoc niching step.
`constraint_handling="cdp"` adds the Constraint Dominance Principle to the
replacement step (upstream `MOEAD` gap fixed this session, reusing the same
`cons_sum` the framework's tournament / rank-and-crowding already use).

Question: does decomposition hold the corrected-front spread where NSGA-II
collapsed to the utopia corner and NSGA-III's reference-direction niching only
delayed the erosion?
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp_nsga2.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["nsga2"] = dict(
    CASE["nsga2"],
    algorithm="moead",
    save_tag="moead_run1",
    hot_start=False,
    overshoot_mu=10.0,        # run1's penalty
    overshoot_ks_cap=1.0,
    moead_neighbours=15,      # T; Zhang & Li default 20, 15 for a small 2-obj front
    # n_ref_partitions=None -> pop_size - 1 -> 65 weight vectors
)
