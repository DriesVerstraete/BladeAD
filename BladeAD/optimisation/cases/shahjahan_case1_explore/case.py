"""THROWAWAY exploratory variant of `shahjahan_case1` -- for getting a feel of
the 3-objective problem and its convergence challenges, NOT a benchmark run.

Thin override of the real case (no duplication -> cannot drift):
  - num_radial 48 -> 32   (temp only; 48 stays the default in shahjahan_case1)
  - n_points 12 -> 8, grid_fm/grid_noise 8 -> 5   (bounded first-feel budget)
  - maxiter 120 kept (real value -- representative convergence behaviour)

Acoustics is forced OFF by the sibling run.py launcher (patches
sweep_common.BASE_OPTS) -- no noise objective in Case 1, and the cruise-noise
constraint doesn't match Shahjahan's hover-noise one anyway. Noise comes back
properly at Case 1N.

Delete this dir (and shahjahan_case1_smoke) once the feel run is done.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["rotor"]["num_radial"] = 32
CASE["sweep"].update(n_points=8, grid_fm=5, grid_noise=5)
