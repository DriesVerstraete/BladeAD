"""THROWAWAY exploratory FPP variant -- same purpose as shahjahan_case1_explore
(feel of the problem + convergence), but fixed-pitch.

Thin override of shahjahan_case1_fpp with the explore knobs (num_radial 32,
n_points 8, grid 5x5; maxiter 120 kept). Acoustics forced OFF by run.py.

Delete this dir once the FPP feel run is done.
"""
import copy

from BladeAD.optimisation.cases.shahjahan_case1_fpp.case import CASE as _BASE

CASE = copy.deepcopy(_BASE)
CASE["rotor"]["num_radial"] = 32
CASE["sweep"].update(n_points=8, grid_fm=5, grid_noise=5, maxiter=200)  # FPP needs more iters
CASE["constraints"]["thrust_tolerance_n"] = 5.0                         # 4% slop OK for a feel run
