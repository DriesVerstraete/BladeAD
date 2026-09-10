"""Unit tests for `RotorNSGA2Setup` -- the optional loose hover-power cap
(`constraints.max_hover_elec_w`) and the acoustic-scout wiring. Pure: loads a
case dict (no solve, no CSDL graph), builds the Setup, checks constraint
bookkeeping.
"""
import os
import unittest

import numpy as np

from BladeAD.optimisation.case import load_case_dict
from BladeAD.optimisation.nsga2_runner import RotorNSGA2Setup

_CASES = os.path.join(os.path.dirname(__file__), "..", "..",
                      "BladeAD", "optimisation", "cases")


def _seed_x(case):
    b = case["bounds"]
    mid = lambda k: 0.5 * (b[k][0] + b[k][1])
    return {
        "chord_cps_m": np.full(case["rotor"]["n_chord_cps"], mid("chord_m")),
        "twist_cps_deg": np.full(case["rotor"]["n_twist_cps"], mid("twist_deg")),
        "hover_rpm": mid("hover_rpm"),
        "cruise_rpm": mid("cruise_rpm"),
        "collective_deg": mid("pitch_deg"),
        "oei_rpm": mid("hover_rpm"),
    }


def _fake_r(case, hover_elec):
    """Minimal forward.evaluate-shaped result dict for `_constraints`."""
    return {
        "cl_cap": 0.99,
        "hover_thrust_target": 1000.0, "hover_thrust": 1000.0,
        "cruise_thrust_target": 200.0, "cruise_thrust": 200.0,
        "hover_stall_ratio": 0.9, "cruise_stall_ratio": 0.9,
        "taper": 0.5, "twist_washout_deg": 5.0,
        "hover_torque_margin": 0.8, "cruise_torque_margin": 0.8,
        "hover_electrical_power": hover_elec, "cruise_electrical_power": 9000.0,
    }


class TestHoverElecCap(unittest.TestCase):
    def _scout_case(self):
        return load_case_dict(os.path.join(_CASES, "shahjahan_case1_fpp_noise_scout"))

    def test_scout_case_shape(self):
        case = self._scout_case()
        self.assertEqual(case["constraints"]["max_hover_elec_w"], 25000.0)
        names = [o["name"] for o in case["objectives"]]
        self.assertEqual(set(names), {"cruise_elec", "cruise_noise"})

    def test_cap_absent_by_default(self):
        case = load_case_dict(os.path.join(_CASES, "shahjahan_case1_fpp_noise"))
        case["constraints"].pop("max_hover_elec_w", None)
        setup = RotorNSGA2Setup(case, _seed_x(case), ks_rho=100.0, overshoot_mu=5.0,
                                overshoot_ks_cap=0.5,
                                objective_names=["cruise_noise", "cruise_elec"])
        self.assertNotIn("hover_elec_cap", setup._con_names)

    def test_cap_present_and_violation_sign(self):
        case = self._scout_case()
        setup = RotorNSGA2Setup(case, _seed_x(case), ks_rho=100.0, overshoot_mu=5.0,
                                overshoot_ks_cap=0.5)
        self.assertEqual(setup._con_names.count("hover_elec_cap"), 1)
        self.assertEqual(setup._con_names[-1], "hover_elec_cap")
        self.assertTrue(setup.with_acoustics)

        g = dict(zip(setup._con_names, setup._constraints(_fake_r(case, 20000.0))))
        self.assertAlmostEqual(g["hover_elec_cap"], (20000.0 - 25000.0) / 25000.0)
        self.assertLess(g["hover_elec_cap"], 0.0)                 # satisfied

        g = dict(zip(setup._con_names, setup._constraints(_fake_r(case, 30000.0))))
        self.assertGreater(g["hover_elec_cap"], 0.0)              # violated


if __name__ == "__main__":
    unittest.main()
