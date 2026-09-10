"""Unit tests for the generic hybrid scout -> refiner driver.

Pure-function coverage only -- no CSDL, no solve, no network. The end-to-end
`run_hybrid` path is exercised by an overnight run, not here.
"""
import unittest

import numpy as np

from BladeAD.optimisation.hybrid import (
    DEFAULTS, _dominates, _merge, _pick_anchors, _resolve_levels,
)
from BladeAD.optimisation.case import ObjectiveSpec


def _point(values, rpm):
    return {
        "feasible": True,
        "cons_sum": 0.0,
        "perf": {"_raw_obj": list(values)},
        "var_dict": {
            "chord_cps_m": [0.1], "twist_cps_deg": [1.0], "hover_rpm": [rpm],
            "cruise_rpm": [1000.0], "collective_deg": [5.0], "oei_rpm": [1000.0],
        },
    }


class TestPickAnchors(unittest.TestCase):
    def test_collapsed_front_still_spans_the_box(self):
        pop = [_point((float(i), float(15 - i)), 1500.0 + i) for i in range(15)]
        anchors = _pick_anchors({"front": [pop[7]], "final_population": pop}, 3,
                                obj_names=["a", "b"])
        self.assertEqual(len(anchors), 3)
        self.assertEqual(len({str(a["var_dict"]) for a in anchors}), 3)
        self.assertGreater(len({a["var_dict"]["hover_rpm"][0] for a in anchors}), 1)

    def test_spread_front_covers_extremes_and_interior(self):
        pop = [_point((float(i), float((19 - i) ** 1.5)), 1500.0 + i) for i in range(20)]
        anchors = _pick_anchors({"front": pop, "final_population": pop}, 4,
                                obj_names=["a", "b"])
        self.assertEqual(len(anchors), 4)
        full = np.array([p["perf"]["_raw_obj"] for p in pop], float)
        lo, ptp = full.min(0), np.where(np.ptp(full, 0) == 0, 1, np.ptp(full, 0))
        xy = (np.array([a["perf"]["_raw_obj"] for a in anchors], float) - lo) / ptp
        pair_min = min(np.linalg.norm(xy[i] - xy[j])
                       for i in range(4) for j in range(i + 1, 4))
        self.assertGreaterEqual(pair_min, 0.1)
        self.assertTrue(np.any(xy[:, 0] <= 0.1) and np.any(xy[:, 0] >= 0.9))
        self.assertTrue(np.any(xy[:, 1] <= 0.1) and np.any(xy[:, 1] >= 0.9))


class TestMerge(unittest.TestCase):
    def _specs(self):
        return [ObjectiveSpec("hover_elec", "", "min", "cap"),
                ObjectiveSpec("cruise_elec", "", "min", "cap"),
                ObjectiveSpec("third", "", "max", "floor")]

    def test_goal_aware_single_dominator(self):
        specs = self._specs()
        winner = {"hover_elec": 1, "cruise_elec": 1, "third": 3}
        others = [{"hover_elec": 2, "cruise_elec": 2, "third": 2},
                  {"hover_elec": 1, "cruise_elec": 3, "third": 2}]
        self.assertEqual(_merge([winner] + others, specs), [winner])
        self.assertTrue(_dominates(winner, others[0], specs))
        self.assertFalse(_dominates(others[0], winner, specs))

    def test_mutually_non_dominated_both_kept(self):
        specs = self._specs()
        a = {"hover_elec": 1, "cruise_elec": 3, "third": 2}
        b = {"hover_elec": 3, "cruise_elec": 1, "third": 2}
        self.assertEqual(len(_merge([a, b], specs)), 2)


class TestConfig(unittest.TestCase):
    def test_defaults_are_generic(self):
        self.assertIs(DEFAULTS["with_acoustics"], False)
        self.assertIs(DEFAULTS["cross_check"], True)
        self.assertEqual(DEFAULTS["levels"], {})
        self.assertIn("scout_objectives", DEFAULTS)
        self.assertEqual(DEFAULTS["scout_workers"], 1)   # serial unless a case opts in
        self.assertNotIn("oei_levels", DEFAULTS)

    def test_resolve_levels_explicit_is_passthrough(self):
        non_proxy = [ObjectiveSpec("x", "", "min", "cap"),
                     ObjectiveSpec("y", "", "max", "floor")]
        cfg = dict(DEFAULTS)
        cfg["levels"] = {"x": [1.0, 2.0, 3.0], "y": [10, 20]}
        levels = _resolve_levels(ctx=None, non_proxy=non_proxy, cfg=cfg,
                                 seed_pkl=None, proxy_data=None)
        self.assertEqual(levels, {"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0]})


if __name__ == "__main__":
    unittest.main()
