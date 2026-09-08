import unittest

import numpy as np

from BladeAD.optimisation.hybrid import DEFAULTS, _dominates, _merge, _pick_anchors
from BladeAD.optimisation.case import ObjectiveSpec


def point(values, rpm):
    return {"feasible": True, "cons_sum": 0.0,
            "perf": {"_raw_obj": list(values)},
            "var_dict": {"chord_cps_m": [0.1], "twist_cps_deg": [1.0],
                         "hover_rpm": [rpm], "cruise_rpm": [1000.0],
                         "collective_deg": [5.0], "oei_rpm": [1000.0]}}


class TestHybrid(unittest.TestCase):
    def test_collapsed_front_pads_extremes(self):
        pop = [point((float(i), float(15 - i)), 1500.0 + i) for i in range(15)]
        anchors = _pick_anchors({"front": [pop[7]], "final_population": pop}, 3,
                                obj_names=["a", "b"])
        self.assertEqual(len(anchors), 3)
        self.assertEqual(len({str(x["var_dict"]) for x in anchors}), 3)
        self.assertGreater(len({x["var_dict"]["hover_rpm"][0] for x in anchors}), 1)

    def test_spread_front_is_normalized(self):
        pop = [point((float(i), float((19 - i) ** 1.5)), 1500.0 + i) for i in range(20)]
        anchors = _pick_anchors({"front": pop, "final_population": pop}, 4,
                                obj_names=["a", "b"])
        xy = np.array([[p["perf"]["_raw_obj"][0], p["perf"]["_raw_obj"][1]] for p in anchors])
        full = np.array([p["perf"]["_raw_obj"] for p in pop])
        xy = (xy - full.min(0)) / np.where(np.ptp(full, 0) == 0, 1, np.ptp(full, 0))
        self.assertGreaterEqual(min(np.linalg.norm(a - b) for i, a in enumerate(xy) for b in xy[i + 1:]), .12)
        self.assertTrue(np.any(xy[:, 0] <= .1))
        self.assertTrue(np.any(xy[:, 0] >= .9))
        self.assertTrue(np.any(xy[:, 1] <= .1))
        self.assertTrue(np.any(xy[:, 1] >= .9))

    def test_goal_aware_merge(self):
        specs = [ObjectiveSpec("hover_elec", "", "min", "cap"),
                 ObjectiveSpec("cruise_elec", "", "min", "cap"),
                 ObjectiveSpec("oei_margin", "", "max", "floor")]
        winner = {"hover_elec": 1, "cruise_elec": 1, "oei_margin": 3}
        others = [{"hover_elec": 2, "cruise_elec": 2, "oei_margin": 2},
                  {"hover_elec": 1, "cruise_elec": 3, "oei_margin": 2}]
        self.assertEqual(_merge([winner] + others, specs), [winner])
        self.assertTrue(_dominates(winner, others[0], specs))

    def test_defaults_include_cross_stack(self):
        self.assertEqual(DEFAULTS["n_anchors"], 3)
        self.assertTrue(DEFAULTS["cross_check"])
        self.assertEqual(DEFAULTS["cross_levels"], [19300.0, 20000.0, 21000.0])


if __name__ == "__main__":
    unittest.main()
