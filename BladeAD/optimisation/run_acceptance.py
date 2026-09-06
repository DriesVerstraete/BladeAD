"""Acceptance driver for the objective-slots workflow (no argparse -- edit
STAGE). Run in the `rotor_design` env:

    cd BladeAD/optimisation
    python -u run_acceptance.py > cases/shahjahan_test/run_acceptance.log 2>&1

STAGE:
  "main"      -- full run_case on cases/shahjahan_test (3-obj, grid method)
  "reordered" -- full run_case on cases/shahjahan_test_reordered
  "twoobj"    -- full run_case on cases/shahjahan_test_2obj
  "strlist"   -- full run_case on cases/shahjahan_test_strlist
  "anchors"   -- anchors only, for every variant (cheap-ish structural check)
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

from BladeAD.optimisation.orchestrator import run_case            # noqa: E402
from BladeAD.optimisation.case import Context                     # noqa: E402
from BladeAD.optimisation.sweep_common import objective_anchor    # noqa: E402

STAGE = "main"

_CASES = {
    "main": "shahjahan_test",
    "reordered": "shahjahan_test_reordered",
    "twoobj": "shahjahan_test_2obj",
    "strlist": "shahjahan_test_strlist",
}


def _anchor_summary(case_dir):
    ctx = Context.load(case_dir)
    from BladeAD.optimisation.case import active_objectives
    for s in active_objectives(ctx.case):
        _, d = objective_anchor(ctx, s.name)
        ov = d["objective_values"]
        print(f"  anchor {s.name:14s}: " + "  ".join(f"{k}={v:.4f}" for k, v in ov.items()))


if __name__ == "__main__":
    if STAGE == "anchors":
        for tag, name in _CASES.items():
            cd = os.path.join(_HERE, "cases", name)
            if os.path.isdir(cd):
                print(f"\n=== {tag} ({name}) ===")
                _anchor_summary(cd)
    else:
        cd = os.path.join(_HERE, "cases", _CASES[STAGE])
        run_case(cd, three_obj_method="grid")
        print("\n--- anchor registry ---")
        with open(os.path.join(cd, "_anchor_registry.json")) as f:
            print(json.dumps(json.load(f), indent=2))
        _anchor_summary(cd)
