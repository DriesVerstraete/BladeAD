"""Re-trace the epsilon-constraint `hover_elec__cruise_elec` edge SEEDED FROM
the NSGA-II front geometry, to test whether the original epsilon front is
truncated by a locally-converged `cruise_elec` anchor.

The NSGA-II run1 converged (collapsed) design sits at cruise electrical power
~8311 W -- ~10% below the original epsilon edge front's minimum (~9278 W) --
and is fully feasible on the epsilon front's own constraint set (thrust exact,
stall <= 0.99, taper, motor torque, OEI margin 1.385 >= 1.143 floor). If SLSQP
seeded from that geometry lands in the same low-cruise-power basin, the original
epsilon front is incomplete (its anchor SLSQP solve was stuck).

No argparse (workspace style). Run:

    conda activate rotor_design
    cd .../999-software/bladead_repo
    python -u -m BladeAD.optimisation.reseed_epsilon_check \
      > BladeAD/optimisation/cases/shahjahan_case1_fpp_reseed/reseed.log 2>&1

Each `solve.run_case_dir` call runs SLSQP with hard thrust equalities (the
epsilon method's normal behaviour) + the OEI floor + stall/taper/torque, so the
points are directly comparable to the original edge front.
"""
import glob
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np


class _Stamp:
    """`[HH:MM:SS]` prefix per line -- the format of `run_case.log`."""

    def __init__(self, s):
        self.s = s
        self._bol = True

    def write(self, t):
        for c in t.splitlines(keepends=True):
            if self._bol and c.strip():
                self.s.write(time.strftime("[%H:%M:%S] "))
            self.s.write(c)
            self._bol = c.endswith("\n")
        return len(t)

    def flush(self):
        self.s.flush()

_HERE = os.path.dirname(os.path.abspath(__file__))
NSGA2_RESULT = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_nsga2", "nsga2_result_run1.pkl")
CASE_DIR = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_reseed")
ORIG_EPS_DIR = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_explore_sm099")

# hover_elec caps to trace (None = no cap: the pure min-cruise-power anchor).
# The original edge front covers ~19121-19306; go well past it toward the
# NSGA-II design's ~20450.
HOVER_ELEC_CAPS = [None, 21500.0, 21000.0, 20500.0, 20000.0, 19500.0, 19300.0, 19100.0, 18900.0]
OEI_RPM_SEED = 1800.0        # torque-feasible OEI point for the NSGA-II geometry


def _pick_geometry(nsga2_result):
    """Lowest-cruise-power feasible individual from the NSGA-II result."""
    r = pickle.load(open(nsga2_result, "rb"))
    feas = [x for x in r["final_population"] if x["feasible"]] or r["front"]
    best = min(feas, key=lambda x: x["raw_obj"][1])          # raw_obj = [He, Ce]
    vd = best["var_dict"]
    return {
        "chord_cps": np.asarray(vd["chord_cps_m"], float),
        "twist_cps": np.deg2rad(np.asarray(vd["twist_cps_deg"], float)),   # solve.py wants radians
        "rpm": float(np.ravel(vd["hover_rpm"])[0]),
        "cruise_rpm": float(np.ravel(vd["cruise_rpm"])[0]),
        "pitch_deg": float(np.ravel(vd["collective_deg"])[0]),             # FPP shared collective
        "cruise_pitch_deg": 0.0,
        "oei_rpm": OEI_RPM_SEED,
        "_from": os.path.basename(nsga2_result), "_He": best["raw_obj"][0], "_Ce": best["raw_obj"][1],
    }


def _orig_edge_points():
    m = (sorted(glob.glob(os.path.join(ORIG_EPS_DIR, "pareto_front_hover_elec__cruise_elec_*.pkl")))
         or sorted(glob.glob(os.path.join(ORIG_EPS_DIR, "pareto_front_cruise_elec__hover_elec_*.pkl"))))
    if not m:
        return []
    d = pickle.load(open(m[-1], "rb"))
    return [(p["hover_elec"], p["cruise_elec"]) for p in d["front"]
            if "hover_elec" in p and "cruise_elec" in p]


def _run_point(seed_path, cap):
    """One SLSQP solve in its OWN subprocess (like `sweep_common.run_solve`) so
    the BEM inner-solver output is captured and discarded -- the log stays as
    clean as `run_case.log`."""
    opts = dict(initial_result_from=seed_path, with_acoustics=False,
                epsilons=({} if cap is None else {"hover_elec": float(cap)}))
    code = ("from BladeAD.optimisation import solve; "
            f"solve.run_case_dir(r'{CASE_DIR}', **{opts!r})")
    print(f"  $ solve({opts})")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], cwd=CASE_DIR,
                       capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise RuntimeError(f"solve failed (exit {p.returncode}, {dt:.0f}s)")
    saved = [ln for ln in p.stdout.splitlines() if ln.startswith("Saved to ")]
    if not saved:
        print(p.stdout[-3000:])
        raise RuntimeError(f"no 'Saved to' line ({dt:.0f}s)")
    pkl_path = saved[-1][len("Saved to "):].strip()
    res = pickle.load(open(pkl_path, "rb"))
    ok = all(res["constraint_checks"].values())
    bad = [k for k, v in res["constraint_checks"].items() if not v]
    print(f"  -> {os.path.basename(pkl_path)}  ({dt:.0f}s"
          f"{', FAILED: ' + ', '.join(bad) if bad else ''})")
    return {
        "cap": cap,
        "hover_elec": res["hover_electrical_power"],
        "cruise_elec": res["cruise_electrical_power"],
        "hover_thrust": res["hover_thrust"], "cruise_thrust": res["cruise_thrust"],
        "oei_thrust_margin": res.get("oei_thrust_margin"),
        "hover_stall_ratio": res["hover_stall_ratio"],
        "feasible": ok, "failed_checks": bad,
    }


def main():
    os.makedirs(CASE_DIR, exist_ok=True)
    geom = _pick_geometry(NSGA2_RESULT)
    seed_path = os.path.join(CASE_DIR, "nsga2_reseed_geom.pkl")
    with open(seed_path, "wb") as f:
        pickle.dump(geom, f)
    print(f"seed geometry from {geom['_from']}: NSGA-II He={geom['_He']:.0f} Ce={geom['_Ce']:.0f}")
    print(f"wrote {seed_path}\n")

    orig = _orig_edge_points()
    if orig:
        he = [p[0] for p in orig]
        ce = [p[1] for p in orig]
        print(f"original epsilon edge front: {len(orig)} pts  "
              f"He [{min(he):.0f}, {max(he):.0f}]  Ce [{min(ce):.0f}, {max(ce):.0f}]\n")

    rows = []
    for cap in HOVER_ELEC_CAPS:
        tag = "no cap (min cruise)" if cap is None else f"hover_elec <= {cap:.0f}"
        print(f"--- {tag} ---")
        try:
            row = _run_point(seed_path, cap)
        except Exception as exc:                          # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
            rows.append({"cap": cap, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        flag = "" if row["feasible"] else f"  INFEASIBLE {row['failed_checks']}"
        print(f"  He={row['hover_elec']:.0f}  Ce={row['cruise_elec']:.0f}  "
              f"Th={row['hover_thrust']:.0f} Tc={row['cruise_thrust']:.1f}  "
              f"OEI={row['oei_thrust_margin']:.3f}  hov_stall={row['hover_stall_ratio']:.3f}{flag}\n")

    feas = [r for r in rows if r.get("feasible")]
    out = {
        "seed_geometry": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in geom.items()},
        "original_epsilon_front": orig,
        "reseeded_points": rows,
        "reseeded_cruise_elec_min": (min(r["cruise_elec"] for r in feas) if feas else None),
        "original_cruise_elec_min": (min(p[1] for p in orig) if orig else None),
    }
    js = os.path.join(CASE_DIR, "reseed_epsilon_check.json")
    with open(js, "w") as f:
        json.dump(out, f, indent=2)

    print("=== summary ===")
    if feas and orig:
        rm = out["reseeded_cruise_elec_min"]
        om = out["original_cruise_elec_min"]
        print(f"original epsilon min cruise power : {om:.0f} W")
        print(f"re-seeded  epsilon min cruise power: {rm:.0f} W   ({100 * (rm - om) / om:+.1f}%)")
        if rm < om - 50:
            print("=> ORIGINAL EPSILON FRONT IS TRUNCATED: SLSQP seeded from the NSGA-II "
                  "geometry reaches a lower-cruise-power basin its own anchor missed.")
        else:
            print("=> original epsilon front holds: the NSGA-II point does not survive "
                  "SLSQP with hard thrust equalities + the full constraint set.")
    print(f"\nSaved {js}")


if __name__ == "__main__":
    sys.stdout = _Stamp(sys.stdout)
    sys.stderr = _Stamp(sys.stderr)
    main()
