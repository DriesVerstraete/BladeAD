"""Overlay an NSGA-II raw front against the epsilon-constraint (SLSQP) edge
front for the same objective pair, and tabulate the roadmap-step-4 comparison:
does the front move, does maxiter-thrash drop, cost vs the epsilon grid.

No argparse (workspace style) -- set the module constants below and run:

    conda run -n rotor_design python -m BladeAD.optimisation.compare_nsga2_vs_epsilon

`NSGA2_PKL`        : a `nsga2_result_<tag>.pkl` from `nsga2_runner.run`.
`EPSILON_FRONT`   : a `pareto_front_<a>__<b>_<tag>.pkl` from the epsilon workflow
                    (`edges.run_edge`), or None to auto-glob the newest matching
                    pair in `EPSILON_CASE_DIR`.
`EPSILON_CASE_DIR`: where to look when `EPSILON_FRONT` is None.

Reads NSGA-II objective values from the per-individual `performance` payload
(raw, un-penalised) -- never from the penalised `obj`.
"""
import glob
import json
import os
import pickle

import numpy as np

# --- run switches --------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
NSGA2_PKL = None            # e.g. ".../shahjahan_case1_fpp_nsga2/nsga2_result_XXXX.pkl"
EPSILON_FRONT = None
# the CORRECTED (re-seeded, from run1's NSGA-II geometry) front -- not the
# original truncated sm099 one. Reads a pareto_front_*.pkl or the feasible
# stage1_*.pkl geometries in the dir (see plot_nsga2_progress._epsilon_pts).
EPSILON_CASE_DIR = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_reseed")
OUT_DIR = None              # default: the NSGA-II case dir
# ------------------------------------------------------------------------


def _newest(pattern):
    m = sorted(glob.glob(pattern), key=os.path.getmtime)
    return m[-1] if m else None


def _hv(points, ref):
    """Hypervolume dominated by `points` w.r.t. `ref` -- moocore assumes
    minimisation of all objectives and `ref` an upper bound dominated by all."""
    import moocore
    pts = np.asarray(points, float)
    if len(pts) == 0:
        return 0.0
    return float(moocore.hypervolume(pts, ref=np.asarray(ref, float)))


def _epsilon_front_points(front_pkl, obj_names):
    with open(front_pkl, "rb") as f:
        d = pickle.load(f)
    pts = []
    for p in d["front"]:
        if all(k in p for k in obj_names):
            pts.append([float(p[k]) for k in obj_names])
    return np.array(pts, float), d.get("objective_labels", obj_names)


def _ideal_hover_power(case):
    """Momentum-theory ideal hover shaft power P = T^1.5 / sqrt(2 rho A). Any
    design reporting less than this is a non-physical BEM artefact (clamped-Re
    post-stall) and must be dropped before the front is read."""
    from .case import isa_atmos_density
    r = case["rotor"]
    A = np.pi * (r["radius_m"] ** 2 - r["hub_radius_m"] ** 2)
    rho = isa_atmos_density(case["operating"]["hover"]["altitude_m"])
    T = case["operating"]["hover"]["thrust_n"]
    return float(T ** 1.5 / np.sqrt(2.0 * rho * A))


def _physical(rows, p_ideal):
    """Keep rows whose reported hover power is >= the momentum-theory ideal."""
    out = []
    for r in rows:
        hp = r.get("perf", {}).get("hover_power")
        if hp is None or hp >= p_ideal:
            out.append(r)
    return out


def compare(nsga2_pkl, epsilon_front=None, epsilon_case_dir=None, out_dir=None):
    with open(nsga2_pkl, "rb") as f:
        nz = pickle.load(f)
    obj_names = nz["obj_names"]
    out_dir = out_dir or nz["case_dir"]

    p_ideal = None
    try:
        from .case import load_case_dict
        p_ideal = _ideal_hover_power(load_case_dict(nz["case_dir"]))
    except Exception:
        pass

    n_obj = len(obj_names)
    # Re-derive the front from the FEASIBLE + PHYSICAL final population: a
    # non-physical point (hover power < ideal) can wrongly dominate real points
    # at run time, so `nz["front"]` cannot just be trusted / filtered.
    feas_rows = [r for r in nz["final_population"] if r["feasible"]]
    if p_ideal:
        kept = _physical(feas_rows, p_ideal)
        if len(kept) != len(feas_rows):
            print(f"NOTE: dropped {len(feas_rows) - len(kept)} non-physical feasible "
                  f"individual(s) (hover power < ideal {p_ideal:.0f} W)")
        feas_rows = kept
    _F = np.array([r["raw_obj"] for r in feas_rows], float).reshape(-1, n_obj)
    from .nsga2_runner import _nondominated as _nd
    front_rows = [feas_rows[i] for i in (_nd(_F) if len(_F) else [])]
    nz_front = np.array([r["raw_obj"] for r in front_rows], float).reshape(-1, n_obj)
    nz_all_feas = _F

    a, b = obj_names[:2]
    eps_dir = epsilon_case_dir or EPSILON_CASE_DIR
    if epsilon_front:                                   # explicit pareto_front pkl
        eps_pts, _ = _epsilon_front_points(epsilon_front, (a, b))
    else:                                               # dir: pareto_front_*.pkl or stage1_*.pkl
        from .plot_nsga2_progress import _epsilon_pts
        eps_pts = _epsilon_pts(eps_dir, a, b)
    eps_pts = np.asarray(eps_pts, float).reshape(-1, 2)

    # shared reference point for hypervolume: nadir over both fronts + 10%
    stack = np.vstack([p for p in (nz_front[:, :2], eps_pts) if len(p)])
    if len(stack):
        lo, hi = stack.min(0), stack.max(0)
        ref = hi + 0.1 * (hi - lo + 1e-9)
    else:
        ref = np.array([1.0, 1.0])

    metrics = {
        "nsga2_pkl": nsga2_pkl,
        "epsilon_front_pkl": epsilon_front,
        "epsilon_source": epsilon_front or eps_dir,
        "objectives": obj_names,
        "nsga2": {
            "n_eval": nz["n_eval"], "wall_s": round(nz["wall_s"], 1),
            "wall_min": round(nz["wall_s"] / 60, 2),
            "pop_size": nz["pop_size"], "n_gen": nz["n_gen"],
            "n_final": nz["n_final"], "n_feasible": nz["n_feasible"],
            "n_front": nz["n_front"], "n_failed_final": nz.get("n_failed_final"),
            "front_hv": round(_hv(nz_front[:, :2], ref), 3) if len(nz_front) else 0.0,
            "front_spans": {n: [float(nz_front[:, i].min()), float(nz_front[:, i].max())]
                            for i, n in enumerate(obj_names)} if len(nz_front) else {},
        },
        "epsilon": {
            "n_points": int(len(eps_pts)),
            "front_hv": round(_hv(eps_pts, ref), 3) if len(eps_pts) else 0.0,
            "front_spans": {n: [float(eps_pts[:, i].min()), float(eps_pts[:, i].max())]
                            for i, n in enumerate(obj_names[:2])} if len(eps_pts) else {},
        },
        "hv_reference_point": ref[:2].tolist(),
    }

    tag = nz["tag"]
    base = os.path.join(out_dir, f"nsga2_vs_epsilon_{tag}")
    with open(base + ".json", "w") as f:
        json.dump(metrics, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if len(eps_pts):
        o = np.argsort(eps_pts[:, 0])
        ax.plot(eps_pts[o, 0], eps_pts[o, 1], "-o", color="#1f77b4",
                label=f"epsilon-SLSQP edge ({len(eps_pts)})", zorder=3)
    if len(nz_all_feas):
        ax.scatter(nz_all_feas[:, 0], nz_all_feas[:, 1], s=18, color="#bbb",
                   label=f"NSGA-II feasible pop ({len(nz_all_feas)})", zorder=1)
    if len(nz_front):
        o = np.argsort(nz_front[:, 0])
        ax.plot(nz_front[o, 0], nz_front[o, 1], "-s", color="#d62728",
                label=f"NSGA-II front ({len(nz_front)})", zorder=2)
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title(f"{os.path.basename(nz['case_dir'])}  --  NSGA-II vs epsilon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=130)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved {base}.json\nSaved {base}.png")
    return metrics


if __name__ == "__main__":
    pkl = NSGA2_PKL or _newest(os.path.join(
        _HERE, "cases", "shahjahan_case1_fpp_nsga2", "nsga2_result_*.pkl"))
    if not pkl:
        raise SystemExit("set NSGA2_PKL (no nsga2_result_*.pkl found to auto-pick)")
    compare(pkl, EPSILON_FRONT, EPSILON_CASE_DIR, OUT_DIR)
