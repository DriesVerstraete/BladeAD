"""Progress scatter for an NSGA-II rotor run: every generation's objective
cloud, coloured by generation (viridis) -- early gens dark, latest bright, so
the cloud is seen contracting onto the front.

Safe to run against a LIVE run: it only reads, and the per-generation history
write is atomic (`_NSGA2Verbose.write_history` -> tmp + os.replace), with a
retry here as belt-and-braces.

No argparse (workspace style). Call `plot(...)` or run the module:

    conda activate rotor_design
    cd .../999-software/bladead_repo
    python -m BladeAD.optimisation.plot_nsga2_progress

`SOURCE` may be a case dir, a `nsga2_result_<tag>.pkl`, or a history pkl.
`nsga2_runner.run()` also calls `plot(out_pkl, filter_feasible=True)` at the end.
"""
import glob
import os
import pickle
import time

import numpy as np

# --- run switches (module-level, not argparse) ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_nsga2")
FILTER_FEASIBLE = True      # True -> only feasible points; False -> all
FRONT_ONLY = False          # True -> per-generation feasible non-dominated set only
SHOW_EPSILON = True         # overlay the epsilon-constraint edge front
# where to look for the epsilon edge front. Reads a `pareto_front_<a>__<b>_*.pkl`
# if present, else the feasible `stage1_*.pkl` geometries in the dir. Default is
# the CORRECTED (re-seeded) front, not the original truncated sm099 one.
EPSILON_CASE_DIR = os.path.join(_HERE, "cases", "shahjahan_case1_fpp_reseed")
# ------------------------------------------------------------------------

_FEAS_TOL = 1e-6

# objectives with goal="max": the runner stores them NEGATED in `_raw_obj`
# (`nsga2_runner.obj_func`), so flip the sign back for display. The ε-front pkls
# store the true (positive) value, so they need no flip and line up with the
# flipped scatter.
_MAXIMISED = {"oei_margin", "fm_hover", "eta_cruise"}


def _read_retry(path, tries=5, delay=0.5):
    last = None
    for _ in range(tries):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError, OSError) as exc:
            last = exc
            time.sleep(delay)
    raise last


def _gens_from_history(hist_pkl):
    data = _read_retry(hist_pkl)
    history = data["history"] if isinstance(data, dict) else data
    gens, obj_names = [], None
    for g, pop in enumerate(history):
        ro, cv = [], []
        for ind in pop:
            pf = getattr(ind, "performance", None) or {}
            if obj_names is None and pf.get("_obj_names"):
                obj_names = list(pf["_obj_names"])
            ro.append(pf.get("_raw_obj", [np.nan, np.nan]))
            cv.append(float(getattr(ind, "cons_sum", np.nan)))
        gens.append({"gen": g, "raw_obj": np.asarray(ro, float),
                     "cons_sum": np.asarray(cv, float)})
    return gens, (obj_names or ["f0", "f1"])


def _resolve(source):
    """-> (generations, obj_names, tag, out_dir, epsilon_case_dir)."""
    if os.path.isdir(source):
        res = sorted(glob.glob(os.path.join(source, "nsga2_result_*.pkl")),
                     key=os.path.getmtime)
        if res:
            source = res[-1]
        else:
            h = sorted(glob.glob(os.path.join(source, "results",
                                              "optimisation_history_*.pkl")),
                       key=os.path.getmtime)
            if not h:
                raise SystemExit(f"no nsga2_result_*.pkl or history pkl under {source}")
            source = h[-1]

    base = os.path.basename(source)
    if base.startswith("nsga2_result_"):
        r = _read_retry(source)
        tag = r.get("tag", "run")
        out_dir = r.get("case_dir", os.path.dirname(source))
        eps_dir = r.get("epsilon_case_dir")
        if r.get("generations"):
            return r["generations"], r["obj_names"], tag, out_dir, eps_dir
        gens, on = _gens_from_history(r["history_pkl"])
        return gens, r.get("obj_names", on), tag, out_dir, eps_dir

    # a history pkl directly
    gens, on = _gens_from_history(source)
    tag = base.replace("optimisation_history_", "").replace(".pkl", "").split("_")[-1]
    out_dir = os.path.dirname(os.path.dirname(source))
    return gens, on, tag, out_dir, os.path.join(os.path.dirname(out_dir),
                                                "shahjahan_case1_fpp_explore_sm099")


def _nondominated(F):
    keep = []
    for i in range(len(F)):
        if not any(j != i and np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                   for j in range(len(F))):
            keep.append(i)
    return keep


# objective name -> the key it lives under in a solve.py `stage1_*.pkl`
_STAGE1_KEY = {"hover_elec": "hover_electrical_power",
               "cruise_elec": "cruise_electrical_power",
               "fm_hover": "figure_of_merit", "eta_cruise": "cruise_efficiency",
               "oei_margin": "oei_thrust_margin"}


def _epsilon_pts(eps_dir, a, b):
    """Corrected epsilon front points (a, b) -- a `pareto_front_<a>__<b>_*.pkl`
    if present, else the feasible `stage1_*.pkl` geometries in the dir."""
    if not eps_dir or not os.path.isdir(eps_dir):
        return np.empty((0, 2))
    m = (sorted(glob.glob(os.path.join(eps_dir, f"pareto_front_{a}__{b}_*.pkl")))
         or sorted(glob.glob(os.path.join(eps_dir, f"pareto_front_{b}__{a}_*.pkl"))))
    if m:
        d = _read_retry(m[-1])
        return np.array([[float(p[a]), float(p[b])] for p in d["front"]
                         if a in p and b in p], float)
    ka, kb = _STAGE1_KEY.get(a, a), _STAGE1_KEY.get(b, b)
    pts = []
    for f in sorted(glob.glob(os.path.join(eps_dir, "stage1_*.pkl"))):
        try:
            g = _read_retry(f)
        except Exception:
            continue
        if not all((g.get("constraint_checks") or {"_": True}).values()):
            continue
        if ka in g and kb in g:
            pts.append([float(g[ka]), float(g[kb])])
    return np.array(pts, float)


def plot(source=SOURCE, *, filter_feasible=True, front_only=False,
         show_epsilon=True, out_path=None, epsilon_case_dir=None, marker_size=45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens, obj_names, tag, out_dir, eps_dir_stamped = _resolve(source)
    # priority: explicit arg > EPSILON_CASE_DIR (the corrected re-seeded front)
    # > whatever the result pkl stamped (may be the old truncated sm099 front).
    eps_dir = (epsilon_case_dir
               or (EPSILON_CASE_DIR if os.path.isdir(EPSILON_CASE_DIR) else None)
               or eps_dir_stamped)
    a, b = obj_names[0], obj_names[1]
    sa = -1.0 if a in _MAXIMISED else 1.0
    sb = -1.0 if b in _MAXIMISED else 1.0

    G, X, Y = [], [], []
    for gd in gens:
        ro = np.asarray(gd["raw_obj"], float)
        cs = np.asarray(gd["cons_sum"], float)
        if ro.ndim != 2 or ro.shape[0] == 0:
            continue
        feas = cs <= _FEAS_TOL
        idx = np.where(feas)[0] if filter_feasible else np.arange(len(ro))
        if front_only and len(idx):
            fidx = np.where(feas)[0]
            if len(fidx):
                nd = _nondominated(ro[fidx, :2])
                idx = fidx[nd]
            else:
                idx = np.array([], int)
        for k in idx:
            G.append(gd["gen"])
            X.append(sa * ro[k, 0])
            Y.append(sb * ro[k, 1])

    G, X, Y = np.array(G), np.array(X), np.array(Y)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if len(G):
        o = np.argsort(G)                       # latest gens drawn on top
        sc = ax.scatter(X[o], Y[o], c=G[o], cmap="viridis", s=marker_size,
                        alpha=0.75, edgecolors="none")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("generation")
    else:
        ax.text(0.5, 0.5, "no points to plot", ha="center", transform=ax.transAxes)

    if show_epsilon:
        ep = _epsilon_pts(eps_dir, a, b)
        if len(ep):
            oo = np.argsort(ep[:, 0])
            ax.plot(ep[oo, 0], ep[oo, 1], "-o", color="#d62728", ms=4, lw=1.2,
                    label=f"ε-SLSQP edge ({len(ep)})", zorder=5)
            ax.legend(fontsize=8)

    kind = ("feasible" if filter_feasible else "all") + ("-front" if front_only else "")
    ax.set_xlabel(a + " (max)" if sa < 0 else a)
    ax.set_ylabel(b + " (max)" if sb < 0 else b)
    ax.set_title(f"NSGA-II progress ({kind})  —  {os.path.basename(out_dir)}  [{tag}]\n"
                 f"{len(gens)} generations, {len(G)} points")
    fig.tight_layout()

    if out_path is None:
        suffix = "" if (filter_feasible and not front_only) else f"_{kind}"
        out_path = os.path.join(out_dir, f"nsga2_progress_{tag}{suffix}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}  ({len(gens)} gens, {len(G)} pts, kind={kind})")
    return out_path


if __name__ == "__main__":
    plot(SOURCE, filter_feasible=FILTER_FEASIBLE, front_only=FRONT_ONLY,
         show_epsilon=SHOW_EPSILON)
