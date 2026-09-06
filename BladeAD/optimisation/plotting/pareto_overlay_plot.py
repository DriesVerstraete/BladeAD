"""Reusable 3-objective-vs-pairwise Pareto overlay figure (FM / cruise
efficiency / hover noise), styled per the SPL matplotlib guide.

Four panels: the 3-D front cloud, then the three pairwise projections of that
cloud (grey) each with the matching dedicated 2-objective front (blue line +
markers) drawn on top. A visible gap between the projected cloud and the
dedicated front on a face = the N-objective sweep is under-resolving that edge.

Reusable by pointing at pkl files -- no CLI args, per the workspace coding
style. Edit the CONFIG dict at the bottom (or import `render` and pass your
own dict). CONFIG shape:

    {
      "fronts": {
        "three_obj":  <pkl path>,          # required; run_sweep_n output
        "fm_eta":     <pkl path> | None,    # optional dedicated 2-obj fronts
        "fm_noise":   <pkl path> | None,
        "eta_noise":  <pkl path> | None,
      },
      # (x_key, y_key) into each front pkl's per-point dict. Defaults below
      # match both this project's shahjahan sweeps and the front-prop sweeps.
      "keys": {
        "three_obj": ("fm_hover", "eta_cruise", "hover_noise_db"),
        "fm_eta":    ("fm_hover", "eta_cruise"),
        "fm_noise":  ("fm_hover", "hover_noise_db"),
        "eta_noise": ("eta_cruise", "hover_noise_db"),
      },
      "out_path": <png path>,
      "title": <str>,
    }

Run with the `rotor_design` env's python.
"""
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from ._style import ps  # noqa: E402

_SENTINEL = "INFEASIBLE_SUBPROBLEM"
FM, ETA, NOISE = 0, 1, 2
_AXIS_LABEL = {FM: "hover FM", ETA: "cruise efficiency", NOISE: "hover noise (dB)"}

_DEFAULT_KEYS = {
    "three_obj": ("fm_hover", "eta_cruise", "hover_noise_db"),
    "fm_eta": ("fm_hover", "eta_cruise"),
    "fm_noise": ("fm_hover", "hover_noise_db"),
    "eta_noise": ("eta_cruise", "hover_noise_db"),
}
# front-prop sweep pkls use different key names for the FM-vs-eta 2-obj front
_KEY_ALIASES = {"fm_hover": ("fm_hover", "obj1_fm_hover", "figure_of_merit"),
                "eta_cruise": ("eta_cruise", "obj2_eta_cruise", "cruise_efficiency"),
                "hover_noise_db": ("hover_noise_db", "hover_total_spl_db")}


def _get(point, key):
    for candidate in _KEY_ALIASES.get(key, (key,)):
        if candidate in point:
            return point[candidate]
    raise KeyError(f"none of {_KEY_ALIASES.get(key, (key,))} in front point {list(point)}")


def _real_points(path):
    front = pickle.load(open(path, "rb"))["front"]
    return [p for p in front if p.get("pkl_path") != _SENTINEL]


def _load_cloud(path, keys):
    pts = _real_points(path)
    return np.array([[_get(p, keys[0]), _get(p, keys[1]), _get(p, keys[2])] for p in pts])


def _load_2obj(path, keys):
    if not path or not os.path.exists(path):
        return None
    pts = _real_points(path)
    if not pts:
        return None
    arr = np.array([[_get(p, keys[0]), _get(p, keys[1])] for p in pts])
    return arr[np.argsort(arr[:, 0])]


def _grid(ax):
    ax.minorticks_on()
    ax.grid(which="major", alpha=1.0, linewidth=0.6)
    ax.grid(which="minor", alpha=0.5, linewidth=0.4)


def _face(ax, cloud, ai, aj, front2d, title):
    ax.scatter(cloud[:, ai], cloud[:, aj], s=42, color=ps.mygray, edgecolor=ps.myblack,
               linewidth=0.4, zorder=2, label="3-obj front (projected)")
    if front2d is not None and len(front2d) >= 2 and np.ptp(front2d[:, 0]) > 1e-6:
        ax.plot(front2d[:, 0], front2d[:, 1], "-", marker=ps.markers[0],
                color=ps.myblue, ms=float(ps.marker_size), lw=ps.line_width_soft,
                zorder=3, label="dedicated 2-obj front")
    ax.set_xlabel(_AXIS_LABEL[ai])
    ax.set_ylabel(_AXIS_LABEL[aj])
    ax.set_title(title)
    _grid(ax)
    ax.legend(fontsize=ps.fontsize_legend, loc="best")


def _render_slots(config):
    """Objective-slot-driven overlay (rotor_pareto_slots): labels, goal
    directions and the C(N,2) panel set come from `config['objectives']` (a list
    of {name, label, goal}); the front pkls store each point's objective values
    keyed by slot name. 3-D scatter panel only for N == 3."""
    import itertools

    objs = config["objectives"]
    names = [o["name"] for o in objs]
    label = {o["name"]: o.get("label", o["name"]) for o in objs}
    n = len(names)

    def _cloud(path):
        return np.array([[p[nm] for nm in names] for p in _real_points(path)])

    def _edge(path, ai, aj):
        if not path or not os.path.exists(path):
            return None
        pts = _real_points(path)
        if not pts:
            return None
        arr = np.array([[p[names[ai]], p[names[aj]]] for p in pts])
        return arr[np.argsort(arr[:, 0])]

    cloud = _cloud(config["fronts"]["interior"])
    edges = config["fronts"].get("edges", {})
    pairs = list(itertools.combinations(range(n), 2))
    n_panels = len(pairs) + (1 if n == 3 else 0)
    ncol = 2 if n_panels <= 4 else 3
    nrow = -(-n_panels // ncol)
    fig = plt.figure(figsize=(6 * ncol, 5 * nrow))
    slot = 1

    if n == 3:
        ax3d = fig.add_subplot(nrow, ncol, slot, projection="3d"); slot += 1
        ax3d.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=42, color=ps.myblue,
                     edgecolor=ps.myblack, linewidth=0.4)
        ax3d.set_xlabel(label[names[0]]); ax3d.set_ylabel(label[names[1]])
        ax3d.set_zlabel(label[names[2]])
        ax3d.set_title(f"interior front ({len(cloud)} pts)")

    # For N == 3, order the pairwise panels so the 2x2 grid reads as a matrix:
    #   [ 3D            ] [ (0,1)  x=obj0 ]
    #   [ (1,2) y=obj2  ] [ (0,2)  x=obj0, y=obj2 ]
    # -> right column shares the x-axis (obj0), bottom row shares the y-axis (obj2).
    panel_pairs = [(0, 1), (1, 2), (0, 2)] if n == 3 else pairs
    ax_by_pair = {}

    for ai, aj in panel_pairs:
        kw = {}
        if n == 3 and (ai, aj) == (0, 2):
            kw = {"sharex": ax_by_pair[(0, 1)], "sharey": ax_by_pair[(1, 2)]}
        ax = fig.add_subplot(nrow, ncol, slot, **kw); slot += 1
        ax_by_pair[(ai, aj)] = ax
        ax.scatter(cloud[:, ai], cloud[:, aj], s=42, color=ps.mygray, edgecolor=ps.myblack,
                   linewidth=0.4, zorder=2, label="interior front (projected)")
        key = f"{names[ai]}__{names[aj]}"
        alt = f"{names[aj]}__{names[ai]}"
        e = _edge(edges.get(key) or edges.get(alt), ai, aj)
        if e is not None and len(e) >= 2 and np.ptp(e[:, 0]) > 1e-6:
            ax.plot(e[:, 0], e[:, 1], "-", marker=ps.markers[0], color=ps.myblue,
                    ms=float(ps.marker_size), lw=ps.line_width_soft, zorder=3,
                    label="dedicated 2-obj edge")
        ax.set_xlabel(label[names[ai]]); ax.set_ylabel(label[names[aj]])
        ax.set_title(f"{label[names[ai]]} vs {label[names[aj]]}")
        _grid(ax)
        ax.legend(fontsize=ps.fontsize_legend, loc="best")
        # shared axes still keep their own labels + ticks (matches the styled
        # reference figure); re-enable the tick labels sharex/sharey suppressed.
        ax.tick_params(labelbottom=True, labelleft=True)

    fig.suptitle(config["title"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = config["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=ps.image_resolution)
    plt.close(fig)
    print(f"Saved {out_path}  ({len(cloud)} interior pts, {n} objectives)")
    return out_path


def render(config):
    if config.get("objectives"):
        return _render_slots(config)
    keys = {**_DEFAULT_KEYS, **config.get("keys", {})}
    fronts = config["fronts"]
    cloud = _load_cloud(fronts["three_obj"], keys["three_obj"])

    fig = plt.figure(figsize=(12, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(cloud[:, FM], cloud[:, ETA], cloud[:, NOISE], s=42, color=ps.myblue,
                 edgecolor=ps.myblack, linewidth=0.4)
    ax3d.set_xlabel(_AXIS_LABEL[FM])
    ax3d.set_ylabel(_AXIS_LABEL[ETA])
    ax3d.set_zlabel(_AXIS_LABEL[NOISE])
    ax3d.set_title(f"3-objective front ({len(cloud)} pts)")

    # Layout: right column shares x = FM (panels 2 & 4); bottom row shares
    # y = noise (panels 3 & 4). sharex/sharey so ticks line up for comparison.
    ax_fm_eta = fig.add_subplot(2, 2, 2)
    ax_eta_noise = fig.add_subplot(2, 2, 3)
    ax_fm_noise = fig.add_subplot(2, 2, 4, sharex=ax_fm_eta, sharey=ax_eta_noise)
    _face(ax_fm_eta, cloud, FM, ETA,
          _load_2obj(fronts.get("fm_eta"), keys["fm_eta"]), "FM vs cruise efficiency")
    _face(ax_eta_noise, cloud, ETA, NOISE,
          _load_2obj(fronts.get("eta_noise"), keys["eta_noise"]), "cruise efficiency vs noise")
    _face(ax_fm_noise, cloud, FM, NOISE,
          _load_2obj(fronts.get("fm_noise"), keys["fm_noise"]), "FM vs noise")

    fig.suptitle(config["title"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = config["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=ps.image_resolution)
    plt.close(fig)
    print(f"Saved {out_path}  ({len(cloud)} 3-obj pts)")
    return out_path


# `render(config)` is called by `BladeAD.optimisation.plot.render_overlay`; the
# legacy hardcoded `keys=` path is kept for back-compat with the front-prop
# overlay (which lives outside this package and passes its own config).
