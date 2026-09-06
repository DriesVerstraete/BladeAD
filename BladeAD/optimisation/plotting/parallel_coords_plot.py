"""Parallel-coordinates plot of the blade design variables across the three
pairwise Pareto sub-fronts -- Shahjahan et al. (2024) Fig. 21 style: one
independently-scaled vertical axis per design variable, ticks on each axis, no
grid, polylines coloured by group, legend below.

Groups here are the three C(3,2) pairwise EDGE sub-fronts (not the interior grid,
not the anchors), so the plot answers: do the FM-eta, FM-noise and eta-noise
edges explore distinct blade families, or the same one?

Config only (no CLI, per project convention). Reads the latest edge pkls in
`CASE_DIR`; each front point's `stage1_*.pkl` carries the design vector.

    /Users/.../miniconda3/envs/rotor_design/bin/python parallel_coords_plot.py

AXIS_SOURCE:
  "cps"     -- the raw Bezier control points the optimiser drives (5 chord + 5
               twist) plus the 4 scalar DVs.
  "profile" -- the evaluated chord / twist distribution sampled at R_STATIONS
               (closer to the paper's figure; physically readable).
"""
import glob
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from ._style import ps  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# config  (no argparse -- edit here; $ROTOR_PARETO_CASE_DIR overrides CASE_DIR)
# --------------------------------------------------------------------------
CASE_DIR = os.environ.get(
    "ROTOR_PARETO_CASE_DIR",
    os.path.abspath(os.path.join(_HERE, "..", "cases", "shahjahan_test")))
FIG_DIR = CASE_DIR                       # write the figure next to the case

AXIS_SOURCE = "cps"                      # "cps" | "profile"
R_STATIONS = (0.15, 0.30, 0.50, 0.75, 1.00)   # used when AXIS_SOURCE == "profile"

# label -> (edge-pkl glob, colour). The three pairwise sub-fronts.
GROUPS = {
    r"FM$-\eta$": ("pareto_front_fm_hover__eta_cruise_*.pkl", ps.myblue),
    r"FM$-$noise": ("pareto_front_fm_hover__hover_noise_*.pkl", ps.myred),
    r"$\eta-$noise": ("pareto_front_eta_cruise__hover_noise_*.pkl", ps.myyellow),
}

LINE_ALPHA = 0.55
N_TICKS = 5                              # value labels per axis


# --------------------------------------------------------------------------
# design-vector extraction
# --------------------------------------------------------------------------
def _axis_defs():
    """List of (extractor(stage1_data) -> float, axis label)."""
    defs = []
    if AXIS_SOURCE == "cps":
        for i in range(5):
            defs.append((lambda d, i=i: float(d["chord_cps"][i]), rf"$c_{i + 1}$"))
        for i in range(5):
            defs.append((lambda d, i=i: np.degrees(float(d["twist_cps"][i])),
                         rf"$\theta_{i + 1}$ [deg]"))
    elif AXIS_SOURCE == "profile":
        for rr in R_STATIONS:
            defs.append((lambda d, rr=rr: float(np.interp(
                rr, d["radial_stations_r_over_R"], d["chord_profile"])),
                f"$c$\n{rr:.2f}"))
        for rr in R_STATIONS:
            defs.append((lambda d, rr=rr: np.degrees(float(np.interp(
                rr, d["radial_stations_r_over_R"], d["twist_profile"]))),
                f"$\\theta$ [deg]\n{rr:.2f}"))
    else:
        raise ValueError(f"AXIS_SOURCE must be 'cps' or 'profile', got {AXIS_SOURCE!r}")
    defs += [
        (lambda d: float(d["rpm"]), r"RPM$_h$"),
        (lambda d: float(d["cruise_rpm"]), r"RPM$_c$"),
        (lambda d: float(d["hover_pitch_deg"]), r"coll$_h$ [deg]"),
        (lambda d: float(d["cruise_pitch_deg"]), r"coll$_c$ [deg]"),
    ]
    return defs


def _tick_decimals(lo, hi):
    """Fewest sig-figs that render lo and hi as different strings (2..6)."""
    for dec in range(2, 7):
        if f"{lo:.{dec}g}" != f"{hi:.{dec}g}":
            return dec
    return 6


def _fmt(val, dec):
    """Plain integer for RPM-scale values, else `dec` sig-figs."""
    if 100.0 <= abs(val) < 1e5:
        return f"{val:.0f}"
    return f"{val:.{dec}g}"


def _load_group(pattern, axis_defs):
    matches = sorted(glob.glob(os.path.join(CASE_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"no edge pkl matching {pattern} in {CASE_DIR}")
    front = pickle.load(open(matches[-1], "rb"))["front"]
    rows = []
    for pt in front:
        sp = os.path.join(CASE_DIR, os.path.basename(pt["pkl_path"]))
        if not os.path.exists(sp):
            print(f"  skip (missing stage1 pkl): {os.path.basename(sp)}")
            continue
        d = pickle.load(open(sp, "rb"))
        rows.append([f(d) for f, _ in axis_defs])
    return np.asarray(rows, dtype=float), os.path.basename(matches[-1])


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def render():
    axis_defs = _axis_defs()
    n = len(axis_defs)

    series = {}
    for label, (pattern, colour) in GROUPS.items():
        X, src = _load_group(pattern, axis_defs)
        series[label] = (X, colour)
        print(f"{label}: {len(X)} points from {src}")

    allrows = np.vstack([X for X, _ in series.values()])
    lo, hi = allrows.min(axis=0), allrows.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    def norm(M):
        return (M - lo) / span

    fig, ax = plt.subplots(figsize=(1.02 * n, 4.4))
    xs = np.arange(n)

    for label, (X, colour) in series.items():
        Y = norm(X)
        for row in Y:
            ax.plot(xs, row, color=colour, lw=ps.line_width_soft * 0.5,
                    alpha=LINE_ALPHA, solid_capstyle="round")
        ax.plot([], [], color=colour, lw=ps.line_width, label=label)  # legend proxy

    for i, (_, lab) in enumerate(axis_defs):
        ax.axvline(i, color=ps.myblack, lw=ps.axes_linewidth * 0.55, zorder=1)
        mid = 0.5 * (lo[i] + hi[i])
        if hi[i] - lo[i] < 1e-3 * max(abs(mid), 1e-9):     # pinned axis -> one label
            ax.text(i - 0.05, 0.5, f"{mid:.4g}*", ha="right", va="center",
                    fontsize=7, color=ps.mygray)
            continue
        dec = _tick_decimals(lo[i], hi[i])
        for frac in np.linspace(0.0, 1.0, N_TICKS):
            val = lo[i] + frac * span[i]
            ax.text(i - 0.05, frac, _fmt(val, dec), ha="right", va="center",
                    fontsize=7, color=ps.myblack)

    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in axis_defs], fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(-0.75, n - 0.35)
    ax.set_ylim(-0.06, 1.10)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=len(series), frameon=False, handlelength=1.6)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = os.path.join(FIG_DIR, f"parallel_coords_edges_{AXIS_SOURCE}_shahjahan_{tag}.png")
    fig.savefig(out, dpi=ps.image_resolution, bbox_inches="tight")
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    render()
