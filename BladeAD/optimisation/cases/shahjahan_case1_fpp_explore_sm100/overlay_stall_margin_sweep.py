"""Throwaway: overlay the 2-objective FPP edge fronts for stall_margin 0.90
(laptop `shahjahan_case1_fpp_explore`) vs 0.99 (desktop
`shahjahan_case1_fpp_explore_sm099`). Read-only on both case dirs.
"""
import glob
import pickle
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPT = ("/Users/dverstraete/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)"
       "/BaseFolder/999-software/bladead_repo/BladeAD/optimisation")
CASES = {
    "sm 0.90 (baseline)": OPT + "/cases/shahjahan_case1_fpp_explore",
    "sm 0.99": OPT + "/cases/shahjahan_case1_fpp_explore_sm099",
    "sm 1.00": OPT + "/cases/shahjahan_case1_fpp_explore_sm100",
    "unconstrained (cl_max 3.0)": OPT + "/cases/shahjahan_case1_fpp_explore_smNONE",
}
COLOR = {"sm 0.90 (baseline)": "#1f77b4", "sm 0.99": "#d62728", "sm 1.00": "#2ca02c",
         "unconstrained (cl_max 3.0)": "#9467bd"}
SENTINEL = "INFEASIBLE_SUBPROBLEM"

PANELS = [
    ("hover_elec", "cruise_elec", "hover_elec__cruise_elec"),
    ("hover_elec", "oei_margin", "hover_elec__oei_margin"),
    ("cruise_elec", "oei_margin", "cruise_elec__oei_margin"),
]
LABEL = {"hover_elec": "hover electrical power (kW)",
         "cruise_elec": "cruise electrical power (kW)",
         "oei_margin": "OEI thrust margin"}


def _front(case_dir, stem):
    hits = sorted(glob.glob(f"{case_dir}/pareto_front_{stem}_*.pkl"))
    if not hits:
        return None
    pts = [p for p in pickle.load(open(hits[-1], "rb"))["front"]
           if p.get("pkl_path") != SENTINEL]
    return pts


def _scale(key, v):
    return v / 1000.0 if key.endswith("_elec") else v


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (xk, yk, stem) in zip(axes, PANELS):
    for name, case_dir in CASES.items():
        pts = _front(case_dir, stem)
        if not pts:
            print(f"  [{name}] {stem}: no front pkl")
            continue
        arr = np.array([[_scale(xk, p[xk]), _scale(yk, p[yk])] for p in pts])
        arr = arr[np.argsort(arr[:, 0])]
        ax.plot(arr[:, 0], arr[:, 1], "-o", color=COLOR[name], ms=5, lw=1.4,
                label=f"{name}  (n={len(arr)})")
    ax.set_xlabel(LABEL[xk])
    ax.set_ylabel(LABEL[yk])
    ax.set_title(f"{LABEL[xk].split(' (')[0]}  vs  {LABEL[yk].split(' (')[0]}")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("FPP Case 1 edge fronts -- hover stall-margin sweep (" + ", ".join(CASES) + ")", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out = CASES["sm 0.99"] + "/pareto_overlay_sm090_vs_sm099.png"
fig.savefig(out, dpi=150)
scratch = ("/private/tmp/claude-502/-Users-dverstraete-Library-CloudStorage-OneDrive-"
           "TheUniversityofSydney-Staff--BaseFolder/2d625801-f33f-4c92-9b03-921188f72668"
           "/scratchpad/pareto_overlay_sm090_vs_sm099.png")
shutil.copy(out, scratch)
print("Saved", out)

# also dump the knee-ish numbers for the report
for name, case_dir in CASES.items():
    print(f"\n=== {name} ===")
    for xk, yk, stem in PANELS:
        pts = _front(case_dir, stem)
        if not pts:
            continue
        print(f"  {stem}:")
        for p in sorted(pts, key=lambda q: q[xk]):
            print(f"    hover={p['hover_elec']/1000:7.3f} kW  "
                  f"cruise={p['cruise_elec']/1000:7.3f} kW  "
                  f"oei={p['oei_margin']:.4f}")
