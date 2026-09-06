"""Overlay figure: the interior N-objective front vs the dedicated pairwise
2-objective edge fronts. Thin wrapper around
`code/rotor_acoustics/pareto_overlay_plot.render`, passing the objective spec so
axis labels, goal directions and the C(N, 2) panel set come from the case rather
than being hardcoded.
"""
import os

from .plotting.pareto_overlay_plot import render

from .case import active_objectives


def render_overlay(ctx, three_obj_pkl, edge_pkls, title=None,
                   out_name="pareto_overlay.png"):
    specs = active_objectives(ctx.case)
    names = [s.name for s in specs]
    edges = {}
    for pair, pkl in edge_pkls.items():
        a, b = sorted(pair, key=names.index)
        edges[f"{a}__{b}"] = pkl
    case_name = os.path.basename(ctx.case_dir)
    cfg = {
        "objectives": [{"name": s.name, "label": s.label, "goal": s.goal} for s in specs],
        "fronts": {"interior": three_obj_pkl, "edges": edges},
        "out_path": ctx.out(out_name),
        "title": title or f"{case_name}: interior N-objective front vs dedicated 2-objective edges",
    }
    return render(cfg)
