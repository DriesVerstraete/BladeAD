"""Locate the SPL Gulliver plot style if it is available on this machine,
otherwise fall back to plain matplotlib with a small brand-neutral palette.

Set ``$SPL_PLOT_STYLE`` to the directory that holds ``plot_settings.py`` plus
the Gulliver TTFs (the fonts are intentionally not vendored into this repo).
Import ``ps`` from here and use ``ps.myblue`` etc.; every attribute the
plotting scripts touch has a fallback so they run with or without the style.
"""
import os
import sys


class _Fallback:
    myblack = (0.0, 0.0, 0.0)
    myblue = "#0F95D7"
    myred = "#e41a1c"
    mygreen = "#4daf4a"
    myyellow = (1.0, 194 / 255, 10 / 255)
    mygray = "#7F7F7F"
    mydarkblue = "#377eb8"
    myorange = "#FF7F0E"
    mybrown = "#8C564B"
    mypurple = "#9467BD"
    colors = [myblack, myblue, myred, myyellow, mygray, mydarkblue, myorange,
              mybrown, mygreen, mypurple, mygray]
    markers = ["o", "s", "^", "p", "v", "*", "x", "D", "H"]
    line_width = 2.25
    line_width_soft = 2.0
    marker_size = "5"
    fontsize = 12
    fontsize_legend = 10
    axes_linewidth = 1.75
    alpha = 1.0
    image_resolution = 300
    shade_color = mygray


def _resolve():
    d = os.environ.get("SPL_PLOT_STYLE")
    if not d or not os.path.isdir(d):
        return None
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import plot_settings as mod  # noqa: F401  (applies rcParams on import)
        return mod
    except Exception:
        return None


ps = _resolve() or _Fallback()
STYLE_ACTIVE = not isinstance(ps, _Fallback)
