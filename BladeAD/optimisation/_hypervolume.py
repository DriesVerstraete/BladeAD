"""Uncrowded Hypervolume Improvement (UHVI) scorer for the N-objective gap
picker in `_sweep.pick_next_target_n`. Vendored 2026-09-07 from AircraftDesign's
`optimisation_framework/optimisation/util/uncrowded_hypervolume.py`.

`moocore` (LGPL-2.1-or-later, small pip wheel: cffi + numpy + platformdirs) is
imported lazily -- only the N>=4 front picker needs it, so the rest of
`BladeAD.optimisation` (the 3-objective epsilon-grid path) imports without it.
`pip install moocore` to enable the N>=4 picker.
"""
import numpy as np


def _dominating_mask(point, archive_points):
    """Boolean mask over `archive_points` rows that weakly-dominate `point`
    (minimisation convention): archive_i <= point in every objective, and
    strictly better in at least one."""
    diff = archive_points - point
    weakly_better = np.all(diff <= 0, axis=1)
    strictly_better = np.any(diff < 0, axis=1)
    return weakly_better & strictly_better


def uncrowded_hypervolume_improvement(point, archive_points, reference_point):
    """Uncrowded Hypervolume Improvement (UHVI), minimisation convention.

    Toure, Hansen, Auger & Brockhoff, "Uncrowded Hypervolume Improvement:
    COMO-CMA-ES and the Sofomore framework", GECCO 2019, Eq. 5:

        UHVI_r(s, S) =  HVI_r(s, S)     if s is not dominated by the
                                          empirical Pareto front of S
                       -d_r(s, S)        if s is dominated by it

    d_r is approximated here as the minimum Euclidean distance from `point` to
    the nearest individual archive point that dominates it (an over-estimate of
    the true infimum boundary distance -- avoids general n-objective staircase
    geometry for a first working version).

    :param point: (n_obj,) objective vector of the candidate being scored.
    :param archive_points: (n_archive, n_obj), may be empty.
    :param reference_point: (n_obj,), must be weakly worse than every point of
        interest in every objective (moocore's hypervolume convention).
    :return: scalar UHVI value (larger is better).
    """
    point = np.asarray(point, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    archive_points = np.asarray(archive_points, dtype=float).reshape(-1, len(point))

    if len(archive_points) > 0:
        mask = _dominating_mask(point, archive_points)
    else:
        mask = np.zeros(0, dtype=bool)

    if np.any(mask):
        dominating = archive_points[mask]
        d = np.min(np.linalg.norm(dominating - point, axis=1))
        return -d

    try:
        import moocore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the N>=4 front picker (pick_next_target_n) needs `moocore`; "
            "`pip install moocore`"
        ) from exc

    if len(archive_points) == 0:
        hv_before = 0.0
    else:
        hv_before = moocore.hypervolume(archive_points, ref=reference_point)

    combined = (np.vstack([archive_points, point[None, :]])
                if len(archive_points) > 0 else point[None, :])
    hv_after = moocore.hypervolume(combined, ref=reference_point)

    return hv_after - hv_before
