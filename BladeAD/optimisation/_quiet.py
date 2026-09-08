"""Suppress the BEM inner-solver chatter (`nonlinear solver: bracketed_search
converged ...`, csdl_alpha non-convergence `state 0 / residual` dumps, numpy
RuntimeWarnings) around a BEM eval / SLSQP solve.

`QUIET_SOLVER` is the workspace-style module switch (no argparse): default True
(silence). Flip to False in code to see the solver output when debugging a
convergence problem.

FD-level redirect (`os.dup2`) so C-extension and held-reference writes are
caught too. Real failures raised by the wrapped code still propagate -- only
stdout/stderr text is dropped.
"""
import contextlib
import os
import sys
import warnings

QUIET_SOLVER = True


@contextlib.contextmanager
def muffled(enabled=QUIET_SOLVER):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (devnull, *saved):
            os.close(fd)
