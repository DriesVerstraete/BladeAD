"""Run the full case-driven Pareto pipeline.

Workspace style: no argparse. Set CASE_DIR to the case directory, then

    cd code/rotor_pareto
    python -u run_case.py > cases/shahjahan_proprotor/run_case.log 2>&1

Every output line is prefixed with an [HH:MM:SS] wall-clock stamp so a stalled
solve is obvious from the log alone (a solve caps at ~3.6 min at maxiter=60, so
a gap much larger than that after a "$ run(...)" line means it is stuck).

Everything (anchors, edge fronts, the 3-obj front, the overlay plot, the
per-case anchor registry) is written into CASE_DIR.
"""
import os
import sys
import time


class _StampWriter:
    """Line-buffered stdout/stderr wrapper that prefixes each line with the
    local time. Shared across stdout and stderr so interleaved output stays in
    order."""

    def __init__(self, stream):
        self._stream = stream
        self._at_line_start = True

    def write(self, text):
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start and chunk.strip():
                self._stream.write(time.strftime("[%H:%M:%S] "))
            self._stream.write(chunk)
            self._at_line_start = chunk.endswith("\n")
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stdout = _StampWriter(sys.stdout)
sys.stderr = _StampWriter(sys.stderr)

_HERE = os.path.dirname(os.path.abspath(__file__))

from BladeAD.optimisation.orchestrator import run_case   # noqa: E402

# --- edit this line to point at a different case -------------------------------
CASE_DIR = os.path.join(_HERE, "cases", "shahjahan_test")

if __name__ == "__main__":
    run_case(CASE_DIR)
