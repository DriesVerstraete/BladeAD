"""Smoke launcher: run_case on the tiny-budget Case 1 smoke variant."""
import os, sys, time


class _Stamp:
    def __init__(self, s): self.s = s; self._bol = True
    def write(self, t):
        for c in t.splitlines(keepends=True):
            if self._bol and c.strip():
                self.s.write(time.strftime("[%H:%M:%S] "))
            self.s.write(c); self._bol = c.endswith("\n")
        return len(t)
    def flush(self): self.s.flush()


sys.stdout = _Stamp(sys.stdout); sys.stderr = _Stamp(sys.stderr)
from BladeAD.optimisation.orchestrator import run_case

HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    run_case(HERE)
