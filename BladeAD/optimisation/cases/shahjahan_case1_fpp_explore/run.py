"""Launch run_case on the exploratory Case 1 variant with acoustics forced OFF.

`sweep_common.BASE_OPTS` is expanded in THIS process before each subprocess
solve is spawned, so patching it here propagates to every solve.
"""
import os, sys, time

import BladeAD.optimisation.sweep_common as _sc
_sc.BASE_OPTS = {"with_acoustics": False}


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

if __name__ == "__main__":
    print(f"BASE_OPTS = {_sc.BASE_OPTS}")
    run_case(os.path.dirname(os.path.abspath(__file__)))
