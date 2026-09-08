"""NSGA-II driver for Shahjahan Case 1 FPP -- reads the `nsga2` block from
`case.py` and calls `nsga2_runner.run`. Per-line HH:MM:SS timestamps; point the
Bash launch at a log file with `> log.txt 2>&1` (never a `| tail` pipe -- this
project's solves silently auto-background, see the project CLAUDE.md).

    conda run -n rotor_design python -u \
      999-software/bladead_repo/BladeAD/optimisation/cases/shahjahan_case1_fpp_nsga2_oeiH/run.py \
      > 999-software/bladead_repo/BladeAD/optimisation/cases/shahjahan_case1_fpp_nsga2_oeiH/run.log 2>&1
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Stamp:
    def __init__(self, s):
        self.s = s
        self._bol = True

    def write(self, t):
        for c in t.splitlines(keepends=True):
            if self._bol and c.strip():
                self.s.write(time.strftime("[%H:%M:%S] "))
            self.s.write(c)
            self._bol = c.endswith("\n")
        return len(t)

    def flush(self):
        self.s.flush()


sys.stdout = _Stamp(sys.stdout)
sys.stderr = _Stamp(sys.stderr)

from BladeAD.optimisation.case import load_case_dict
from BladeAD.optimisation import nsga2_runner

if __name__ == "__main__":
    opts = dict(load_case_dict(_HERE).get("nsga2", {}))
    print(f"nsga2 opts = {opts}")
    nsga2_runner.run(_HERE, **opts)
