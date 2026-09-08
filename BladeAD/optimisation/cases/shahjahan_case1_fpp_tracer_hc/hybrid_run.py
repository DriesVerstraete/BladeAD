"""Run the hybrid scout-to-refiner pipeline.

Direct invocation: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u hybrid_run.py > hybrid.log 2>&1
"""
import os
import sys
import time
from BladeAD.optimisation.hybrid import run_hybrid

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Stamp:
    def __init__(self, stream):
        self.stream = stream
        self.beginning = True

    def write(self, text):
        for line in text.splitlines(keepends=True):
            if self.beginning and line.strip():
                self.stream.write(time.strftime("[%H:%M:%S] "))
            self.stream.write(line)
            self.beginning = line.endswith("\n")
        return len(text)

    def flush(self):
        self.stream.flush()


sys.stdout = _Stamp(sys.stdout)
sys.stderr = _Stamp(sys.stderr)

if __name__ == "__main__":
    run_hybrid(_HERE)
