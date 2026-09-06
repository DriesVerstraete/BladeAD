"""Generate NeuralFoil CL/CD/mach_crit/analysis_confidence tables for the
Shahjahan proprotor's real 4-section airfoil set (MH126/MH113/MH115/MH121, per
shahjahan-rotor/spec-shahjahan-propulsion-unit.md section 8): 15 Re nodes
(30e3..9e6), alpha +/-180 deg at 0.5 deg step, 360-degree post-stall blend on.
`mach_crit` is NeuralFoil's incompressible-Cpmin-derived critical Mach (alpha/Re
dependent) -- stored here so the local Mach-correction layer never re-runs
NeuralFoil. CL/CD are incompressible (mach=0); compressibility is applied
downstream. No real XFOIL ground truth exists for these four locally --
training-table generation only, not an accuracy check. `model_size="xxxlarge"`
(NeuralFoil's largest net) -- this runs once per airfoil offline (~0.2 s/Re node,
seconds total), so there is no reason to use a smaller, less accurate net.
Regenerate (deterministic) with the spl-bricks env's python; the CSVs alongside
are the checked-in database the Shahjahan airfoil model reads:
    /opt/anaconda3/envs/spl-bricks/bin/python build_neuralfoil_tables.py
"""
import csv
import os

import aerosandbox as asb
import numpy as onp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# root -> tip. Loaded from the committed .dat files in dat/, NOT asb's name-based
# database -- asb's built-in "mh126" resolves to a different airfoil (t/c 17.9%,
# not this project's 25.0% root section); confirmed by direct comparison, 2026-09-05.
AIRFOILS = {
    "MH126": f"{THIS_DIR}/dat/mh126.dat",
    "MH113": f"{THIS_DIR}/dat/mh113.dat",
    "MH115": f"{THIS_DIR}/dat/mh115.dat",
    "MH121": f"{THIS_DIR}/dat/mh121.dat",
}
REYNOLDS_NODES = (30e3, 50e3, 75e3, 100e3, 150e3, 200e3, 300e3, 500e3, 750e3, 1e6, 1.5e6, 2e6, 3e6, 6e6, 9e6)
# widened from the MH117-matching (150e3..6e6) range, 2026-09-05 -- a full-rotor hover+cruise BEM
# sweep hit stations outside that range (_BSplineSurface hard-fails outside its Re domain, no
# clamp like it has for alpha); this range covers realistic low-Re root and high-Re tip stations.
ALPHA_DEG = onp.arange(-180.0, 180.0 + 0.25, 0.5)
# widened from -90/90, 2026-09-05 -- an acoustic-only optimization trial drove a blade station
# past 90 deg (_prepare() hard-fails above the table's max alpha, unlike the soft clamp below
# the low end); +/-180 matches Pass 3's own verified NeuralFoil range.
OUT_DIR = THIS_DIR                                            # tables live next to this generator


def main():
    for name, dat_path in AIRFOILS.items():
        airfoil = asb.Airfoil(name, coordinates=dat_path)
        rows = []
        for reynolds in REYNOLDS_NODES:
            result = airfoil.get_aero_from_neuralfoil(
                alpha=ALPHA_DEG, Re=reynolds, mach=0.0, model_size="xxxlarge",
                include_360_deg_effects=True,
            )
            for i, alpha in enumerate(ALPHA_DEG):
                rows.append((
                    reynolds, float(alpha), float(result["CL"][i]), float(result["CD"][i]),
                    float(result["mach_crit"][i]), float(result["analysis_confidence"][i]),
                ))
        out_path = f"{OUT_DIR}/{name}_neuralfoil_table.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("reynolds", "alpha", "CL", "CD", "mach_crit", "analysis_confidence"))
            writer.writerows(rows)
        print(f"{name}: wrote {len(rows)} rows to {out_path}  t/c={airfoil.max_thickness():.4f}")


if __name__ == "__main__":
    main()
