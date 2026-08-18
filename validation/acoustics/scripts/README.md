# Acoustic validation scripts

Scripts added here must state their required environment and use only paths relative to the
BladeAD repository. RCAIDE baseline generation and BladeAD validation must remain separate
processes because the project uses separate `rcaide` and `rotor_design` environments.

Validation scripts report signed error, mean absolute error, maximum absolute error, and overall
SPL error as applicable. Plotting is supplementary and never replaces numerical assertions.

Generate the initial tool-independent fixtures from the pinned RCAIDE checkout with:

```bash
python -u extract_rcaide_fixtures.py --rcaide-root /path/to/RCAIDE_LEADS
```

Run this command in the `rcaide` environment. Review changes to existing CSV files before
accepting them; the script is reproducible extraction machinery, not authority to overwrite data.

Generate untouched source-tool predictions separately with:

```bash
python -u generate_rcaide_baselines.py \
  --rcaide-root /path/to/RCAIDE_LEADS \
  --source-commit <full-commit-hash>
```

This runs RCAIDE's own validation driver without changing its model settings and captures its
outputs into source-labelled compressed NPZ files. It can take several minutes.

Run the BladeAD F8745-D4 acoustic-radiation comparison in the `rotor_design` environment with:

```bash
python run_bladead_f8745_validation.py
```

This uses signed aerodynamic disk loads already frozen in the RCAIDE line-source archive and
evaluates BladeAD Lowson plus Barry–Magliozzi noise. The signed distributions include negative
root-region elements but integrate exactly to RCAIDE's positive per-blade totals. It does not
import RCAIDE or claim to validate BladeAD BEM aerodynamics.

Run the aligned-inflow Hanson loading-plus-thickness comparison with:

```bash
python run_bladead_f8745_hanson_validation.py
```

This uses the archived F8745 airfoil thickness shape and keeps the physical BladeAD line-load
convention separate from a diagnostic reproduction of RCAIDE's legacy loading-adapter scaling.

Audit the load adapter and propagation convention with:

```bash
python audit_f8745_interface.py
```

This writes the parity audit and convection-sensitivity table without modifying model constants or
acceptance thresholds.

Bound the effect of aerodynamic-source magnitude and fixed-integral radial redistribution with:

```bash
python audit_f8745_load_sensitivity.py
```

The ±10% magnitude cases and ±20% linear spanwise redistribution cases are sensitivity bounds,
not experimental uncertainty intervals. The radial cases preserve total thrust and torque.

Reproduce RCAIDE's line-source loading, thickness, final pressure convention, and archived
wing-wake adjustment term by term with:

```bash
python audit_f8745_hanson_terms.py
```

The +15 dB wing-wake adjustment is reproduced only for audit parity and is never applied inside
the BladeAD production acoustic graph.

Run the RCAIDE-independent F8475 geometry-to-noise comparison with:

```bash
python run_bladead_f8475_bem_hanson_validation.py
```

This power-matches each published case by changing the blade angle at three-quarter radius, then
passes BladeAD BEM sectional loads directly to BladeAD Hanson acoustics at the paper's nominal
4 m microphone distance. It uses a Clark-Y ZeroD polar fitted to the available Re=1,000,000 XFOIL
table; pressure and Reynolds-number variation remain explicit limitations in the report.

Audit F8475 thickness quadrature and pressure conventions with:

```bash
python audit_f8475_thickness_pressure.py
```

This compares coherent RMS, coherent peak, and diagnostic magnitude-only synthesis, then checks
chordwise and radial integration convergence without tuning the production acoustic model.

Run the corrected-condition RCAIDE plane-source comparison in the `rcaide` environment with:

```bash
python run_rcaide_f8475_corrected_validation.py --rcaide-root /path/to/RCAIDE_LEADS
```

Run `run_bladead_f8475_lowson_validation.py` in the `rotor_design` environment, then merge Lowson,
Hanson, and RCAIDE results with `compare_f8475_corrected_models.py`.
