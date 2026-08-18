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
