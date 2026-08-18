# DJI 9443 fixture data dictionary

- `operating_conditions.csv`: frozen 5400 RPM hover condition and measured thrust coefficient.
- `chord_distribution.csv`: raw nondimensional chord stations, `c/R` versus `r/R`.
- `twist_distribution.csv`: raw blade twist in degrees versus `r/R`.

The current fixture intentionally contains no experimental acoustic CSV. Those values are
`not reported` in the locally recovered machine-readable sources and must not be inferred from
FLOWUnsteady predictions or copied from an uncorrected microphone-plane comparison.
