# Acoustic validation fixtures

Fixtures are immutable inputs to validation and regression tests. Correct a discovered
transcription error in a reviewed commit that identifies the original and corrected values;
do not silently regenerate or overwrite a fixture.

Each case directory contains:

- `provenance.md` describing sources, licensing, assumptions, and unresolved questions;
- geometry in SI units with the original nondimensional quantities retained where available;
- operating conditions and observer definitions with explicit frames and angle conventions;
- experimental spectra in dedicated CSV files;
- externally generated predictions in source-labelled NPZ or CSV files.

The initial cases are F8745-D4 for tonal validation and APC 11x4 for broadband validation.
`lowson_hg_matlab/` preserves a pinned combined Sears-loading plus Barry–Magliozzi thickness
comparison and its separately labelled experimental points; its README records why the combined
curve is not yet a loading-only acceptance target.
