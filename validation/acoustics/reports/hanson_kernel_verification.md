# Hanson line-source kernel verification

**Date:** 2026-08-18

**Production path:** `BladeAD/core/acoustics/tonal/hanson.py`

## Implemented scope

- Hanson spanwise non-compact loading radiation with chordwise-compact loading.
- Uniform aligned inflow expressed in the rotor-axis frame.
- Arbitrary non-negative loading harmonics and positive blade-passing modes.
- Real-arithmetic CSDL-alpha graph with separate real and imaginary harmonic pressure.
- Line loads in N/m, nondimensional radial integration, and no hidden per-blade scaling.

The loading kernel does not include non-zero inflow angle coordinate transforms. The separate
Hanson thickness kernel, BladeAD BEM load adapter, API model selection, and F8745 validation path
now exist.

## Primal verification

`tests/acoustics/test_hanson.py` independently evaluates the equation with NumPy complex
arithmetic and SciPy Bessel functions. It compares every node, observer, acoustic mode, loading
harmonic, and radial contribution before also checking the summed complex pressure. The CSDL and
reference values agree to `rtol=2e-12`, with `atol=1e-15` for the resolved contributions.

## Derivative verification

The active test checks both real and imaginary pressure derivatives with respect to:

- axial line-load harmonics; and
- observer polar angle, including Bessel argument, source phase, directivity, convection, and
  propagation phase effects.

The test uses `csdl.derivative_utils.verify_derivatives` at a finite-difference step of `1e-6`
with `raise_on_error=True`.

## Regression result

Command:

```text
conda run -n rotor_design pytest -q tests/acoustics
```

Result: `37 passed` in 2.42 s. Existing CSDL/NumPy warnings remain; no new test failure occurs.

## Next validation increment

The line-load adapter and opt-in public API path now exist. The next increment is to compare their
intermediate force harmonics, Bessel arguments, complex pressure, and harmonic SPL against the
archived RCAIDE F8745 line-source fields before evaluating experimental accuracy.

## Public API boundary

`RotorAcousticSettings(tonal_model="hanson_line")` selects the Hanson path while `"lowson"`
remains the default. The initial Hanson API accepts only load harmonic zero and rejects Sears.
Hanson thickness requires explicit normalized chordwise shape and quadrature inputs; no hidden
airfoil-shape default is used. The model uses only the mesh-velocity projection onto the rotor thrust axis;
transverse inflow and general angle-of-attack Hanson geometry are not modeled.

The real BEM regression checks finite outputs and CSDL derivatives of Hanson total tonal SPL with
respect to RPM and observer position.
