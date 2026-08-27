# Box-beam structural-model verification

## Scope

`BladeAD.core.structures` contains a differentiable first-order rotating-blade model with:

- a rectangular hollow box spar with independent cap and web thickness profiles;
- sectional area, second moments, axial stiffness, and two bending stiffnesses;
- blade and complete-rotor blade mass;
- centrifugal axial force;
- flapwise and edgewise shear forces and bending moments;
- four raw corner normal stresses and separate cap/web shear stresses; and
- smooth tensile, compressive, shear, and overall utilization aggregates.

BladeAD BEM sectional thrust and drag are complete-rotor element loads. The coupling divides them by
blade count and applies BladeAD's radial integration weights. Each azimuth is retained as a separate
structural load case. Internal loads are evaluated at element inboard faces so the physical hub load
is retained rather than being evaluated half an element outboard.

## Reproducible test

From the BladeAD repository in an environment containing CSDL-alpha:

```bash
python -m pytest tests/structures/test_box_beam.py -q
```

Tested on 2026-08-27 with Python 3.11 and the SPL `rotor_design` environment:

```text
6 passed in 1.31s
```

The run emitted dependency-level CSDL-alpha/NumPy warnings and one intermediate BEM divide-by-zero
warning, but no model-specific warning or failure.

The complete BladeAD regression suite was then rerun with the structural tests included:

```text
73 passed in 94.19s
```

The test suite covers:

- hollow-rectangle area and second moments against independent closed-form equations;
- uniform distributed flapwise and edgewise cantilever loads against exact root shear and moment;
- rotating uniform-mass centrifugal force against its exact radial integral;
- distributed-load mesh convergence;
- direct coupling to real `BEMModel` outputs with multiple azimuths;
- invalid material and sectional-load convention inputs; and
- finite-difference verification of mass and smooth-utilization derivatives with respect to
  aerodynamic load, RPM, spar width, and cap thickness.

## Required optimisation constraints

The model deliberately does not clip invalid geometry. Optimisation problems must constrain
`section.inner_width` and `section.inner_height` to remain positive and should apply manufacturing
minimums to cap/web thickness and outer dimensions. Structural utilization is feasible at or below
one; raw stresses must remain in reported results alongside smooth aggregates.

## Limits

This is a static, small-deflection, equivalent-isotropic box-beam model. It does not yet include
laminate axes or coupling, Tsai--Wu failure, foam/skin mass or load sharing, airfoil pitching moment,
torsion, deflection, buckling, fatigue, joints, modal separation, or thermal effects. Cap/web shear
uses average two-wall shear areas. Material properties and allowables are caller inputs; the model
does not distribute a validated material system.
