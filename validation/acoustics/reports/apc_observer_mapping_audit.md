# APC 11×4 observer-mapping audit

**Date:** 2026-08-26
**Source:** pinned RCAIDE `frequency_domain_test.py` and frozen APC observer fixture

RCAIDE constructs five microphones on a 1.905 m semicircle using a driver parameter `theta` from
45° to 135°. For downstream observers (`x ≥ 0`), the experimental directivity angle from the
rotor plane is

`angle_from_plane = atan2(x, |y|) = theta - 90°`.

| Experimental label | RCAIDE index | Driver parameter | Cartesian position (m) | Derived angle from plane |
|---:|---:|---:|---|---:|
| 22.5° | 3 | 112.5° | (0.729012, -1.759991, 0) | 22.5° |
| 45° | 4 | 135° | (1.347038, -1.347038, 0) | 45° |

The source plots these exact indices as the 22.5° and 45° experimental microphones. Its local
verification arrays and error-field names instead say `60deg` and `90deg`; those names are stale
because they contradict both the plot labels and the physical coordinates.

`validation/acoustics/scripts/audit_apc_observer_mapping.py` reproduces the mapping from the
frozen Cartesian fixture and fails if either derived angle differs by more than `1e-10` degrees.

## Disposition

The APC observer mapping is resolved. This removes the geometry blocker but does not clear the
broadband physical-validation gate: BladeAD currently raises `NotImplementedError` when broadband
is enabled, so Gill–Lee still needs implementation, derivative verification, and comparison with
the APC spectra.
