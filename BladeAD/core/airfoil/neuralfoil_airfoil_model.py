"""NeuralFoil forward pass and AeroSandbox Mach correction ported to CSDL.

Reuses NeuralFoil's existing trained weights (vendored under
`data/neuralfoil_weights/`, see that folder's README for provenance) -- no retraining. Ports
the incompressible-net path of `neuralfoil.main.get_aero_from_kulfan_parameters()`
(NeuralFoil v0.3.3, Peter Sharpe, MIT license) as native CSDL `Variable` arithmetic, so CSDL's
own reverse-mode AD gives exact derivatives through the neural net and smooth Mach correction.
The piecewise wave-drag law uses a narrow `csdl.CustomExplicitOperation` because CSDL has no hard
conditional primitive; its branch derivatives are analytic.

KNOWN APPROXIMATION, deliberately kept (Pass 1b re-assessment, see
`06-rotor-optimisation/neuralfoil-csdl-port/findings-pass1.md`): `csdl_alpha` has no true hard
elementwise min/max/abs -- `csdl.minimum`, `csdl.maximum`, and `csdl.absolute` are all smoothed
(log-sum-exp-style) approximations with a `rho` smoothing parameter (default `rho=20`, verified in
Pass 1a to produce real, non-trivial error -- e.g. `Top_Xtr` off by 0.03 absolute at some test
points). `rho=1000` narrows this to <1.3e-4 absolute and is used at every clip/abs call site here.
**This only affects `Top_Xtr`/`Bot_Xtr` and the boundary-layer `theta`/`H` outputs**. The raw
`CL`/`CD`/`CM`/`analysis_confidence` never call clip/abs and match the reference exactly
(~1e-15 absolute). A `csdl.CustomExplicitOperation`
exact-clip/exact-abs replacement was prototyped in Pass 1b and deliberately NOT kept: `ue/vinf`
(what Pass 2's `Cpmin_0`/`mach_crit`/`mach_dd` will be built from, per AeroSandbox's
`KulfanAirfoil.get_aero_from_neuralfoil()`) is a raw net-output slice that never passes through
clip/abs at all, so the approximation is structurally unreachable from Mcrit/Mdd -- confirmed
quantitatively (not just by code inspection) to ~3e-15 absolute agreement across 144 airfoil/
alpha/Re combinations, independent of which clip/abs implementation is active. A hand-derived
custom-op with manually-specified derivatives is a maintenance liability (same class as the
acoustics Bessel-function gap already tracked in this project) not worth paying for outputs
nothing downstream consumes. Revisit only if a future output that genuinely depends on `Top_Xtr`/
`Bot_Xtr`/`theta`/`H` needs tighter-than-1e-4 exactness.

**Derivative caveat, separate from the value-accuracy point above**: Pass 1b's AD-vs-FD check
(`_check_derivatives()`) only exercises `CL`/`CD`/`CM` -- it does NOT check derivatives of
`Top_Xtr`/`Bot_Xtr`/`theta`/`H` themselves. Those flow through the smoothed clip/abs, so their
*derivatives* (not just their values) carry the `rho=1000` approximation too, and near a clip
boundary (`Top_Xtr` at/near 0 or 1) a smoothed derivative can be proportionally worse than the
value error even though the value itself checks out fine. **If a future module ever needs
d(Top_Xtr)/d(alpha) or similar** (e.g. a transition-location-dependent drag/BL coupling, not
anything currently planned) -- unlike Mcrit/Mdd, which structurally never touches this at all --
that is exactly the case the exact-primitive custom op (prototyped, then dropped, in Pass 1b)
would matter for. Revisit then, not before.

Airfoil normalisation/rotation, control surfaces, and AeroSandbox's 360-degree post-stall blend
are included. Explicitly NOT in this module (see
`06-rotor-optimisation/neuralfoil-csdl-port/port-plan.md` for the full staged plan):
  - Kulfan/CST shape reconstruction. Kulfan parameters and normalization metadata arrive as
    already-fitted numeric constants
    (from `asb.Airfoil(name).to_kulfan_airfoil(n_weights_per_side=8)`, run once, offline, in
    plain Python) -- this module never reconstructs x,y coordinates.

Run self-test (verifies against the real Python `neuralfoil` package):
    /opt/anaconda3/envs/rotor_design/bin/python neuralfoil_airfoil_model.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csdl_alpha as csdl
import numpy as onp

_WEIGHTS_DIR = Path(__file__).parent / "data" / "neuralfoil_weights"
_N_BL = 32  # `Data.N` in the reference package -- boundary-layer points per surface
_DEG2RAD = onp.pi / 180.0

_INPUT_DIST = dict(onp.load(_WEIGHTS_DIR / "scaled_input_distribution.npz"))
_INPUT_DIST["N_inputs"] = len(_INPUT_DIST["mean_inputs_scaled"])


def available_model_sizes() -> set[str]:
    return {p.stem.removeprefix("nn-") for p in _WEIGHTS_DIR.glob("nn-*.npz")}


def _load_nn_parameters(model_size: str) -> dict[str, onp.ndarray]:
    path = _WEIGHTS_DIR / f"nn-{model_size}.npz"
    if not path.is_file():
        raise ValueError(f"Invalid model_size={model_size!r}. Must be one of {available_model_sizes()}.")
    return dict(onp.load(path))


def _sigmoid(x):
    # Reference clips to +-_ln_eps (float32 dynamic range) to suppress exp() overflow; +-80 is
    # comfortably past exp() saturation in float64 and keeps this branch-free / AD-friendly.
    # csdl.minimum/maximum need matching-shape operands, not a bare scalar -- broadcast explicitly.
    lo = onp.full(x.shape, -80.0)
    hi = onp.full(x.shape, 80.0)
    xc = csdl.minimum(csdl.maximum(x, lo, rho=1000.0), hi, rho=1000.0)  # KNOWN APPROXIMATION, see module docstring
    return 1.0 / (1.0 + csdl.exp(-xc))


def _swish(x):
    return x * _sigmoid(x)


def _blend(switch, high, low):
    weight_high = 0.5 + 0.5 * csdl.tanh(switch)
    return high * weight_high + low * (1.0 - weight_high)


def _softmax_pair(a, b, softness=1.0):
    return softness * csdl.log(csdl.exp(a / softness) + csdl.exp(b / softness))


def _post_stall_coefficients(alpha_deg):
    alpha_rad = alpha_deg * _DEG2RAD
    sina = csdl.sin(alpha_rad)
    cosa = csdl.cos(alpha_rad)
    cd90 = 2.08 + 8.36e-2 * cosa + 4.06e-1 * cosa ** 2
    cn = cd90 * sina
    ct = (9.00e-2 - 1.78e-1 * cosa - 2.98e-1 * cosa ** 3) * sina ** 2
    return cn * cosa + ct * sina, cn * sina - ct * cosa, 0.0 * sina


class _PeriodicAngleOperation(csdl.CustomExplicitOperation):
    def evaluate(self, alpha_deg):
        self.declare_input("alpha_deg", alpha_deg)
        output = self.create_output("periodic_alpha_deg", alpha_deg.shape)
        indices = onp.arange(alpha_deg.size)
        self.declare_derivative_parameters(
            "periodic_alpha_deg", "alpha_deg", rows=indices, cols=indices
        )
        return output

    def compute(self, inputs, outputs):
        outputs["periodic_alpha_deg"] = (inputs["alpha_deg"] + 180.0) % 360.0 - 180.0

    def compute_derivatives(self, inputs, outputs, derivatives):
        derivatives["periodic_alpha_deg", "alpha_deg"] = onp.ones(inputs["alpha_deg"].size)


@dataclass(frozen=True)
class NeuralFoilControlSurface:
    """Fixed hinge location and downward-positive deflection in degrees."""

    hinge_point: float
    deflection: float | csdl.Variable


class _WaveDragOperation(csdl.CustomExplicitOperation):
    def evaluate(self, mach, mach_crit, t_over_c):
        self.declare_input("mach", mach)
        self.declare_input("mach_crit", mach_crit)
        self.declare_input("t_over_c", t_over_c)
        if mach.shape != mach_crit.shape or mach.shape != t_over_c.shape:
            raise ValueError("mach, mach_crit, and t_over_c must have identical shapes.")
        indices = onp.arange(mach.size)
        output = self.create_output("CD_wave", mach.shape)
        for name in ("mach", "mach_crit", "t_over_c"):
            self.declare_derivative_parameters("CD_wave", name, rows=indices, cols=indices)
        return output

    @staticmethod
    def _values_and_derivatives(mach, mach_crit, t_over_c):
        mach = onp.asarray(mach)
        mach_crit = onp.asarray(mach_crit)
        t_over_c = onp.asarray(t_over_c)
        mach_dd = mach_crit + (0.1 / 320.0) ** (1.0 / 3.0)
        values = onp.empty_like(mach)
        d_mach = onp.zeros_like(mach)
        d_mach_crit = onp.zeros_like(mach)
        d_t_over_c = onp.zeros_like(mach)

        before_crit = mach < mach_crit
        before_dd = (~before_crit) & (mach < mach_dd)
        transonic = (~before_crit) & (~before_dd) & (mach < 1.1)
        supersonic = ~(before_crit | before_dd | transonic)
        values[before_crit] = 0.0

        delta = mach[before_dd] - mach_crit[before_dd]
        values[before_dd] = 80.0 * delta ** 4
        d_mach[before_dd] = 320.0 * delta ** 3
        d_mach_crit[before_dd] = -320.0 * delta ** 3

        x = mach[transonic]
        x_a = mach_dd[transonic]
        thickness = t_over_c[transonic]
        interval = 1.1 - x_a
        position = (x - x_a) / interval
        blend = 0.5 + 0.5 * onp.cos(onp.pi * position)
        line_a = (x - x_a) * 0.1 + 80.0 * (0.1 / 320.0) ** (4.0 / 3.0)
        line_b_factor = 0.8 - 6.4 * (x - 1.1)
        line_b = thickness * line_b_factor
        values[transonic] = blend * line_a + (1.0 - blend) * line_b
        d_blend_dx = -0.5 * onp.pi * onp.sin(onp.pi * position) / interval
        d_mach[transonic] = d_blend_dx * (line_a - line_b) + blend * 0.1 - (1.0 - blend) * 6.4 * thickness
        d_position_dxa = (x - 1.1) / interval ** 2
        d_blend_dxa = -0.5 * onp.pi * onp.sin(onp.pi * position) * d_position_dxa
        d_mach_crit[transonic] = d_blend_dxa * (line_a - line_b) - blend * 0.1
        d_t_over_c[transonic] = (1.0 - blend) * line_b_factor

        switch = 40.0 * (mach[supersonic] - 1.1)
        tanh_switch = onp.tanh(switch)
        weight_high = 0.5 + 0.5 * tanh_switch
        factor = 0.96 - 0.32 * weight_high
        values[supersonic] = t_over_c[supersonic] * factor
        d_mach[supersonic] = -6.4 * t_over_c[supersonic] * (1.0 - tanh_switch ** 2)
        d_t_over_c[supersonic] = factor
        return values, d_mach, d_mach_crit, d_t_over_c

    def compute(self, inputs, outputs):
        outputs["CD_wave"] = self._values_and_derivatives(
            inputs["mach"], inputs["mach_crit"], inputs["t_over_c"])[0]

    def compute_derivatives(self, inputs, outputs, derivatives):
        values = self._values_and_derivatives(inputs["mach"], inputs["mach_crit"], inputs["t_over_c"])
        derivatives["CD_wave", "mach"] = values[1].ravel()
        derivatives["CD_wave", "mach_crit"] = values[2].ravel()
        derivatives["CD_wave", "t_over_c"] = values[3].ravel()


def _squared_mahalanobis_distance(x):
    """x: (N_inputs, N_cases) CSDL Variable (already in `net()`'s transposed convention)."""
    n_cases = x.shape[1]
    mean = csdl.expand(_INPUT_DIST["mean_inputs_scaled"], (_INPUT_DIST["N_inputs"], n_cases), "i->ij")
    x_minus_mean = x - mean
    weighted = csdl.matmat(_INPUT_DIST["inv_cov_inputs_scaled"], x_minus_mean)  # (N_inputs, N_cases)
    return csdl.sum(x_minus_mean * weighted, axes=(0,))  # (N_cases,)


def _net(x, nn_params):
    """x: (N_inputs, N_cases) -> y: (N_outputs, N_cases). Ports `main.py`'s `net()`."""
    n_cases = x.shape[1]
    layer_indices = sorted({int(k.split(".")[1]) for k in nn_params if k.startswith("net.")})
    for pos, i in enumerate(layer_indices):
        w = nn_params[f"net.{i}.weight"]
        b = nn_params[f"net.{i}.bias"]
        b_expanded = csdl.expand(b, (b.shape[0], n_cases), "i->ij")
        x = csdl.matmat(w, x) + b_expanded
        if pos != len(layer_indices) - 1:  # no activation on the last layer
            x = _swish(x)
    return x


def _row(v, n_cases):
    """Broadcast a scalar (float) or a length-n_cases CSDL Variable/array to shape (1, n_cases)."""
    if isinstance(v, (int, float, onp.number)):
        v = onp.full(n_cases, float(v))
    return csdl.reshape(v, (1, n_cases))


def _shape_of(v):
    if hasattr(v, "shape"):
        return tuple(v.shape)
    if onp.isscalar(v):
        return ()
    return (len(v),)


def _dup_case0_1d(v):
    """Duplicate a shape-(1,) input to shape-(2,) by repeating case 0 -- part of the
    n_cases=1 workaround below. Handles both CSDL Variables (AD-transparent -- no detach,
    the duplicate stays on the same graph so gradients still flow to the original) and plain
    numpy arrays/Python scalars."""
    if hasattr(v, "value"):
        row = csdl.reshape(v, (1,))
        return csdl.reshape(csdl.vstack([row, row]), (2,))
    arr = onp.atleast_1d(onp.asarray(v, dtype=float))
    return onp.concatenate([arr, arr])


def _dup_case0_2d(v, n_cols):
    """Same as `_dup_case0_1d` but for a shape-(1, n_cols) input (the Kulfan weight arrays)."""
    if hasattr(v, "value"):
        row = csdl.reshape(v, (1, n_cols))
        return csdl.vstack([row, row])
    arr = onp.asarray(v, dtype=float).reshape(1, n_cols)
    return onp.vstack([arr, arr])


def get_aero_from_kulfan_parameters(
    upper_weights, lower_weights, leading_edge_weight, TE_thickness,
    alpha_deg, Re, mach, max_thickness, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0,
    model_size="small", control_surfaces=None, include_360_deg_effects=True,
):
    """CSDL port of AeroSandbox's Kulfan NeuralFoil wrapper.

    Per-station (vectorised) inputs -- `n_cases` stations evaluated in one call:
      upper_weights, lower_weights : (n_cases, 8) CSDL Variable or array
      leading_edge_weight, TE_thickness, alpha_deg, Re, mach : (n_cases,) Variable or array
      max_thickness : precomputed thickness/chord scalar or (n_cases,) Variable/array
      n_crit, xtr_upper, xtr_lower : Python float (broadcast) or (n_cases,) CSDL Variable/array
      control_surfaces : optional list of `NeuralFoilControlSurface` objects
      include_360_deg_effects : include AeroSandbox's post-stall blend, default True

    `alpha_deg` in degrees, matching the reference package's convention. Returns a dict of CSDL
    Variables, each shape (n_cases,): `analysis_confidence`, `CL`, `CD`, `CM`, `Top_Xtr`,
    `Bot_Xtr`, and `upper_bl_theta_i` / `upper_bl_H_i` / `upper_bl_ue/vinf_i` /
    `lower_bl_theta_i` / `lower_bl_H_i` / `lower_bl_ue/vinf_i` for i in range(32).

    `alpha_deg`/`Re` etc. are expected as plain 1-D arrays or CSDL Variables of shape exactly
    `(n_cases,)`; `upper_weights`/`lower_weights` exactly `(n_cases, 8)`. Shapes are validated
    up front (Pass 1b hardening -- Pass 1a left this unchecked, risking a `(n_cases, 1)` or
    `(1, n_cases)` input silently misinferring `n_cases` or broadcasting wrong).

    `n_cases=1` is handled by an internal duplicate-and-slice workaround (Pass 1c) for a real
    `csdl_alpha` bug found in Pass 1b: a `(1,1)`-shape Variable silently squeezes to `(1,)` on
    scalar multiplication, which otherwise breaks this function's internal `csdl.vstack` at
    exactly `n_cases=1`. Confirmed reachable in production BladeAD usage:
    `BEM.compute_inflow_angle`'s `memory_efficiency=True` path calls `airfoil_model.evaluate()`
    once per blade station with a scalar `Re[i, j, k]` slice, i.e. `n_cases=1`. See
    `06-rotor-optimisation/neuralfoil-csdl-port/findings-pass1c.md`.
    """
    def _validated_n_cases():
        alpha_shape = _shape_of(alpha_deg)
        if len(alpha_shape) != 1:
            raise ValueError(f"alpha_deg must be 1-D, shape (n_cases,); got {alpha_shape}.")
        n = alpha_shape[0]
        if n == 0:
            raise ValueError("alpha_deg must contain at least one case; got shape (0,).")
        for name, v in (
            ("Re", Re),
            ("leading_edge_weight", leading_edge_weight),
            ("TE_thickness", TE_thickness),
            ("mach", mach),
            ("max_thickness", max_thickness),
            ("n_crit", n_crit),
            ("xtr_upper", xtr_upper),
            ("xtr_lower", xtr_lower),
        ):
            shape = _shape_of(v)
            if shape not in ((n,), ()):  # allow a true scalar for a broadcast constant
                raise ValueError(f"{name} must be shape ({n},) or a scalar; got {shape}.")
        for name, v in (("upper_weights", upper_weights), ("lower_weights", lower_weights)):
            shape = _shape_of(v)
            if shape != (n, 8):
                raise ValueError(f"{name} must be shape ({n}, 8); got {shape}.")
        return n

    n_cases = _validated_n_cases()
    if control_surfaces is None:
        control_surfaces = []

    if n_cases == 1:
        padded = dict(
            upper_weights=_dup_case0_2d(upper_weights, 8),
            lower_weights=_dup_case0_2d(lower_weights, 8),
            leading_edge_weight=(_dup_case0_1d(leading_edge_weight)
                                  if _shape_of(leading_edge_weight) == (1,) else leading_edge_weight),
            TE_thickness=(_dup_case0_1d(TE_thickness)
                          if _shape_of(TE_thickness) == (1,) else TE_thickness),
            alpha_deg=_dup_case0_1d(alpha_deg),
            Re=_dup_case0_1d(Re) if _shape_of(Re) == (1,) else Re,
            mach=_dup_case0_1d(mach) if _shape_of(mach) == (1,) else mach,
            max_thickness=(_dup_case0_1d(max_thickness)
                           if _shape_of(max_thickness) == (1,) else max_thickness),
            n_crit=n_crit, xtr_upper=xtr_upper, xtr_lower=xtr_lower, model_size=model_size,
            control_surfaces=[
                NeuralFoilControlSurface(
                    hinge_point=surface.hinge_point,
                    deflection=(
                        _dup_case0_1d(surface.deflection)
                        if _shape_of(surface.deflection) == (1,)
                        else surface.deflection
                    ),
                )
                for surface in control_surfaces
            ],
            include_360_deg_effects=include_360_deg_effects,
        )
        full = get_aero_from_kulfan_parameters(**padded)
        return {k: csdl.reshape(v[0:1], (1,)) for k, v in full.items()}

    nn_params = _load_nn_parameters(model_size)

    alpha_periodic = _PeriodicAngleOperation().evaluate(
        csdl.reshape(_row(alpha_deg, n_cases), (n_cases,))
    )
    effective_d_alpha = 0.0
    effective_cd_multiplier = 1.0
    for surface in control_surfaces:
        if not onp.isfinite(surface.hinge_point) or not 0.0 <= surface.hinge_point <= 1.0:
            raise ValueError("control-surface hinge_point must be finite and between zero and one")
        deflection = surface.deflection
        deflection_shape = _shape_of(deflection)
        if deflection_shape == (1,) and n_cases > 1:
            deflection = csdl.expand(deflection, (n_cases,))
        elif deflection_shape not in ((), (n_cases,)):
            raise ValueError(
                f"control-surface deflection must be scalar or shape ({n_cases},); "
                f"got {deflection_shape}"
            )
        effectiveness = 1.0 - max(0.0, surface.hinge_point + 1.0e-16) ** 2.751428551177291
        effective_d_alpha = effective_d_alpha + deflection * effectiveness
        deflection_ratio_squared = (deflection / 11.5) ** 2
        effective_cd_multiplier = effective_cd_multiplier * (
            2.0 + deflection_ratio_squared - (1.0 + deflection_ratio_squared) ** 0.5
        )
    alpha_for_net = alpha_periodic + effective_d_alpha

    upper_t = csdl.transpose(upper_weights)  # (8, n_cases)
    lower_t = csdl.transpose(lower_weights)  # (8, n_cases)
    alpha_rad = _row(alpha_for_net, n_cases) * _DEG2RAD
    sin2a = csdl.sin(2.0 * alpha_rad)
    cosa = csdl.cos(alpha_rad)

    rows = (
        [csdl.reshape(upper_t[i, :], (1, n_cases)) for i in range(8)]
        + [csdl.reshape(lower_t[i, :], (1, n_cases)) for i in range(8)]
        + [
            _row(leading_edge_weight, n_cases),
            _row(TE_thickness, n_cases) * 50.0,
            sin2a,
            cosa,
            1.0 - cosa ** 2,
            (csdl.log(_row(Re, n_cases)) - 12.5) / 3.5,
            (_row(n_crit, n_cases) - 9.0) / 4.5,
            _row(xtr_upper, n_cases),
            _row(xtr_lower, n_cases),
        ]
    )
    x = csdl.vstack(rows)  # (25, n_cases), already in `net()`'s transposed convention

    def _confidence_adjusted(x_in):
        y = _net(x_in, nn_params)
        adj = _squared_mahalanobis_distance(x_in) / (2.0 * _INPUT_DIST["N_inputs"])
        y0 = csdl.reshape(y[0, :], (n_cases,)) - adj
        return y, y0

    y, y0 = _confidence_adjusted(x)

    ### Flip x per the reference's alpha-symmetry embedding, evaluate again, un-flip the outputs.
    x_flipped = csdl.vstack(
        [csdl.reshape(x[i + 8, :], (1, n_cases)) * -1.0 for i in range(8)]     # upper <- -lower
        + [csdl.reshape(x[i, :], (1, n_cases)) * -1.0 for i in range(8)]      # lower <- -upper
        + [
            csdl.reshape(x[16, :], (1, n_cases)) * -1.0,                     # LE weight flips
            csdl.reshape(x[17, :], (1, n_cases)),                            # TE thickness unchanged
            csdl.reshape(x[18, :], (1, n_cases)) * -1.0,                     # sin(2a) flips
            csdl.reshape(x[19, :], (1, n_cases)),                            # cos(a) unchanged
            csdl.reshape(x[20, :], (1, n_cases)),                            # 1-cos^2(a) unchanged
            csdl.reshape(x[21, :], (1, n_cases)),                            # Re unchanged
            csdl.reshape(x[22, :], (1, n_cases)),                            # n_crit unchanged
            csdl.reshape(x[24, :], (1, n_cases)),                            # xtr_upper <- xtr_lower
            csdl.reshape(x[23, :], (1, n_cases)),                            # xtr_lower <- xtr_upper
        ]
    )
    y_flipped, y_flipped_0 = _confidence_adjusted(x_flipped)

    N = _N_BL
    rows_unflipped = [csdl.reshape(y_flipped_0, (1, n_cases))]
    rows_unflipped.append(csdl.reshape(y_flipped[1, :], (1, n_cases)) * -1.0)   # CL
    rows_unflipped.append(csdl.reshape(y_flipped[2, :], (1, n_cases)))          # CD (unscaled)
    rows_unflipped.append(csdl.reshape(y_flipped[3, :], (1, n_cases)) * -1.0)   # CM
    rows_unflipped.append(csdl.reshape(y_flipped[5, :], (1, n_cases)))          # Top_Xtr <- Bot_Xtr
    rows_unflipped.append(csdl.reshape(y_flipped[4, :], (1, n_cases)))          # Bot_Xtr <- Top_Xtr
    # upper<->lower BL blocks, in order: theta(N), H(N), ue/vinf(N), each surface
    rows_unflipped += [csdl.reshape(y_flipped[6 + N * 3 + i, :], (1, n_cases)) for i in range(N)]      # upper_theta
    rows_unflipped += [csdl.reshape(y_flipped[6 + N * 4 + i, :], (1, n_cases)) for i in range(N)]      # upper_H
    rows_unflipped += [csdl.reshape(y_flipped[6 + N * 5 + i, :], (1, n_cases)) * -1.0 for i in range(N)]  # upper_ue/vinf
    rows_unflipped += [csdl.reshape(y_flipped[6 + i, :], (1, n_cases)) for i in range(N)]               # lower_theta
    rows_unflipped += [csdl.reshape(y_flipped[6 + N + i, :], (1, n_cases)) for i in range(N)]           # lower_H
    rows_unflipped += [csdl.reshape(y_flipped[6 + N * 2 + i, :], (1, n_cases)) * -1.0 for i in range(N)]  # lower_ue/vinf
    y_unflipped = csdl.vstack(rows_unflipped)

    y0_full = csdl.reshape(y0, (1, n_cases))
    y_rest = y[1:, :]
    y_full = csdl.vstack([y0_full, y_rest])

    y_fused = (y_full + y_unflipped) / 2.0
    confidence = _sigmoid(csdl.reshape(y_fused[0, :], (n_cases,)))
    zeros_1d, ones_1d = onp.zeros(n_cases), onp.ones(n_cases)
    top_xtr_raw = csdl.reshape(y_fused[4, :], (n_cases,))
    top_xtr = csdl.minimum(csdl.maximum(top_xtr_raw, zeros_1d, rho=1000.0), ones_1d, rho=1000.0)  # KNOWN APPROXIMATION
    bot_xtr = csdl.minimum(csdl.maximum(csdl.reshape(y_fused[5, :], (n_cases,)), zeros_1d, rho=1000.0), ones_1d, rho=1000.0)  # KNOWN APPROXIMATION

    CL = csdl.reshape(y_fused[1, :], (n_cases,)) / 2.0
    CD = csdl.exp((csdl.reshape(y_fused[2, :], (n_cases,)) - 2.0) * 2.0)
    CD = CD * effective_cd_multiplier
    CM = csdl.reshape(y_fused[3, :], (n_cases,)) / 20.0

    re_1d = csdl.reshape(_row(Re, n_cases), (n_cases,))
    Re_col = csdl.expand(re_1d, (N, n_cases), "j->ij")
    results = {"analysis_confidence": confidence, "CL": CL, "CD": CD, "CM": CM,
              "Top_Xtr": top_xtr, "Bot_Xtr": bot_xtr}
    for surface, base in (("upper", 6), ("lower", 6 + N * 3)):
        theta_raw = y_fused[base:base + N, :]
        h_raw = y_fused[base + N:base + N * 2, :]
        ue_vinf = y_fused[base + N * 2:base + N * 3, :]
        theta = (10.0 ** theta_raw - 0.1) / (csdl.absolute(ue_vinf, rho=1000.0) * Re_col)  # KNOWN APPROXIMATION
        h = 2.6 * csdl.exp(h_raw)
        for i in range(N):
            results[f"{surface}_bl_theta_{i}"] = csdl.reshape(theta[i, :], (n_cases,))
            results[f"{surface}_bl_H_{i}"] = csdl.reshape(h[i, :], (n_cases,))
            results[f"{surface}_bl_ue/vinf_{i}"] = csdl.reshape(ue_vinf[i, :], (n_cases,))

    ue_rows = [results[f"{surface}_bl_ue/vinf_{i}"] for surface in ("upper", "lower") for i in range(N)]
    cp_candidates = csdl.vstack([csdl.reshape(1.0 - ue ** 2, (1, n_cases)) for ue in ue_rows])
    cpmin_0 = csdl.minimum(cp_candidates, axes=(0,), rho=100.0)
    if include_360_deg_effects:
        cl_separated, cd_separated, cm_separated = _post_stall_coefficients(alpha_periodic)
        is_separated = _softmax_pair(
            alpha_periodic - 20.0, -20.0 - alpha_periodic
        ) / 3.0
        CL = _blend(is_separated, cl_separated, CL)
        skin_friction = 0.074 / re_1d ** (1.0 / 5.0)
        CD = csdl.exp(
            _blend(
                is_separated,
                csdl.log(cd_separated + skin_friction),
                csdl.log(CD),
            )
        )
        CM = _blend(is_separated, cm_separated, CM)
        alpha_periodic_rad = alpha_periodic * _DEG2RAD
        sin_alpha = csdl.sin(alpha_periodic_rad)
        cpmin_0 = _blend(
            is_separated, -1.0 - 0.5 * sin_alpha ** 2, cpmin_0
        )
        top_xtr = _blend(
            is_separated, 0.5 - 0.5 * csdl.tanh(10.0 * sin_alpha), top_xtr
        )
        bot_xtr = _blend(
            is_separated, 0.5 + 0.5 * csdl.tanh(10.0 * sin_alpha), bot_xtr
        )
    cpmin_0 = csdl.minimum(cpmin_0, onp.zeros(n_cases), rho=1000.0)
    mach_crit = (
        1.011571026701678 - cpmin_0
        + 0.6582431351007195 * (-cpmin_0) ** 0.6724789439840343
    ) ** -0.5504677038358711
    mach_dd = mach_crit + (0.1 / 320.0) ** (1.0 / 3.0)

    mach = csdl.reshape(_row(mach, n_cases), (n_cases,))
    beta_squared_ideal = 1.0 - mach ** 2
    beta = csdl.maximum(beta_squared_ideal, -beta_squared_ideal, rho=2.0) ** 0.5
    CL = CL / beta
    CM = CM / beta
    cpmin = cpmin_0 / beta
    buffet_factor = _blend(50.0 * (mach - (mach_dd + 0.04)), _blend((mach - 1.0) / 0.1, 1.0, 0.5), 1.0)
    cla_supersonic_ratio_factor = _blend((mach - 1.0) / 0.1, 4.0 / (2.0 * onp.pi), 1.0)
    CL = CL * buffet_factor * cla_supersonic_ratio_factor

    thickness = csdl.reshape(_row(max_thickness, n_cases), (n_cases,))
    CD = CD + _WaveDragOperation().evaluate(mach, mach_crit, thickness)
    has_aerodynamic_center_shift = (mach - (mach_dd + 0.06)) / 0.06
    if include_360_deg_effects:
        has_aerodynamic_center_shift = _softmax_pair(
            is_separated, has_aerodynamic_center_shift, softness=0.1
        )
    alpha_rad_1d = alpha_periodic * _DEG2RAD
    CM = CM + _blend(
        has_aerodynamic_center_shift,
        -0.25 * csdl.cos(alpha_rad_1d) * CL - 0.25 * csdl.sin(alpha_rad_1d) * CD,
        0.0,
    )
    results.update(CL=CL, CD=CD, CM=CM, Cpmin=cpmin, Cpmin_0=cpmin_0,
                   Top_Xtr=top_xtr, Bot_Xtr=bot_xtr,
                   mach_crit=mach_crit, mach_dd=mach_dd)
    return results


class NeuralFoilAirfoilModel:
    """CSDL-native airfoil model matching AeroSandbox's outer Airfoil wrapper.

    Matches this package's radian-based `evaluate(alpha, Re, Ma)` interface. Normalized Kulfan
    parameters, maximum thickness, and normalization metadata are fixed at construction time.
    """

    def __init__(self, kulfan_parameters: dict, model_size: str = "small",
                n_crit: float = 9.0, xtr_upper: float = 1.0, xtr_lower: float = 1.0,
                control_surfaces=None, include_360_deg_effects: bool = True):
        self.upper_weights = onp.asarray(kulfan_parameters["upper_weights"], dtype=float)
        self.lower_weights = onp.asarray(kulfan_parameters["lower_weights"], dtype=float)
        if self.upper_weights.shape != (8,) or self.lower_weights.shape != (8,):
            raise ValueError(
                "NeuralFoil's neural networks expect exactly 8 CST weights per side, got "
                f"upper={self.upper_weights.shape}, lower={self.lower_weights.shape}.")
        self.leading_edge_weight = float(kulfan_parameters["leading_edge_weight"])
        self.TE_thickness = float(kulfan_parameters["TE_thickness"])
        required_geometry = (
            "max_thickness", "rotation_angle", "x_translation",
            "y_translation", "scale_factor",
        )
        missing = [name for name in required_geometry if name not in kulfan_parameters]
        if missing:
            raise ValueError(
                "kulfan_parameters must include offline geometry constants: "
                + ", ".join(missing)
            )
        self.max_thickness = float(kulfan_parameters["max_thickness"])
        self.rotation_angle = float(kulfan_parameters["rotation_angle"])
        self.x_translation = float(kulfan_parameters["x_translation"])
        self.y_translation = float(kulfan_parameters["y_translation"])
        self.scale_factor = float(kulfan_parameters["scale_factor"])
        if not onp.isfinite(self.scale_factor) or self.scale_factor <= 0.0:
            raise ValueError("scale_factor must be finite and positive")
        self.model_size = model_size
        self.n_crit, self.xtr_upper, self.xtr_lower = n_crit, xtr_upper, xtr_lower
        self.control_surfaces = [] if control_surfaces is None else list(control_surfaces)
        self.include_360_deg_effects = include_360_deg_effects

    def evaluate_all(self, alpha, Re, Ma):
        """`alpha`/`Re` may be any shape -- BladeAD's real BEM call sites pass either a 1-D
        vector or the full `(num_nodes, num_radial, num_azimuthal)` field (confirmed from
        `BEM.compute_inflow_angle`'s non-memory-efficient path, Pass 1c). Flattened to 1-D for
        `get_aero_from_kulfan_parameters` (which only knows a flat `n_cases` axis), then
        reshaped back to the caller's original shape on the way out."""
        shape = alpha.shape if hasattr(alpha, "shape") else (len(alpha),)
        re_shape = Re.shape if hasattr(Re, "shape") else (len(Re),)
        ma_shape = Ma.shape if hasattr(Ma, "shape") else (len(Ma),)
        if re_shape != shape or ma_shape != shape:
            raise ValueError(f"alpha, Re, and Ma must have the same shape; got {shape}, {re_shape}, and {ma_shape}.")
        n_cases = int(onp.prod(shape))
        alpha_flat = csdl.reshape(alpha, (n_cases,)) if hasattr(alpha, "value") else onp.asarray(alpha).reshape(n_cases)
        Re_flat = csdl.reshape(Re, (n_cases,)) if hasattr(Re, "value") else onp.asarray(Re).reshape(n_cases)
        Ma_flat = csdl.reshape(Ma, (n_cases,)) if hasattr(Ma, "value") else onp.asarray(Ma).reshape(n_cases)
        alpha_normalized_deg = alpha_flat / _DEG2RAD + self.rotation_angle
        re_normalized = Re_flat / self.scale_factor
        upper = onp.tile(self.upper_weights, (n_cases, 1))
        lower = onp.tile(self.lower_weights, (n_cases, 1))
        aero = get_aero_from_kulfan_parameters(
            upper, lower, self.leading_edge_weight, self.TE_thickness,
            alpha_normalized_deg, re_normalized,
            Ma_flat, self.max_thickness,
            n_crit=self.n_crit, xtr_upper=self.xtr_upper, xtr_lower=self.xtr_lower,
            model_size=self.model_size, control_surfaces=self.control_surfaces,
            include_360_deg_effects=self.include_360_deg_effects)
        delta_alpha_rad = self.rotation_angle * _DEG2RAD
        x_translation_qc = (
            -self.x_translation
            + 0.25 / self.scale_factor * onp.cos(delta_alpha_rad)
            - 0.25
        )
        y_translation_qc = (
            -self.y_translation
            + 0.25 / self.scale_factor * onp.sin(-delta_alpha_rad)
        )
        aero["CM"] = (
            aero["CM"] - aero["CL"] * x_translation_qc
            + aero["CD"] * y_translation_qc
        )
        if shape != (n_cases,):
            aero = {name: csdl.reshape(value, shape) for name, value in aero.items()}
        return aero

    def evaluate(self, alpha, Re, Ma):
        aero = self.evaluate_all(alpha, Re, Ma)
        return aero["CL"], aero["CD"]


def _gen_reference_cases(model_size, alphas, res, machs, airfoils):
    """Precompute Kulfan parameters + reference outputs in the `spl-bricks` env (has
    asb+neuralfoil) for one `model_size`. Needs that env's interpreter as a subprocess --
    this module itself has no asb/neuralfoil dependency."""
    import subprocess
    import json

    gen_script = f"""
import json, numpy as np
import aerosandbox as asb
import neuralfoil as nf

cases = []
alphas = {alphas!r}
res = {res!r}
machs = {machs!r}
for name in {airfoils!r}:
    af = asb.Airfoil(name)
    ka = af.to_kulfan_airfoil(n_weights_per_side=8)
    kulfan = dict(upper_weights=list(ka.upper_weights), lower_weights=list(ka.lower_weights),
                  leading_edge_weight=float(ka.leading_edge_weight), TE_thickness=float(ka.TE_thickness),
                  max_thickness=float(ka.max_thickness()))
    for a in alphas:
        for re in res:
            for mach in machs:
                ref = ka.get_aero_from_neuralfoil(alpha=a, Re=re, mach=mach,
                                                  model_size={model_size!r},
                                                  include_360_deg_effects=False)
                cases.append({{"airfoil": name, "kulfan": kulfan, "alpha": a, "Re": re,
                              "mach": mach,
                              "ref": {{k: float(np.asarray(v).reshape(-1)[0]) for k, v in ref.items()}}}})
print(json.dumps(cases))
"""
    out = subprocess.run(
        ["/opt/anaconda3/envs/spl-bricks/bin/python", "-c", gen_script],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _selftest():
    """Verify against the real Python `neuralfoil` package, for every vendored model size
    (Pass 1b, item 3.2/3.3). Needs the `spl-bricks` env's `neuralfoil`/`aerosandbox`
    importable -- run with the `rotor_design` interpreter, which shells out to `spl-bricks`
    for the reference values."""
    alphas = [-8.0, -2.0, 0.0, 4.0, 10.0, 16.0]
    res = [2.0e5, 1.0e6, 3.0e6]
    machs = [0.0, 0.5, 0.75, 0.9, 1.15]
    airfoils = ["mh117", "mh60", "naca0012"]

    # CL/CD/CM/analysis_confidence never touch the smoothed clip/abs ops -- exact-tolerance gate.
    # Top_Xtr/Bot_Xtr and the boundary-layer outputs DO (see module docstring, "KNOWN
    # APPROXIMATION") -- gated on a looser absolute error, since relative error is meaningless
    # near a reference value of exactly 0 or 1 (a fully-clipped case).
    exact_keys = ("CL", "CD", "CM", "analysis_confidence", "Cpmin", "Cpmin_0", "mach_crit", "mach_dd")
    approx_keys = ("Top_Xtr", "Bot_Xtr")

    all_ok = True
    for model_size in sorted(available_model_sizes()):
        cases = _gen_reference_cases(model_size, alphas, res, machs, airfoils)

        upper = onp.array([c["kulfan"]["upper_weights"] for c in cases])
        lower = onp.array([c["kulfan"]["lower_weights"] for c in cases])
        le = onp.array([c["kulfan"]["leading_edge_weight"] for c in cases])
        te = onp.array([c["kulfan"]["TE_thickness"] for c in cases])
        alpha = onp.array([c["alpha"] for c in cases])
        Re = onp.array([c["Re"] for c in cases])
        mach = onp.array([c["mach"] for c in cases])
        max_thickness = onp.array([c["kulfan"]["max_thickness"] for c in cases])

        recorder = csdl.Recorder(inline=True)
        recorder.start()
        aero = get_aero_from_kulfan_parameters(
            upper, lower, le, te, alpha, Re, mach, max_thickness,
            model_size=model_size, include_360_deg_effects=False)
        recorder.stop()

        print(f"model_size={model_size!r} ({len(cases)} cases)")
        max_rel, max_abs = {}, {}
        for key in exact_keys + approx_keys:
            got = onp.asarray(aero[key].value).reshape(-1)
            ref = onp.array([c["ref"][key] for c in cases])
            rel = onp.abs(got - ref) / (onp.abs(ref) + 1e-9)
            max_rel[key] = float(onp.max(rel))
            max_abs[key] = float(onp.max(onp.abs(got - ref)))
            print(f"  {key:<20} max abs diff {max_abs[key]:.3e}  max rel diff {max_rel[key]:.3e}")

        bl_keys = [f"upper_bl_theta_{i}" for i in (0, 15, 31)] + [f"lower_bl_ue/vinf_{i}" for i in (0, 15, 31)]
        bl_abs = {}
        for key in bl_keys:
            if key not in cases[0]["ref"]:
                continue
            got = onp.asarray(aero[key].value).reshape(-1)
            ref = onp.array([c["ref"][key] for c in cases])
            rel = onp.abs(got - ref) / (onp.abs(ref) + 1e-9)
            bl_abs[key] = float(onp.max(onp.abs(got - ref)))
            print(f"  {key:<20} max abs diff {bl_abs[key]:.3e}  max rel diff {onp.max(rel):.3e}")

        # Absolute tolerance for the exact-arithmetic outputs too -- a relative-diff gate is
        # meaningless when the reference value itself is near zero (amplifies machine-epsilon noise).
        ok = (all(max_abs[k] < 1e-10 for k in exact_keys)
              and all(max_abs[k] < 1e-3 for k in approx_keys)
              and all(v < 1e-3 for v in bl_abs.values()))
        print(f"  model_size={model_size!r}", "OK" if ok else "FAIL")
        all_ok = all_ok and ok

    print("neuralfoil_airfoil_model PASS 1b multi-size self-test", "OK" if all_ok else "FAIL")
    return all_ok


def _eval_case(upper, lower, le, te, max_thickness, alpha, Re, mach, model_size="small", want_derivs=False):
    """Run one case's forward pass in its own Recorder/graph, returning (CL, CD, CM) as plain
    floats, and optionally the AD Jacobian of each w.r.t. every input (as a dict of
    {input_name: 1-D numpy array}).

    Padded to n_cases=2 (case 0 = the real case, case 1 = an arbitrary fixed dummy) rather than
    evaluated at n_cases=1 directly -- **`csdl_alpha` has a real bug/quirk found during this
    check: multiplying a `(1,1)`-shape Variable by a Python scalar silently squeezes it to
    `(1,)`**, which breaks `csdl.vstack` inside `get_aero_from_kulfan_parameters` at exactly
    n_cases=1 (confirmed minimal repro: `csdl.Variable(value=[[0.5]]) * 50.0` -> shape `(1,)`,
    not `(1,1)`; at n_cases=2 the same op correctly preserves `(2,1)`). Since the per-case
    formula has no cross-case coupling (each station's aero depends only on its own inputs),
    padding with a dummy second case and reading off case 0 gives an exact single-case answer
    without hitting the squeeze bug -- confirmed the off-diagonal Jacobian blocks between case 0
    and case 1 are structurally exact zero below. This n_cases=1 limitation should be tracked as
    a known gap (Pass 1c or a follow-up) since a real BEM call could plausibly evaluate a single
    station."""
    n_cases = 2
    upper2 = onp.vstack([onp.asarray(upper, dtype=float).reshape(1, 8), onp.zeros((1, 8))])
    lower2 = onp.vstack([onp.asarray(lower, dtype=float).reshape(1, 8), onp.zeros((1, 8))])
    le2 = onp.array([le, 0.0], dtype=float)
    te2 = onp.array([te, 0.01], dtype=float)
    alpha2 = onp.array([alpha, 0.0], dtype=float)
    Re2 = onp.array([Re, 1.0e6], dtype=float)
    mach2 = onp.array([mach, 0.3], dtype=float)
    thickness2 = onp.array([max_thickness, 0.12], dtype=float)

    rec = csdl.Recorder(inline=True)
    rec.start()
    upper_v = csdl.Variable(value=upper2)
    lower_v = csdl.Variable(value=lower2)
    le_v = csdl.Variable(value=le2)
    te_v = csdl.Variable(value=te2)
    alpha_v = csdl.Variable(value=alpha2)
    Re_v = csdl.Variable(value=Re2)
    mach_v = csdl.Variable(value=mach2)
    aero = get_aero_from_kulfan_parameters(
        upper_v, lower_v, le_v, te_v, alpha_v, Re_v, mach_v, thickness2,
        model_size=model_size, include_360_deg_effects=False)
    CL, CD, CM = aero["CL"], aero["CD"], aero["CM"]
    out_vals = {"CL": float(CL.value[0]), "CD": float(CD.value[0]), "CM": float(CM.value[0])}

    if not want_derivs:
        rec.stop()
        return out_vals, None

    in_vars = {"upper_weights": upper_v, "lower_weights": lower_v, "leading_edge_weight": le_v,
               "TE_thickness": te_v, "alpha_deg": alpha_v, "Re": Re_v, "mach": mach_v}
    derivs = csdl.derivative([CL, CD, CM], list(in_vars.values()))
    rec.stop()

    ad = {}
    for out_name, out_var in zip(("CL", "CD", "CM"), (CL, CD, CM)):
        ad[out_name] = {}
        for name, v in in_vars.items():
            jac = onp.asarray(derivs[out_var, v].value)  # (n_cases, n_in_flat)
            n_in_per_case = jac.shape[1] // n_cases
            ad[out_name][name] = jac[0, :n_in_per_case]  # case-0 row, case-0 columns only
            off_diag = jac[0, n_in_per_case:]  # case-0 row, case-1 columns -- must be exact 0
            assert onp.all(off_diag == 0.0), f"unexpected cross-case coupling: {out_name}/{name}"
    return out_vals, ad


def _check_derivatives():
    """Pass 1b item 3.4: check CSDL's AD-computed derivatives of CL/CD/CM w.r.t. alpha_deg, Re,
    and each of the 18 Kulfan-parameter inputs, against central-difference FD, at several test
    points spanning the same airfoils/alpha/Re ranges as `_selftest()`."""
    cases = _gen_reference_cases("small", alphas=[2.0], res=[3.0e5],
                                  machs=[0.3, 0.7, 0.9, 1.15],
                                  airfoils=["mh117", "naca0012"])

    flat_specs = [("upper_weights", 8), ("lower_weights", 8), ("leading_edge_weight", 1),
                  ("TE_thickness", 1), ("alpha_deg", 1), ("Re", 1), ("mach", 1)]

    overall_max_rel = 0.0
    overall_worst = None
    max_rel_by_input = {name: 0.0 for name, _ in flat_specs}
    for case in cases:
        base = dict(upper=onp.array(case["kulfan"]["upper_weights"], dtype=float),
                    lower=onp.array(case["kulfan"]["lower_weights"], dtype=float),
                    le=float(case["kulfan"]["leading_edge_weight"]),
                    te=float(case["kulfan"]["TE_thickness"]),
                    max_thickness=float(case["kulfan"]["max_thickness"]),
                    alpha=float(case["alpha"]), Re=float(case["Re"]), mach=float(case["mach"]))
        arg_key = {"upper_weights": "upper", "lower_weights": "lower",
                   "leading_edge_weight": "le", "TE_thickness": "te",
                   "alpha_deg": "alpha", "Re": "Re", "mach": "mach"}

        _, ad = _eval_case(**base, want_derivs=True)

        for in_name, n_comp in flat_specs:
            key = arg_key[in_name]
            base_val = onp.atleast_1d(base[key]).astype(float)
            for comp in range(n_comp):
                h = 1e-6 * max(abs(base_val[comp]), 1.0)
                plus, minus = dict(base), dict(base)
                v_plus, v_minus = base_val.copy(), base_val.copy()
                v_plus[comp] += h
                v_minus[comp] -= h
                plus[key] = v_plus if n_comp > 1 else float(v_plus[0])
                minus[key] = v_minus if n_comp > 1 else float(v_minus[0])
                out_plus, _ = _eval_case(**plus, want_derivs=False)
                out_minus, _ = _eval_case(**minus, want_derivs=False)

                for out_name in ("CL", "CD", "CM"):
                    fd = (out_plus[out_name] - out_minus[out_name]) / (2 * h)
                    ad_val = ad[out_name][in_name][comp]
                    denom = max(abs(fd), abs(ad_val), 1e-8)
                    rel_err = abs(fd - ad_val) / denom
                    max_rel_by_input[in_name] = max(max_rel_by_input[in_name], rel_err)
                    if rel_err > overall_max_rel:
                        overall_max_rel = rel_err
                        overall_worst = (case["airfoil"], case["alpha"], case["Re"],
                                          out_name, in_name, comp, ad_val, fd)

    print(f"AD-vs-FD derivative check: {len(cases)} cases x 7 inputs x 3 outputs")
    print(f"  max relative error: {overall_max_rel:.3e}")
    print(f"  max relative error w.r.t. mach: {max_rel_by_input['mach']:.3e}")
    print(f"  worst case: {overall_worst}")
    ok = overall_max_rel < 1e-4
    print("neuralfoil_airfoil_model PASS 1b derivative check", "OK" if ok else "FAIL")
    return ok


def _gen_pass3_reference_cases(control_surface=None):
    import json
    import subprocess

    surface_expression = "None"
    if control_surface is not None:
        surface_expression = (
            "[asb.ControlSurface(hinge_point="
            f"{control_surface.hinge_point!r}, deflection={control_surface.deflection!r})]"
        )
    script = f"""
import aerosandbox as asb
import json
import numpy as np

cases = []
for name in ["mh117", "naca0012"]:
    airfoil = asb.Airfoil(name)
    normalization = airfoil.normalize(return_dict=True)
    kulfan = normalization["airfoil"].to_kulfan_airfoil(
        n_weights_per_side=8, normalize_coordinates=False
    )
    parameters = dict(
        upper_weights=list(kulfan.upper_weights),
        lower_weights=list(kulfan.lower_weights),
        leading_edge_weight=float(kulfan.leading_edge_weight),
        TE_thickness=float(kulfan.TE_thickness),
        max_thickness=float(kulfan.max_thickness()),
        rotation_angle=float(normalization["rotation_angle"]),
        x_translation=float(normalization["x_translation"]),
        y_translation=float(normalization["y_translation"]),
        scale_factor=float(normalization["scale_factor"]),
    )
    for alpha in [-40, -23, -20, -17, 0, 17, 20, 23, 40, 90, 140, 179, 220]:
        for reynolds in [3e5, 1e6]:
            for mach in [0.0, 0.5678, 0.9]:
                result = airfoil.get_aero_from_neuralfoil(
                    alpha=alpha, Re=reynolds, mach=mach, model_size="small",
                    control_surfaces={surface_expression}, include_360_deg_effects=True,
                )
                cases.append(dict(
                    airfoil=name, parameters=parameters, alpha=alpha, Re=reynolds, mach=mach,
                    reference={{key: float(np.asarray(value).reshape(-1)[0])
                               for key, value in result.items()}},
                ))
print(json.dumps(cases))
"""
    result = subprocess.run(
        ["/opt/anaconda3/envs/spl-bricks/bin/python", "-c", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _evaluate_pass3_cases(cases, control_surfaces=None):
    outputs = []
    for airfoil_name in sorted({case["airfoil"] for case in cases}):
        selected = [case for case in cases if case["airfoil"] == airfoil_name]
        model = NeuralFoilAirfoilModel(
            selected[0]["parameters"], model_size="small",
            control_surfaces=control_surfaces, include_360_deg_effects=True,
        )
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        aero = model.evaluate_all(
            onp.deg2rad([case["alpha"] for case in selected]),
            onp.array([case["Re"] for case in selected]),
            onp.array([case["mach"] for case in selected]),
        )
        recorder.stop()
        for index, case in enumerate(selected):
            outputs.append((case, {key: float(onp.asarray(value.value)[index])
                                   for key, value in aero.items()}))
    return outputs


def _check_pass3_values():
    keys = ("CL", "CD", "CM", "Cpmin", "Cpmin_0", "mach_crit", "mach_dd",
            "analysis_confidence", "Top_Xtr", "Bot_Xtr")
    cases = _gen_pass3_reference_cases()
    comparisons = _evaluate_pass3_cases(cases)
    max_abs = {key: 0.0 for key in keys}
    max_rel = {key: 0.0 for key in keys}
    for case, output in comparisons:
        for key in keys:
            reference = case["reference"][key]
            error = abs(output[key] - reference)
            max_abs[key] = max(max_abs[key], error)
            max_rel[key] = max(max_rel[key], error / (abs(reference) + 1e-9))

    control = NeuralFoilControlSurface(hinge_point=0.75, deflection=12.0)
    control_cases = _gen_pass3_reference_cases(control)
    control_comparisons = _evaluate_pass3_cases(control_cases, [control])
    control_max_abs = {key: 0.0 for key in keys}
    for case, output in control_comparisons:
        for key in keys:
            control_max_abs[key] = max(
                control_max_abs[key], abs(output[key] - case["reference"][key])
            )

    sample = cases[:4]
    implicit_empty = _evaluate_pass3_cases(sample)
    explicit_empty = _evaluate_pass3_cases(sample, [])
    empty_identical = all(
        first_output[key] == second_output[key]
        for (_, first_output), (_, second_output) in zip(implicit_empty, explicit_empty)
        for key in keys
    )

    print(f"PASS 3 outer-Airfoil value check: {len(cases)} cases across 2 airfoils")
    for key in keys:
        print(f"  {key:<20} max abs {max_abs[key]:.3e}  max rel {max_rel[key]:.3e}")
    print("PASS 3 control-surface max absolute errors")
    for key in keys:
        print(f"  {key:<20} {control_max_abs[key]:.3e}")
    print(f"PASS 3 explicit-empty control surfaces bit-identical: {empty_identical}")
    ok = (
        all(max_abs[key] < 1e-10 for key in keys[:8])
        and all(max_abs[key] < 1e-3 for key in keys[8:])
        and all(control_max_abs[key] < 1e-10 for key in keys[:8])
        and all(control_max_abs[key] < 1e-3 for key in keys[8:])
        and empty_identical
    )
    print("neuralfoil_airfoil_model PASS 3 value check", "OK" if ok else "FAIL")
    return ok


def _check_pass3_alpha_derivatives():
    cases = _gen_pass3_reference_cases()
    parameters = next(case["parameters"] for case in cases if case["airfoil"] == "mh117")
    model = NeuralFoilAirfoilModel(parameters, model_size="small")
    maximum_relative_error = 0.0
    worst = None
    for alpha_deg in (-23.0, -20.0, -17.0, 17.0, 20.0, 23.0, 60.0, 140.0, 220.0):
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        alpha = csdl.Variable(value=onp.deg2rad(onp.array([alpha_deg, 0.0])))
        aero = model.evaluate_all(alpha, onp.array([989832.0, 1e6]), onp.array([0.5678, 0.3]))
        derivatives = csdl.derivative(
            [aero[key] for key in ("CL", "CD", "CM")], [alpha]
        )
        recorder.stop()
        step = 1e-5
        values = {}
        for offset_name, offset in (("plus", step), ("minus", -step)):
            rec = csdl.Recorder(inline=True)
            rec.start()
            shifted = model.evaluate_all(
                onp.array([onp.deg2rad(alpha_deg) + offset, 0.0]),
                onp.array([989832.0, 1e6]), onp.array([0.5678, 0.3]),
            )
            values[offset_name] = {key: float(shifted[key].value[0]) for key in ("CL", "CD", "CM")}
            rec.stop()
        for key in ("CL", "CD", "CM"):
            ad = float(onp.asarray(derivatives[aero[key], alpha].value)[0, 0])
            fd = (values["plus"][key] - values["minus"][key]) / (2.0 * step)
            relative_error = abs(ad - fd) / max(abs(ad), abs(fd), 1e-8)
            if relative_error > maximum_relative_error:
                maximum_relative_error = relative_error
                worst = (alpha_deg, key, ad, fd)
    print(f"PASS 3 alpha AD-vs-FD max relative error: {maximum_relative_error:.3e}")
    print(f"PASS 3 alpha AD-vs-FD worst case: {worst}")
    ok = maximum_relative_error < 1e-4
    print("neuralfoil_airfoil_model PASS 3 derivative check", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok_multisize = _selftest()
    ok_derivs = _check_derivatives()
    ok_pass3_values = _check_pass3_values()
    ok_pass3_derivs = _check_pass3_alpha_derivatives()
    raise SystemExit(
        0 if (ok_multisize and ok_derivs and ok_pass3_values and ok_pass3_derivs) else 1
    )
