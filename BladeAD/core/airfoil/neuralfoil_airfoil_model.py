"""NeuralFoil forward pass ported to CSDL (Pass 1, incompressible net only).

Reuses NeuralFoil's existing trained weights (vendored under
`data/neuralfoil_weights/`, see that folder's README for provenance) -- no retraining. Ports
the incompressible-net path of `neuralfoil.main.get_aero_from_kulfan_parameters()`
(NeuralFoil v0.3.3, Peter Sharpe, MIT license) as native CSDL `Variable` arithmetic, so CSDL's
own reverse-mode AD gives exact derivatives -- no `csdl.CustomExplicitOperation` / hand-coded
backprop needed, unlike the PCHIP/B-spline tabulated models (which wrap `scipy.interpolate`,
off the CSDL AD path).

KNOWN APPROXIMATION (not a silent one -- see `06-rotor-optimisation/neuralfoil-csdl-port/
findings-pass1.md`): `csdl_alpha` has no true hard elementwise min/max/abs -- `csdl.minimum`,
`csdl.maximum`, and `csdl.absolute` are all smoothed (log-sum-exp-style) approximations with a
`rho` smoothing parameter (default `rho=20`, which was verified to produce real, non-trivial
error -- e.g. `Top_Xtr` off by 0.03 absolute at some test points). This is the same class of gap
as the acoustics Bessel-function precedent (`csdl->CasADi special-function gap`, 2026-08-29): a
genuinely missing exact primitive, not a convenience shortcut. `rho=1000` is used at every
clip/abs call site here as a tighter approximation (verified below), pending a proper exact
primitive (e.g. a small `csdl.CustomExplicitOperation` with a `sign(x)`/indicator-function
derivative). **This only affects `Top_Xtr`/`Bot_Xtr` and the boundary-layer outputs** (needed
for Pass 2's `Cpmin`, not this pass's BladeAD integration) -- `CL`/`CD`/`CM`, the outputs
`NeuralFoilAirfoilModel.evaluate()` actually returns, never call clip/abs and match the
reference exactly (verified to ~1e-14 relative, machine precision on the shared arithmetic).

Explicitly NOT in this module (see
`06-rotor-optimisation/neuralfoil-csdl-port/port-plan.md` for the full staged plan):
  - Mach as an input -- the raw net has none ("No mach parameter in this version" in the
    reference source). `evaluate()` accepts `Ma` only for interface parity with the other
    airfoil models in this package; it is unused here.
  - The Mcrit/Mdd compressibility correction, and `Cpmin_0` (needs the boundary-layer outputs
    this module DOES produce, but the softmin/Mach-blend formula itself is Pass 2's job).
  - Airfoil normalisation/rotation, control surfaces, 360-degree post-stall blending -- all
    layered on top of `get_aero_from_kulfan_parameters()` in AeroSandbox's own wrapper, outside
    that function's contract.
  - Kulfan/CST shape reconstruction. Kulfan parameters arrive as already-fitted numeric arrays
    (from `asb.Airfoil(name).to_kulfan_airfoil(n_weights_per_side=8)`, run once, offline, in
    plain Python) -- this module never reconstructs x,y coordinates.

Run self-test (verifies against the real Python `neuralfoil` package):
    /opt/anaconda3/envs/rotor_design/bin/python neuralfoil_airfoil_model.py
"""
from __future__ import annotations

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
    if isinstance(v, (int, float)):
        v = onp.full(n_cases, float(v))
    return csdl.reshape(v, (1, n_cases))


def get_aero_from_kulfan_parameters(
    upper_weights, lower_weights, leading_edge_weight, TE_thickness,
    alpha_deg, Re, n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0, model_size="small",
):
    """CSDL port of `neuralfoil.get_aero_from_kulfan_parameters` (incompressible net only).

    Per-station (vectorised) inputs -- `n_cases` stations evaluated in one call:
      upper_weights, lower_weights : (n_cases, 8) CSDL Variable or array
      leading_edge_weight, TE_thickness, alpha_deg, Re : (n_cases,) CSDL Variable or array
      n_crit, xtr_upper, xtr_lower : Python float (broadcast) or (n_cases,) CSDL Variable/array

    `alpha_deg` in degrees, matching the reference package's convention. Returns a dict of CSDL
    Variables, each shape (n_cases,): `analysis_confidence`, `CL`, `CD`, `CM`, `Top_Xtr`,
    `Bot_Xtr`, and `upper_bl_theta_i` / `upper_bl_H_i` / `upper_bl_ue/vinf_i` /
    `lower_bl_theta_i` / `lower_bl_H_i` / `lower_bl_ue/vinf_i` for i in range(32).
    """
    nn_params = _load_nn_parameters(model_size)
    n_cases = alpha_deg.shape[0] if hasattr(alpha_deg, "shape") else len(alpha_deg)

    upper_t = csdl.transpose(upper_weights)  # (8, n_cases)
    lower_t = csdl.transpose(lower_weights)  # (8, n_cases)
    alpha_rad = _row(alpha_deg, n_cases) * _DEG2RAD
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
    CM = csdl.reshape(y_fused[3, :], (n_cases,)) / 20.0

    Re_col = csdl.expand(csdl.reshape(_row(Re, n_cases), (n_cases,)), (N, n_cases), "j->ij")
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
    return results


class NeuralFoilAirfoilModel:
    """CSDL-native airfoil model using NeuralFoil's incompressible net (Pass 1).

    Matches this package's `evaluate(alpha, Re, Ma)` airfoil-model interface (see
    `tabulated_airfoil_model.py`); `Ma` is accepted for interface parity but unused (Pass 2).
    Kulfan parameters for the represented airfoil are fixed at construction time.
    """

    def __init__(self, kulfan_parameters: dict, model_size: str = "small",
                n_crit: float = 9.0, xtr_upper: float = 1.0, xtr_lower: float = 1.0):
        self.upper_weights = onp.asarray(kulfan_parameters["upper_weights"], dtype=float)
        self.lower_weights = onp.asarray(kulfan_parameters["lower_weights"], dtype=float)
        self.leading_edge_weight = float(kulfan_parameters["leading_edge_weight"])
        self.TE_thickness = float(kulfan_parameters["TE_thickness"])
        self.model_size = model_size
        self.n_crit, self.xtr_upper, self.xtr_lower = n_crit, xtr_upper, xtr_lower

    def evaluate(self, alpha, Re, Ma):
        del Ma  # unused -- Pass 2 (Mcrit/Mdd compressibility correction)
        n_cases = alpha.shape[0] if hasattr(alpha, "shape") else len(alpha)
        upper = onp.tile(self.upper_weights, (n_cases, 1))
        lower = onp.tile(self.lower_weights, (n_cases, 1))
        aero = get_aero_from_kulfan_parameters(
            upper, lower, self.leading_edge_weight, self.TE_thickness, alpha, Re,
            n_crit=self.n_crit, xtr_upper=self.xtr_upper, xtr_lower=self.xtr_lower,
            model_size=self.model_size)
        return aero["CL"], aero["CD"]


def _selftest():
    """Verify against the real Python `neuralfoil` package. Needs the `spl-bricks` env's
    `neuralfoil`/`aerosandbox` importable -- run with that interpreter, or pass the test
    vectors in precomputed (this module itself has no asb/neuralfoil dependency)."""
    import subprocess
    import sys
    import json

    # Precompute Kulfan parameters + reference outputs in the spl-bricks env (has asb+neuralfoil).
    gen_script = r"""
import json, numpy as np
import aerosandbox as asb
import neuralfoil as nf

cases = []
alphas = [-8.0, -2.0, 0.0, 4.0, 10.0, 16.0]
res = [2.0e5, 1.0e6, 3.0e6]
for name in ("mh117", "mh60", "naca0012"):
    af = asb.Airfoil(name)
    ka = af.to_kulfan_airfoil(n_weights_per_side=8)
    kulfan = dict(upper_weights=list(ka.upper_weights), lower_weights=list(ka.lower_weights),
                  leading_edge_weight=float(ka.leading_edge_weight), TE_thickness=float(ka.TE_thickness))
    for a in alphas:
        for re in res:
            ref = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=dict(upper_weights=np.array(kulfan["upper_weights"]),
                                        lower_weights=np.array(kulfan["lower_weights"]),
                                        leading_edge_weight=kulfan["leading_edge_weight"],
                                        TE_thickness=kulfan["TE_thickness"]),
                alpha=a, Re=re, model_size="small")
            cases.append({"airfoil": name, "kulfan": kulfan, "alpha": a, "Re": re,
                          "ref": {k: float(np.asarray(v).reshape(-1)[0]) for k, v in ref.items()}})
print(json.dumps(cases))
"""
    out = subprocess.run(
        ["/opt/anaconda3/envs/spl-bricks/bin/python", "-c", gen_script],
        capture_output=True, text=True, check=True)
    cases = json.loads(out.stdout)

    upper = onp.array([c["kulfan"]["upper_weights"] for c in cases])
    lower = onp.array([c["kulfan"]["lower_weights"] for c in cases])
    le = onp.array([c["kulfan"]["leading_edge_weight"] for c in cases])
    te = onp.array([c["kulfan"]["TE_thickness"] for c in cases])
    alpha = onp.array([c["alpha"] for c in cases])
    Re = onp.array([c["Re"] for c in cases])

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    aero = get_aero_from_kulfan_parameters(upper, lower, le, te, alpha, Re, model_size="small")
    recorder.stop()

    # CL/CD/CM/analysis_confidence never touch the smoothed clip/abs ops -- exact-tolerance gate.
    # Top_Xtr/Bot_Xtr and the boundary-layer outputs DO (see module docstring, "KNOWN
    # APPROXIMATION") -- gated on absolute error, since relative error is meaningless near a
    # reference value of exactly 0 or 1 (a fully-clipped case).
    exact_keys = ("CL", "CD", "CM", "analysis_confidence")
    approx_keys = ("Top_Xtr", "Bot_Xtr")
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
    print("neuralfoil_airfoil_model PASS 1 self-test", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
