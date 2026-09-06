"""Compressibility layer on top of BladeAD's incompressible NeuralFoil B-spline
polar tables. Case-agnostic -- depends only on a table path + a section t/c.

Applies AeroSandbox's `get_aero_from_neuralfoil` Mach correction (the same method
`neuralfoil_airfoil_model.py`'s Pass 2 implements) using a per-section critical
Mach `mach_crit(alpha, Re)` read from the table -- so NeuralFoil is never called
at runtime:

  beta  = max(1 - Ma^2, -(1 - Ma^2), rho=2)^0.5        # soft-abs, stays real past M=1
  Cl    = Cl0 / beta  * buffet_factor * supersonic_factor
  Cd    = Cd0 + WaveDrag(Ma, mach_crit, t/c)           # NO Prandtl-Glauert on Cd -- matches the source
  mach_dd = mach_crit + (0.1/320)^(1/3)

The 4-branch piecewise wave-drag law (`_WaveDragOperation`) is copied verbatim
from `BladeAD/core/airfoil/neuralfoil_airfoil_model.py` (itself a port of the
AeroSandbox reference); CSDL has no hard conditional so it stays a narrow custom
op with analytic diagonal derivatives.

`mach_crit` in the table is NeuralFoil's incompressible-Cpmin-derived value
(alpha/Re dependent: a deeper suction peak lowers it), stored by the table
generator alongside CL/CD.
"""
import csv

import numpy as onp
import csdl_alpha as csdl

from BladeAD.core.airfoil.tabulated_airfoil_model import (
    _BSplineSurface, _TabulatedAirfoilModel, _TabulatedAirfoilOperation,
)
from scipy.interpolate import RectBivariateSpline

_MACH_DD_OFFSET = (0.1 / 320.0) ** (1.0 / 3.0)


class _MachCritBSplineSurface(_BSplineSurface):
    """`_BSplineSurface` plus a third spline for `mach_crit(log Re, alpha)`."""

    def __init__(self, table_path):
        super().__init__(table_path)
        rows = list(csv.DictReader(open(table_path)))
        re = onp.array([float(r["reynolds"]) for r in rows])
        al = onp.array([float(r["alpha"]) for r in rows])
        mc = onp.array([float(r["mach_crit"]) for r in rows])
        order = onp.lexsort((al, re))
        grid = mc[order].reshape(len(self.reynolds), len(self.alpha_deg))
        self._mach_crit_spline = RectBivariateSpline(
            self.log_reynolds, self.alpha_deg, grid, kx=3, ky=3, s=0)

    def predict_mach_crit(self, alpha, reynolds):
        shape, alpha_deg, reynolds_flat, below = self._prepare(alpha, reynolds)
        log_re = onp.log(reynolds_flat)
        mc = self._mach_crit_spline.ev(log_re, alpha_deg)
        dmc_da = self._mach_crit_spline.ev(log_re, alpha_deg, dy=1) * (180.0 / onp.pi)
        dmc_dre = self._mach_crit_spline.ev(log_re, alpha_deg, dx=1) / reynolds_flat
        dmc_da[below] = 0.0
        return tuple(x.reshape(shape) for x in (mc, dmc_da, dmc_dre))


class _MachCritOperation(csdl.CustomExplicitOperation):
    """mach_crit(alpha, Re) as a differentiable CSDL op (alpha/Re derivatives only)."""

    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def evaluate(self, alpha, Re):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        indices = onp.arange(alpha.size)
        out = self.create_output("mach_crit", alpha.shape)
        self.declare_derivative_parameters("mach_crit", "alpha", rows=indices, cols=indices)
        self.declare_derivative_parameters("mach_crit", "Re", rows=indices, cols=indices)
        return out

    def compute(self, inputs, outputs):
        outputs["mach_crit"] = self.surface.predict_mach_crit(inputs["alpha"], inputs["Re"])[0]

    def compute_derivatives(self, inputs, outputs, derivatives):
        _, dmc_da, dmc_dre = self.surface.predict_mach_crit(inputs["alpha"], inputs["Re"])
        derivatives["mach_crit", "alpha"] = dmc_da.ravel()
        derivatives["mach_crit", "Re"] = dmc_dre.ravel()


class _WaveDragOperation(csdl.CustomExplicitOperation):
    # Copied verbatim from BladeAD/core/airfoil/neuralfoil_airfoil_model.py (Pass 2).
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
        mach_dd = mach_crit + _MACH_DD_OFFSET
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
        line_a = (x - x_a) * 0.1 + 80.0 * _MACH_DD_OFFSET ** 4
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
        v = self._values_and_derivatives(inputs["mach"], inputs["mach_crit"], inputs["t_over_c"])
        derivatives["CD_wave", "mach"] = v[1].ravel()
        derivatives["CD_wave", "mach_crit"] = v[2].ravel()
        derivatives["CD_wave", "t_over_c"] = v[3].ravel()


def _blend(switch, high, low):
    weight_high = 0.5 + 0.5 * csdl.tanh(switch)
    return high * weight_high + low * (1.0 - weight_high)


class MachCorrectedBSplineAirfoilModel(_TabulatedAirfoilModel):
    """Incompressible NeuralFoil B-spline polar + AeroSandbox Mach correction.

    Parameters
    ----------
    table_path : str
        `<name>_neuralfoil_table.csv` with a `mach_crit` column.
    t_over_c : float
        section maximum thickness / chord, for the wave-drag branch.
    clamp_reynolds : bool
        clamp Re to the table span rather than hard-failing when a gradient-based
        optimiser transiently probes outside it (the converged design's Re is
        checked in-range separately). Default True.
    """

    surface_class = _MachCritBSplineSurface

    def __init__(self, table_path=None, t_over_c=0.12, clamp_reynolds=True):
        super().__init__(table_path)
        self.t_over_c = float(t_over_c)
        self.surface.clamp_reynolds = bool(clamp_reynolds)

    def evaluate(self, alpha, Re, Ma):
        cl0, cd0 = _TabulatedAirfoilOperation(self.surface).evaluate(alpha, Re, Ma)
        mach_crit = _MachCritOperation(self.surface).evaluate(alpha, Re)
        mach_dd = mach_crit + _MACH_DD_OFFSET

        # Ideal Prandtl-Glauert sqrt(1 - Ma^2), floored at Ma ~ 0.97 so it stays real and the
        # amplification is capped near sonic. The AeroSandbox reference uses a softness-0.5
        # soft-abs here (matched bit-for-bit by the NeuralFoil port); that inflates beta by
        # ~3-8% over Ma 0.5-0.65 -- deliberately tightened here, our tips run right in that band.
        beta = csdl.maximum(1.0 - Ma ** 2, Ma * 0.0 + 0.05, rho=20.0) ** 0.5

        cl = cl0 / beta
        buffet_factor = _blend(50.0 * (Ma - (mach_dd + 0.04)),
                               _blend((Ma - 1.0) / 0.1, 1.0, 0.5), 1.0)
        supersonic_factor = _blend((Ma - 1.0) / 0.1, 4.0 / (2.0 * onp.pi), 1.0)
        cl = cl * buffet_factor * supersonic_factor

        t_over_c = Ma * 0.0 + self.t_over_c            # broadcast the section scalar to Ma's shape
        cd = cd0 + _WaveDragOperation().evaluate(Ma, mach_crit, t_over_c)
        return cl, cd
