from pathlib import Path

import csdl_alpha as csdl
import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline


_DEFAULT_MH117_TABLE = (
    Path(__file__).parent / "data" / "mh117_completed_viterna_blend_3deg.csv"
)


class _PolarSurface:
    def __init__(self, table_path):
        data = np.genfromtxt(table_path, delimiter=",", names=True, encoding=None)
        self.reynolds = np.unique(data["reynolds"])
        self.alpha_deg = np.unique(data["alpha"])
        expected = len(self.reynolds) * len(self.alpha_deg)
        if len(data) != expected:
            raise ValueError("The polar table must contain a complete Reynolds-alpha grid.")

        order = np.lexsort((data["alpha"], data["reynolds"]))
        self.cl = data["CL"][order].reshape(len(self.reynolds), len(self.alpha_deg))
        self.cd = data["CD"][order].reshape(len(self.reynolds), len(self.alpha_deg))
        self.log_reynolds = np.log(self.reynolds)

    def _prepare(self, alpha, reynolds):
        alpha = np.asarray(alpha, dtype=float)
        reynolds = np.asarray(reynolds, dtype=float)
        if alpha.shape != reynolds.shape:
            raise ValueError("alpha and Re must have identical shapes.")
        requested_alpha_deg = np.rad2deg(alpha.ravel())
        below_physical_domain = requested_alpha_deg < self.alpha_deg[0]
        alpha_deg = np.maximum(requested_alpha_deg, self.alpha_deg[0])
        reynolds_flat = reynolds.ravel()
        if np.any(reynolds_flat < self.reynolds[0]) or np.any(
            reynolds_flat > self.reynolds[-1]
        ):
            raise ValueError(
                f"Re is outside [{self.reynolds[0]:g}, {self.reynolds[-1]:g}]."
            )
        if np.any(requested_alpha_deg > self.alpha_deg[-1]):
            raise ValueError(
                f"alpha exceeds {self.alpha_deg[-1]:g} deg."
            )
        return alpha.shape, alpha_deg, reynolds_flat, below_physical_domain


class _RecursivePchipSurface(_PolarSurface):
    @staticmethod
    def _pchip_slopes_and_direction(x, y, dy):
        h = np.diff(x)
        secant = np.diff(y) / h
        d_secant = np.diff(dy) / h
        slopes = np.zeros_like(y)
        d_slopes = np.zeros_like(y)

        for i in range(1, len(y) - 1):
            if secant[i - 1] == 0 or secant[i] == 0 or np.sign(secant[i - 1]) != np.sign(secant[i]):
                continue
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            denominator = w1 / secant[i - 1] + w2 / secant[i]
            slopes[i] = (w1 + w2) / denominator
            d_denominator = (
                -w1 * d_secant[i - 1] / secant[i - 1] ** 2
                - w2 * d_secant[i] / secant[i] ** 2
            )
            d_slopes[i] = -(w1 + w2) * d_denominator / denominator**2

        for target, first, second in ((0, 0, 1), (-1, -1, -2)):
            h0 = h[first]
            h1 = h[second]
            m0 = secant[first]
            m1 = secant[second]
            dm0 = d_secant[first]
            dm1 = d_secant[second]
            slope = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            d_slope = ((2 * h0 + h1) * dm0 - h0 * dm1) / (h0 + h1)
            if np.sign(slope) != np.sign(m0):
                slope = 0.0
                d_slope = 0.0
            elif np.sign(m0) != np.sign(m1) and abs(slope) > 3 * abs(m0):
                slope = 3 * m0
                d_slope = 3 * dm0
            slopes[target] = slope
            d_slopes[target] = d_slope
        return slopes, d_slopes

    @classmethod
    def _pchip_value_and_direction(cls, x, y, dy, xq):
        slopes, d_slopes = cls._pchip_slopes_and_direction(x, y, dy)
        interval = min(max(np.searchsorted(x, xq) - 1, 0), len(x) - 2)
        h = x[interval + 1] - x[interval]
        t = (xq - x[interval]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        value = (
            h00 * y[interval]
            + h10 * h * slopes[interval]
            + h01 * y[interval + 1]
            + h11 * h * slopes[interval + 1]
        )
        direction = (
            h00 * dy[interval]
            + h10 * h * d_slopes[interval]
            + h01 * dy[interval + 1]
            + h11 * h * d_slopes[interval + 1]
        )
        return value, direction

    def _coefficient(self, values, alpha_deg, reynolds):
        alpha_splines = [
            PchipInterpolator(self.alpha_deg, row, extrapolate=False) for row in values
        ]
        coefficient = np.empty_like(alpha_deg)
        d_coefficient_d_alpha_deg = np.empty_like(alpha_deg)
        d_coefficient_d_reynolds = np.empty_like(alpha_deg)
        for i, (angle, re_value) in enumerate(zip(alpha_deg, reynolds)):
            values_at_alpha = np.array([spline(angle) for spline in alpha_splines])
            alpha_slopes = np.array(
                [spline.derivative()(angle) for spline in alpha_splines]
            )
            reynolds_spline = PchipInterpolator(
                self.log_reynolds, values_at_alpha, extrapolate=False
            )
            log_re = np.log(re_value)
            coefficient[i], d_coefficient_d_alpha_deg[i] = (
                self._pchip_value_and_direction(
                    self.log_reynolds, values_at_alpha, alpha_slopes, log_re
                )
            )
            d_coefficient_d_reynolds[i] = (
                reynolds_spline.derivative()(log_re) / re_value
            )
        return coefficient, d_coefficient_d_alpha_deg, d_coefficient_d_reynolds

    def predict(self, alpha, reynolds):
        shape, alpha_deg, reynolds_flat, below_physical_domain = self._prepare(
            alpha, reynolds
        )
        cl, dcl_da_deg, dcl_dre = self._coefficient(
            self.cl, alpha_deg, reynolds_flat
        )
        cd, dcd_da_deg, dcd_dre = self._coefficient(
            self.cd, alpha_deg, reynolds_flat
        )
        degrees_per_radian = 180.0 / np.pi
        dcl_da_deg[below_physical_domain] = 0.0
        dcd_da_deg[below_physical_domain] = 0.0
        return tuple(
            item.reshape(shape)
            for item in (
                cl,
                cd,
                dcl_da_deg * degrees_per_radian,
                dcd_da_deg * degrees_per_radian,
                dcl_dre,
                dcd_dre,
            )
        )


class _BSplineSurface(_PolarSurface):
    def __init__(self, table_path):
        super().__init__(table_path)
        self._cl_spline = RectBivariateSpline(
            self.log_reynolds, self.alpha_deg, self.cl, kx=3, ky=3, s=0
        )
        self._cd_spline = RectBivariateSpline(
            self.log_reynolds, self.alpha_deg, self.cd, kx=3, ky=3, s=0
        )

    def predict(self, alpha, reynolds):
        shape, alpha_deg, reynolds_flat, below_physical_domain = self._prepare(
            alpha, reynolds
        )
        log_re = np.log(reynolds_flat)
        cl = self._cl_spline.ev(log_re, alpha_deg)
        cd = self._cd_spline.ev(log_re, alpha_deg)
        degrees_per_radian = 180.0 / np.pi
        dcl_da = self._cl_spline.ev(log_re, alpha_deg, dy=1) * degrees_per_radian
        dcd_da = self._cd_spline.ev(log_re, alpha_deg, dy=1) * degrees_per_radian
        dcl_da[below_physical_domain] = 0.0
        dcd_da[below_physical_domain] = 0.0
        dcl_dre = self._cl_spline.ev(log_re, alpha_deg, dx=1) / reynolds_flat
        dcd_dre = self._cd_spline.ev(log_re, alpha_deg, dx=1) / reynolds_flat
        return tuple(
            item.reshape(shape)
            for item in (cl, cd, dcl_da, dcd_da, dcl_dre, dcd_dre)
        )


class _TabulatedAirfoilOperation(csdl.CustomExplicitOperation):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)
        shape = alpha.shape
        if shape != Re.shape or shape != Ma.shape:
            raise ValueError("alpha, Re, and Ma must have identical shapes.")
        if len(shape) not in (1, 2, 3):
            raise NotImplementedError("Only one-, two-, and three-dimensional inputs are supported.")
        indices = np.arange(np.prod(shape))
        cl = self.create_output("Cl", shape)
        cd = self.create_output("Cd", shape)
        for output in ("Cl", "Cd"):
            self.declare_derivative_parameters(
                output, "alpha", rows=indices, cols=indices
            )
            self.declare_derivative_parameters(output, "Re", rows=indices, cols=indices)
            self.declare_derivative_parameters(output, "Ma", dependent=False)
        return cl, cd

    def compute(self, inputs, outputs):
        cl, cd, _, _, _, _ = self.surface.predict(inputs["alpha"], inputs["Re"])
        outputs["Cl"] = cl
        outputs["Cd"] = cd

    def compute_derivatives(self, inputs, outputs, derivatives):
        _, _, dcl_da, dcd_da, dcl_dre, dcd_dre = self.surface.predict(
            inputs["alpha"], inputs["Re"]
        )
        derivatives["Cl", "alpha"] = dcl_da.ravel()
        derivatives["Cd", "alpha"] = dcd_da.ravel()
        derivatives["Cl", "Re"] = dcl_dre.ravel()
        derivatives["Cd", "Re"] = dcd_dre.ravel()


class _TabulatedAirfoilModel:
    surface_class = None

    def __init__(self, table_path=None):
        if table_path is None:
            table_path = _DEFAULT_MH117_TABLE
        table_path = Path(table_path)
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        self.surface = self.surface_class(table_path)

    def evaluate(self, alpha, Re, Ma):
        return _TabulatedAirfoilOperation(self.surface).evaluate(alpha, Re, Ma)

    def predict(self, alpha, Re, derivatives=False):
        values = self.surface.predict(alpha, Re)
        return values if derivatives else values[:2]


class MH117PchipAirfoilModel(_TabulatedAirfoilModel):
    """MH117 recursive PCHIP polar with constant support below its table domain."""

    surface_class = _RecursivePchipSurface


class MH117BSplineAirfoilModel(_TabulatedAirfoilModel):
    """MH117 cubic B-spline polar with constant support below its table domain."""

    surface_class = _BSplineSurface
