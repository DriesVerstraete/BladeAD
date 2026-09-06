import csdl_alpha as csdl
import numpy as np
import scipy.sparse
from BladeAD.utils.parameterization_custom_op import BSplineParameterizationExplicitOperation


def get_bspline_mtx(num_cp, num_pt, order=4, sample_locations=None):
    order = min(order, num_cp)

    knots = np.zeros(num_cp + order)
    knots[order-1:num_cp+1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0

    if sample_locations is None:
        t_vec = np.linspace(0, 1, num_pt)
    else:
        t_vec = np.asarray(sample_locations, dtype=float).reshape(-1)
        if t_vec.shape[0] != num_pt:
            raise ValueError(
                f"sample_locations has length {t_vec.shape[0]}, expected num_pt={num_pt}"
            )
        t_vec = np.clip(t_vec, 0.0, 1.0)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()


    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_pt, num_cp),
    )


class BsplineParameterization:
    """B-spline parameterization for radial profiles.


    Parameters
    ----------
    num_radial : int
        Number of radial stations.
    num_cp : int
        Number of B-spline control points.
    order : int, optional
        Order of B-spline, by default 4.
    sample_locations : np.ndarray, optional
        Parameter values in [0, 1] at which to sample the B-spline, length
        ``num_radial``. Default None -> uniform ``np.linspace(0, 1, num_radial)``
        (unchanged legacy behaviour). Pass the normalized physical radial
        station locations here when using a non-uniform BEM grid so control
        points map to fixed span fractions regardless of station spacing.

    Raises
    ------
    ValueError
        B-spline order cannot be greater than the number of control points.
    ValueError
        Number of control points cannot be greater than the number of radial stations.
    """
    def __init__(
            self,
            num_radial,
            num_cp,
            order: int=4,
            sample_locations=None,
        ) -> None:
        csdl.check_parameter(num_cp, "num_radial", types=int)
        csdl.check_parameter(num_cp, "num_cp", types=int)
        csdl.check_parameter(order, "order", types=int)
        
        if order > num_cp:
            raise ValueError("B-spline order cannot be greater than the number of control points.")

        if num_cp > num_radial:
            raise ValueError("number of control points cannot be greater than the number of radial stations.")

        self.num_cp = num_cp
        self.order = order
        self.num_radial = num_radial
        self.b_spline_mat = get_bspline_mtx(num_cp, num_radial, order, sample_locations)

    def evaluate_radial_profile(self, control_points : csdl.Variable):
        """Evaluate radial profile using B-spline parameterization.

        Parameters
        ----------
        control_points : csdl.Variable
            B-spline control points.

        Returns
        -------
        csdl.Variable
            Radial profile.
        """
        b_spline_exp_op = BSplineParameterizationExplicitOperation(
            num_radial=self.num_radial, 
            num_cp=self.num_cp,
            jac=self.b_spline_mat,
        )

        return b_spline_exp_op.evaluate(control_points)


