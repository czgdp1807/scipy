#!/usr/bin/env python3
"""
Regrid (2-D smoothing B-splines via separable 1-D FITPACK kernels)
==================================================================

1) Overview
-----------
This module fits a bivariate tensor-product B-spline surface to gridded data
`Z[i, j]` sampled on strictly increasing coordinates `x[i]`, `y[j]`. It mirrors
the adaptive spirit of FITPACK's REGRID/fpgrre (knot growth + smoothing
parameter search to meet a target residual `s`) while keeping the control flow
explicit in Python and returning a 2-D `NdBSpline`.

**Key conventions**
- **Penalty scaling is 1/p** - *same as FITPACK*. The augmented system stacks
  `(D / p)` under the data matrix. Smaller `p` -> stronger smoothing; larger `p`
  -> weaker smoothing (approaching interpolation).
- `p == -1` is used **as a sentinel for `p = ∞`** (interpolatory limit): when
  seen, the solver omits penalty rows entirely.

2) Mathematics (and how this differs from FITPACK's REGRID)
-----------------------------------------------------------
Let `A_x`, `A_y` be banded 1-D design matrices and `D_x`, `D_y` the banded
1-D roughness (difference) matrices returned by FITPACK/Dierckx's 1-D APIs
(`data_matrix`, `disc`). The 2-D smoothing objective is

    minimize  ||A c - z||^2 + (1/p) ||D c||^2,            (1)

implemented by the **augmented** least squares system

    [ A ] c ~ [ z ]
    [D/p]     [ 0 ].

**Same as FITPACK:** Equation (1) uses the *1/p* convention.
**Different in this module:** We realize the 2-D problem by composing **1-D**
banded operators in two separable passes (x then y), instead of calling a
monolithic 2-D routine. This makes the mathematics transparent while producing
the same normal equations structure that REGRID targets internally.

3) Residual energy: definition and use
--------------------------------------
After solving on the current knots (initially at the interpolatory limit,
`p = ∞`), we evaluate the surface and form

    R = Z - Zhat,                         fp = sum(R[i, j]^2).

We then **project** residual energy to knot spans along each axis:

- Row energy  `row_energy[i] = sum(R[i, j]^2, j)`  -> accumulated into `fpintx`.
- Column energy `col_energy[j] = sum(R[i, j]^2, i)` -> accumulated into `fpinty`.

These per-span energies guide **adaptive knot insertion**: the algorithm picks
high-energy spans (skipping zero-length ones) and inserts data-aligned knots
(e.g., median sample within the span), with simple batch sizing and headroom
limits. This is conceptually the same idea as REGRID's `fpint` arrays; here it
is implemented explicitly and vectorized in Python for clarity.

4) Solver used here vs. FITPACK's REGRID
----------------------------------------
**This module (separable 1-D composition):**
- Uses **1-D FITPACK/Dierckx kernels** (`data_matrix`, `disc`, `qr_reduce`,
  `fpback`) to solve 2-D via two passes:
  1) augment/QR/backsolve along **x**, producing an intermediate;
  2) augment/QR/backsolve along **y**, producing the coefficient grid `C`.
- The augmented rows are stacked as `[A; D/p]` (1/p scaling, **same as FITPACK**).
- If `fp > s` after knot growth, performs a **scalar search in `p`** using a
  ratio-of-roots routine; `p == -1` is treated as `p = ∞` for the interpolatory
  reference without penalty rows.

**FITPACK REGRID (monolithic 2-D routine):**
- Implements the same mathematical objective and **1/p** penalty scaling,
  but inside a specialized, fully 2-D Fortran routine with in-place Givens/QR
  and a rational update for `p`.
- Handles residual partitioning, knot insertion, and `p` updates internally.

**Practical difference:** We build the 2-D solve from **1-D building blocks**,
which keeps each stage observable/testable and uses the same low-level kernels
as FITPACK, but without relying on the single monolithic REGRID entry point.

5) Execution flow (who calls whom)
----------------------------------
1. **`regrid_python`**
   - Validates inputs (strictly increasing `x`, `y`; grid shape; `s >= 0`),
     normalizes `bbox`,
   - Delegates to `_regrid_python_fitpack`.

2. **`_regrid_python_fitpack`**
   - Applies `bbox` -> `(x_fit, y_fit, Z_fit)`.
   - Initializes clamped knots using `_generate_knots`:
     - if `s == 0`: interpolatory setup (no internal knots),
     - else: open-uniform ends without interior knots.
   - **Loop:**
     - Build 1-D banded matrices via `build_matrices` (`data_matrix`, `disc`).
     - Call `_solve_2d_fitpack` with `p = ∞` (sentinel `-1`) on the current knots.
     - If `s == 0`: return; else if `fp < s`: break; else compute residuals,
       decide new knots with `_generate_knots` (x and y alternate), and continue.
   - If the loop ends with `fp > s`, call `_p_search_hit_s` to tune `p`
     to achieve `fp(p) ~ s`, and return the resulting `NdBSpline`.

3. **`_p_search_hit_s`**
   - Wraps the problem in `F` (maps `p` -> `fp(p)` by invoking `_solve_2d_fitpack`).
   - Uses `root_rati` to find `p*` with `fp(p*) ~ s`, caching `C`.

4. **`_solve_2d_fitpack`**
   - Performs the two separable 1-D augmented QR solves (`[A; D/p]`) along x then y,
     evaluates `fp`, and returns `(C, fp)`.

The final `(tx, ty, C)` are packaged as an `NdBSpline` for convenient evaluation.
"""
import numpy as np
from scipy.interpolate import BSpline, NdBSpline
from scipy.interpolate._fitpack_repro import (
    root_rati, disc, add_knot, _not_a_knot)
from . import _dierckx

def ndbspline_call_like_bivariate(ndbs, x, y, dx=0, dy=0, grid=True):
    """
    Evaluate a 2D ``NdBSpline`` like a classical bivariate API.

    Parameters
    ----------
    ndbs : NdBSpline
        A 2D spline object (``len(ndbs.t) == 2``).
    x, y : array_like
        Sample locations. If ``grid=True``, these must be 1-D strictly
        increasing vectors. If ``grid=False``, they can be broadcastable
        arrays of the same shape.
    dx, dy : int, optional
        Derivative orders along x and y respectively, by default 0.
    grid : bool, optional
        If True, evaluate on the cartesian product of ``x`` and ``y``;
        otherwise treat ``(x, y)`` as paired coordinates, by default True.

    Returns
    -------
    ndarray or (ndarray, dict)
        Evaluated values with shape:
        - ``(len(x), len(y), ...)`` if ``grid=True``.
        - ``x.shape + ...`` if ``grid=False``.

    Raises
    ------
    ValueError
        If ``ndbs`` is not 2D, derivatives are negative, or monotonicity checks fail.

    Notes
    -----
    This is a thin convenience wrapper around ``NdBSpline.__call__`` with input
    validation and optional profiling.
    """
    if len(ndbs.t) != 2:
        raise ValueError("ndbs must be a 2D NdBSpline (len(t) == 2).")
    if not (isinstance(dx, int | np.integer) and isinstance(dy, int | np.integer)):
        raise ValueError("dx and dy must be integers.")
    if dx < 0 or dy < 0:
        raise ValueError("order of derivative must be positive or zero")

    trailing = ndbs.c.shape[2:]

    if grid:
        x = np.asarray(x)
        y = np.asarray(y)

        if x.size == 0 or y.size == 0:
            vals = np.zeros((x.size, y.size) + trailing, dtype=ndbs.c.dtype)
            return vals

        if (x.size >= 2) and (not np.all(np.diff(x) >= 0.0)):
            raise ValueError("x must be strictly increasing when `grid` is True")
        if (y.size >= 2) and (not np.all(np.diff(y) >= 0.0)):
            raise ValueError("y must be strictly increasing when `grid` is True")

        X, Y = np.meshgrid(x, y, indexing="ij")
        xi = np.stack((X, Y), axis=-1)  # (len(x), len(y), 2)

        vals = ndbs(xi, nu=(dx, dy), extrapolate=ndbs.extrapolate)

        return vals
    else:
        x = np.asarray(x)
        y = np.asarray(y)

        if x.shape != y.shape:
            x, y = np.broadcast_arrays(x, y)

        if x.size == 0:
            return np.zeros(x.shape + trailing, dtype=ndbs.c.dtype)
        xi = np.stack((x.ravel(), y.ravel()), axis=-1)
        vals = ndbs(xi, nu=(dx, dy), extrapolate=ndbs.extrapolate)
        return vals.reshape(x.shape + trailing)

def return_NdBSpline(fp, tck, degrees):
    """
    Build a 2D ``NdBSpline`` from knot vectors and a coefficient grid.

    Parameters
    ----------
    fp : float
        Residual sum of squares of the produced fit (kept for upstream use).
    tck : tuple
        Tuple ``(tx, ty, C)`` where ``tx``, ``ty`` are knot vectors and ``C``
        is a coefficient array with shape ``(nx - kx - 1, ny - ky - 1)`` or
        a compatible shape that can be reshaped to that.
    degrees : tuple of int
        Degrees ``(kx, ky)`` along x and y.

    Returns
    -------
    NdBSpline
        The constructed 2D spline.

    Notes
    -----
    Only repacks the coefficient grid; ``fp`` is not used internally here.
    """
    nx, ny = len(tck[0]), len(tck[1])
    kx, ky = degrees
    c = tck[2].reshape(nx - kx - 1, ny - ky - 1)
    return NdBSpline((tck[0], tck[1]), c, degrees)


def _stack_augmented_fitpack(A, offs_a, D, offs_d, nc, k, p):
    """
    Stack data and smoothing-penalty rows for banded QR, using 1/p weighting.

    Parameters
    ----------
    A : ndarray, shape (m, bw_a)
        Banded data/design matrix for one axis (from `_dierckx.data_matrix`).
    offs_a : ndarray
        Row-wise band offsets for `A`.
    D : ndarray, shape (r, bw_d)
        Banded roughness (difference) penalty matrix for the same axis
        (from `disc`).
    offs_d : ndarray
        Row-wise band offsets for `D`.
    nc : int
        Number of top (data) rows from `A` to include.
    k : int
        Spline degree (used only for sizing in the current implementation).
    p : float
        Smoothing parameter. The effective penalization term is scaled as **1/p**:
        larger `p` means *less* smoothing (approaching interpolation).
        If `p == -1`, it signals *p -> inf*, i.e. a pure interpolatory system
        with **no** penalty rows appended.

    Returns
    -------
    AA : ndarray
        Augmented banded matrix with `A` stacked over `(D / p)` when `p != -1`.
    offset : ndarray
        Concatenated band offsets for the augmented matrix.
    nc : int
        Returned unchanged for downstream convenience.

    Notes
    -----
    This formulation follows the FITPACK convention where the smoothing equation

        min ||A c - z||^2 + (1/p) ||D c||^2

    is implemented via stacking `D / p` beneath `A`. Setting `p = -1` signals
    infinite `p` (interpolation, no smoothing penalty).
    """
    if p == -1:
        return A, offs_a, nc

    nz = k + 1
    AA = np.zeros((nc + D.shape[0], k+2), dtype=float)
    AA[:nc, :nz] = A[:nc, :]
    AA[nc:, :] = D / p
    offset = np.r_[offs_a, offs_d]
    return AA, offset, nc


def _solve_2d_fitpack(Ax, offs_x, ncx,
                      Dx, offs_dx,
                      Ay, offs_y, ncy, Q,
                      Dy, offs_dy, p,
                      kx, tx, x_x,
                      ky, ty, x_y, z):
    """
    Solve the 2-D tensor-product spline system using separable banded QR.

    ================================================================
    Mathematical model (step by step, plain text)
    ================================================================

    Shapes:
        Z      : (mx, my)  -> original data
        Ax, Ay : design matrices for x and y
        Dx, Dy : roughness penalty matrices for x and y
        C      : (nx, ny)  -> spline coefficients to solve for

    Surface approximation:
        Zhat = Ax * C * Ay^T

    Objective (smoothing formulation):
        minimize ||Ax*C*Ay^T - Z||^2 + (1/p)*(||Dx*C||^2 + ||C*Dy^T||^2)

    In practice (FITPACK-style separable approach), we solve this in two stages:

    --------------------------------------------------------
    Stage 1 (x-direction solve for all y-columns together):
    --------------------------------------------------------

        For each column of Z:
            minimize ||Ax*T - Z||^2 + (1/p)*||Dx*T||^2

        This is equivalent to the augmented least-squares system:
            [Ax]       [Z]
            [Dx/p] * T = [0]

        i.e.  minimize || [Ax; Dx/p]*T - [Z; 0] ||^2

        The solution T is obtained by QR reduction and back-substitution.

    --------------------------------------------------------
    Stage 2 (y-direction solve using transposed result):
    --------------------------------------------------------
        Now treat T^T as the new RHS for the y-direction:
            minimize ||Ay*C^T - T^T||^2 + (1/p)*||Dy*C^T||^2

        Equivalent to augmented system:
            [Ay]       [T^T]
            [Dy/p] * C^T = [0]

        i.e.  minimize || [Ay; Dy/p]*C^T - [T^T; 0] ||^2

        Solving this gives C^T (then transposed back to C).

    --------------------------------------------------------
    Interpolation limit:
    --------------------------------------------------------
        If p == -1, penalties are omitted (Dx, Dy are not stacked).
        The solver behaves as a near-interpolating system.

    --------------------------------------------------------
    Residual computation:
    --------------------------------------------------------
        Zhat = Ax * C * Ay^T
        fp   = sum((Z - Zhat)^2)

    Parameters
    ----------
    Ax, Ay : ndarray
        Banded data matrices for the x and y axes.
    offs_x, offs_y : ndarray
        Band offsets for `Ax` and `Ay`.
    ncx, ncy : int
        Number of top (data) rows in `Ax` and `Ay`.
    Dx, Dy : ndarray
        Banded roughness penalty matrices for x and y.
    offs_dx, offs_dy : ndarray
        Band offsets for `Dx` and `Dy`.
    ncdx, ncdy : int
        Penalty matrix row counts (from `disc`), informational only.
    Q : ndarray, shape (mx, my)
        RHS data grid (copied from `Z`).
    p : float
        Smoothing parameter. The penalty term is scaled as **1/p**.
        Setting `p == -1` signals *p -> inf* (interpolation, omit penalty).
    kx, ky : int
        Spline degrees along x and y.
    tx, ty : ndarray
        Knot vectors along x and y.
    x_x, x_y : ndarray
        Sample coordinates.
    z : ndarray
        Original data grid for residual evaluation.

    Returns
    -------
    C : ndarray
        2-D B-spline coefficient grid.
    fp : float
        Residual sum of squares between fitted surface and `z`.

    Notes
    -----
    This performs two separable QR solves (x then y), each augmented by
    `(D / p)` when `p != -1`.  Setting `p = -1` skips all penalty rows,
    yielding an interpolatory surface.  The resulting coefficients and residual
    follow the same conventions as FITPACK's `fpgrre`.
    """
    # Dummy unit weights for FITPACK fpback APIs.
    w_x = np.ones_like(x_x)
    w_y = np.ones_like(x_y)

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L97-L105
    # Build the augmented banded matrix for x:
    #   - If p != -1, stack (Dx / p) under Ax for FITPACK-style smoothing.
    #   - If p == -1, _stack_augmented_fitpack omits the penalty part entirely.
    # Returns:
    #   Ax_aug      : augmented banded matrix (data [+ penalty]).
    #   offset_aug_x: band offsets compatible with Ax_aug.
    #   nc_augx     : number of top data rows within Ax_aug (== ncx).
    Ax_aug, offset_aug_x, nc_augx = _stack_augmented_fitpack(
        Ax, offs_x, Dx, offs_dx, ncx, kx, p)

    # Same for y: build Ay_aug with (Dy / p) stacked if p != -1.
    Ay_aug, offset_aug_y, nc_augy = _stack_augmented_fitpack(
        Ay, offs_y, Dy, offs_dy, ncy, ky, p)

    # If we stacked penalty rows on the x side, the RHS must be padded with zeros
    # to match the augmented row count for the QR reduction call.
    if p != -1: # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L97
        # Dx.shape[0] is the number of penalty rows; add that many zero rows
        # so Ax_aug and Q have compatible leading dimensions for in-place QR.
        # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L110-L118
        Q = np.vstack([Q, np.zeros((Dx.shape[0], Q.shape[1]), dtype=float)])

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L106-L175
    # Perform in-place banded QR reduction of the x-augmented system:
    # This orthogonalizes/eliminates along x for all RHS columns in Q simultaneously.
    # After this, fpback can do x-direction back-substitution to
    # get c^T (partial coeffs).
    _dierckx.qr_reduce(Ax_aug, offset_aug_x, nc_augx, Q)

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L246-L253
    # Back-substitute along x to solve the reduced system:
    #   cT has shape (ny_data_like, ncoef_x) in this calling pattern, i.e. per y-column.
    # The API uses:
    #   Ax_aug, nc_augx: reduced upper structure
    #   x_x, tx, kx, w_x: x-sample grid, knot vector, degree, and (unit) weights
    #   Q: RHS (current)
    T, _, _ = _dierckx.fpback(
        Ax_aug, nc_augx, x_x,
        Q, tx, kx, w_x,
        Q, False
    )

    # We now want to treat the *y*-direction solve with these as the new RHS.
    # Transpose so each column corresponds to a y-solve RHS consistently.
    Q = np.ascontiguousarray(T.T)

    # If we stacked penalty rows on the y side, pad RHS with zeros to match Ay_aug.
    if p != -1: # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L97
        # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L110-L118
        Q = np.vstack([Q, np.zeros((Dy.shape[0], Q.shape[1]), dtype=float)])

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L176-L245
    # Perform in-place banded QR reduction along y for all columns of Q.
    _dierckx.qr_reduce(Ay_aug, offset_aug_y, nc_augy, Q)

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpgrre.f#L254-L269
    # Final back-substitution along y:
    # Returns:
    #   C  : coefficient matrix (orientation matches Ay/BSpline expectations here)
    #   fp : FITPACK's internal residual metric from the y-solve
    #        (we recompute below anyway)
    C, _, fp = _dierckx.fpback(
        Ay_aug, nc_augy,
        x_y, Q, ty, ky, w_y,    # y-grid, y-knots, degree, weights
        Q,                      # RHS -> solution becomes coefficients along y
        False
    )

    # Build explicit (dense) design matrices to evaluate the fitted surface:
    # _Ax: [mx × nx_coef], _Ay: [my × ny_coef]
    _Ax = BSpline.design_matrix(x_x, tx, kx, extrapolate=False)
    _Ay = BSpline.design_matrix(x_y, ty, ky, extrapolate=False)

    # Evaluate the fitted surface: zhat = Ax * C^T * Ay^T
    # Note: C currently aligns so that C.T matches x-first multiplication order.
    zhat = _Ax @ C.T @ _Ay.T

    # Compute the residual sum of squares against the original data z.
    fp = np.sum(np.square(z - zhat))

    # Return coefficients in the conventional (nx_coef, ny_coef) orientation and fp.
    return C.T, fp

class F:
    """
    Callable wrapper for computing `fp(p)` for a fixed spline configuration.

    Parameters
    ----------
    Ax, Ay : ndarray
        Banded data matrices.
    offs_x, offs_y : ndarray
        Band offsets for the data matrices.
    ncx, ncy : int
        Number of data rows in `Ax` and `Ay`.
    Dx, Dy : ndarray
        Banded penalty matrices.
    offs_dx, offs_dy : ndarray
        Band offsets for the penalty matrices.
    ncdx, ncdy : int
        Penalty matrix row counts.
    kx, ky : int
        Degrees along x and y.
    tx, ty : ndarray
        Knot vectors along x and y.
    x_x, x_y : ndarray
        Sample coordinates.
    w_x, w_y : ndarray
        Weights (usually ones).
    z : ndarray
        Data grid for computing the residual.

    Attributes
    ----------
    C : ndarray
        Coefficient matrix from the most recent solve.
    fp : float
        Residual value from the most recent solve.

    Notes
    -----
    The penalty is applied as **1/p**, so smaller `p` values yield heavier
    smoothing. Setting `p == -1` corresponds to *p = inf*, i.e. interpolation.
    Intended for use by `_p_search_hit_s` to iteratively evaluate `fp(p)`.
    """

    def __init__(self, Ax, offs_x, ncx,
                 Dx, offs_dx, Ay,
                 offs_y, ncy, Q, Dy,
                 offs_dy, kx,
                 tx, x_x, ky, ty,
                 x_y, z):
        self.Ax = Ax
        self.offs_x = offs_x
        self.ncx = ncx
        self.Dx = Dx
        self.offs_dx = offs_dx
        self.Ay = Ay
        self.offs_y = offs_y
        self.ncy = ncy
        self.Q = Q
        self.Dy = Dy
        self.offs_dy = offs_dy
        self.kx = kx
        self.tx = tx
        self.x_x = x_x
        self.ky = ky
        self.ty = ty
        self.x_y = x_y
        self.z = z

    def __call__(self, p):
        C, fp = _solve_2d_fitpack(
            self.Ax.copy(), self.offs_x.copy(), self.ncx,
            self.Dx.copy(), self.offs_dx.copy(),
            self.Ay.copy(), self.offs_y.copy(), self.ncy,
            self.Q.copy(), self.Dy.copy(), self.offs_dy.copy(),
            p, self.kx, self.tx, self.x_x,
            self.ky, self.ty, self.x_y, self.z)
        self.C = C
        self.fp = fp
        return fp

# https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpregr.f#L301-L367
def _p_search_hit_s(
    Ax, offs_x, ncx, Dx, offs_dx, Ay,
    offs_y, ncy, Q, Dy, offs_dy, kx,
    tx, x_x, ky, ty, x_y, z, s, fp0, *,
    p_init=1.0, tol_rel=1e-3, maxit=40, verbose=False):
    """
    Search for a smoothing parameter `p` such that `fp(p) ~ s`.

    Parameters
    ----------
    Ax, Ay, Dx, Dy, offs_x, offs_y, offs_dx,
    offs_dy, ncx, ncy, ncdx, ncdy :
        See `_solve_2d_fitpack`.
    Q : ndarray
        RHS data grid (copy of `Z`).
    kx, ky : int
        Spline degrees.
    tx, ty : ndarray
        Knot vectors.
    x_x, x_y : ndarray
        Sample coordinates.
    w_x, w_y : ndarray
        Sample weights.
    z : ndarray
        Original data grid for residuals.
    s : float
        Target smoothing residual (`fp` target).
    fp0 : float or None
        Residual at `p = inf` (interpolatory limit,
                               represented by `p == -1`).
    p_init : float, optional
        Starting guess for the finite `p` search, default 1.0.
    tol_rel : float, optional
        Relative tolerance for matching `fp(p)` to `s`.
    maxit : int, optional
        Maximum iterations for the root search.
    verbose : bool, optional
        Print diagnostic output if True.

    Returns
    -------
    p_star : float
        Smoothing parameter for which `fp(p_star)` ~ `s`.
    C_star : ndarray
        Coefficient grid corresponding to `p_star`.
    fp_star : float
        Residual at `p_star`.

    Notes
    -----
    The solver treats `p == -1` as *p = inf* (interpolatory, no penalty).
    For finite `p`, the penalty scales as **1/p** - smaller `p` increases
    smoothing. A ratio-of-roots search (`root_rati`) iteratively adjusts `p`
    until the residual `fp(p)` matches the target `s` within tolerance.
    """

    fp_at = F(Ax, offs_x, ncx, Dx, offs_dx, Ay,
              offs_y, ncy, Q, Dy, offs_dy, kx,
              tx, x_x, ky, ty, x_y, z)

    def g(p):
        return fp_at(p) - s

    fpms = g(-1)

    bracket = ((0.0, fp0 - s), (np.inf, fpms))
    ftol = max(s * tol_rel, 1e-12)

    r = root_rati(g, p_init, bracket, ftol, maxit=maxit)
    p_star = r.root
    fp_star = fp_at(p_star)
    C_star = fp_at.C
    if verbose:
        print(f"[psearch] root_rati -> p={p_star:.6e}, fp={fp_star:.6e}")

    return p_star, C_star, fp_star

def _apply_bbox_grid(x, y, Z, bbox):
    """
    Restrict (x, y, Z) to a rectangular bounding box.

    Parameters
    ----------
    x, y : ndarray
        Monotonic sample coordinates.
    Z : ndarray, shape (len(x), len(y))
        Data grid.
    bbox : sequence of 4 scalars or None
        ``(xb, xe, yb, ye)``; any element may be None to skip clipping.

    Returns
    -------
    x_fit, y_fit, Z_fit : ndarray
        Sliced arrays restricted to bbox.
    ix, iy : slice or ndarray
        Indexers mapping from full arrays to the restricted ones.

    Raises
    ------
    ValueError
        If bbox is invalid or excludes all samples along an axis.
    """
    if all([bboxi is None for bboxi in bbox]):
        return x, y, Z, slice(None), slice(None)

    xb, xe, yb, ye = bbox
    if not (xb < xe and yb < ye):
        raise ValueError("bbox must satisfy xb < xe and yb < ye")

    ix = np.where((x >= xb) & (x <= xe))[0]
    iy = np.where((y >= yb) & (y <= ye))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError("bbox excludes all samples in x or y.")

    return x[ix], y[iy], Z[np.ix_(ix, iy)], np.s_[ix], np.s_[iy]

def build_matrices(x, y, z, tx, ty, kx, ky):

    w_x = np.ones_like(x)
    w_y = np.ones_like(y)

    Ax, offset_x, nc_x = _dierckx.data_matrix(x, tx, kx, w_x)
    Ay, offset_y, nc_y = _dierckx.data_matrix(y, ty, ky, w_y)
    Drx, offset_dx, nc_dx = disc(tx, kx)
    Dry, offset_dy, nc_dy = disc(ty, ky)
    Q = z.copy()

    return (Ax, offset_x, nc_x,
            Ay, offset_y, nc_y,
            Drx, offset_dx, nc_dx,
            Dry, offset_dy, nc_dy, Q)

TOL = 0.001

def _generate_knots(x, xb, xe, k, s, nmin=None, nmax=None,
                    nest=None, t=None, fp=None, fpold=None,
                    residuals=None, nplus=None):
    """
    Knot-growth helper for 1-D smoothing B-splines.

    What this function does
    -----------------------
    - Grows a non-periodic knot vector t iteratively for a target smoothing s,
      following the classic FITPACK knot-augmentation heuristics (fpcurf family).
    - Uses the same stopping rule as FITPACK: if |fp - s| < acc or fp < s
      (with acc = s*TOL), accept current t.
    - When more knots are needed, computes nplus (how many knots to insert next)
      based on the previous decrease in residual (fpold - fp), then calls add_knot
      up to nplus times.
    - Provides special handling for s == 0 (interpolation): returns a not-a-knot
      vector immediately.

    How it compares with _fitpack_repro.py::_generate_knots_impl
    -------------------------------------------------------------

    Similarities:
      1) Same growth logic and thresholds:
         - acc = s*TOL
         - Accept if |fp - s| < acc or fp < s
         - nplus update rule derived from delta = fpold - fp, with doubling/halving caps
      2) Same bounds for non-periodic splines:
         - nmin = 2*(k+1)  (LSQ polynomial stage)
         - nmax = m + k + 1 (interpolating spline)
      3) Same storage guard:
         - If nest is too small (< 2*(k+1)), raise; if n hits nest, stop and return t
      4) Same end behavior at the "interpolating" cap:
         - When n >= nmax, switch to an interpolation-layout knot
           vector (here: not-a-knot)

    Differences:
      1) API style:
         - _generate_knots_impl is a generator that YIELDS a sequence of trial knot
           vectors (t) on each iteration and internally recomputes residuals/fp.
         - This function is a stateful helper that RETURNS updated values and expects
           the caller to manage the loop, residuals, fp, and fpold between calls.
             * First call: pass t=None to initialize; returns (t, nest, nmin, nmax)
             * Subsequent calls: pass updated fp, fpold, residuals, nplus, and current t
               to either accept, grow knots, or stop.
      2) Periodicity:
         - _generate_knots_impl supports periodic=True (with per = xe - xb, periodic
           wrapping of boundary knots, and periodic-specific acceptance checks).
         - This function is non-periodic only (no periodic branch, no per, no periodic
           boundary updates). It uses not-a-knot when reaching nmax.
      3) Residual computation:
         - _generate_knots_impl calls an internal
           _get_residuals(x, y, t, k, w, periodic) each iteration and
           yields after setting fp and residuals.
         - This function does NOT compute residuals/fp. The caller must compute them
           externally and pass:
             * residuals: from the last solve with current t
             * fp: current sum of squared residuals
             * fpold: previous fp (for nplus update)
      4) Return values:
         - _generate_knots_impl yields t multiple times and eventually returns None.
         - This function returns different tuples depending on phase:
             * Initialization (t is None): (t, nest, nmin, nmax)
             * Acceptance (|fp - s| < acc or fp < s): returns None (caller stops)
             * Growth step: (t, nplus)
             * Early stop due to n >= nmax or n >= nest: (t, nplus) or (t,) as coded

    Parameters
    ----------
    x : 1-D ndarray
        Strictly increasing sample coordinates.
    xb, xe : float
        Domain endpoints used to seed the initial no-internal-knots vector.
    k : int
        Spline degree.
    s : float
        Smoothing target (> 0 for smoothing; if s == 0, do pure interpolation).
    nmin, nmax : int, optional
        Lower/upper bounds on knot count. If t is None, these are derived as:
          nmin = 2*(k+1), nmax = m + k + 1 (m = x.size).
    nest : int, optional
        Storage cap for knots. If t is None and nest is None, it defaults to
        max(m + k + 1, 2*k + 3). Must satisfy nest >= 2*(k+1).
    t : 1-D ndarray, optional
        Current knot vector. If None, the function initializes t as
        [xb]*(k+1) + [xe]*(k+1).
    fp, fpold : float, optional
        Current and previous residual sums of squares (needed after initialization).
    residuals : 1-D ndarray, optional
        Most recent residual signal used by add_knot to place the next knot.
    nplus : int, optional
        Previous iteration's proposed number of knots to add (used in nplus update).

    Returns
    -------
    On first call (t is None):
        t, nest, nmin, nmax
            Initialized knot vector and derived limits.
    On acceptance (|fp - s| < acc or fp < s):
        None
            Caller should stop; current t is acceptable.
    On growth step:
        t, nplus
            Updated knot vector after inserting up to nplus knots;
            also the nplus chosen.
    On early stop due to n >= nmax or n >= nest:
        t, nplus   (or just t depending on branch)
            Final knot vector respecting the cap.

    Notes
    -----
    - Use this as a building block in an outer loop:
        1) Call with t=None to initialize (and receive nest, nmin, nmax).
        2) Fit/evaluate to compute fp and residuals for current t.
        3) Call again with updated fp, fpold, residuals to either accept, or
           get back an updated t (and nplus) to continue.
    - For s == 0, this routine returns a not-a-knot vector immediately and
      expects the caller to perform interpolation.
    - This helper mirrors FITPACK's growth heuristics but intentionally leaves
      residual computation and periodic handling to the caller/higher level.
    """

    if s == 0:
        if nest is not None:
            raise ValueError("s == 0 is interpolation only")
        # For special-case k=1 (e.g., Lyche and Morken, Eq.(2.16)),
        # _not_a_knot produces desired knot vector
        return _not_a_knot(x, k), None, None, None

    acc = s * TOL
    m = x.size    # the number of data points

    if t is None:
        if nest is None:
            # the max number of knots. This is set in _fitpack_impl.py line 274
            # and fitpack.pyf line 198
            # Ref: https://github.com/scipy/scipy/blob/596b586e25e34bd842b575bac134b4d6924c6556/scipy/interpolate/_fitpack_impl.py#L260-L263
            nest = max(m + k + 1, 2*k + 3)
        else:
            if nest < 2*(k + 1):
                raise ValueError(f"`nest` too small: {nest = } < 2*(k+1) = {2*(k+1)}.")

        nmin = 2*(k + 1)    # the number of knots for an LSQ polynomial approximation
        nmax = m + k + 1  # the number of knots for the spline interpolation

        fp = 0.0
        fpold = 0.0

        # start from no internal knots
        t = np.asarray([xb]*(k+1) + [xe]*(k+1))

        return t, nest, nmin, nmax

    n = t.size

    fpms = fp - s

    # c  test whether the approximation sinf(x) is an acceptable solution.
    # c  if f(p=inf) < s accept the choice of knots.
    if (abs(fpms) < acc) or (fpms < 0):
        return

    # ### c  increase the number of knots. ###

    # c  determine the number of knots nplus we are going to add.
    if n == nmin:
        # the first iteration
        nplus = 1
    else:
        delta = fpold - fp
        npl1 = int(nplus * fpms / delta) if delta > acc else nplus*2
        nplus = min(nplus*2, max(npl1, nplus//2, 1))

    # actually add knots
    for j in range(nplus):
        t = add_knot(x, t, k, residuals)

        # check if we have enough knots already

        n = t.shape[0]
        # c  if n = nmax, sinf(x) is an interpolating spline.
        # c  if n=nmax we locate the knots as for interpolation.
        if n >= nmax:
            return _not_a_knot(x, k), nplus

        # c  if n=nest we cannot increase the number of knots because of
        # c  the storage capacity limitation.
        if n >= nest:
            return t, nplus

    return t, nplus

def _regrid_python_fitpack(
    x, y, Z, *, kx=3, ky=3, s=0.0,
    maxit=50, nestx=None, nesty=None,
    bbox=[None]*4, verbose=False):
    """
    Core adaptive bivariate spline fitter using the 1/p-penalty convention.

    Parameters
    ----------
    x, y : array_like
        Strictly increasing coordinate vectors.
    Z : array_like, shape (len(x), len(y))
        Data grid.
    kx, ky : int, optional
        Spline degrees along x and y, default 3 (cubic).
    s : float, optional
        Target residual (`fp` target). `s = 0` requests an interpolatory
        surface; `s > 0` triggers smoothing with penalty weight **1/p**.
    maxit : int, optional
        Maximum iterations for the `p`-search when smoothing, default 50.
    nestx, nesty : int or None
        Max coefficient counts per axis (nesting limits).
    bbox : sequence of 4 scalars
        Optional domain limits `(xb, xe, yb, ye)`. Use `None` entries to skip.
    verbose : bool, optional
        Print detailed iteration logs if True.

    Returns
    -------
    NdBSpline
        Fitted 2-D spline surface.

    Notes
    -----
    The internal smoothing parameter `p` follows the **inverse**-penalty
    rule: penalty term is 1/p.  Hence, larger `p` -> weaker smoothing
    (approaching interpolation), while smaller `p` -> stronger smoothing.
    A sentinel value `p == -1` is interpreted as *p = inf*, corresponding to
    an exact (interpolatory) fit.

    The iterative process adaptively grows knot vectors based on residual
    energy and optionally performs a 1-D search over `p` to satisfy `fp ~ s`.
    """
    x_fit, y_fit, Z_fit, _, _ = _apply_bbox_grid(x, y, Z, bbox)

    if x_fit.size < (kx + 1) or y_fit.size < (ky + 1):
        raise ValueError(
            f"Not enough samples inside bbox for degrees (kx={kx}, ky={ky}). "
            f"Need at least k+1 per axis: ({kx+1}, {ky+1}). "
            f"Got ({x_fit.size}, {y_fit.size})."
        )

    xb = float(x_fit[0] if bbox[0] is None else bbox[0])
    xe = float(x_fit[-1] if bbox[1] is None else bbox[1])
    yb = float(y_fit[0] if bbox[2] is None else bbox[2])
    ye = float(y_fit[-1] if bbox[3] is None else bbox[3])

    tx, nestx, nminx, nmaxx = _generate_knots(x_fit, xb, xe, kx, s, nest=nestx)
    ty, nesty, nminy, nmaxy = _generate_knots(y_fit, yb, ye, ky, s, nest=nesty)

    if s == 0.0:
        (Ax, offset_x, nc_x,
         Ay, offset_y, nc_y,
         Drx, offset_dx, _,
         Dry, offset_dy, _, Q) = build_matrices(
             x_fit, y_fit, Z, tx, ty, kx, ky)
        C0, fp  = _solve_2d_fitpack(Ax, offset_x, nc_x,
                                    Drx, offset_dx,
                                    Ay, offset_y, nc_y, Q,
                                    Dry, offset_dy, -1,
                                    kx, tx, x_fit, ky, ty,
                                    y_fit, Z_fit)
        return return_NdBSpline(fp, (tx, ty, C0), (kx, ky))

    moves = 0
    fpold = None
    last_axis = "y"
    mpm = len(x) + len(y)
    nx = len(tx)
    ny = len(ty)
    mx_head = max(0, nestx - nx) if nestx is not None else None
    my_head = max(0, nesty - ny) if nesty is not None else None
    fp0 = None
    nplusx = None
    nplusy = None

    # https://github.com/scipy/scipy/blob/v1.16.2/scipy/interpolate/fitpack/fpregr.f#L51-L300
    for it in range(mpm):
        nx, ny = len(tx), len(ty)

        (Ax, offset_x, nc_x,
         Ay, offset_y, nc_y,
         Drx, offset_dx, _,
         Dry, offset_dy, _, Q) = build_matrices(
             x_fit, y_fit, Z, tx, ty, kx, ky)
        C0, fp  = _solve_2d_fitpack(Ax, offset_x, nc_x,
                                    Drx, offset_dx,
                                    Ay, offset_y, nc_y, Q,
                                    Dry, offset_dy, -1,
                                    kx, tx, x_fit, ky, ty,
                                    y_fit, Z_fit)

        if len(tx) == nminx and len(ty) == nminy:
            fp0 = fp

        if verbose:
            print(f"[it={it}] p=0 fp={fp:.6e} s={s:.6e} "
                  f"nx={nx} ny={ny} moves={moves}/{mpm}")
            print(f"    headroom: mx_head={mx_head} my_head={my_head} "
                  f"(nestx={nestx}, nesty={nesty})")

        if fp < s:
            break

        _Ax = BSpline.design_matrix(x_fit, tx, kx, extrapolate=False)
        _Ay = BSpline.design_matrix(y_fit, ty, ky, extrapolate=False)
        Z0  = (_Ax @ C0) @ _Ay.T
        R = Z_fit - Z0

        added_x = None
        added_y = None
        if last_axis == "y":
            len_tx_before = len(tx)
            tx, nplusx = _generate_knots(
                x_fit, xb, xe, kx, s, nmin=nminx, nmax=nmaxx,
                nest=nestx, t=tx, fp=fp, fpold=fpold,
                residuals=np.ascontiguousarray(np.sum(R**2, 1)),
                nplus=nplusx)
            added_x = len(tx) - len_tx_before
            if verbose:
                print(f"    Inserted {added_x} knots in X")
            last_axis = "x"
        else:
            len_ty_before = len(ty)
            ty, nplusy = _generate_knots(
                y_fit, yb, ye, ky, s, nmin=nminy, nmax=nmaxy,
                nest=nesty, t=ty, fp=fp, fpold=fpold,
                residuals=np.ascontiguousarray(np.sum(R**2, 0)),
                nplus=nplusy)
            added_y = len(ty) - len_ty_before
            if verbose:
                print(f"    Inserted {added_y} knots in Y")
            last_axis = "y"

        fpold = fp

    if verbose:
        print(f"Growth end with fp={fp:.6e} > s={s:.6e} "
              f"({'nest exhausted' if (mx_head==0 and my_head==0) else 'early stop'})")

    if len(tx) != nminx or len(ty) != nminy:
        (Ax, offset_x, nc_x,
        Ay, offset_y, nc_y,
        Drx, offset_dx, _,
        Dry, offset_dy, _, Q) = build_matrices(
            x_fit, y_fit, Z, tx, ty, kx, ky)
        _, C_sm, fp_sm = _p_search_hit_s(Ax, offset_x, nc_x,
                                        Drx, offset_dx,
                                        Ay, offset_y, nc_y, Q,
                                        Dry, offset_dy,
                                        kx, tx, x_fit, ky,
                                        ty, y_fit,
                                        Z_fit, s, fp0, maxit=maxit)
        return return_NdBSpline(fp_sm, (tx, ty, C_sm), (kx, ky))
    else:
        return return_NdBSpline(fp, (tx, ty, C0), (kx, ky))


def regrid_python(x, y, Z, *,
    kx=3, ky=3, s=0.0, maxit=50, nestx=None,
    nesty=None, bbox=[None]*4, verbose=False):
    """
    Public interface for 2-D smoothing B-spline fitting (1/p penalty form).

    Parameters
    ----------
    x, y : array_like
        Strictly increasing 1-D coordinate vectors.
    Z : array_like, shape (len(x), len(y))
        Data grid.
    kx, ky : int, optional
        Spline degrees along x and y, default cubic (3).
    s : float, optional
        Target smoothing residual (`fp` target). Must satisfy `s >= 0`.
        The underlying formulation uses a **1/p** penalty, meaning:
        - small `p` -> heavy smoothing,
        - large `p` -> light smoothing (approaching interpolation).
        Setting `p == -1` internally denotes *p = inf*, i.e. a pure interpolant.
    maxit : int, optional
        Maximum iterations for `p`-search if invoked.
    nestx, nesty : int or None
        Nesting limits for coefficient counts per axis.
    bbox : sequence of 4 scalars
        Optional bounding box `(xb, xe, yb, ye)`; use `None` entries to disable.
    verbose : bool, optional
        Print iteration diagnostics if True.

    Returns
    -------
    NdBSpline
        Fitted bivariate spline surface.

    Notes
    -----
    This validates input, enforces monotonicity, and calls
    `_regrid_python_fitpack`.  The fit obeys the 1/p-penalty convention used
    throughout this module, and `p == -1` is treated as the interpolatory
    (infinite-p) case.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, float)
    bbox = np.ravel(bbox)

    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x must be strictly increasing")
    if not np.all(np.diff(y) > 0.0):
        raise ValueError("y must be strictly increasing")
    if x.size != Z.shape[0]:
        raise ValueError("x dimension of z must have same number of elements as x")
    if y.size != Z.shape[1]:
        raise ValueError("y dimension of z must have same number of elements as y")
    if s is not None and not (s >= 0.0):
        raise ValueError("s should be s >= 0.0")
    if not bbox.shape == (4,):
        raise ValueError(f"bbox shape should be (4,), found: {bbox.shape}")

    return _regrid_python_fitpack(
        x, y, Z, kx=kx, ky=ky, s=s, maxit=maxit,
        nestx=nestx, nesty=nesty, bbox=bbox,
        verbose=verbose)
