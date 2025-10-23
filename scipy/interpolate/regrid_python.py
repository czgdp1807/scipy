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

- Row energy  `row_energy[i] = sum(R[i, j]^2, j)`  → accumulated into `fpintx`.
- Column energy `col_energy[j] = sum(R[i, j]^2, i)` → accumulated into `fpinty`.

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
1. `regrid_python`
   - Validates monotonic `x`, `y`, grid shapes, `s ≥ 0`, normalizes `bbox`.
   - Dispatches to `_regrid_python_fitpack`.

2. `_regrid_python_fitpack`
   - Applies `bbox` to get `(x_fit, y_fit, Z_fit)`.
   - Initializes clamped knots (interpolatory if `s == 0`, open-uniform otherwise).
   - **Loop:**
     - Build 1-D banded matrices via `build_matrices` (calls `data_matrix`, `disc`).
     - Call `_solve_2d_fitpack` with `p = ∞` (sentinel `-1`) for the current knots.
     - If `fp ≤ s`, return. Otherwise compute `fpintx/fpinty`,
       decide batches, and insert knots.
   - If growth saturates with `fp > s`, run `_p_search_hit_s` to tune `p`.

3. `_p_search_hit_s`
   - Wraps the problem in `F`, which evaluates `fp(p)` by invoking `_solve_2d_fitpack`.
   - Uses `root_rati` to find `p*` with `fp(p*) ~ s`, caching the resulting `C`.

4. `_solve_2d_fitpack`
   - Performs the two separable 1-D augmented QR solves (`[A; D/p]`) along x then y,
     returns coefficients `C` and residual `fp`.

The final `(tx, ty, C)` are packaged as an `NdBSpline` for convenient evaluation.
"""
import numpy as np
from scipy.interpolate import BSpline, NdBSpline
from scipy.interpolate._fitpack_repro import root_rati, disc
import time
from . import _dierckx

from time import perf_counter
from contextlib import contextmanager

class BlockTimer:
    """
    Lightweight wall-clock timer for named code blocks.

    Use as a context manager to accumulate time under a user-defined name
    and later print a sorted timing report.

    Examples
    --------
    >>> t = BlockTimer()
    >>> with t("assemble"):
    ...     ...  # do work
    >>> with t("solve"):
    ...     ...  # do work
    >>> t.report()
    """

    def __init__(self):
        """
        Initialize the timer store.

        Notes
        -----
        Internally maintains a dict ``times: Dict[str, float]`` that maps
        a block name to cumulative seconds.
        """
        self.times = {}

    @contextmanager
    def __call__(self, name):
        """
        Time a code block and accumulate its duration under ``name``.

        Parameters
        ----------
        name : str
            Bucket name under which the elapsed time is accumulated.

        Yields
        ------
        None
            Context manager for timing.

        Notes
        -----
        Nested usage is supported; times are accumulated independently.
        """
        t0 = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (perf_counter() - t0)

    def report(self, round_to=5):
        """
        Print and return a sorted timing table.

        Parameters
        ----------
        round_to : int, optional
            Number of decimal places for pretty-printing, by default 5.

        Returns
        -------
        list of tuple
            List of ``(name, seconds, percent)`` entries, sorted by seconds
            in descending order. The percentages sum to ~100%.

        Notes
        -----
        If no time was recorded, a small epsilon is used to avoid division by zero.
        """
        total = sum(self.times.values()) or 1e-12
        rows = []
        for k, v in sorted(self.times.items(), key=lambda kv: kv[1], reverse=True):
            pct = 100.0 * v / total
            rows.append((k, v, pct))
        # pretty print
        print("\n=== Block timing ===")
        for k, v, pct in rows:
            print(f"{k:>7}: {v:.{round_to}f}s  ({pct:.{round_to}f}%)")
        print(f"  total: {total:.{round_to}f}s")
        return rows

def _memory_percent_increase_xi_vs_xyZ(nx, ny, s_coord, s_Z):
    """
    Percent memory increase of using stacked 2D coordinates vs separate (x, y, Z).

    Computes the extra memory percentage when calling a spline with an ``(N, 2)``
    stacked coordinate array ``xi`` (where ``N = nx * ny``) instead of passing
    separate ``x``, ``y`` and a grid ``Z``.

    Parameters
    ----------
    nx, ny : int
        Grid sizes along x and y.
    s_coord : int
        Itemsize (in bytes) for coordinate dtype (e.g., 8 for float64).
    s_Z : int
        Itemsize (in bytes) for Z dtype (e.g., 8 for float64).

    Returns
    -------
    float
        Percentage increase in memory footprint (0..100+).

    Notes
    -----
    Base memory ~ ``(nx + ny) * s_coord + (nx * ny) * s_Z``.
    Extra memory for ``xi`` ~ ``(2 * nx * ny - (nx + ny)) * s_coord``.
    """
    base = (nx + ny) * s_coord + (nx * ny) * s_Z
    extra = (2 * nx * ny - (nx + ny)) * s_coord
    return (extra / base) * 100.0 if base > 0 else 0.0

def ndbspline_call_like_bivariate(ndbs, x, y, dx=0, dy=0, grid=True,
                                  return_profiling_data=False, Z_for_memory=None):
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
    return_profiling_data : bool, optional
        If True, return a dict with coarse timings of mesh and evaluation,
        by default False.
    Z_for_memory : array_like, optional
        If provided, used only to estimate memory overhead of stacked coords.

    Returns
    -------
    ndarray or (ndarray, dict)
        Evaluated values with shape:
        - ``(len(x), len(y), ...)`` if ``grid=True``.
        - ``x.shape + ...`` if ``grid=False``.
        If ``return_profiling_data=True``, also returns a dict of timings and
        optional memory overhead in percent.

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
    prof = {}

    if grid:
        t0 = time.perf_counter()
        x = np.asarray(x)
        y = np.asarray(y)

        if x.size == 0 or y.size == 0:
            vals = np.zeros((x.size, y.size) + trailing, dtype=ndbs.c.dtype)
            if return_profiling_data:
                prof.update(dict(block1_s=0.0, block2_s=0.0, total_s=0.0))
                if Z_for_memory is not None:
                    prof["mem_pct_extra"] = 0.0
                return vals, prof
            return vals

        if (x.size >= 2) and (not np.all(np.diff(x) >= 0.0)):
            raise ValueError("x must be strictly increasing when `grid` is True")
        if (y.size >= 2) and (not np.all(np.diff(y) >= 0.0)):
            raise ValueError("y must be strictly increasing when `grid` is True")

        X, Y = np.meshgrid(x, y, indexing="ij")
        xi = np.stack((X, Y), axis=-1)  # (len(x), len(y), 2)
        t1 = time.perf_counter()

        vals = ndbs(xi, nu=(dx, dy), extrapolate=ndbs.extrapolate)
        t2 = time.perf_counter()

        if return_profiling_data:
            prof["block1_s"] = t1 - t0
            prof["block2_s"] = t2 - t1
            prof["total_s"]  = t2 - t0
            if Z_for_memory is not None:
                s_coord = x.dtype.itemsize
                s_Z = Z_for_memory.dtype.itemsize
                prof["mem_pct_extra"] = _memory_percent_increase_xi_vs_xyZ(
                    len(x), len(y), s_coord, s_Z)
            return vals, prof
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

def will_square_overflow(x, dtype=np.float64):
    """
    Check if squaring values would overflow the given dtype.

    Parameters
    ----------
    x : array_like
        Input values to test.
    dtype : dtype, optional
        Floating dtype to test against, by default ``np.float64``.

    Returns
    -------
    ndarray of bool
        Boolean mask where True indicates ``x**2`` would overflow ``dtype``.

    Notes
    -----
    Compares ``abs(x)`` to ``sqrt(np.finfo(dtype).max)`` elementwise.
    """
    max_val = np.finfo(dtype).max
    return abs(x) > np.sqrt(max_val)

def fp_residual(Z, Zhat):
    """
    Compute FITPACK-style residual sum of squares ``fp``.

    Parameters
    ----------
    Z : array_like
        Target data (observed grid).
    Zhat : array_like
        Predicted/fitted grid, broadcastable to ``Z``.

    Returns
    -------
    float
        Residual sum of squares, or ``inf`` if non-finite or overflow detected.

    Notes
    -----
    Guards against NaNs/Infs and overflow by early returning ``inf``.
    """
    R = Z - Zhat

    if not np.isfinite(R).all():
        return float("inf")

    if np.isnan(R).any() or will_square_overflow(R).any():
        return float("inf")

    return np.sum(np.square(R))

def _solve_2d_fitpack(Ax, offs_x, ncx,
                      Dx, offs_dx,
                      Ay, offs_y, ncy, Q,
                      Dy, offs_dy, p,
                      kx, tx, x_x,
                      ky, ty, x_y, z):
    """
    Solve the 2-D tensor-product spline system using separable banded QR.

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
    w_x = np.ones_like(x_x)
    w_y = np.ones_like(x_y)

    Ax_aug, offset_aug_x, nc_augx = _stack_augmented_fitpack(
        Ax, offs_x, Dx, offs_dx, ncx, kx, p)
    Ay_aug, offset_aug_y, nc_augy = _stack_augmented_fitpack(
        Ay, offs_y, Dy, offs_dy, ncy, ky, p)

    if p != -1:
        Q = np.vstack([Q, np.zeros((Dx.shape[0], Q.shape[1]), dtype=np.float64)])

    _dierckx.qr_reduce(Ax_aug, offset_aug_x, nc_augx, Q)

    Q_t = np.ascontiguousarray(Q)
    cT, _, _ = _dierckx.fpback(
        Ax_aug, nc_augx, x_x,
        Q_t, tx, kx, w_x,
        Q_t, False
    )
    W = cT.T

    if p != -1:
        W = np.vstack([W, np.zeros((Dy.shape[0], W.shape[1]), dtype=np.float64)])

    WT = np.ascontiguousarray(W)
    _dierckx.qr_reduce(Ay_aug, offset_aug_y, nc_augy, WT)
    W1 = np.ascontiguousarray(WT)

    C, _, fp = _dierckx.fpback(
        Ay_aug, nc_augy,
        x_y, W1, ty, ky, w_y,    # y = W1 (dummy)
        W1,                      # yw = RHS = W1 → returns C
        False
    )

    C = C.T
    _Ax = BSpline.design_matrix(x_x, tx, kx, extrapolate=False)
    _Ay = BSpline.design_matrix(x_y, ty, ky, extrapolate=False)
    zhat = (_Ax @ C) @ _Ay.T
    fp = fp_residual(z, zhat)
    return C, fp

def _span_indices_from_knots(x, t, k):
    """
    Map each ``x`` to its B-spline span index (clamped).

    Parameters
    ----------
    x : array_like
        Query points.
    t : array_like
        Knot vector.
    k : int
        Spline degree.

    Returns
    -------
    ndarray of int
        Span indices ``j`` such that ``t[j] <= x < t[j+1]``, clamped to
        ``[k, len(t) - k - 2]``.
    """
    j = np.searchsorted(t, x, side="right") - 1
    j = np.clip(j, k, len(t) - k - 2)
    return j

def _fpint_axis_from_residual_rows(x, t, k, row_energy):
    """
    Accumulate residual row energies into knot-span bins.

    Parameters
    ----------
    x : array_like
        Axis sample positions.
    t : array_like
        Knot vector along the same axis.
    k : int
        Spline degree.
    row_energy : array_like, shape (len(x),)
        Energy per row, e.g. ``sum(R**2, axis=1)``.

    Returns
    -------
    ndarray
        ``fpint`` array of length ``len(t) - 1`` with per-span energy totals.
        Zero for degenerate spans where ``t[i+1] == t[i]``.
    """
    t = np.asarray(t, float)
    valid_span = (t[1:] > t[:-1])
    fpintx = np.zeros(len(t) - 1, dtype=float)

    spans = _span_indices_from_knots(x, t, k)
    np.add.at(fpintx, spans, row_energy)

    fpintx[~valid_span] = 0.0
    return fpintx

def _fpint_xy(x, y, t_x, t_y, kx, ky, R):
    """
    Project residual energy along x and y knot spans.

    Parameters
    ----------
    x, y : array_like
        Sample positions along x and y.
    t_x, t_y : array_like
        Knot vectors along x and y.
    kx, ky : int
        Spline degrees.
    R : ndarray, shape (len(x), len(y))
        Residual grid ``Z - Zhat``.

    Returns
    -------
    fpintx, fpinty : ndarray
        Per-span energy arrays for x and y axes, respectively.
    """
    row_energy = np.einsum("ij,ij->i", R, R)
    col_energy = np.einsum("ij,ij->j", R, R)
    fpintx = _fpint_axis_from_residual_rows(x, t_x, kx, row_energy)
    fpinty = _fpint_axis_from_residual_rows(y, t_y, ky, col_energy)
    return fpintx, fpinty

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
    p_star = float(r.root)
    fp_star = fp_at(p_star)
    C_star = fp_at.C
    if verbose:
        print(f"[psearch] root_rati -> p={p_star:.6e}, fp={fp_star:.6e}")

    return p_star, C_star, fp_star

def _enforce_clamped(tx, k, xb, xe):
    """
    Enforce clamped ends on a knot vector.

    Parameters
    ----------
    tx : array_like
        Knot vector (will be copied).
    k : int
        Degree.
    xb, xe : float
        Domain endpoints.

    Returns
    -------
    ndarray
        Knot vector with first/last ``k+1`` entries set to ``xb``/``xe``.
    """
    t = np.asarray(tx, float).copy()
    t[:k+1]    = xb
    t[-(k+1):] = xe
    return t

def _decide_batch_counts(fp, fpold, s,
                         fpintx, fpinty,
                         mx_head, my_head,
                         *, batch_cap=12):
    """
    Decide how many x- and y-knots to insert next (batch growth heuristic).

    Parameters
    ----------
    fp : float
        Current residual.
    fpold : float or None
        Previous residual (for estimating gain).
    s : float
        Target residual.
    fpintx, fpinty : ndarray
        Per-span residual energies along x and y.
    mx_head, my_head : int
        Remaining headroom (max additional knots) along x and y.
    batch_cap : int, optional
        Maximum total insertions per iteration, by default 12.

    Returns
    -------
    nplx, nply : int
        Planned insertions along x and y.

    Notes
    -----
    Uses deficit/gain ratio and energy split to allocate between axes, with guards.
    """
    if not (np.isfinite(fp) and (fpold is None or
                                 np.isfinite(fpold)) and np.isfinite(s)):
        nplx = 1 if mx_head > 0 else 0
        nply = 1 if my_head > 0 else 0
        return nplx, nply

    deficit = max(fp - s, 0.0)
    if fpold is None:
        wx = float(np.sum(fpintx))
        wy = float(np.sum(fpinty))
        if (mx_head <= 0) and (my_head <= 0):
            return 0, 0
        if wy > wx:
            return (0, 1 if my_head > 0 else 0)
        else:
            return (1 if mx_head > 0 else 0, 0)

    gain = max(fpold - fp, 0.0)

    GAIN_FLOOR = 1e-12
    if gain <= GAIN_FLOOR:
        n_total = 1
    else:
        ratio = min(deficit / gain, 1e6)
        n_total = int(np.clip(np.ceil(ratio / 2.0), 1, batch_cap))

    wx = float(np.sum(fpintx))
    wy = float(np.sum(fpinty))
    if wx + wy <= 0:
        nplx_raw = n_total // 2
    else:
        nplx_raw = int(np.round(n_total * (wx / (wx + wy))))
    nply_raw = n_total - nplx_raw

    nplx = int(min(max(nplx_raw, 0), mx_head))
    nply = int(min(max(nply_raw, 0), my_head))
    if (nplx + nply) == 0 and (mx_head > 0 or my_head > 0):
        if mx_head >= my_head:
            nplx = min(1, mx_head)
        else:
            nply = min(1, my_head)
    return nplx, nply

def _top_spans(fpint, n, forbid_zero_len_mask):
    """
    Select up to ``n`` top energy spans, honoring a validity mask.

    Parameters
    ----------
    fpint : ndarray
        Energy per span.
    n : int
        Maximum number to select.
    forbid_zero_len_mask : ndarray of bool or None
        Mask of valid spans (True = valid). If None, all are considered.

    Returns
    -------
    list of int
        Selected span indices in descending energy order (up to ``n``).
    """
    idx = np.argsort(fpint)[::-1]
    out = []
    for j in idx:
        if len(out) >= n:
            break
        if forbid_zero_len_mask is not None and not forbid_zero_len_mask[j]:
            continue
        out.append(int(j))
    return out

def _pick_data_knot_in_span(x, t, jspan):
    """
    Pick a representative data abscissa within a knot span.

    Parameters
    ----------
    x : ndarray
        Sorted data abscissae.
    t : ndarray
        Knot vector.
    jspan : int
        Span index.

    Returns
    -------
    float or None
        Candidate knot location (e.g., a median-like sample) or ``None`` if
        no data falls in the span.
    """
    lo, hi = t[jspan], t[jspan+1]
    idx = np.where((x >= lo) & (x < hi))[0]
    if idx.size == 0:
        return None
    return float(x[idx[idx.size//2]])

def _batch_insert_axis(x, t, k, fpint, n_add, xb, xe, nest_limit):
    """
    Insert up to ``n_add`` knots along an axis guided by span energies.

    Parameters
    ----------
    x : ndarray
        Data abscissae.
    t : ndarray
        Current knot vector.
    k : int
        Degree.
    fpint : ndarray
        Per-span residual energies.
    n_add : int
        Requested number of insertions.
    xb, xe : float
        Domain endpoints.
    nest_limit : int or None
        Maximum allowed number of coefficients; if exceeded, stops.

    Returns
    -------
    t_new : ndarray
        Updated knot vector.
    added : int
        Actual number of insertions performed.

    Notes
    -----
    Skips degenerate spans, duplicates, and respects nesting/headroom limits.
    """
    if n_add <= 0:
        return t, 0
    valid_span = (t[1:] > t[:-1])
    spans = _top_spans(fpint, n_add*2, valid_span)
    added = 0
    for js in spans:
        if added >= n_add:
            break
        cand = _pick_data_knot_in_span(x, t, js)
        if cand is None:
            continue
        cand = float(np.clip(cand, xb, xe))
        if not (t[k] + 1e-12 < cand < t[-k-1] - 1e-12):
            continue
        if np.any(np.isclose(t, cand, atol=1e-12)):
            continue
        t2 = np.sort(np.r_[t, cand])
        n_old = len(t) - k - 1
        n_new = len(t2) - k - 1
        if n_new <= n_old:
            continue
        if nest_limit is not None and n_new > nest_limit:
            break
        t = t2
        added += 1
    return t, added

def make_interpolatory_knots(x, k):
    """
    Build clamped interpolatory knot vector from strictly increasing samples.

    Parameters
    ----------
    x : ndarray
        Strictly increasing sample locations.
    k : int
        Degree.

    Returns
    -------
    ndarray
        Knot vector with ``k+1`` repeats at both ends and internal knots at
        ``x[left : -right]`` (if available), where ``left = (k+1)//2``.
    """
    x = np.asarray(x, float)
    if x.size < k + 1:
        raise ValueError("need at least k+1 samples")

    left  = (k + 1) // 2
    right = (k + 1) - left

    t_start = np.repeat(x[0], k + 1)
    t_end   = np.repeat(x[-1], k + 1)

    if x.size > (left + right):
        ti = x[left : x.size - right]
    else:
        ti = np.array([], dtype=float)

    t = np.r_[t_start, ti, t_end].astype(float)
    return t

def make_open_uniform_knots(x, k, n_internal, *,
                            xb=None, xe=None):
    """
    Build an open uniform knot vector on [xb, xe] with optional interior knots.

    Parameters
    ----------
    x : ndarray
        Sample locations (used to compute quantiles if ``n_internal > 0``).
    k : int
        Degree.
    n_internal : int
        Number of interior knots; if > 0, placed at empirical quantiles of
        samples within [xb, xe].
    xb, xe : float, optional
        Domain endpoints. Defaults to ``x[0]`` and ``x[-1]``.

    Returns
    -------
    ndarray
        Clamped knot vector with ``k+1`` repeats at both ends and
        ``n_internal`` interior knots if requested.

    Raises
    ------
    ValueError
        If no samples fall inside the requested domain.
    """
    x = np.asarray(x)
    if xb is None:
        xb = float(x[0])
    if xe is None:
        xe = float(x[-1])
    if not (xb < xe):
        raise ValueError("xb must be < xe")

    t_start = np.repeat(xb, k + 1)
    t_end = np.repeat(xe, k + 1)

    if n_internal and n_internal > 0:
        x_in = x[(x >= xb) & (x <= xe)]
        if x_in.size == 0:
            raise ValueError("No x samples fall inside [xb, xe].")
        ti = np.quantile(x_in, np.linspace(0, 1, n_internal + 2)[1:-1])
        return np.r_[t_start, ti, t_end].astype(float)

    return np.r_[t_start, t_end].astype(float)

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
    x_fit = x_fit.astype(float)
    y_fit = y_fit.astype(float)
    mx, my = Z_fit.shape

    if x_fit.size < (kx + 1) or y_fit.size < (ky + 1):
        raise ValueError(
            f"Not enough samples inside bbox for degrees (kx={kx}, ky={ky}). "
            f"Need at least k+1 per axis: ({kx+1}, {ky+1}). "
            f"Got ({x_fit.size}, {y_fit.size})."
        )

    if nestx is None:
        nestx = len(x_fit) + kx + 1
    if nesty is None:
        nesty = len(y_fit) + ky + 1

    xb = float(x_fit[0] if bbox[0] is None else bbox[0])
    xe = float(x_fit[-1] if bbox[1] is None else bbox[1])
    yb = float(y_fit[0] if bbox[2] is None else bbox[2])
    ye = float(y_fit[-1] if bbox[3] is None else bbox[3])

    if s == 0.0:
        tx = make_interpolatory_knots(x_fit, kx)
        ty = make_interpolatory_knots(y_fit, ky)
    else:
        tx = make_open_uniform_knots(x_fit, kx, 0, xb=xb, xe=xe)
        ty = make_open_uniform_knots(y_fit, ky, 0, xb=yb, xe=ye)

    moves = 0
    fpold = None
    last_axis = "y"
    mpm = len(x) + len(y) + 1
    nx = len(tx)
    ny = len(ty)
    mx_head = max(0, nestx - nx) if nestx is not None else None
    my_head = max(0, nesty - ny) if nesty is not None else None
    nminx = 2*(kx + 1)
    nminy = 2*(ky + 1)
    nmaxx = len(x_fit)
    nmaxy = len(y_fit)
    fp0 = None

    for it in range(len(x) + len(y) + 1):
        tx = _enforce_clamped(tx, kx, xb, xe)
        ty = _enforce_clamped(ty, ky, yb, ye)
        nx, ny = len(tx), len(ty)

        (Ax, offset_x, nc_x,
         Ay, offset_y, nc_y,
         Drx, offset_dx, nc_dx,
         Dry, offset_dy, nc_dy, Q) = build_matrices(
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

        if s == 0.0 or fp <= s:
            if verbose:
                print(f"  Target reached at it={it} (fp = {fp} <= s = {s})")
            return return_NdBSpline(fp, (tx, ty, C0), (kx, ky))

        _Ax = BSpline.design_matrix(x_fit, tx, kx, extrapolate=False)
        _Ay = BSpline.design_matrix(y_fit, ty, ky, extrapolate=False)
        Z0  = (_Ax @ C0) @ _Ay.T
        R = Z_fit - Z0
        fpintx, fpinty = _fpint_xy(x_fit, y_fit, tx, ty, kx, ky, R)

        if fpold is None:
            wx, wy = np.sum(fpintx), np.sum(fpinty)
            nplx = 1 if (wx >= wy) else 0
            nply = 1 if (wy > wx) else 0
            if verbose:
                print(f"First growth step: wx={wx:.3e} wy={wy:.3e} "
                      f"-> nplx={nplx} nply={nply}")
        else:
            nplx, nply = _decide_batch_counts(fp, fpold, s, fpintx, fpinty,
                                              mx_head or 0, my_head or 0,
                                              batch_cap=12)
            if verbose:
                print(f"  Batch decision: fpold={fpold:.6e} -> fp={fp:.6e}, "
                      f"nplx={nplx}, nply={nply}")

        grew_total = 0
        if last_axis == "y" and nplx:
            tx, added_x = _batch_insert_axis(x_fit, tx, kx, fpintx, nplx, xb, xe, nestx)
            if verbose and added_x:
                print(f"    Inserted {added_x} knots in X")
            grew_total += added_x
            if added_x:
                last_axis = "x"
        if nply:
            ty, added_y = _batch_insert_axis(y_fit, ty, ky, fpinty, nply, yb, ye, nesty)
            if verbose and added_y:
                print(f"    Inserted {added_y} knots in Y")
            grew_total += added_y
            if added_y:
                last_axis = "y"
        if last_axis == "x" and nplx:
            tx, added_x = _batch_insert_axis(x_fit, tx, kx, fpintx, nplx, xb, xe, nestx)
            if verbose and added_x:
                print(f"    Inserted {added_x} knots in X (post-Y)")
            grew_total += added_x
            if added_x:
                last_axis = "x"

        if grew_total == 0:
            if verbose:
                print("[interp-exit] cannot add more knots "
                      "-> building interpolatory knots & returning p=0")
            tx = make_interpolatory_knots(x_fit, kx)
            ty = make_interpolatory_knots(y_fit, ky)

            (Ax, offset_x, nc_x,
             Ay, offset_y, nc_y,
             Drx, offset_dx, nc_dx,
             Dry, offset_dy, nc_dy, Q) = build_matrices(
                x_fit, y_fit, Z, tx, ty, kx, ky)
            C0, fp = _solve_2d_fitpack(Ax, offset_x, nc_x,
                                  Drx, offset_dx,
                                  Ay, offset_y, nc_y, Q,
                                  Dry, offset_dy, -1,
                                  kx, tx, x_fit, ky,
                                  ty, y_fit, Z_fit)
            if fp < s:
                (Ax, offset_x, nc_x,
                 Ay, offset_y, nc_y,
                 Drx, offset_dx, _,
                 Dry, offset_dy, _, Q) = build_matrices(
                    x_fit, y_fit, Z, tx, ty, kx, ky)
                _, C_sm, fp_sm = _p_search_hit_s(Ax, offset_x, nc_x,
                                  Drx, offset_dx,
                                  Ay, offset_y, nc_y, Q,
                                  Dry, offset_dy,
                                  kx, tx, x_fit, ky, ty, y_fit,
                                  Z_fit, s, fp0, p_init=1, verbose=verbose)
                return return_NdBSpline(fp_sm, (tx, ty, C_sm), (kx, ky))
            else:
                return return_NdBSpline(fp, (tx, ty, C0), (kx, ky))

        moves += grew_total
        if moves >= mpm:
            if verbose:
                print("Reached mpm limit, stopping growth")
            break

        fpold = fp

        if len(tx) >= nmaxx or len(ty) >= nmaxy:
            if verbose:
                print(f"[interp-exit] reached nmaxx/nmaxy: nx={len(tx)} ny={len(ty)} "
                      "-> building interpolatory knots & returning p=0")
            tx = make_interpolatory_knots(x_fit, kx)
            ty = make_interpolatory_knots(y_fit, ky)

            (Ax, offset_x, nc_x,
             Ay, offset_y, nc_y,
             Drx, offset_dx, _,
             Dry, offset_dy, _, Q) = build_matrices(
                x_fit, y_fit, Z, tx, ty, kx, ky)
            C0, fp = _solve_2d_fitpack(Ax, offset_x, nc_x,
                                  Drx, offset_dx,
                                  Ay, offset_y, nc_y, Q,
                                  Dry, offset_dy, -1,
                                  kx, tx, x_fit, ky,
                                  ty, y_fit, Z_fit)
            if fp < s:
                (Ax, offset_x, nc_x,
                 Ay, offset_y, nc_y,
                 Drx, offset_dx, _,
                 Dry, offset_dy, _, Q) = build_matrices(
                    x_fit, y_fit, Z, tx, ty, kx, ky)
                _, C_sm, fp_sm = _p_search_hit_s(Ax, offset_x, nc_x,
                                  Drx, offset_dx,
                                  Ay, offset_y, nc_y, Q,
                                  Dry, offset_dy,
                                  kx, tx, x_fit, ky, ty, y_fit,
                                  Z_fit, s, fp0, verbose=verbose)
                return return_NdBSpline(fp_sm, (tx, ty, C_sm), (kx, ky))
            else:
                return return_NdBSpline(fp, (tx, ty, C0), (kx, ky))

    if verbose:
        print(f"Growth end with fp={fp:.6e} > s={s:.6e} "
              f"({'nest exhausted' if (mx_head==0 and my_head==0) else 'early stop'})")

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

    x = np.asarray(x)
    y = np.asarray(y)
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
