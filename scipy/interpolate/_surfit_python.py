#!/usr/bin/env python3
"""
SURFIT-style tensor-product B-spline smoothing (banded, Python prototype)

This file contains a clear and practical re-implementation of the SURFIT
algorithm idea in plain NumPy and SciPy. It fits a smooth tensor-product
B-spline surface S(x, y) to scattered data by balancing data fit and
curvature. The fit is controlled by a smoothing parameter p and a target
weighted residual sum s.


-----------------------------------------------------------------------
Step-by-step algorithm used in this prototype
-----------------------------------------------------------------------

1) Build initial knot vectors
   - Start with open clamped knots with no interior knots in x and y.
   - Degrees are kx and ky. With no interior knots, the number of
     basis functions starts at kx + 1 by ky + 1.

2) Optional axis swap to reduce band width
   - Estimate the band width for the tensor-product system.
   - If swapping x and y leads to a smaller band width, swap them.
   - This reduces memory and speeds up the banded solver.

3) Assemble the linear system for a fixed set of knots
   - Data term: build A^T W A and A^T W z in band storage.
   - Penalty term: build R that penalizes curvature in x and y.
     We use R = kron(Qx, My) + kron(Mx, Qy), where
       Mx, My are the mass matrices (integrals of basis products)
       Qx, Qy are the roughness matrices (integrals of second derivative products)

4) Search for the smoothing parameter p
   - Solve (A^T W A + p R) c = A^T W z for different p.
   - Evaluate fp(p) = sum_i w_i * (z_i - S(x_i, y_i))^2.
   - Find p so that fp is close to the target s. We bracket the root and
     use a simple rational step, with a safe geometric fallback.

5) Check stopping conditions
   - If fp <= s * (1 + rtol), accept the current knots and return the spline.
   - If the number of coefficients already exceeds the number of data
     points, return the current fit.
   - If we hit the iteration limit for knot insertion, return the current fit.

6) If not done, insert one knot where it helps most
   - Compute residual energy (total weighted squared error
     still present in a span) per knot span in x and in y.
   - Pick the span with the highest energy, subject to remaining capacity
     in max_ncoef_x or max_ncoef_y.
   - Place the new knot at the energy centroid of that span, with a guard
     that prevents creation of very thin sliver intervals. If a list of
     candidate positions is supplied (for example, unique data coordinates),
     snap to the nearest candidate inside the span.
   - Go back to step 3 with the updated knot vector.

7) Output
   - Return a small wrapper object that stores knots, coefficients,
     degrees, chosen p, and achieved fp. Evaluation can be done using
     the stored (tx, ty, c) or by any compatible SciPy helper.


-----------------------------------------------------------------------
Which function does which step
-----------------------------------------------------------------------

Inputs and orchestration
- surfit_python(...): public entry point. Validates inputs, sets defaults,
  chooses the bounding box and the default s when needed, and then calls
  the banded driver _surfit_python_banded.

- _surfit_python_banded(...): main driver loop that implements steps 1 through 6.
  It builds knots, optionally swaps axes to reduce band width, assembles the
  data and penalty terms in band storage, searches for p, checks stopping
  conditions, and if needed inserts one interior knot and repeats.

Knot handling and basis evaluation
- open_clamped_knots(...): builds an open clamped knot vector on a given interval.
- find_span(...): binary search to find the B-spline knot span index for x or y.
- ders_basis_funs(...): returns the nonzero basis functions and derivatives at
  a point, up to a requested derivative order. With n=0 it returns the basis only.

Penalty matrices and quadrature
- univariate_M_Q(...): builds the univariate mass matrix M and roughness matrix Q
  for a given knot vector and degree using Gauss-Legendre quadrature per span.

System assembly and solver
- _assemble_data_normal_banded(...): builds the data normal equations A^T W A and
  the right-hand side A^T W z in symmetric band storage.
- _assemble_penalty_banded(...): builds the penalty matrix R in symmetric band
  storage using R = kron(Qx, My) + kron(Mx, Qy).
- _solve_banded(...): solves a symmetric positive definite band system via
  scipy.linalg.solveh_banded with upper storage.

Smoothing parameter search
- _p_search_banded(...): finds p so that fp(p) is close to s, using
  bracketing, a rational step similar in spirit to FITPACK fprati,
  and a geometric fallback. It calls the band solver at trial p values.

Residual processing and knot insertion
- evaluate_scatter_from_coeffs(...): evaluates the fitted surface S at all data
  points using cached local basis values.
- _interval_energy_centroids(...): computes the sum of weighted squared residuals
  per knot span and their energy-weighted coordinate sums. These are used to
  place a new knot at the centroid coord / fpint.
- _insert_centroid_with_sliver_guard(...): inserts one knot inside a chosen span.
  It uses the energy centroid, snaps to the nearest candidate if provided, and
  rejects placements that would create very thin sliver intervals.
- residual_histograms(...): helper for building residual energy per span. The main
  driver uses _interval_energy_centroids directly.

Other helpers
- _maybe_swap_axes_for_bandwidth(...): optionally swaps x and y, and the corresponding
  knots and degrees, to reduce the band width before assembly.
- SmoothBivariateSplinePython: light wrapper that stores (tx, ty, c), degrees,
  chosen p, and achieved fp. This mirrors the SciPy object enough for simple use.


-----------------------------------------------------------------------
How this prototype differs from FITPACK fpsurf (Fortran)
-----------------------------------------------------------------------

The original routine is: recursive subroutine fpsurf(...)
It is long and very optimized. Below are the main conceptual differences.

A) How the linear system is formed and solved
- FITPACK constructs and triangularizes the observation matrix using Givens
  rotations in place (calls fpgivs and fprota). It also has explicit rank
  checks and can compute minimum norm solutions when the system is rank
  deficient (calls fprank).
- This prototype never forms or rotates a tall observation matrix. It builds
  the symmetric normal equations directly in symmetric band storage:
  A^T W A for the data term, and an explicit penalty matrix R, then solves
  (A^T W A + p R) c = A^T W z using solveh_banded.
- Rank deficiency: FITPACK scales by dmax, checks diagonal magnitudes, and can
  fall back to a minimum norm solution. This prototype does not perform a full
  rank-revealing path. It adds a tiny ridge 1e-12 to the main diagonal before
  solving. This is simpler but less robust in truly rank-deficient cases.

B) How the smoothness penalty is represented
- FITPACK extends the rotated observation matrix with extra rows so that the
  solution behaves like a polynomial of degree kx in x and ky in y, with those
  extra rows weighted by 1/p. It also uses discontinuity jumps of higher order
  derivatives at interior knots (see fpdisc and the arrays bx, by).
- This prototype builds the penalty as an explicit symmetric positive definite
  matrix R = kron(Qx, My) + kron(Mx, Qy). The univariate matrices M and Q
  are computed by Gauss-Legendre integration of basis products and second
  derivative products. The penalty enters as p R in the normal equations.

C) p-search logic
- FITPACK uses a careful iteration with constants con4 and con9 to expand or
  shrink p and relies on fprati to update p from three evaluations, with proof
  style bracketing and control on fp-s.
- This prototype follows the same spirit. It explicitly brackets a root of
  fp(p) - s by growing p by a factor of 10 when needed, uses a simple rational
  step like fprati, and falls back to the geometric mean when the model step
  is not safe. The constants and exact flow differ, but the idea is the same.

D) Data ordering and paneling
- FITPACK groups data points into panels (rectangles formed by knot lines),
  orders them by panel with fporde, and reuses local values spx and spy.
- This prototype uses a simpler path. It precomputes the nonzero basis values
  per point and assembles the banded normal equations directly. This is easier
  to read but may be slower on very large problems.

E) Knot insertion policy
- Both systems look at residual energy per interval and place a new knot at
  an energy centroid arg = coord / fpint, with a sliver guard:
  do not insert if the new interval would be more than about 10 to 1 smaller
  than its neighbor.
- This prototype can snap the new knot to the nearest candidate inside the
  span, for example a unique data value, which can help numerical stability.
  FITPACK does not snap to data in this way.

F) Axis swap to reduce band width
- Both systems may swap x and y to reduce the band width of the system. In
  FITPACK this is tracked with ichang and also affects which arrays are
  copied. This prototype performs a simple check and swap using
  _maybe_swap_axes_for_bandwidth.

G) Error codes and restart options
- FITPACK reports many detailed error codes in ier and supports options like
  iopt to restart from previous knots, or to accept the least squares spline
  directly for iopt = -1.
- This prototype does not support these options or error codes. It always
  starts from open clamped knots, and returns the best fit it found under
  the chosen limits. It does not expose ier style codes.

H) Discontinuity jump tables
- FITPACK uses fpdisc to compute derivative jump tables bx and by at interior
  knots, and uses them in the extended system.
- This prototype does not compute jump tables. It directly integrates to form
  M and Q and then builds R from those matrices.

I) Rank-1 data guard
- This prototype checks if the data are almost colinear in the plane of
  (x, y). If that happens it freezes interior knots by capping the maximum
  number of coefficients in each direction to at least degree+1. This mirrors
  a practical behavior seen in tests. FITPACK handles these cases within its
  robust linear algebra and ordering logic rather than by this guard.


-----------------------------------------------------------------------
Notes and practical tips
-----------------------------------------------------------------------

- Degrees: Typical choice is kx = ky = 3 for a cubic surface.
- Default target s: When s is None we set s = m - sqrt(2*m). This mirrors a
  common rule of thumb used in many spline packages.
- Band storage layout: We use SciPy upper storage for symmetric band matrices.
  ab[u, j] stores the main diagonal. ab[u-1, 1:] stores the first super
  diagonal, and so on. Helper _add_ab writes into this layout.
- Memory layouts for coefficients: For evaluation we follow Fortran order
  flattening of the coefficient grid (column-major). Inside the banded p-search
  we sometimes keep a C-order view for convenience and convert as needed.
- Sliver guard: The knot insertion code rejects placements that create an
  interval more than about 10 times smaller than its neighbor. This keeps
  condition numbers under control.
- Stopping logic: We accept the fit when fp <= s * (1 + rtol). You can tighten
  rtol for a closer match to s. If the system becomes too large relative to m,
  or the knot insertion budget is exhausted, we return the current best.


-----------------------------------------------------------------------
Rough map from FITPACK calls to this prototype
-----------------------------------------------------------------------

- fpbspl         -> ders_basis_funs with n = 0 (basis values only)
- fpgivs,fprota  -> not used; we solve symmetric band systems instead
- fpback         -> replaced by solveh_banded on the banded normal equations
- fpdisc         -> not used; we build M and Q by numerical quadrature
- fprati         -> mirrored by _fprati_next and the bracketing loop
- residual fpint, coord and arg logic -> mirrored by
  _interval_energy_centroids and _insert_centroid_with_sliver_guard
- axis swap logic with ichang -> mirrored by _maybe_swap_axes_for_bandwidth


-----------------------------------------------------------------------
Caveats
-----------------------------------------------------------------------

- This prototype favors clarity. It can be slower than FITPACK on very
  large problems because it does not do in-place rotations.
- Rank deficient or nearly rank deficient systems may need stronger regularization
  than the tiny ridge. If you see instability, increase weights, reduce degrees,
  or keep p away from zero.
- The penalty uses second derivatives in both directions. You can change quad_n
  in univariate_M_Q to trade accuracy for speed.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import SmoothBivariateSpline
from scipy.linalg import solveh_banded


def find_span(t, p, x):
    """
    Return the knot span index j such that t[j] <= x < t[j+1].

    Parameters
    ----------
    t : array_like
        Nondecreasing knot vector of length ncoef + p + 1.
    p : int
        Spline degree. For cubic p = 3.
    x : float
        Query location.

    Returns
    -------
    j : int
        Knot span index. It is in the range [p, ncoef - 1].

    Notes
    -----
    This is the standard binary search used in B spline evaluation.
    It clamps to the first or last span when x is outside the open interval.
    """
    ncoef = len(t) - p - 1
    if x <= t[p]:
        return p
    if x >= t[ncoef]:
        return ncoef - 1
    lo, hi = p, ncoef
    while True:
        mid = (lo + hi) // 2
        if x < t[mid]:
            hi = mid
        elif x >= t[mid + 1]:
            lo = mid
        else:
            return mid


def ders_basis_funs(i, x, p, U, n):
    """
    Evaluate nonzero B spline basis functions and their derivatives up to order n.

    Parameters
    ----------
    i : int
        Knot span index so that U[i] <= x < U[i+1].
    x : float
        Query location.
    p : int
        Spline degree.
    U : array_like
        Knot vector.
    n : int
        Highest derivative to return. Use n = 0 for only the basis.

    Returns
    -------
    ders : ndarray, shape (n+1, p+1)
        ders[k, j] is the k th derivative of N_{i-p+j,p}(x).
        ders[0] holds the basis values.

    Notes
    -----
    This is a direct translation of the algorithm in Piegl and Tiller.
    It uses the triangular table ndu and a small 2 by (p+1) buffer a.
    """
    ndu = np.zeros((p + 1, p + 1), dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = x - U[i + 1 - j]
        right[j] = U[i + j] - x
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = 0.0 if ndu[j, r] == 0.0 else ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    ders = np.zeros((n + 1, p + 1), dtype=float)
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    a = np.zeros((2, p + 1), dtype=float)
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    rfact = float(p)
    for k in range(1, n + 1):
        for j in range(p + 1):
            ders[k, j] *= rfact
        rfact *= (p - k)
    return ders


def open_clamped_knots(a, b, p, ncoef):
    """
    Build an open clamped knot vector on [a, b] for a given degree and ncoef.

    Parameters
    ----------
    a, b : float
        Domain bounds. Must satisfy a < b.
    p : int
        Degree of the spline.
    ncoef : int
        Number of coefficients. Must be greater than p.

    Returns
    -------
    t : ndarray
        Knot vector of length ncoef + p + 1. The first p+1 values are a.
        The last p+1 values are b. The interior knots are uniform if any.

    Notes
    -----
    If there are no interior knots the vector is just [a,...,a, b,...,b].
    """
    if ncoef <= p:
        raise ValueError("ncoef must be > degree")
    n_inner = ncoef - p - 1
    t0 = np.full(p + 1, a, dtype=float)
    t1 = np.full(p + 1, b, dtype=float)
    if n_inner <= 0:
        return np.concatenate([t0, t1])
    inner = np.linspace(a, b, n_inner + 2)[1:-1]
    return np.concatenate([t0, inner, t1])


def univariate_M_Q(t, p, quad_n=8):
    """
    Build univariate mass M and roughness Q matrices for a spline basis.

    Parameters
    ----------
    t : array_like
        Knot vector.
    p : int
        Degree of the spline.
    quad_n : int, default 8
        Number of Gauss Legendre points per knot span.

    Returns
    -------
    M : ndarray, shape (ncoef, ncoef)
        M[i,j] = integral N_i(x) N_j(x) dx on the domain.
    Q : ndarray, shape (ncoef, ncoef)
        Q[i,j] = integral N_i''(x) N_j''(x) dx on the domain.

    Notes
    -----
    The integrals are a sum over spans of simple Gauss Legendre rules.
    Only the local p by p block in each active span is updated.
    """
    ncoef = len(t) - p - 1
    M = np.zeros((ncoef, ncoef), dtype=float)
    Q = np.zeros((ncoef, ncoef), dtype=float)
    xi, wi = leggauss(quad_n)

    for j in range(p, ncoef):
        a, b = t[j], t[j + 1]
        if b <= a:
            continue
        xm = 0.5 * (a + b)
        xr = 0.5 * (b - a)
        for q in range(quad_n):
            x = xm + xr * xi[q]
            ders = ders_basis_funs(j, x, p, t, n=2)
            N = ders[0]
            N2 = ders[2]
            i0 = j - p

            for aidx in range(p + 1):
                ia = i0 + aidx
                Na, Na2 = N[aidx], N2[aidx]
                for bidx in range(p + 1):
                    ib = i0 + bidx
                    Nb, Nb2 = N[bidx], N2[bidx]
                    wq = wi[q] * xr
                    M[ia, ib] += wq * (Na * Nb)
                    Q[ia, ib] += wq * (Na2 * Nb2)

    M = 0.5 * (M + M.T)
    Q = 0.5 * (Q + Q.T)
    return M, Q


def assemble_data_normal(x, y, z, w, tx, ty, kx, ky):
    """
    Assemble dense normal equations for the data term only.

    Parameters
    ----------
    x, y : array_like, shape (m,)
        Sample coordinates.
    z : array_like, shape (m,)
        Sample values.
    w : array_like, shape (m,)
        Nonnegative weights.
    tx, ty : arrays
        Knot vectors in x and y.
    kx, ky : int
        Degrees in x and y.

    Returns
    -------
    ATA : ndarray, shape (nx*ny, nx*ny)
        Upper triangular part of A^T W A stored densely and then symmetrized.
    ATz : ndarray, shape (nx*ny,)
        Right hand side A^T W z.
    basis_x, basis_y : list
        Cached local basis info per point. Each entry is (start_index, values).

    Notes
    -----
    This version is dense and used for clarity and testing.
    The banded version is used in production.
    """
    m = x.size
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    ncoef = nx * ny

    ATA = np.zeros((ncoef, ncoef), dtype=float)
    ATz = np.zeros(ncoef, dtype=float)

    basis_x = []
    basis_y = []
    for k in range(m):
        jx = find_span(tx, kx, float(x[k]))
        dersx = ders_basis_funs(jx, float(x[k]), kx, tx, n=0)
        Nx = dersx[0]
        ix0 = jx - kx
        basis_x.append((ix0, Nx.copy()))

        jy = find_span(ty, ky, float(y[k]))
        dersy = ders_basis_funs(jy, float(y[k]), ky, ty, n=0)
        Ny = dersy[0]
        iy0 = jy - ky
        basis_y.append((iy0, Ny.copy()))

    for k in range(m):
        wk = w[k]
        zk = z[k]
        ix0, Nx = basis_x[k]
        iy0, Ny = basis_y[k]

        for ax in range(kx + 1):
            i = ix0 + ax
            if i < 0 or i >= nx:
                continue
            vx = Nx[ax]
            for ay in range(ky + 1):
                j = iy0 + ay
                if j < 0 or j >= ny:
                    continue
                vy = Ny[ay]
                L = i + nx * j
                ATz[L] += wk * zk * (vx * vy)

        for ax in range(kx + 1):
            i = ix0 + ax
            if i < 0 or i >= nx:
                continue
            vx = Nx[ax]
            for ay in range(ky + 1):
                j = iy0 + ay
                if j < 0 or j >= ny:
                    continue
                vy = Ny[ay]
                L = i + nx * j
                Vij = vx * vy
                for bx in range(kx + 1):
                    ip = ix0 + bx
                    if ip < 0 or ip >= nx:
                        continue
                    vxp = Nx[bx]
                    for by in range(ky + 1):
                        jp = iy0 + by
                        if jp < 0 or jp >= ny:
                            continue
                        vyp = Ny[by]
                        Lp = ip + nx * jp
                        if Lp < L:
                            continue
                        ATA[L, Lp] += wk * Vij * (vxp * vyp)

    iu = np.triu_indices(ncoef, k=1)
    ATA[(iu[1], iu[0])] = ATA[iu]
    return ATA, ATz, basis_x, basis_y


def evaluate_scatter_from_coeffs(c, nx, ny, basis_x, basis_y):
    """
    Evaluate the tensor product spline at all data points.

    Parameters
    ----------
    c : array_like, shape (nx*ny,)
        Coefficients in Fortran order flattening, that is column major.
    nx, ny : int
        Number of basis functions in x and y.
    basis_x, basis_y : list
        Cached local basis info per point as produced by assemble functions.

    Returns
    -------
    out : ndarray, shape (m,)
        Predicted values at each input point.
    """
    m = len(basis_x)
    out = np.zeros(m, dtype=float)
    for k in range(m):
        ix0, Nx = basis_x[k]
        iy0, Ny = basis_y[k]
        val = 0.0
        for ax, vx in enumerate(Nx):
            i = ix0 + ax
            if i < 0 or i >= nx:
                continue
            for ay, vy in enumerate(Ny):
                j = iy0 + ay
                if j < 0 or j >= ny:
                    continue
                L = i + nx * j
                val += c[L] * (vx * vy)
        out[k] = val
    return out


def _interval_energy_centroids(x, y, res2, tx, ty, kx, ky):
    """
    Compute per span residual energy and energy weighted centroids.

    Parameters
    ----------
    x, y : arrays of points
    res2 : array
        Weighted squared residuals per point.
    tx, ty : knot vectors
    kx, ky : degrees

    Returns
    -------
    fpint_x, coord_x, fpint_y, coord_y : ndarrays
        For each knot interval j, fpint holds the sum of energy in that span
        and coord holds the sum of energy times coordinate. The centroid is
        coord[j] / fpint[j] when fpint[j] > 0.
    """
    nxk = len(tx) - 1
    nyk = len(ty) - 1
    fpint_x = np.zeros(nxk, float)
    coord_x = np.zeros(nxk, float)
    fpint_y = np.zeros(nyk, float)
    coord_y = np.zeros(nyk, float)

    for xi, yi, e in zip(x, y, res2):
        jx = find_span(tx, kx, float(xi))
        jy = find_span(ty, ky, float(yi))
        fpint_x[jx] += e
        coord_x[jx] += e * float(xi)
        fpint_y[jy] += e
        coord_y[jy] += e * float(yi)

    return fpint_x, coord_x, fpint_y, coord_y


def _insert_centroid_with_sliver_guard(t, k, span_j, num, den, candidates=None):
    """
    Insert a knot at the energy centroid of a span with a sliver guard.

    Parameters
    ----------
    t : array_like
        Knot vector.
    k : int
        Degree.
    span_j : int
        Interval index. The span is [t[j], t[j+1]).
    num : float
        Sum of energies in this span. If not positive, no insert is done.
    den : float
        Sum of energies times coordinate in this span.
    candidates : array_like or None
        Optional list of candidate positions, usually unique data values.
        If given, the inserted knot snaps to the nearest candidate inside.

    Returns
    -------
    t_new : ndarray or None
        A new knot vector with one value inserted, or None if not inserted.

    Notes
    -----
    The sliver guard avoids creating a very small interval next to a larger
    one. It rejects if the ratio of the two sides is worse than 10 to 1.
    It also nudges away from exact duplicates at machine precision.
    """
    if num <= 0.0:
        return None

    a, b = float(t[span_j]), float(t[span_j + 1])
    arg = float(den / num)

    if not (a < arg < b):
        eps = 1e-12 * max(1.0, (b - a))
        arg = min(max(arg, a + eps), b - eps)

    if candidates is not None:
        cand = np.asarray(candidates, float)
        cand = cand[(cand > a) & (cand < b)]
        if cand.size:
            arg = float(cand[np.argmin(np.abs(cand - arg))])

    fac1 = b - arg
    fac2 = arg - a
    if fac1 > 10.0 * fac2 or fac2 > 10.0 * fac1:
        return None

    j = span_j + 1

    if np.isclose(arg, t[j - 1]) or np.isclose(arg, t[j]):
        eps = 1e-12 * max(1.0, (b - a))
        arg = min(max(arg, a + eps), b - eps)

    return np.insert(t, j, arg)


def _fprati_next(p1, f1, p2, f2, p3, f3):
    """
    Compute the next p by solving a simple rational model of fp - s.

    Parameters
    ----------
    p1, p2, p3 : float
        Three p values that bracket the root. p1 < p3 and f1 > 0, f3 < 0.
    f1, f2, f3 : float
        Values of fp - s at those p.

    Returns
    -------
    p : float or None
        Proposed next p inside the bracket. None if the step is not valid.

    Notes
    -----
    This is the same idea as FITPACK fprati. It fits a model
    a + b p + c p f(p) = f(p) and takes p = -a / b.
    """
    M = np.array([
        [1.0, p1, -f1 * p1],
        [1.0, p2, -f2 * p2],
        [1.0, p3, -f3 * p3],
    ], dtype=float)
    rhs = np.array([f1, f2, f3], dtype=float)
    try:
        a, b, c = np.linalg.solve(M, rhs)
        if b == 0.0:
            return None
        p = -a / b

        lo, hi = (p1, p3) if p1 < p3 else (p3, p1)
        if not np.isfinite(p) or p <= 0:
            return None

        p = float(np.clip(p, lo * (1 + 1e-12), hi * (1 - 1e-12)))
        return p
    except np.linalg.LinAlgError:
        return None


def _geom_mid(p_lo, p_hi):
    """
    Geometric mean of two positive values with floor for safety.

    Parameters
    ----------
    p_lo, p_hi : float
        Positive values with p_lo <= p_hi.

    Returns
    -------
    mid : float
        sqrt(max(p_lo, 1e-300) * max(p_hi, 1e-300)).
    """
    return float(np.sqrt(max(p_lo, 1e-300) * max(p_hi, 1e-300)))


def p_search(ATA, ATz, R, nx, ny, basis_x, basis_y, z, w,
             s_target, p0=1.0, rtol=1e-3, maxit=25, verbose=False):
    """
    Search for smoothing parameter p so that fp is close to s_target.

    Parameters
    ----------
    ATA : ndarray
        Dense symmetric matrix for data term normal equations.
    ATz : ndarray
        Right hand side.
    R : ndarray
        Dense symmetric penalty matrix.
    nx, ny : int
        Basis sizes in x and y.
    basis_x, basis_y : list
        Cached basis per point.
    z, w : arrays
        Observations and weights.
    s_target : float or None
        Target weighted residual sum. If None or <= 0, use p = 0.
    p0 : float
        Initial p to try when bracketing.
    rtol : float
        Relative tolerance on fp versus target.
    maxit : int
        Maximum number of iterations.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    p : float
        Chosen p.
    c : ndarray
        Coefficients at that p.
    fp : float
        Achieved weighted residual sum.

    Notes
    -----
    The function fp(p) is nondecreasing. We bracket the root of fp(p) - s.
    We then use a rational step with fallback to geometric bisection.
    """
    n = ATA.shape[0]
    I = np.eye(n)

    def solve_at(p):
        Areg = ATA + p * R + 1e-12 * I
        c = np.linalg.solve(Areg, ATz)
        zhat = evaluate_scatter_from_coeffs(c, nx, ny, basis_x, basis_y)
        fp = float(np.sum(w * (z - zhat) ** 2))
        return fp, c

    if s_target is None or s_target <= 0.0:
        fp0, c0 = solve_at(0.0)
        return 0.0, c0, fp0

    p1 = 0.0
    f1, c1 = solve_at(p1)
    if verbose:
        print(f"[psearch] p=0 fp={f1:.6e} target={s_target:.6e}")
    f1 -= s_target

    if f1 <= rtol * s_target:
        return 0.0, c1, f1 + s_target

    p3 = max(p0, 1.0)
    f3, c3 = solve_at(p3)
    f3 -= s_target
    grow = 10.0
    n_grow = 0
    while f3 > 0.0 and n_grow < 25:
        p3 *= grow
        f3, c3 = solve_at(p3)
        f3 -= s_target
        n_grow += 1
        if verbose:
            print(f"[psearch] grow p={p3:.3e} -> fp-s={f3:+.3e}")

    if f3 > 0.0:
        if abs(f1) <= abs(f3):
            return p1, c1, f1 + s_target
        else:
            return p3, c3, f3 + s_target

    p2 = _geom_mid(max(p1, 1e-300), p3)
    f2, c2 = solve_at(p2)
    f2 -= s_target
    if verbose:
        print(f"[psearch] bracket p1={p1:.3e}(+), p3={p3:.3e}(-), "
              f"try p2={p2:.3e} f2={f2:+.3e}")

    for _ in range(maxit):

        if abs(f2) <= rtol * s_target:
            return p2, c2, f2 + s_target

        if not (f1 > f2 > f3):
            p2 = _geom_mid(max(p1, 1e-300), p3)
            f2, c2 = solve_at(p2)
            f2 -= s_target
            if verbose:
                print(f"[psearch] repair -> p2={p2:.3e} f2={f2:+.3e}")

        p_new = _fprati_next(p1, f1, p2, f2, p3, f3)
        if p_new is None or not (min(p1, p3) < p_new < max(p1, p3)):
            p_new = _geom_mid(max(p1, 1e-300), p3)

        f_new, c_new = solve_at(p_new)
        f_new -= s_target
        if verbose:
            print(f"[psearch] step p={p_new:.3e} f={f_new:+.3e}")

        if f_new > 0.0:
            p1, f1, c1 = p_new, f_new, c_new
        else:
            p3, f3, c3 = p_new, f_new, c_new

        if abs(f1) < abs(f3):
            p2, f2, c2 = p1, f1, c1
        else:
            p2, f2, c2 = p3, f3, c3

    if abs(f1) <= abs(f3):
        return p1, c1, f1 + s_target
    else:
        return p3, c3, f3 + s_target


def residual_histograms(x, y, res2, tx, ty, kx, ky):
    """
    Build residual energy histograms per knot interval in x and y.

    Parameters
    ----------
    x, y : arrays of points
    res2 : array
        Weighted squared residual per point.
    tx, ty : knot vectors
    kx, ky : degrees

    Returns
    -------
    Ex, Ey : ndarrays
        Energy per interval in x and y.
    jx_list, jy_list : ndarrays
        Span index of each point in x and y.
    """
    nxk = len(tx) - 1
    nyk = len(ty) - 1
    Ex = np.zeros(nxk, dtype=float)
    Ey = np.zeros(nyk, dtype=float)
    jx_list = np.empty(x.size, dtype=int)
    jy_list = np.empty(y.size, dtype=int)
    for idx, (xi, yi, e) in enumerate(zip(x, y, res2)):
        jx = find_span(tx, kx, float(xi))
        jy = find_span(ty, ky, float(yi))
        Ex[jx] += e
        Ey[jy] += e
        jx_list[idx] = jx
        jy_list[idx] = jy
    return Ex, Ey, jx_list, jy_list


def insert_at_data_median(t, k, j_span, data_in_span):
    """
    Insert a knot near the median of data in a span, with a small tolerance.

    Parameters
    ----------
    t : array_like
        Knot vector.
    k : int
        Degree.
    j_span : int
        Interval index.
    data_in_span : array_like
        Data values inside the span to compute a median from.

    Returns
    -------
    t_new : ndarray
        Knot vector with one value inserted, or the original if the new value
        would be too close to an existing knot.

    Notes
    -----
    If there are no data inside the span the midpoint is used.
    A small tolerance based on average interior knot spacing avoids duplicates.
    """
    a, b = t[j_span], t[j_span + 1]
    if data_in_span.size:
        u = float(np.median(data_in_span))
    else:
        u = 0.5 * (a + b)

    inner = max(1, len(t) - 2 * (k + 1) + 1)
    avg = (t[-k - 1] - t[k]) / inner if inner > 0 else (b - a)
    eps = max(1e-12, 0.02 * avg)
    j = int(np.searchsorted(t, u))
    if j > 0 and abs(u - t[j - 1]) < eps:
        return t
    if j < len(t) and abs(t[j] - u) < eps:
        return t
    return np.insert(t, j, u)


class SmoothBivariateSplinePython(SmoothBivariateSpline):
    """
    Light wrapper that stores knots, coefficients, degrees, p, and fp.

    Notes
    -----
    This class mirrors the SciPy object enough for simple evaluation use.
    It keeps tck as (tx, ty, c_flat) with c flattened in Fortran order.
    """

    def __init__(self, tx, ty, C, kx, ky, p, fp):
        """
        Create a spline object from knots and a coefficient grid.

        Parameters
        ----------
        tx, ty : arrays
            Knot vectors in x and y.
        C : ndarray, shape (nx, ny)
            Coefficient grid in Fortran order. That is columns are contiguous
            when flattened.
        kx, ky : int
            Degrees in x and y.
        p : float
            Smoothing parameter used to fit these coefficients.
        fp : float
            Achieved weighted residual sum.
        """
        tx = np.asarray(tx, float)
        ty = np.asarray(ty, float)
        C = np.asarray(C, float)
        nx = tx.size - kx - 1
        ny = ty.size - ky - 1
        if C.shape != (nx, ny):
            raise ValueError(f"C shape {C.shape} != ({nx},{ny}) from knots")
        c_flat = np.ravel(C, order="C")
        self.tck = (tx, ty, c_flat)
        self.degrees = (kx, ky)
        self.p = float(p)
        self.fp = float(fp)


def _validate_input(x, y, z, w, kx, ky, eps):
    """
    Validate and normalize basic inputs.

    Parameters
    ----------
    x, y, z : arrays
        Data coordinates and values.
    w : array or None
        Weights. If None it will be created as ones.
    kx, ky : int
        Degrees in x and y.
    eps : float or None
        Optional small parameter reserved for future use.

    Returns
    -------
    x, y, z, w : ndarrays
        Normalized arrays.

    Raises
    ------
    ValueError
        If shapes are inconsistent or degrees are not supported.

    Notes
    -----
    The function only checks shapes and basic conditions. It does not sort or
    deduplicate inputs. Degrees must be nonnegative.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    if not x.size == y.size == z.size:
        raise ValueError("x, y, and z should have a same length")

    if w is not None:
        w = np.asarray(w)
        if x.size != w.size:
            raise ValueError("x, y, z, and w should have a same length")
        elif not np.all(w >= 0.0):
            raise ValueError("w should be positive")
    if (eps is not None) and (not 0.0 < eps < 1.0):
        raise ValueError("eps should be between (0, 1)")
    if not x.size >= (kx + 1) * (ky + 1):
        raise ValueError(
            "The length of x, y and z should be at least (kx+1) * (ky+1)"
        )

    return x, y, z, w


def _alloc_symm_band(n, u):
    """
    Allocate a symmetric band matrix in upper storage (first version).

    Parameters
    ----------
    n : int
        Order of the square matrix.
    u : int
        Number of super diagonals to store.

    Returns
    -------
    ab : ndarray, shape (u+1, n)
        Banded array container. ab[0] is the main diagonal if using lower=False.

    Notes
    -----
    This version appears before the convention switch and is kept for clarity.
    It is later redefined with the exact storage layout used by solveh_banded.
    """
    return np.zeros((u + 1, n), dtype=float)


def _add_ab(ab, u, i, j, val):
    """
    Add val to position (i, j) in a symmetric band matrix (first version).

    Parameters
    ----------
    ab : ndarray
        Banded matrix storage.
    u : int
        Number of super diagonals stored.
    i, j : int
        Row and column indexes.
    val : float
        Value to add.

    Notes
    -----
    This helper respects symmetry and only updates the stored triangle.
    This first version uses a simple top aligned convention.
    """
    if j < i:
        i, j = j, i
    k = j - i
    if k <= u:
        ab[k, i] += val


def _kron_banded_add(ab, u, A, B, ny):
    """
    Accumulate ab += kron(A, B) into band storage, using upper triangle.

    Parameters
    ----------
    ab : ndarray
        Target band matrix in upper storage.
    u : int
        Band width in super diagonals.
    A : ndarray
        Square matrix for x direction.
    B : ndarray
        Square matrix for y direction.
    ny : int
        Size of the B factor. Used to compute block offsets.

    Notes
    -----
    For each block pair we update only the upper triangle and within the band.
    This first version matches the first storage convention above.
    """
    nx = A.shape[0]
    ny_ = B.shape[0]
    assert ny_ == ny

    for ia in range(nx):
        for ja in range(ia, nx):
            a = A[ia, ja]
            if a == 0.0:
                continue
            row_block = ia * ny
            col_block = ja * ny

            for ib in range(ny):
                Li = row_block + ib
                for jb in range(ib, ny):
                    Lj = col_block + jb
                    val = a * B[ib, jb]
                    k = Lj - Li
                    if 0 <= k <= u:
                        ab[k, Li] += val


def _assemble_data_normal_banded(x, y, z, w, tx, ty, kx, ky):
    """
    Assemble the data normal equations in symmetric band storage.

    Parameters
    ----------
    x, y, z, w : arrays
        Input points, values, and weights.
    tx, ty : arrays
        Knot vectors.
    kx, ky : int
        Degrees.

    Returns
    -------
    ab : ndarray
        Upper triangle band storage of A^T W A.
    ATz : ndarray
        Right hand side A^T W z.
    basis_x, basis_y : list
        Cached basis per point as tuples (start_index, values).
    nx, ny : int
        Basis sizes.
    u_data : int
        Data band width in super diagonals.

    Notes
    -----
    The band width is kx * ny + ky for the tensor product with degrees kx, ky.
    """
    m = x.size
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    ncof = nx * ny
    u_data = kx * ny + ky

    ab = _alloc_symm_band(ncof, u_data)
    ATz = np.zeros(ncof, dtype=float)

    basis_x = []
    basis_y = []
    for k in range(m):
        jx = find_span(tx, kx, float(x[k]))
        Nx = ders_basis_funs(jx, float(x[k]), kx, tx, n=0)[0]
        ix0 = jx - kx
        basis_x.append((ix0, Nx.copy()))

        jy = find_span(ty, ky, float(y[k]))
        Ny = ders_basis_funs(jy, float(y[k]), ky, ty, n=0)[0]
        iy0 = jy - ky
        basis_y.append((iy0, Ny.copy()))

    for k in range(m):
        wk = w[k]
        zk = z[k]
        ix0, Nx = basis_x[k]
        iy0, Ny = basis_y[k]

        active = []
        vals = []
        for ax in range(kx + 1):
            i = ix0 + ax
            if not (0 <= i < nx):
                continue
            vx = Nx[ax]
            for ay in range(ky + 1):
                j = iy0 + ay
                if not (0 <= j < ny):
                    continue
                vy = Ny[ay]
                Lb = i * ny + j
                v = vx * vy
                active.append(Lb)
                vals.append(v)

        wz = wk * zk
        for idx, Lb in enumerate(active):
            ATz[Lb] += wz * vals[idx]

        for a in range(len(active)):
            La = active[a]
            va = vals[a]
            for b in range(a, len(active)):
                Lb = active[b]
                vb = vals[b]
                _add_ab(ab, u_data, La, Lb, wk * va * vb)

    return ab, ATz, basis_x, basis_y, nx, ny, u_data


def _assemble_penalty_banded(tx, ty, kx, ky, quad_n=6):
    """
    Assemble the roughness penalty R in symmetric band storage.

    Parameters
    ----------
    tx, ty : arrays
        Knot vectors in x and y.
    kx, ky : int
        Degrees.
    quad_n : int
        Number of Gauss points per span for the integrals.

    Returns
    -------
    abR : ndarray
        Band storage of R = kron(Qx, My) + kron(Mx, Qy).
    u_p : int
        Band width of the penalty part.

    Notes
    -----
    The univariate matrices M and Q come from univariate_M_Q.
    """
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    ncof = nx * ny

    Mx, Qx = univariate_M_Q(tx, kx, quad_n=quad_n)
    My, Qy = univariate_M_Q(ty, ky, quad_n=quad_n)

    u_p = kx * ny + ky
    abR = _alloc_symm_band(ncof, u_p)

    _kron_banded_add(abR, u_p, Qx, My, ny)
    _kron_banded_add(abR, u_p, Mx, Qy, ny)

    return abR, u_p


def _solve_banded(ab_total, u, ATz):
    """
    Solve a symmetric positive definite band system.

    Parameters
    ----------
    ab_total : ndarray
        Symmetric band matrix in upper storage as used by SciPy.
    u : int
        Number of super diagonals.
    ATz : ndarray
        Right hand side.

    Returns
    -------
    c : ndarray
        Solution vector.

    Notes
    -----
    This is a thin wrapper around scipy.linalg.solveh_banded with lower=False.
    """
    return solveh_banded(ab_total, ATz, lower=False, check_finite=False)


# From here on we redefine helpers to match SciPy upper storage layout exactly.

def _alloc_symm_band(n, u):
    """
    Allocate a symmetric band matrix in SciPy upper storage.

    Parameters
    ----------
    n : int
        Matrix size.
    u : int
        Number of super diagonals.

    Returns
    -------
    ab : ndarray, shape (u+1, n)
        ab[u, :] is the main diagonal. ab[u-1, 1:] is the first super diagonal.
    """
    return np.zeros((u + 1, n), dtype=float)


def _add_ab(ab, u, i, j, val):
    """
    Add val to position (i, j) for SciPy upper band layout.

    Parameters
    ----------
    ab : ndarray
        Band matrix.
    u : int
        Number of super diagonals.
    i, j : int
        Row and column.
    val : float
        Value to add.

    Notes
    -----
    The element (i, j) maps to ab[u - (j - i), j] when j >= i and j - i <= u.
    """
    if j < i:
        i, j = j, i
    k = j - i
    if 0 <= k <= u:
        ab[u - k, j] += val


def _kron_banded_add(ab, u, A, B, ny):
    """
    Add the upper triangle of kron(A, B) into ab with SciPy band layout.

    Parameters
    ----------
    ab : ndarray
        Target band matrix.
    u : int
        Number of super diagonals.
    A, B : ndarrays
        Factors of the Kronecker product.
    ny : int
        Size of B.

    Notes
    -----
    Only updates entries inside the band and in the upper triangle.
    """
    nx = A.shape[0]
    ny_ = B.shape[0]
    assert ny_ == ny

    for ia in range(nx):
        for ja in range(ia, nx):
            a = A[ia, ja]
            if a == 0.0:
                continue
            row_block = ia * ny
            col_block = ja * ny
            for ib in range(ny):
                Li = row_block + ib
                for jb in range(ib, ny):
                    Lj = col_block + jb
                    val = a * B[ib, jb]
                    _add_ab(ab, u, Li, Lj, val)


def _p_search_banded(
    abATA, u_data, ATz,
    abR, u_p, nx, ny,
    basis_x, basis_y, z, w,
    s_target, p0=1.0, rtol=1e-3,
    maxit=25, verbose=False
):
    """
    Band friendly p search. Same goal as p_search but works with band storage.

    Parameters
    ----------
    abATA : ndarray
        Data normal equations in band storage.
    u_data : int
        Band width for data part.
    ATz : ndarray
        Right hand side.
    abR : ndarray
        Penalty matrix in band storage.
    u_p : int
        Band width for penalty part.
    nx, ny : int
        Basis sizes.
    basis_x, basis_y : list
        Cached basis per point.
    z, w : arrays
        Observations and weights.
    s_target : float or None
        Target residual sum.
    p0 : float
        Initial p guess for bracketing.
    rtol : float
        Relative tolerance for fp.
    maxit : int
        Max iterations.
    verbose : bool
        If True print progress.

    Returns
    -------
    p : float
        Chosen smoothing parameter.
    c_Lb : ndarray
        Coefficients in C order flattening for internal use here.
    fp : float
        Achieved weighted residual sum.

    Notes
    -----
    The solve is done by solveh_banded on abATA + p * abR with a tiny ridge.
    """
    n = ATz.size
    u = max(u_data, u_p)

    def convert_c_Lb_to_Fortran_flat(c_Lb, nx, ny):
        """
        Convert a C order flat vector back to Fortran order flattening.

        Returns
        -------
        cF : ndarray
            Fortran order flattening used by evaluation helpers.
        """
        C = c_Lb.reshape(nx, ny, order="C")
        cF = np.ravel(C, order="F")
        return cF

    def _pad_or_copy(ab, cur_u):
        """
        Pad a band matrix to a larger band width by adding empty rows on top.
        """
        if cur_u == u:
            return ab.copy()
        pad = _alloc_symm_band(n, u)
        pad[u - cur_u:, :] = ab
        return pad

    abATAu = _pad_or_copy(abATA, u_data)
    abRu = _pad_or_copy(abR, u_p)

    def fp_at(p):
        """
        Compute fp and coefficients at a given p for the band system.
        """
        ab = abATAu + p * abRu
        ab[u, :] += 1e-12
        c_Lb = _solve_banded(ab, u, ATz)
        cF = convert_c_Lb_to_Fortran_flat(c_Lb, nx, ny)
        zhat = evaluate_scatter_from_coeffs(cF, nx, ny, basis_x, basis_y)
        fp = float(np.sum(w * (z - zhat) ** 2))
        return fp, c_Lb

    if s_target is None or s_target <= 0.0:
        fp0, c0_Lb = fp_at(0.0)
        return 0.0, c0_Lb, fp0

    p1 = 0.0
    f1, c1 = fp_at(p1)
    f1 -= s_target
    if verbose:
        print(f"[banded psearch] p=0 fp-s={f1:+.3e}")
    if f1 <= rtol * s_target:
        return 0.0, c1, f1 + s_target

    p3 = max(p0, 1.0)
    f3, c3 = fp_at(p3)
    f3 -= s_target
    grow = 10.0
    ng = 0
    while f3 > 0.0 and ng < 25:
        p3 *= grow
        f3, c3 = fp_at(p3)
        f3 -= s_target
        ng += 1
        if verbose:
            print(f"[banded psearch] grow p={p3:.3e} fp-s={f3:+.3e}")

    if f3 > 0.0:
        if abs(f1) <= abs(f3):
            return (p1, c1, f1 + s_target)
        else:
            return (p3, c3, f3 + s_target)

    def _geom_mid(pl, ph):
        return float(np.sqrt(max(pl, 1e-300) * max(ph, 1e-300)))

    def _fprati_next(p1, f1, p2, f2, p3, f3):
        M = np.array([
            [1.0, p1, -f1 * p1],
            [1.0, p2, -f2 * p2],
            [1.0, p3, -f3 * p3],
        ])
        rhs = np.array([f1, f2, f3])
        try:
            a, b, c = np.linalg.solve(M, rhs)
            if b == 0.0:
                return None
            p = -a / b
            lo, hi = (p1, p3) if p1 < p3 else (p3, p1)
            if not np.isfinite(p) or p <= 0:
                return None
            return float(np.clip(p, lo * (1 + 1e-12), hi * (1 - 1e-12)))
        except np.linalg.LinAlgError:
            return None

    p2 = _geom_mid(p1, p3)
    f2, c2 = fp_at(p2)
    f2 -= s_target

    for _ in range(maxit):
        if abs(f2) <= rtol * s_target:
            return p2, c2, f2 + s_target
        if not (f1 > f2 > f3):
            p2 = _geom_mid(p1, p3)
            f2, c2 = fp_at(p2)
            f2 -= s_target
        p_new = _fprati_next(p1, f1, p2, f2, p3, f3)
        if p_new is None:
            p_new = _geom_mid(p1, p3)
        f_new, c_new = fp_at(p_new)
        f_new -= s_target
        if f_new > 0.0:
            p1, f1, c1 = p_new, f_new, c_new
        else:
            p3, f3, c3 = p_new, f_new, c_new

        if abs(f1) < abs(f3):
            p2, f2, c2 = p1, f1, c1
        else:
            p2, f2, c2 = p3, f3, c3

    return (p1, c1, f1 + s_target) if abs(f1) <= abs(f3) else (p3, c3, f3 + s_target)


def _maybe_swap_axes_for_bandwidth(x, y, tx, ty, kx, ky):
    """
    Optionally swap x and y to reduce the band width.

    Parameters
    ----------
    x, y : arrays
        Data coordinates.
    tx, ty : arrays
        Knot vectors.
    kx, ky : int
        Degrees.

    Returns
    -------
    x2, y2, tx2, ty2, kx2, ky2, swapped : tuple
        Possibly swapped arrays and a boolean flag.

    Notes
    -----
    For a tensor product the band width depends on which dimension is first.
    Choosing the smaller of the two possible widths can save memory.
    """
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    band_x_first = kx * (ny - (ky + 1)) + ky
    band_y_first = ky * (nx - (kx + 1)) + kx
    if band_y_first < band_x_first:
        return (y, x, ty.copy(), tx.copy(), ky, kx, True)
    return (x, y, tx, ty, kx, ky, False)


def _surfit_python_banded(
    x, y, z, w, kx, ky,
    s, max_ncoef_x, max_ncoef_y,
    max_knots_iter, p_init, rtol,
    bbox, verbose, eps
):
    """
    Fit a smooth tensor product B spline using banded linear algebra.

    Parameters
    ----------
    x, y, z, w : arrays
        Input points, values, and weights.
    kx, ky : int
        Degrees in x and y.
    s : float
        Target weighted residual sum. If zero the fit tries to interpolate
        subject to numerical limits.
    max_ncoef_x, max_ncoef_y : int or None
        Upper bounds on the number of coefficients in x and y. If None it is
        chosen based on unique data and a small cap.
    max_knots_iter : int
        Maximum number of adaptive knot insertion iterations.
    p_init : float
        Initial p for bracketing.
    rtol : float
        Relative tolerance on fp versus s.
    bbox : tuple
        Domain bounds (xb, xe, yb, ye).
    verbose : bool
        If True print progress.
    eps : float
        Reserved for future small tolerances.

    Returns
    -------
    spl : SmoothBivariateSplinePython
        Resulting spline with knots, coefficients, degree, p, and fp.

    Notes
    -----
    Each iteration builds the data normal equations and the penalty in band
    form, searches p, and checks the target. If not met it inserts one knot
    where the energy is highest, with a guard against sliver intervals.
    """
    xb, xe, yb, ye = bbox

    ncoef_x = kx + 1
    ncoef_y = ky + 1
    tx = open_clamped_knots(xb, xe, kx, ncoef_x)
    ty = open_clamped_knots(yb, ye, ky, ncoef_y)

    x, y, tx, ty, kx, ky, _ = _maybe_swap_axes_for_bandwidth(x, y, tx, ty, kx, ky)

    if max_ncoef_x is None:
        max_ncoef_x = int(min(max(ncoef_x, np.unique(x).size), max(16, x.size)))
    if max_ncoef_y is None:
        max_ncoef_y = int(min(max(ncoef_y, np.unique(y).size), max(16, y.size)))

    rank1 = np.linalg.matrix_rank(np.c_[x - x.mean(), y - y.mean()]) < 2
    if rank1:
        max_ncoef_x = max(ncoef_x, kx + 1)
        max_ncoef_y = max(ncoef_y, ky + 1)

    p = p_init
    fp = np.nan
    c_Lb = None

    for it in range(max_knots_iter + 1):
        abATA, ATz, basis_x, basis_y, nx, ny, u_data = _assemble_data_normal_banded(
            x, y, z, w, tx, ty, kx, ky
        )
        abR, u_p = _assemble_penalty_banded(tx, ty, kx, ky, quad_n=6)

        p, c_Lb, fp = _p_search_banded(
            abATA, u_data, ATz,
            abR, u_p, nx, ny,
            basis_x, basis_y, z, w,
            s_target=s, p0=p_init,
            rtol=rtol, maxit=25, verbose=verbose
        )

        if verbose:
            print(f"[banded knots iter {it}] p={p:.3e} fp={fp:.6e} "
                  f"target={s:.6e} nx={nx} ny={ny}")

        Cmat = c_Lb.reshape(nx, ny, order="C")
        cF = np.ravel(Cmat, order="F")
        if fp <= s * (1.0 + rtol):
            C = np.asarray(cF).reshape(nx, ny, order="F")
            return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                               C=C, kx=kx, ky=ky, p=p, fp=fp)

        if (nx * ny) > x.size:
            C = np.asarray(cF).reshape(nx, ny, order="F")
            return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                               C=C, kx=kx, ky=ky, p=p, fp=fp)

        if (nx >= max_ncoef_x and ny >= max_ncoef_y) or it == max_knots_iter:
            C = np.asarray(cF).reshape(nx, ny, order="F")
            return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                               C=C, kx=kx, ky=ky, p=p, fp=fp)

        zhat = evaluate_scatter_from_coeffs(cF, nx, ny, basis_x, basis_y)
        res2 = w * (z - zhat) ** 2

        fpint_x, coord_x, fpint_y, coord_y = _interval_energy_centroids(
            x, y, res2, tx, ty, kx, ky
        )

        int_x_lo = kx
        int_x_hi = len(tx) - kx - 2
        int_y_lo = ky
        int_y_hi = len(ty) - ky - 2

        can_x = (nx < max_ncoef_x) and (int_x_lo <= int_x_hi)
        can_y = (ny < max_ncoef_y) and (int_y_lo <= int_y_hi)

        pick_axis = None
        pick_span = None
        pick_val = -np.inf

        if can_x:
            jx = int(np.argmax(fpint_x[int_x_lo:int_x_hi + 1]) + int_x_lo)
            if fpint_x[jx] > pick_val:
                pick_axis, pick_span, pick_val = "x", jx, fpint_x[jx]
        if can_y:
            jy = int(np.argmax(fpint_y[int_y_lo:int_y_hi + 1]) + int_y_lo)
            if fpint_y[jy] > pick_val:
                pick_axis, pick_span, pick_val = "y", jy, fpint_y[jy]

        if pick_axis is None or pick_val <= 0.0:
            C = np.asarray(cF).reshape(nx, ny, order="F")
            return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                               C=C, kx=kx, ky=ky, p=p, fp=fp)

        if pick_axis == "x":
            tx_new = _insert_centroid_with_sliver_guard(
                tx, kx, pick_span, num=fpint_x[pick_span],
                den=coord_x[pick_span], candidates=np.unique(x)
            )
            if tx_new is None:
                if (can_y and
                    fpint_y[
                        int(np.argmax(fpint_y[int_y_lo:int_y_hi + 1])
                            + int_y_lo)
                    ] > 0.0):
                    jy2 = int(np.argmax(fpint_y[int_y_lo:int_y_hi + 1]) + int_y_lo)
                    ty_new = _insert_centroid_with_sliver_guard(
                        ty, ky, jy2,
                        num=fpint_y[jy2], den=coord_y[jy2],
                        candidates=np.unique(y)
                    )
                    if ty_new is not None:
                        ty = ty_new
                        continue
                C = np.asarray(cF).reshape(nx, ny, order="F")
                return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                                   C=C, kx=kx, ky=ky, p=p, fp=fp)
            tx = tx_new
            continue
        else:
            ty_new = _insert_centroid_with_sliver_guard(
                ty, ky, pick_span, num=fpint_y[pick_span],
                den=coord_y[pick_span], candidates=np.unique(y)
            )
            if ty_new is None:
                if can_x and fpint_x[
                    int(
                        np.argmax(fpint_x[int_x_lo:int_x_hi + 1])
                        +
                        int_x_lo
                    )] > 0.0:
                    jx2 = int(np.argmax(fpint_x[int_x_lo:int_x_hi + 1]) + int_x_lo)
                    tx_new = _insert_centroid_with_sliver_guard(
                        tx, kx, jx2, num=fpint_x[jx2],
                        den=coord_x[jx2], candidates=np.unique(x)
                    )
                    if tx_new is not None:
                        tx = tx_new
                        continue
                C = np.asarray(cF).reshape(nx, ny, order="F")
                return SmoothBivariateSplinePython(tx=tx.copy(), ty=ty.copy(),
                                                   C=C, kx=kx, ky=ky, p=p, fp=fp)
            ty = ty_new
            continue


def surfit_python(
    x, y, z, w=None,
    kx=3, ky=3, s=None,
    max_ncoef_x=None,
    max_ncoef_y=None,
    max_knots_iter=None,
    p_init=1.0, rtol=1e-3,
    atol=0.0, bbox=[None]*4,
    verbose=False, eps=1e-16
) -> SmoothBivariateSplinePython:
    """
    Public entry point. Fit a smooth tensor product spline to scattered data.

    Parameters
    ----------
    x, y, z : arrays
        Input data coordinates and values. All must have the same length.
    w : array or None
        Nonnegative weights. If None all ones are used.
    kx, ky : int
        Degrees in x and y. Typical choice is 3 and 3.
    s : float or None
        Target weighted residual sum. If None a default of m - sqrt(2 m) is used.
    max_ncoef_x, max_ncoef_y : int or None
        Upper bounds on coefficients in each direction. If None reasonable
        defaults are chosen from unique data counts and sample size.
    max_knots_iter : int or None
        Max number of adaptive knot insertions. If None it is set to m.
    p_init : float
        Initial p for the bracket step.
    rtol : float
        Relative tolerance on fp versus s.
    atol : float
        Placeholder for absolute tolerance. Not used here.
    bbox : list of 4 floats or None
        [xb, xe, yb, ye]. If any entry is None it is taken from data min and max.
    verbose : bool
        If True print progress logs.
    eps : float
        Reserved for future small tolerances.

    Returns
    -------
    spline : SmoothBivariateSplinePython
        Fitted spline object ready for evaluation by SciPy tools that accept
        a tck like structure, or for custom evaluation using this module.

    Notes
    -----
    This function validates inputs, sets defaults, and calls the banded solver.
    """
    x, y, z, w = _validate_input(x, y, z, w, kx, ky, eps)
    bbox = np.ravel(bbox)
    if not bbox.shape == (4,):
        raise ValueError("bbox shape should be (4,)")
    if s is not None and not s >= 0.0:
        raise ValueError("s should be s >= 0.0")

    m = x.size
    if w is None:
        w = np.ones(m, float)
    else:
        w = np.asarray(w, float).ravel()
        if w.size != m:
            raise ValueError("x, y, z, and w should have a same length")

    xb, xe, yb, ye = bbox

    if any(bboxi is None for bboxi in bbox):
        xb, xe = float(np.min(x)), float(np.max(x))
        yb, ye = float(np.min(y)), float(np.max(y))
    else:
        xb, xe, yb, ye = map(float, bbox)

    if s is None:
        s = max(0.0, float(m) - float(np.sqrt(2.0 * m)))

    if max_knots_iter is None:
        max_knots_iter = int(m)

    return _surfit_python_banded(
        x=x, y=y, z=z, w=w,
        kx=kx, ky=ky, s=s,
        max_ncoef_x=max_ncoef_x,
        max_ncoef_y=max_ncoef_y,
        max_knots_iter=max_knots_iter,
        p_init=p_init, rtol=rtol,
        bbox=(xb, xe, yb, ye),
        verbose=verbose, eps=eps
    )
