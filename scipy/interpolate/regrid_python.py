import numpy as np
from scipy.interpolate import RectBivariateSpline, BSpline
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.interpolate._fitpack_repro import disc

class RectBivariateSplinePython(RectBivariateSpline):

    def __init__(self, fp, tck, degrees):
        nx, ny = len(tck[0]), len(tck[1])
        kx, ky = degrees
        c = tck[2].reshape(nx - kx - 1, ny - ky - 1).T.flatten()
        self.fp = fp
        self.tck = (tck[0], tck[1], c)
        self.degrees = degrees

def initial_knots(data, degree, is_interpolation):
    m = len(data)
    k = degree
    k3 = k//2 + 2
    mk1 = m - k - 1
    if is_interpolation:
        interior = data[k3 - 1:k3 + mk1 - 1] if m > 2 else []
        return np.concatenate([
            np.full(k + 1, data[0]),
            interior,
            np.full(k + 1, data[-1])
        ])
    else:
        return np.concatenate([
            np.full(k + 1, data[0]),
            np.full(k + 1, data[-1])
        ])

def compute_fp(A, c, rhs):
    return np.sum((A @ c - rhs) ** 2)


def compute_fpint_and_nrdata(residuals, grid, t, k):
    nint = len(t) - 2 * (k + 1)
    fpint = np.zeros(nint)
    nrdata = np.zeros(nint, dtype=int)
    interval_edges = t[k + 1:-k - 1]
    idx = np.searchsorted(interval_edges, grid, side='right') - 1
    valid = (idx >= 0) & (idx < nint)
    for i in range(len(grid)):
        if valid[i]:
            j = idx[i]
            fpint[j] += residuals[i] ** 2
            nrdata[j] += 1
    return fpint, nrdata


def fpknot_like_insert(data, t, k, fpint, nrdata, nrint, istart, nest):
    m = len(data)
    n = len(t)
    khalf = (n - nrint - 1) // 2
    zero = 0.0

    number = 0
    maxpt = 0
    maxbeg = 0
    fpmax = zero
    jbegin = istart

    for j in range(nrint):
        jpoint = nrdata[j]
        if fpmax < fpint[j] and jpoint != 0:
            fpmax = fpint[j]
            number = j
            maxpt = jpoint
            maxbeg = jbegin
        jbegin += jpoint + 1

    if maxpt == 0 or number + khalf + 1 >= nest:
        return t, fpint, nrdata, n, nrint

    ihalf = maxpt // 2 + 1
    nrx = min(maxbeg + ihalf, m - 1)
    new_knot = data[nrx]
    next = number + 1

    if next <= nrint:
        for j in reversed(range(next, nrint + 1)):
            jj = next + nrint - j
            if jj < len(fpint) - 1:
                fpint[jj + 1] = fpint[jj]
                nrdata[jj + 1] = nrdata[jj]
                jk = jj + khalf
                if jk + 1 < nest:
                    t[jk + 1] = t[jk]

    nrdata[number] = ihalf - 1
    nrdata[next] = maxpt - ihalf
    am = maxpt
    an1 = nrdata[number]
    an2 = nrdata[next]
    if am > 0:
        fpint[number] = fpmax * an1 / am
        fpint[next] = fpmax * an2 / am

    jk = next + khalf
    if jk < len(t):
        t = np.insert(t, jk, new_knot)
        n += 1
        nrint += 1

    return t, fpint, nrdata, n, nrint


def rational_root_update(p1, f1, p2, f2, p3, f3):
    denom = (f2 - f3) * (p2 - p1) - (f2 - f1) * (p2 - p3)
    if abs(denom) < 1e-12:
        return max(0.5 * (p1 + p3), 1e-6)
    num = (f2 - f1) * (p2 - p3) * p2 - (f2 - f3) * (p2 - p1) * p2
    return max(num / denom, 1e-6)

def compute_b_spline_penalty(t, k):
    return disc(t, k)

def construct_augmented_system(x, y, z, tx, ty, kx, ky, p):
    # Observation matrices
    spx = BSpline.design_matrix(x, tx, kx).toarray()
    spy = BSpline.design_matrix(y, ty, ky).toarray()

    # Penalty matrices
    bx = compute_b_spline_penalty(tx, kx)[0]
    by = compute_b_spline_penalty(ty, ky)[0]

    # Augment
    if p > 0:
        Ax = np.vstack([spx, (1 / p) * bx]) if bx.shape[0] > 0 else spx
        Ay = np.vstack([spy, (1 / p) * by]) if by.shape[0] > 0 else spy
    else:
        Ax, Ay = spx, spy

    A = kron(Ay, Ax)
    rhs = z.T.flatten()
    q = np.concatenate([rhs, np.zeros(A.shape[0] - len(rhs))])
    return A, q

def regrid_python(x, y, z, kx=3, ky=3, s=0.0, tol=1e-3, maxit=25):
    mx, my = len(x), len(y)
    z = z.reshape((mx, my))
    rhs = z.T.flatten()

    is_interp = s == 0.0
    acc = tol * s

    tx = initial_knots(x, kx, is_interp)
    ty = initial_knots(y, ky, is_interp)
    max_tx = mx + kx + 1
    max_ty = my + ky + 1
    nestx = max_tx + 10
    nesty = max_ty + 10

    nx = len(tx)
    ny = len(ty)
    nminx = 2 * (kx + 1)
    nminy = 2 * (ky + 1)
    nxe = min(max_tx, nestx)
    nye = min(max_ty, nesty)
    nrintx = nx - nminx + 1
    nrinty = ny - nminy + 1

    fpold = 0.0
    fp0 = 0.0
    reducx = 0.0
    reducy = 0.0
    lastdi = 0
    nplusx = 0
    nplusy = 0
    p = -1

    mpm = mx + my
    for it in range(mpm):
        A, q = construct_augmented_system(x, y, z, tx, ty, kx, ky, p)
        c = lsqr(csr_matrix(A), q)[0]
        fp = compute_fp(A, c, q)

        if it == 0:
            fp0 = fp

        if is_interp or abs(fp - s) < acc:
            return RectBivariateSplinePython(
                fp, (tx, ty, c), (kx, ky))

        if nx == max_tx and ny == max_ty:
            fp = 0
            break

        if nx == nxe and ny == nye:
            break

        if lastdi == 1:
            reducx = fpold - fp
        elif lastdi == 2:
            reducy = fpold - fp

        fpold = fp
        fpms = fp - s

        nplx = 1
        if nx != nminx:
            npl1 = nplusx * 2
            if reducx > acc:
                npl1 = int(nplusx * fpms / reducx)
            nplx = max(min(nplusx * 2, max(npl1, nplusx // 2, 1)), 1)

        nply = 1
        if ny != nminy:
            npl1 = nplusy * 2
            if reducy > acc:
                npl1 = int(nplusy * fpms / reducy)
            nply = max(min(nplusy * 2, max(npl1, nplusy // 2, 1)), 1)

        if (lastdi != 2 and nx < nxe) or ny >= nye:
            lastdi = 1
            nplusx = nplx
            for _ in range(nplusx):
                z_resid = (A @ c - rhs).reshape((mx, my))
                fpintx, nrdatax = compute_fpint_and_nrdata(
                    z_resid.mean(axis=1), x, tx, kx
                )
                fpintx = np.pad(fpintx, (0, max(0, nestx - len(fpintx))))
                nrdatax = np.pad(nrdatax, (0, max(0, nestx - len(nrdatax))))
                tx, fpintx, nrdatax, nx, nrintx = fpknot_like_insert(
                    x, tx, kx, fpintx, nrdatax, nrintx, 0, nestx
                )
                if nx == nxe:
                    break
        else:
            lastdi = 2
            nplusy = nply
            for _ in range(nplusy):
                z_resid = (A @ c - rhs).reshape((mx, my))
                fpinty, nrdatay = compute_fpint_and_nrdata(
                    z_resid.mean(axis=0), y, ty, ky
                )
                fpinty = np.pad(fpinty, (0, max(0, nesty - len(fpinty))))
                nrdatay = np.pad(nrdatay, (0, max(0, nesty - len(nrdatay))))
                ty, fpinty, nrdatay, ny, nrinty = fpknot_like_insert(
                    y, ty, ky, fpinty, nrdatay, nrinty, 0, nesty
                )
                if ny == nye:
                    break

    # Second loop: root-finding for smoothing parameter p
    p1, f1 = 0.0, fp0 - s
    p3, f3 = 1.0, fp - s
    p = 0.5 * (p1 + p3)

    for _ in range(maxit):
        A, q = construct_augmented_system(x, y, z, tx, ty, kx, ky, p)
        c = lsqr(csr_matrix(A), q)[0]
        fp = compute_fp(A, c, rhs)
        fpms = fp - s
        if abs(fpms) < acc:
            return RectBivariateSplinePython(
                fp, (tx, ty, c), (kx, ky))

        if fpms > 0:
            p1, f1 = p, fpms
        else:
            p3, f3 = p, fpms

        p = rational_root_update(p1, f1, p, fpms, p3, f3)

    return RectBivariateSplinePython(
        fp, (tx, ty, c), (kx, ky))
