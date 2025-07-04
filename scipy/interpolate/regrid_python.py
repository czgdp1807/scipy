import numpy as np
from scipy.interpolate import NdBSpline
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import lsqr


def initial_knots(data, degree, is_interpolation):
    m = len(data)
    k1 = degree + 1
    if is_interpolation:
        interior = data[1:-1] if m > 2 else []
        return np.concatenate([
            np.full(k1, data[0]),
            interior,
            np.full(k1, data[-1])
        ])
    else:
        return np.concatenate([
            np.full(k1, data[0]),
            np.full(k1, data[-1])
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


def regrid_python(x, y, z, kx=3, ky=3, s=0.0, tol=1e-3, maxit=25):
    mx, my = len(x), len(y)
    z = z.reshape((mx, my))
    rhs = z.T.flatten()

    is_interp = s == 0.0
    acc = tol * max(s, 1e-12)

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

    mpm = mx + my
    for it in range(mpm):
        Bx = NdBSpline.design_matrix(x[:, None], (tx,), kx)
        By = NdBSpline.design_matrix(y[:, None], (ty,), ky)
        A = kron(By, Bx)
        c = lsqr(csr_matrix(A), rhs)[0]
        fp = compute_fp(A, c, rhs)

        if it == 0:
            fp0 = fp

        if is_interp or abs(fp - s) < acc:
            return NdBSpline(
                (tx, ty),
                c.reshape(nx - kx - 1, ny - ky - 1),
                k=(kx, ky))

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
        A = kron(NdBSpline.design_matrix(y[:, None], (ty,), ky),
                 NdBSpline.design_matrix(x[:, None], (tx,), kx))
        c = lsqr(csr_matrix(A / (1 + p)), rhs / (1 + p))[0]
        fp = compute_fp(A, c, rhs)
        fpms = fp - s
        if abs(fpms) < acc:
            return NdBSpline(
                (tx, ty),
                c.reshape(nx - kx - 1, ny - ky - 1),
                k=(kx, ky)
            )

        if fpms > 0:
            p1, f1 = p, fpms
        else:
            p3, f3 = p, fpms

        p = rational_root_update(p1, f1, p, fpms, p3, f3)

    return NdBSpline(
        (tx, ty),
        c.reshape(nx - kx - 1, ny - ky - 1),
        k=(kx, ky)
    )
