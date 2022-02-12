import numpy as np


def solve_radiosity(FF, E, rho=1, albedo_placement='right',
                    method='jacobi', tol=None):
    if tol is None:
        tol = np.finfo(E.dtype).resolution
        tol *= abs(E).max() # use a relative tolerance
    if albedo_placement not in {'left', 'right'}:
        raise Exception('albedo_placement must be "left" or "right"')
    if method == 'jacobi':
        if albedo_placement == 'left':
            return _solve_radiosity_jacobi_left(FF, E, rho, tol)
        else:
            return _solve_radiosity_jacobi_right(FF, E, rho, tol)
    elif method == 'cg':
        if albedo_placement == 'left':
            return _solve_radiosity_cg_left(FF, E, rho, tol)
        else:
            return _solve_radiosity_cg_right(FF, E, rho, tol)
    else:
        raise Exception('method must be one of: jacobi, cg')


def _solve_radiosity_jacobi_left(FF, E, rho, tol):
    B = E.copy()
    niter = 0
    while True:
        niter += 1
        B1 = E + rho*(FF@B)
        if abs(B1 - B).max() <= tol:
            break
        B = B1
    return B, niter

def _solve_radiosity_jacobi_right(FF, E, rho, tol):
    B = E.copy()
    niter = 0
    while True:
        niter += 1
        B1 = E + FF@(rho*B)
        if abs(B1 - B).max() <= tol:
            break
        B = B1
    return B, niter

def _solve_radiosity_cg_left(FF, E, rho, tol):
    # r = b - A@x
    # niter = 0
    # if |r| <= tol:
    #     return x, niter
    # p = r
    # while True:
    #     Ap = A@p
    #     rr = r@r
    #     a = rr/(Ap@p)
    #     x1 = x + a@p
    #     r1 = r - a@Ap
    #     if |r1| <= tol:
    #         return x1, niter + 1
    #     b = r1@r1/rr
    #     p1 = r1 + b*p
    #     niter += 1
    # assert False

    X, R, niter = np.zeros_like(E), E, 0

    if abs(R).max() <= tol:
        return X, niter

    P = R
    while True:
        #     Ap = A@p
        KK_P = P - rho*(FF@P)

        #     rr = r@r
        R_R = R@R

        #     a = rr/(Ap@p)
        alpha = R_R/(KK_P@P)

        #     x1 = x + a@p
        X1 = X + alpha*P

        #     r1 = r - a@Ap
        R1 = R - alpha*KK_P

        #     if |r1| <= tol:
        #         return x1, niter + 1
        if abs(R1).max() <= tol:
            return X1, niter + 1

        #     b = r1@r1/rr
        beta = R1@R1/R_R

        #     p1 = r1 + b*p
        P1 = R1 + beta*P

        #     niter += 1
        niter += 1

        X = X1
        R = R1
        P = P1

    assert False

def _solve_radiosity_cg_right(FF, E, rho, tol):
    assert False
