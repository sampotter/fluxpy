import numpy as np


def solve_radiosity(FF, E, rho=1, albedo_placement='right',
                    method='jacobi', tol=None):
    if tol is None:
        tol = np.finfo(E.dtype).resolution
    if albedo_placement not in {'left', 'right'}:
        raise Exception('albedo_placement must be "left" or "right"')
    if method != 'jacobi':
        raise Exception('method must be "jacobi"')
    B = E.copy()
    niter = 0
    while True:
        niter += 1
        if albedo_placement == 'left':
            B1 = E + rho*(FF@B)
        else:
            B1 = E + FF@(rho*B)
        dB = B1 - B
        B = B1
        if abs(dB).max()/abs(E).max() <= tol:
            break
    return B, niter
