import numpy as np


def solve_kernel_system(L, b, rho=1, tol=np.finfo(np.float64).eps):
    x = b
    dx = L@(rho*b)
    nmul = 1
    err = np.linalg.norm(b - x + dx)
    while np.linalg.norm(err) > tol:
        x = b + dx
        dx = L@(rho*b)
        nmul += 1
        prev_err = err
        err = np.linalg.norm(b - x + dx)
        if abs(err - prev_err) < tol:
            break
    return x, nmul
