import numpy as np


from scipy.constants import sigma # Stefan-Boltzmann constant


from flux.solve import solve_kernel_system


def get_T(L, Q, rho, emiss, tol=np.finfo(np.float64).eps):
    Q, nmul1 = solve_kernel_system(L, Q, rho, tol)
    Q = np.maximum(0, Q)
    Q *= 1 - rho
    tmp, nmul2 = solve_kernel_system(L, Q, 1, tol)
    tmp = np.maximum(0, tmp)
    Q = (1 - emiss)*Q + emiss*tmp
    Q /= emiss*sigma
    Q = Q**(1/4)
    return Q, nmul1 + nmul2
