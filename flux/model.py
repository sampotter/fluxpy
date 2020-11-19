import numpy as np


from scipy.constants import sigma # Stefan-Boltzmann constant


from flux.solve import solve_radiosity


def compute_steady_state_temp(FF, E, rho, emiss,
                              clamp=True, tol=np.finfo(np.float64).eps):
    B = solve_radiosity(FF, E, rho)[0]
    if clamp:
        B = np.maximum(0, B)
    Q = solve_radiosity(FF, FF@((1 - rho)*B))[0]
    if clamp:
        Q = np.maximum(0, Q)
    T = (((1 - rho)*B + emiss*Q)/(emiss*sigma))**0.25
    return T
