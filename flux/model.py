import numpy as np

from scipy.constants import sigma # Stefan-Boltzmann constant

from flux.solve import solve_radiosity

def compute_steady_state_temp(FF, E, rho, emiss,
                              clamp=True, tol=np.finfo(np.float64).eps):
    if E.ndim == 1:
        B = solve_radiosity(FF, E, rho)[0]
        if clamp:
            B = np.maximum(0, B)
        IR = FF@((1 - rho)*B)
        Q = solve_radiosity(FF, IR)[0]
        if clamp:
            Q = np.maximum(0, Q)
        return (((1 - rho)*B + emiss*Q)/(emiss*sigma))**0.25
    elif E.ndim == 2:
        # solve "visible" section for all time-steps
        B_arr = solve_radiosity(FF, E, rho)[0]
        if len(B_arr.shape)==1:
            B_arr = B_arr[:,np.newaxis]
        # solve IR section for each time step sequentially
        T = []
        for B in B_arr.T:
            if clamp:
                B = np.maximum(0, B)
            IR = FF@((1 - rho)*B)
            Q = solve_radiosity(FF, IR)[0]
            if clamp:
                Q = np.maximum(0, Q)
            T.append((((1 - rho)*B + emiss*Q)/(emiss*sigma))**0.25)
        return np.array(T)
    else:
        raise Exception('E should be a vector or a 2D array')
