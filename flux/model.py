import numpy as np

from scipy.constants import sigma # Stefan-Boltzmann constant

from flux.solve import solve_radiosity

def compute_steady_state_temp(FF, E, rho, emiss, Fsurf=0.,
                              clamp=True, tol=np.finfo(np.float64).eps):
    if E.ndim == 1:
        B = solve_radiosity(FF, E, rho)[0]
        if clamp:
            B = np.maximum(0, B)
        IR = FF@((1 - rho)*B + Fsurf)
        Q = solve_radiosity(FF, IR)[0]
        if clamp:
            Q = np.maximum(0, Q)
        tot = (1 - rho)*B + emiss*Q + Fsurf
        if clamp:
            tot = np.maximum(0, tot)
        return (tot/(emiss*sigma))**0.25
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

def update_incoming_radiances(FF, E, rho, emiss, Qrefl, QIR, Tsurf, clamp=True):
    """
    Compute the fluxes/irradiances at time step
    Args:
        FF: form factor matrix (constant,
        E: direct flux at time t
        rho: albedo
        emiss: emissivity
        Qrefl: reflected flux at previous time step
        QIR: IR flux at previous time step
        Tsurf: surface temperature at previous time step
        clamp: boolean

    Returns:
        Qrefl_np1, QIR_np1
    """
    # eq 17 radiosity paper
    Qrefl_np1 = rho * FF @ (E + Qrefl)
    if clamp:
        Qrefl_np1 = np.maximum(0, Qrefl_np1)
    # eq 18 radiosity paper
    QIR_np1 = FF @ (emiss*sigma*Tsurf**4 + (1-emiss)*QIR)
    if clamp:
        QIR_np1 = np.maximum(0, QIR_np1)

    return Qrefl_np1, QIR_np1