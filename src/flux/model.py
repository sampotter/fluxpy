import numpy as np

from scipy.constants import Stefan_Boltzmann as sigSB

from flux.solve import solve_radiosity
from flux.thermal import PccThermalModel1D

def compute_steady_state_temp(FF, E, rho, emiss, Fsurf=0.,
                              clamp=True, tol=np.finfo(np.float64).eps,
                              method='jacobi'):
    if E.ndim == 1:
        # solve "visible"
        B = solve_radiosity(FF, E, rho, 'right', method)[0]
        if clamp:
            B = np.maximum(0, B)
        IR = FF@((1 - rho)*B + Fsurf)
        # solve IR
        Q = solve_radiosity(FF, IR, 1, 'right', method)[0]
        if clamp:
            Q = np.maximum(0, Q)
        tot = (1 - rho)*B + emiss*Q + Fsurf
        if clamp:
            tot = np.maximum(0, tot)
        return (tot/(emiss*sigSB))**0.25
    elif E.ndim == 2:
        # solve "visible" section for all time-steps
        B_arr = solve_radiosity(FF, E, rho, 'right', method)[0]
        if len(B_arr.shape)==1:
            B_arr = B_arr[:,np.newaxis]
        # solve IR section for each time step sequentially
        T = []
        for B in B_arr.T:
            if clamp:
                B = np.maximum(0, B)
            IR = FF@((1 - rho)*B)
            Q = solve_radiosity(FF, IR, 1, 'right', method)[0]
            if clamp:
                Q = np.maximum(0, Q)
            T.append((((1 - rho)*B + emiss*Q)/(emiss*sigSB))**0.25)
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
    QIR_np1 = FF @ (emiss*sigSB*Tsurf**4 + (1-emiss)*QIR)
    if clamp:
        QIR_np1 = np.maximum(0, QIR_np1)

    return Qrefl_np1, QIR_np1


def update_incoming_radiances_wsvd(E, albedo, emiss, Qrefl, QIR, Tsurf, Vt, w, U):
    """
    Compute the scattered irradiances based on SVD matrices
    Args:
        E: direct solar irradiance at current time step
        albedo: albedo at current time step
        emiss: emissivity
        Qrefl: reflected flux at previous time step
        QIR: IR flux at previous time step
        Tsurf: surface temperature at previous time step
        Vt, w, U: SVD matrices
    """
    # eq 17 radiosity paper, short-wavelength
    #Qrefl_np1 = FF @ ( albedo * (E + Qrefl))  # (NxN) * (1xN) = (1xN)
    if w.size>1:
        tmp1 = Vt @ ( albedo * (E + Qrefl) )  # (NxT) * (1xN) = (1xT)
        tmp2 = np.multiply( w, tmp1)          # (T) * (1xT) = (1xT)
        Qrefl_np1 = U @ tmp2                  # (TxN) * (1xT) = (1xN)
    else: # only one mode
        tmp1 = np.dot(Vt, albedo * (E + Qrefl) )
        tmp2 = np.multiply( w, tmp1)
        Qrefl_np1 = np.multiply(U, tmp2)

    # eq 18 radiosity paper, long-wavelength
    #QIR_np1 = FF @ (emiss*sigSB*Tsurf**4 + (1-emiss)*QIR)
    if w.size>1:
        tmp1 = Vt @ ( emiss*sigSB*Tsurf**4 + (1-emiss)*QIR )
        tmp2 = np.multiply( w, tmp1)
        QIR_np1 = U @ tmp2
    else: # only one mode
        tmp1 = np.dot( Vt, emiss*sigSB*Tsurf**4 + (1-emiss)*QIR )
        tmp2 = np.multiply( w, tmp1 )
        QIR_np1 = np.multiply( U, tmp2 )

    return Qrefl_np1, QIR_np1

class ThermalModel:

    def __init__(self, FF, t, D, F0, rho, method='1mvp', return_flux=False, **kwargs):
        if D.ndim != 2 or D.shape[1] != 3:
            raise ValueError('sun positions ("D") should be an N x 3 ndarray')
        if np.mod(D.shape[0],t.shape[0]) != 0:
            raise ValueError('number of sun positions ("D") should be equal or multiple of number of time points ("t")')

        self._FF = FF
        self._shape_model = kwargs['shape_model'] \
            if 'shape_model' in kwargs else FF.shape_model
        self._t = t
        self._D = D
        self._F0 = F0
        self._rho = rho
        self._source_num_faces = int(self._D.shape[0]/self._t.shape[0])
        self._return_flux = return_flux

        num_faces = self._shape_model.num_faces

        if 'z' not in kwargs:
            raise ValueError('failed to pass skin depths (keyword argument "z")')
        z = kwargs['z']

        def promote_to_array_if_necessary(value, shape, dtype=None):
            if isinstance(value, (int, float)):
                return np.full(shape, value, dtype)
            elif isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError('invalid shape')
                return value
            else:
                raise TypeError(f'bad type: {type(value)}')

        assert not isinstance(z, int)

        if 'T0' not in kwargs:
            raise ValueError('failed to pass initial temperature (keyword argument "T0")')
        T0 = kwargs['T0']
        T0 = promote_to_array_if_necessary(T0, (num_faces, z.size), np.float64)

        if 'ti' not in kwargs:
            raise ValueError('failed to pass thermal inertia (keyword argument "ti")')
        ti = kwargs['ti']
        ti = promote_to_array_if_necessary(ti, (z.size,), np.float64)

        if 'rhoc' not in kwargs:
            raise ValueError('failed to pass thermal conductivity (keyword argument "rhoc")')
        rhoc = kwargs['rhoc']
        rhoc = promote_to_array_if_necessary(rhoc, (z.size,), np.float64)

        if 'emiss' not in kwargs:
            raise ValueError('failed to pass emissivity (keyword argument "emiss")')
        emiss = kwargs['emiss']
        emiss = promote_to_array_if_necessary(emiss, (num_faces,), np.float64)

        if 'Fgeotherm' not in kwargs:
            raise ValueError('failed to pass geothermal flux (keyword argument "Fgeotherm")')
        Fgeotherm = kwargs['Fgeotherm']
        Fgeotherm = promote_to_array_if_necessary(Fgeotherm, (num_faces,), np.float64)

        # provisionalT surf and Q from direct illumination only
        # print(D.shape)
        if self._source_num_faces == 1:
            E0 = self._shape_model.get_direct_irradiance(F0[0], D[0])
        else:
            E0 = self._shape_model.get_direct_irradiance(F0[:self._source_num_faces], D[:self._source_num_faces])

        Q0 = (1 - rho)*E0 + Fgeotherm
        self._Tsurf = (Q0/(sigSB*emiss))**0.25

        # set up bundle of 1D thermal models
        self._pcc_thermal_model_1d = PccThermalModel1D(
            z, T0, ti, rhoc, emiss, Fgeotherm, Q0, self._Tsurf, bcond='Q')

        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # advance iteration count and set current iteration index
        self._iter_count += 1
        i = self._iter_count

        # stop the iteration if we're out of sun directions
        if (self._source_num_faces == 1) & (i == self._D.shape[0]):
            raise StopIteration()
        elif (self._source_num_faces > 1) & (self._source_num_faces*i == self._D.shape[0]):
            raise StopIteration()

        # get the current sun direction and time
        if self._source_num_faces == 1:
            f, d, delta_t = self._F0[i], self._D[i], self._t[i] - self._t[i - 1]
        else:
            f, d, delta_t = self._F0[i*self._source_num_faces:(i+1)*self._source_num_faces,:], \
                            self._D[i*self._source_num_faces:(i+1)*self._source_num_faces,:], \
                            self._t[i] - self._t[i - 1]

        # compute the current insolation
        E = self._shape_model.get_direct_irradiance(f, d)

        # get the previous fluxes and surface temperature
        Qrefl_prev = 0 if i == 1 else self._Qrefl
        QIR_prev = 0 if i == 1 else self._QIR
        Tsurf_prev = self._Tsurf

        emiss = self._pcc_thermal_model_1d.emiss

        # get the next reflected and infrared fluxes
        Qrefl, QIR = update_incoming_radiances(
            self._FF, E, self._rho, emiss, Qrefl=Qrefl_prev,
            QIR=QIR_prev, Tsurf=Tsurf_prev)

        # step the thermal model
        Q = (1 - self._rho)*(E + Qrefl) + emiss*QIR
        self._pcc_thermal_model_1d.step(delta_t, Q)

        # prepare for the next iteration
        self._Qrefl = Qrefl
        self._QIR = QIR
        self._Tsurf = self._pcc_thermal_model_1d.T[:, 0]

        if self._return_flux:
            return self._pcc_thermal_model_1d.T, E, self._Qrefl, self._QIR
        else:
            return self._pcc_thermal_model_1d.T
