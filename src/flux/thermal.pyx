import logging
import numpy as np

cdef extern from "pcc/thrmlLib.h":
    void conductionQ(int nz, double z[], double dt, double Qn, double
		     Qnp1, double T[], double ti[], double rhoc[],
		     double emiss, double Fgeotherm, double *Fsurf)

    void conductionT(int nz, double z[], double dt, double T[], double Tsurf,
		    double Tsurfp1, double ti[], double rhoc[], double Fgeotherm,
		    double *Fsurf)

cdef class PccThermalModel1D:
    cdef:
        int _num_layers
        int _num_faces
        int _num_layers_interior

        double _current_time

        double[::1] _emiss
        double[::1] _z
        double[::1] _ti
        double[::1] _rhoc
        double[::1] _Fgeotherm
        double[::1] _Qprev
        double[::1] _Fsurf
        double[:, ::1] _T
        double[::1] _Tsurfprev

        str _bcond

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_faces(self):
        return self._num_faces

    @property
    def num_layers_interior(self):
        return self._num_layers_interior

    @property
    def current_time(self):
        return self._current_time

    @property
    def emiss(self):
        return np.asarray(self._emiss)

    @property
    def z(self):
        return np.asarray(self._z)

    @property
    def ti(self):
        return np.asarray(self._ti)

    @property
    def rhoc(self):
        return np.asarray(self._rhoc)

    @property
    def Fgeotherm(self):
        return np.asarray(self._Fgeotherm)

    @property
    def Qprev(self):
        return np.asarray(self._Qprev)

    @property
    def Fsurf(self):
        return np.asarray(self._Fsurf)

    @property
    def T(self):
        return np.asarray(self._T)

    @property
    def Tsurfprev(self):
        return np.asarray(self._Tsurfprev)

    @property
    def bcond(self):
        return self._bcond

    def __cinit__(self,
                  double[::1] z,
                  double[:, ::1] T0,
                  double[::1] ti,
                  double[::1] rhoc,
                  double[::1] emiss,
                  double[::1] Fgeotherm,
                  double[::1] Q0,
                  double[::1] Tsurfprev,
                  bcond='Q'):
        if bcond not in {'Q', 'T'}:
            raise ValueError('bcond should be "Q" or "T"')
        self._bcond = bcond

        self._num_faces = T0.shape[0]
        self._num_layers = z.size
        self._num_layers_interior = self._num_layers - 1
        self._current_time = 0

        if not all(0 <= _ <= 1 for _ in emiss):
            raise ValueError('emissivity values ("emiss") should be in range [0, 1]')
        if emiss.ndim != 1 or emiss.size != self.num_faces:
            raise ValueError('"emiss" should be a 1D array with length == T0.shape[0]')
        self._emiss = emiss

        if z[0] != 0:
            raise RuntimeError('z[0] is required to be zero')
        self._z = z
        # TODO: should z[i] be positive or negative for i > 0?

        self._ti = ti
        if (self.ti <= 0.0).any():
            raise ValueError('thermal inertia ("ti") values should be positive')
        if self.ti.ndim != 1 or self.ti.size != self.num_layers:
            raise ValueError('"ti" should be a 1D array with length == z.size')

        self._rhoc = rhoc
        if (self.rhoc <= 0).any():
            raise ValueError('volumetric heat capacity ("rhoc") values should be positive')
        if self.rhoc.ndim != 1 or self.rhoc.size != self.num_layers:
            raise ValueError('"rhoc" should be a 1D array with length == z.size')

        self._Fgeotherm = Fgeotherm
        if (self.Fgeotherm < 0).any():
            raise ValueError('geothermal fluxes ("Fgeotherm") should be positive')
        if self.Fgeotherm.ndim != 1 or self.Fgeotherm.size != self.num_faces:
            raise ValueError('"Fgeotherm" should be a 1D array with length == T0.shape[0]')

        self._Tsurfprev = Tsurfprev
        if (self.Tsurfprev < 0).any():
            raise ValueError('surface temperatures ("Tsurfprev") should be nonnegative')
        if self.Tsurfprev.ndim != 1 or self.Tsurfprev.size != self.num_faces:
            raise ValueError('"Tsurfprev" should be a 1D array with length == T0.shape[0]')

        self._Qprev = Q0
        if (self.Qprev < 0).any():
            raise ValueError('initial irradiance ("Q0") should be nonnegative')
        if self.Qprev.ndim != 1 or self.Qprev.size != self.num_faces:
            raise ValueError('"Q0" should be a 1D array with length == T0.shape[0]')

        self._Fsurf = np.empty_like(self._Qprev)
        self._Fsurf[:] = np.nan

        self._T = T0
        if (self.T < 0).any():
            raise ValueError('initial temperature ("T0") should be nonnegative')
        if self.T.ndim != 2 or self.T.shape[1] != self.num_layers:
            raise ValueError('"T0" should be a 2D array with T0.shape[1] == z.size')

    # step for conductionQ
    cpdef step(self, double dt, X):
        cdef int i

        if self.bcond == 'Q':
            if X.ndim != 1 or X.size != self._Qprev.size:
                raise RuntimeError(f'input flux of wrong size: X.size == {X.size}')

            if not np.isfinite(X).all():
                raise RuntimeError('trying to step with nonfinite input fluxes')

            if (X < 0).any():
                raise RuntimeError('trying to step with negative fluxes')

            for i in range(self._num_faces):
                conductionQ(
                    self.num_layers_interior, # number of grid points below surface
                    &self._z[0],              # depth below surface
                    dt,                       # time step
                    self._Qprev[i],           # net solar insolation at previous time step
                    X[i],                     # net solar insolation at current time step
                    &self._T[i, 0],           # vertical temperature profile
                    &self._ti[0],             # thermal inertia
                    &self._rhoc[0],           # volumetric heat capacity
                    self._emiss[i],           # emissivity
                    self._Fgeotherm[i],       # geothermal heat flux at bottom boundary
                    &self._Fsurf[i])          # heat flux at surface (output)

            self._Qprev = X

        elif self.bcond == 'T':
            if X.ndim != 1 or X.size != self._Tsurfprev.size:
                raise RuntimeError(f'input flux of wrong size: X.size == {X.size}')

            if not np.isfinite(X).all():
                raise RuntimeError('trying to step with nonfinite input temps')

            if (X < 0).any():
                raise RuntimeError('trying to step with negative input temps')

            for i in range(self._num_faces):
                conductionT(
                    self.num_layers_interior, # number of grid points below surface
                    &self._z[0],              # depth below surface
                    dt,                       # time step
                    &self._T[i, 0],           # vertical temperature profile (in/out)
                    self._Tsurfprev[i],       # surf temp at previous time step
                    X[i],                     # surf temp at current time step
                    &self._ti[0],             # thermal inertia
                    &self._rhoc[0],           # volumetric heat capacity
                    self._Fgeotherm[i],       # geothermal heat flux at bottom boundary
                    &self._Fsurf[i])          # heat flux at surface (output)

            self._Tsurfprev = X

        else:
            raise RuntimeError(f'got unexpected BC mode: "{self.bcond}"')

        if not np.isfinite(self.T).all():
            raise RuntimeError('computed nonfinite temperatures while stepping')

        if (self.T < 0).any():
            raise RuntimeError('computed negative temperatures while stepping')

        self._current_time += dt

def setgrid(nz,zfac,zmax):
    """
    construct regularly or geometrically spaced 1D grid
    z(n)-z(1) = 2*z(1)*(zfac**(n-1)-1)/(zfac-1)
    choice of z(1) and z(2) is compatible with conductionQ
    Args:
        nz: number of layers
        zfac: spacing factor (?)
        zmax: maximum depth of layers

    Returns:
    z: nz layers
    """

    z = [0]
    if zfac>1.:
        dz = zmax/(3.+2.*zfac*(zfac**(nz-2)-1.)/(zfac-1.))
        z.append(dz)
        z.append(3*z[1])
        for i in range(nz+1)[3:]: # nz+1 for compatibility, to get z[-1]=zmax
            z.append((1+zfac)*z[i-1] - zfac*z[i-2])
    else:
        dz = zmax/nz
        z.extend([(i - 0.5) * dz for i in range(nz+1)[1:]]) # nz+1 for compatibility, to get z[-1]=zmax
    # here too, we want z to start "underground"
    return np.array(z)[1:]

# analytical version of conductionT
def analytT_(z,T0,rhoc,ti,P,dt):
    """
    Compare with analytical solution for sinusoidal surface temperature
    oscillation and semi-infinite domain
    Args:
        z: depth of sub-surface layers

    Returns:
        temperatures T at layers z
    """
    Ta = 30.

    delta = ti / rhoc * np.sqrt(P / np.pi) # skin depth
    w = 2 * np.pi / P
    T = []
    for t in np.arange(0,P,dt)[:]:
        T.append(T0 + Ta * np.exp(-z / delta) * np.sin(z / delta - w * t))

    return np.vstack(T)

def flux_noatm(R,decl,latitude,HA,SlopeAngle,azFac):
    """
    Calculates incoming solar flux without atmosphere
    Args:
        R: distance from sun (AU)
        decl: planetocentric solar declination (radians)
        latitude: (radians)
        HA: hour angle (radians from noon, clockwise)
        SlopeAngle: >0, (radians)
        azFac: azimuth of topographic gradient (radians east of north),
                azFac=0 is south-facing
    """

    from math import sin, cos, sqrt, acos, pi

    So=1365.  # solar constant [W/m^2]

    c1 = cos(latitude)*cos(decl)
    s1 = sin(latitude)*sin(decl)
    # beta = 90 minus incidence angle for horizontal surface
    # beta = elevation of sun above (horizontal) horizon
    sinbeta = c1*cos(HA) + s1

    cosbeta = sqrt(1-sinbeta**2)
    # hour angle -> azimuth
    buf = ( sin(decl)-sin(latitude)*sinbeta ) / (cos(latitude)*cosbeta)
    # buf can be NaN if cosbeta = 0
    if buf>+1.:
        buf=+1.0  # roundoff
    if buf<-1.:
        buf=-1.0  # roundoff
    azSun = acos(buf)
    if sin(HA)>=0:
        azSun = 2*pi-azSun

    # theta = 90 minus incidence angle for sloped surface
    sintheta = cos(SlopeAngle)*sinbeta - \
        sin(SlopeAngle)*cosbeta*cos(azSun-azFac)
    if cosbeta==0.:   # sun in zenith
        sintheta = cos(SlopeAngle)*sinbeta
    if sintheta<0.:
        sintheta = 0. # horizon
    if sinbeta<0.:
        sintheta=0.   # horizontal horizon at infinity

    flux_noatm = sintheta*So/(R**2)

    return flux_noatm
