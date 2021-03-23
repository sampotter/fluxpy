# cython: embedsignature=True

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
        int nfaces
        int nz
        double t, emissivity
        double[::1] z
        double[::1] ti
        double[::1] rhoc
        double[::1] Fgeotherm
        double[::1] Qprev
        double[::1] Fsurf
        double[:, ::1] T

        double[::1] Tsurfprev
        str bcond


    @property
    def nfaces(self):
        return self.nfaces

    @property
    def nz(self):
        return self.nz

    @property
    def t(self):
        return self.t

    @property
    def emissivity(self):
        return self.emissivity

    @property
    def z(self):
        return np.asarray(self.z[1:])

    @property
    def ti(self):
        return np.asarray(self.ti[1:])

    @property
    def rhoc(self):
        return np.asarray(self.rhoc[1:])

    @property
    def Fgeotherm(self):
        return np.asarray(self.Fgeotherm)

    @property
    def Qprev(self):
        return np.asarray(self.Qprev)

    @property
    def Fsurf(self):
        return np.asarray(self.Fsurf)

    @property
    def T(self):
        return np.asarray(self.T)

    @property
    def Tsurfprev(self):
        return np.asarray(self.Tsurfprev)

    def __cinit__(self, int nfaces, double[::1] z, double T0, double
                  ti, double rhoc, double emissivity=0.0, Fgeotherm=0.0, Qprev=0.0,
                  Tsurfprev=0.0, bcond='Q'):
        self.nfaces = nfaces
        self.nz = z.size
        self.t = 0
        self.emissivity = emissivity
        self.bcond = bcond

        self.z = np.empty((self.nz + 1,), dtype=np.float64)
        self.z[0] = 0.
        self.z[1:] = z[...]

        self.ti = np.empty((self.nz + 1,), dtype=np.float64)
        self.ti[:] = ti

        self.rhoc = np.empty((self.nz + 1,), dtype=np.float64)
        self.rhoc[:] = rhoc

        self.Fgeotherm = np.empty((self.nfaces,), dtype=np.float64)
        self.Fgeotherm[...] = Fgeotherm

        # conductionT arg
        self.Tsurfprev = np.empty((self.nfaces,), dtype=np.float64)
        self.Tsurfprev[...] = Tsurfprev

        self.Qprev = np.empty((self.nfaces,), dtype=np.float64)
        self.Qprev[...] = Qprev

        self.Fsurf = np.empty((self.nfaces,), dtype=np.float64)
        self.Fsurf[...] = 0

        self.T = np.empty((self.nfaces, self.nz + 1), dtype=np.float64)
        self.T[...] = T0

    # step for conductionQ
    cpdef step(self, double dt, double[::1] X):
        cdef int i

        if self.bcond == 'Q':
            # security check on emissivity
            if self.emissivity == 0.0:
                logging.warning("*** Running conductionQ with emissivity=0")
                exit()

            for i in range(self.nfaces):
                conductionQ(
                    self.nz, # number of grid points
                    &self.z[0], # depth below surface
                    dt,
                    self.Qprev[i],
                    X[i],
                    &self.T[i, 0],
                    &self.ti[0],
                    &self.rhoc[0],
                    self.emissivity,
                    self.Fgeotherm[i],
                    &self.Fsurf[i]
                )
            self.Qprev[...] = X[...]

        elif self.bcond == 'T':
                for i in range(self.nfaces):
                    conductionT(
                        self.nz, # number of grid points
                        &self.z[0], # depth below surface
                        dt,
                        &self.T[i, 0],
                        self.Tsurfprev[i],
                        X[i],
                        &self.ti[0],
                        &self.rhoc[0],
                        self.Fgeotherm[i],
                        &self.Fsurf[i]
                    )

                self.Tsurfprev[...] = X[...]

        else:
            logging.error("** unknown bcond parameter value: it should be Q or T")

        self.t += dt

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
#**********************************************************************
#   flux_noatm: calculates incoming solar flux without atmosphere
#     R: distance from sun (AU)
#     decl: planetocentric solar declination (radians)
#     latitude: (radians)
#     HA: hour angle (radians from noon, clockwise)
#     SlopeAngle: >0, (radians)
#     azFac: azimuth of topographic gradient (radians east of north)
#            azFac=0 is south-facing
#**********************************************************************
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
