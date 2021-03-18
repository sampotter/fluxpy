# cython: embedsignature=True

import numpy as np

cdef extern from "pcc/thrmlLib.h":
    void conductionQ(int nz, double z[], double dt, double Qn, double
		     Qnp1, double T[], double ti[], double rhoc[],
		     double emiss, double Fgeotherm, double *Fsurf)

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

    def __cinit__(self, int nfaces, double[::1] z, double T0, double
                  ti, double rhoc, double emissivity, Fgeotherm=0.0):
        self.nfaces = nfaces
        self.nz = z.size
        self.t = 0
        self.emissivity = emissivity

        self.z = np.empty((self.nz + 1,), dtype=np.float64)
        self.z[1:] = z[...]

        self.ti = np.empty((self.nz + 1,), dtype=np.float64)
        self.ti[1:] = ti

        self.rhoc = np.empty((self.nz + 1,), dtype=np.float64)
        self.rhoc[1:] = rhoc

        self.Fgeotherm = np.empty((self.nfaces,), dtype=np.float64)
        self.Fgeotherm[...] = Fgeotherm

        self.Qprev = np.empty((self.nfaces,), dtype=np.float64)
        self.Qprev[...] = 0

        self.Fsurf = np.empty((self.nfaces,), dtype=np.float64)
        self.Fsurf[...] = 0

        self.T = np.empty((self.nfaces, self.nz + 1), dtype=np.float64)
        self.T[...] = T0

    cpdef step(self, double dt, double[::1] Q):
        cdef int i
        for i in range(self.nfaces):
            conductionQ(
                self.nz, # number of grid points
                &self.z[0], # depth below surface
                dt,
                self.Qprev[i],
                Q[i],
                &self.T[i, 0],
                &self.ti[0],
                &self.rhoc[0],
                self.emissivity,
                self.Fgeotherm[i],
                &self.Fsurf[i]
            )
        self.Qprev[...] = Q[...]
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
    dz = zmax/nz

    z = [0]
    if zfac>1.:
        dz = zmax/(3.+2.*zfac*(zfac**(nz-2)-1.)/(zfac-1.))
        z.append(dz)
        z.append(3*z[1])
        for i in range(nz+1)[3:]: # nz+1 for compatibility, to get z[-1]=zmax
            z.append((1+zfac)*z[i-1] - zfac*z[i-2])
    else:
        z.extend([(i - 0.5) * dz for i in range(nz+1)[1:]]) # nz+1 for compatibility, to get z[-1]=zmax
    # here too, we want z to start "underground"
    return np.array(z)[1:]