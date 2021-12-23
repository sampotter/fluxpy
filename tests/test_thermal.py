import csv
import numpy as np
import pathlib
import unittest

from flux.thermal import setgrid, flux_noatm, PccThermalModel1D

class ThermalTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = pathlib.Path(__file__).parent.absolute()/'data'
        self.pcc_thermal_model_1d_path = \
            self.data_path/'thermal'/'pcc_thermal_model_1d'

        np.seterr('raise')

    def test_pcc_thermalQ_model_1d(self):
        path = self.pcc_thermal_model_1d_path/'Tprofile_conductionQ'
        path1 = self.pcc_thermal_model_1d_path/'Tsurface_conductionQ'

        # import template output from pcc/Python
        profile = []
        for p in [path,path1]:
            template = []
            with open(p) as csvDataFile:
                csvReader = csv.reader(csvDataFile,delimiter=" ",quoting=csv.QUOTE_NONNUMERIC)
                for row in csvReader:
                    template.append([x for x in row if x!=''])
            profile.append(template)

        Tprofile = np.vstack(profile[0]).astype('float')[-1] # T profile at last time step
        Tsurface = np.array(profile[1]) # (time/Period,T[0],T[nz])

        # generate 1D grid
        nz = 60
        zfac = 1.05
        zmax = 2.5
        z = setgrid(nz=nz, zfac=zfac, zmax=zmax)
        # print(z)

        # prepare testcranKQ-like input to compute Qn
        stepspersol = 120; period = 88775.244 * 670  # [seconds]
        albedo = 0.2; Rau = 1.52; Decl = 0.; HA = 0.
        latitude = 5.; latitude = np.deg2rad(latitude)
        Qn = (1 - albedo) * flux_noatm(Rau, Decl, latitude, HA, 0., 0.)

        T0 = np.empty((1, z.size + 1))
        T0[...] = 210

        # set-up thermal model
        model = PccThermalModel1D(nfaces=1,z=z,T0=T0,ti=120.,rhoc=960000.,
                                  emissivity=1.,Fgeotherm=0.2, Qprev=Qn, bcond='Q')

        dt = 495661.77900000004
        nsteps = 50000

        # iterate to t = t0+i*dt
        Tsurface_output = []; Tprofile_output = []
        for i in range(nsteps+1)[:]:

            # set "temporary" Qnp1 (from IR model only)
            time = (i + 1) * dt  # time at n+1;
            HA = 2 * np.pi * (time / period % 1.)  # hour angle
            Qnp1 = (1 - albedo) * flux_noatm(Rau, Decl, latitude, HA, 0., 0.)

            # update model to t+=dt
            model.step(dt, np.array([Qnp1]))

            if i%3 == 0:
                Tsurface_output.append([time / period,model.T[0][0],model.T[0][nz]])

            if (i > nsteps-stepspersol):
                if i%10 == 0:
                    Tprofile_output.append(model.T[0])

        # reformat PccThermalModel1D output at last step and validate with template
        Tprofile_output = np.vstack(Tprofile_output)[-1]
        Tsurface_output = np.vstack(Tsurface_output)

        # direct call to python version with same arguments (for testing)
        # # set initial conditions for python reference model
        # T0 = 210; T_condQ = np.repeat(np.float(T0),nz+1); Fsurf_condQ = 0
        # for i in range(nsteps)[:]:
        #
        #     # set "temporary" Qnp1 (from IR model only)
        #     time = (i + 1) * dt  # time at n+1;
        #     HA = 2 * np.pi * (time / Period % 1.)  # hour angle
        #     Qnp1 = (1 - albedo) * flux_noatm(Rau, Decl, latitude, HA, 0., 0.)
        #
        #     # launch test model
        #     conductionQ(nz, np.hstack([[0.],z]), dt, Qn, Qnp1, T_condQ, np.repeat(model.ti[0],nz+1),
        #                 np.repeat(model.rhoc[0],nz+1), model.emissivity, model.Fgeotherm[0], Fsurf_condQ)
        #     Qn = Qnp1
        #
        # template = np.vstack([np.hstack([[0.],z]),T_condQ]).T
        # validation = np.round(model_output.T,7) - np.round(template,7) # round at "reasonable" precision

        # check if Tprofile is the same in both cases (at last step, mK)
        self.assertLess(np.abs(np.sum(np.round(Tprofile_output,3)-np.round(Tprofile,3))), 1.e-3)

        # check if Tsurface (and Tbottom layer) is the same in both cases
        self.assertLess(np.abs(np.sum(np.round(Tsurface_output,3)-np.round(Tsurface,3))), 1.e-3)

if __name__ == '__main__':
    unittest.main()
