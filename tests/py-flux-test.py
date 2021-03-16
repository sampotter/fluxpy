import unittest
import csv

import numpy as np
from flux.thermal import PccThermalModel1D, setgrid

class ThermalTestCase(unittest.TestCase):
    def test_pcc_thermal_model_1d(self):

        # import template output (["index","depth","temperature","heat_flux"])
        template = []
        with open('tests/template.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
            for row in csvReader:
                template.append(row)
        template = np.array(template)[:,1:-1] # only read depth and temperature columns

        # generate 1D grid
        nz = 60
        zfac = 1.05
        zmax = 2.5
        z = setgrid(nz=nz, zfac=zfac, zmax=zmax)
        # print(z)

        # set-up thermal model
        model = PccThermalModel1D(nfaces=1,z=z,T0=210.,ti=120.,rhoc=960000.,emissivity=1.,Fgeotherm=0.2)

        # simulated Q input at t steps (dt=495661.76666666666 s)
        Qsim = np.array([470.84688077122127, 470.20160144930207,
                        468.26753214941181,465.04997402136058])
        dt = 495661.76666666666
        nsteps = 1 # corresponding to step 2 output in testcrankQ

        # TODO this is the way I found to set Q for t=0, but it impacts Tsurf.
        # TODO why can't we set model.Qprev = Qsim[0] for the first step? Maybe it doesn't matter, but for testing...
        model.step(0., np.array([Qsim[0]]))

        # iterate to t = t0+i*dt
        for i in range(nsteps):
            # print input model state
            # print(f"input of step {i}",model.t)
            # print("T",model.T)
            # print("Fsurf",model.Fsurf)

            # set "temporary" Qnp1 (from IR model only)
            Qnp1 = Qsim[i+1]
            # update model to t+=dt
            model.step(dt, np.array([Qnp1]))

            # print output model state
            # print("output of step", i)
            # print("###############")
            # print("t",model.t)
            # print("T",model.T)
            # print("Fsurf",model.Fsurf)

        model_output = np.vstack([np.array([0] + [x for x in model.z]),model.T[0]])

        validation = np.round(model_output.T,7) - np.round(template,7)

        # check if z is the same in both cases
        assert np.abs(np.sum(validation[:,0])) < 1.e-6
        # check if T is the same in both cases
        assert np.abs(np.sum(validation[:,1])) < 1.e-6

if __name__ == '__main__':
    unittest.main()