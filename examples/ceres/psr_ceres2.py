#!/usr/bin/env python3

# finds areas in permanent shadow
# sun directions are circular orbit
# z=0 must represent the equatorial plane
# loops through declinations/obliquities

import matplotlib.pyplot as plt
import numpy as np
import pickle

from flux.plot import tripcolor_vector
from generalorbit import equatorial2sundir

    
NT = 360  #  number of time steps


    
if __name__ == '__main__':

    # load topo 
    with open('mesh.bin', 'rb') as f:   # enter file name here
        shape_model = pickle.load(f)

    V = shape_model.V
    F = shape_model.F
    P = shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])
    #print('V',V.shape,'F',F.shape,'P',P.shape)
    
    # Define constants used in the simulation
    F0 = 1365 # Solar constant
    Rau = 2.76750591  # semimajor axis of Ceres orbit

    HA = np.linspace(0, 2*np.pi, NT, endpoint=False)

    # list of declinations
    decl = [2.0, 4.0, 10.0, 20.0]
    decl = [2., 4., 6., 8., 10., 12., 18., 20.]
    

    EmaxBig = np.empty( ( F.shape[0], len(decl) ) )
    
    for n in range(len(decl)):

        print('')
        print('solar declination:', decl[n] )
        
        Emax = np.zeros(F.shape[0])

        #for j in arange(decl[n],-0.1,-2):  # several positions along orbit
            
        decl_tmp = np.deg2rad(decl[n])  # for north pole
        #decl_tmp = -np.deg2rad(decl[n])  # for south pole
            
        for i in range(NT): # loop through hour angles

            # Direction of sun
            dir_sun = np.array( equatorial2sundir( decl_tmp, HA[i]) )
            
            # Compute the direct irradiance and find the elements which are in shadow.
            E = shape_model.get_direct_irradiance(F0/Rau**2, dir_sun)
            #print(i, np.max(E))

            # Keep track of maximum irradiance
            Emax = np.maximum.reduce([Emax, E])
        
            
        # Calculate areas
        kpsr = np.where(Emax==0)[0]
        #I_psr = Emax == 0
        #print('')
        print( np.min(Emax), np.max(Emax) )
        print('Total area:', shape_model.A.sum()  )
        print('PSR area:', shape_model.A[kpsr].sum() )

        EmaxBig[:,n] = Emax

        
    # Make a plot of maximum direct irradiance
    fig, ax = tripcolor_vector(V, F, Emax, cmap='gray')
    fig.savefig('Emax.png')
    plt.close(fig)
    print('- wrote Emax.png')

    
    # Save map data
    tmp = np.vstack((P.T,EmaxBig.T)).T
    #np.savetxt('emaxs.dat',tmp, fmt='%.4f %.4f %.4f %6.2f %6.2f %6.2f %6.2f')
    np.savetxt('emaxs.dat',tmp, fmt='%.4f %.4f %.4f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f')

   
