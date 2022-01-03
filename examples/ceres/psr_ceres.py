#!/usr/bin/env python3

# finds areas in permanent shadow
# sun directions are from spice kernel for Ceres

import matplotlib.pyplot as plt
import numpy as np
import pickle
import spiceypy as spice

from flux.plot import tripcolor_vector

    
if __name__ == '__main__':

    # initialize SPICE
    spice.kclear()
    spice.furnsh('simple.furnsh')
    spice.furnsh('kernels/dawn_ceres_SPC151123.tpc')
    spice.furnsh('kernels/sb_ceres_grv_171219.bsp')

    # Define time window
    # northern summer solstice, declination +4
    et0 = spice.str2et('2015 JUL 24 08:00:00.00')
    et1 = spice.str2et('2015 JUL 24 18:00:00.00')
    # southern summer solstice, declination -4
    #et0 = spice.str2et('2017 DEC 23 05:00:00.00')
    #et1 = spice.str2et('2017 DEC 23 15:00:00.00')

    et = np.linspace(et0, et1, 10, endpoint=False)

    # Sun positions over time period
    possun = spice.spkpos('SUN', et, 'CERES_FIXED', 'LT+S', 'CERES')[0]
    radsun = np.sqrt(np.sum(possun[:, :2]**2, axis=1))
    AU = 149.59787e6 # km
    print('distance range (AU)', np.min(radsun)/AU, np.max(radsun)/AU )
    
    # load topo 
    with open('mesh.bin', 'rb') as f:   # enter file name here
        shape_model = pickle.load(f)

    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    P = shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])
    print('V',V.shape,'F',F.shape,'P',P.shape)

    # Define constants used in the simulation
    F0 = 1365 # Solar constant

    # This assumes z=0 represents the equatorial plane
    x = P[:,0]; y = P[:,1]; z = P[:,2]
    latloc = np.arctan( z / np.sqrt(x**2 + y**2) )
    print('latitude range of shape model', np.rad2deg(np.min(latloc)), np.rad2deg(np.max(latloc)) )
    
    Emax = np.zeros(F.shape[0])

    print('')
    
    for i in range(len(et)):

        Rau = radsun[i] / AU  # km -> au
        
        # Direction of sun
        dir_sun = possun[i,:] / radsun[i]
        
        # Compute the direct irradiance and find the elements which are in shadow.
        E = shape_model.get_direct_irradiance(F0/Rau**2, dir_sun)
        #I_shadow = E == 0
        print(i, np.max(E))

        # Keep track of maximum irradiance
        Emax = np.maximum.reduce([Emax, E])
        
        # output snapshots
        if len(et)<=10:
            fn = 'E' + str(i) + '.png'
            fig, ax = tripcolor_vector(V, F, E, cmap='jet')
            fig.savefig(fn)
            plt.close(fig)


    # Calculate areas
    kpsr = np.where(Emax==0)[0]
    print('')
    print( np.min(Emax), np.max(Emax) )
    print('Total area:', shape_model.A.sum()  )
    print('PSR area:', shape_model.A[kpsr].sum() )

    # Save map
    np.savez('emax', V=V, F=F, P=P, Emax=Emax, areas=shape_model.A)

    # Make plot of maximum direct irradiance
    fig, ax = tripcolor_vector(V, F, Emax, cmap='gray')
    fig.savefig('Emax.png')
    plt.close(fig)
    print('- wrote Emax.png')
