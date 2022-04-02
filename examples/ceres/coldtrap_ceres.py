#!/usr/bin/env python3

# Calculates equilibrium temperature based on FF for a full rotation


import matplotlib.pyplot as plt
import numpy as np
#import os
import time
import json
import pickle
import spiceypy as spice
import scipy.sparse

from flux.model import compute_steady_state_temp
    

if __name__ == '__main__':

    # load uncompressed FF previously generated
    FF = scipy.sparse.load_npz('FF.npz')
    with open('mesh.bin', 'rb') as f:
        shape_model = pickle.load(f)

    # load compressed FF previously generated
    #FF = CompressedFormFactorMatrix.from_file(FF_path)
    #shape_model = FF.shape_model

    print('- loaded FF')
    V = shape_model.V
    F = shape_model.F
    #N = shape_model.N
    P = shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])

    # initialize SPICE
    spice.kclear()
    spice.furnsh('ceres.furnsh')

    # Define time window
    # northern summer solstice, declination +4
    et0 = spice.str2et('2015 JUL 24 08:00:00.00')
    et1 = spice.str2et('2015 JUL 24 18:00:00.00')
    # southern summer solstice, declination -4
    #et0 = spice.str2et('2017 DEC 23 05:00:00.00')
    #et1 = spice.str2et('2017 DEC 23 15:00:00.00')

    et = np.linspace(et0, et1, 180, endpoint=False)

    # Sun positions over time period
    possun = spice.spkpos('SUN', et, 'CERES_FIXED', 'LT+S', 'CERES')[0]
    radsun = np.sqrt(np.sum(possun[:, :3]**2, axis=1))
    AU = 149.59787e6 # km
    print('distance range (AU)', np.min(radsun)/AU, np.max(radsun)/AU )
    
    # Define constants used in the simulation:
    F0 = 1365 # Solar constant
    albedo = 0.1
    emiss = 0.98

    Emax = np.zeros(F.shape[0])
    Tmax = np.zeros(F.shape[0])
    
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

        # Calculate equilibrium temperatures
        T = compute_steady_state_temp(FF, E, albedo, emiss)
        Tmax = np.maximum.reduce([Tmax, T])

        
    stats = {
        'F0': F0,
        'albedo': albedo,
        'emiss': emiss,
        'num_faces': F.shape[0],
    }
    with open('params.json', 'w') as f:
        json.dump(stats, f)
    print('- wrote params.json')
    
    # Make plot of direct irradiance
    vv = np.vstack( (V[:,0], V[:,1]) )  # same as V[:,:2].T
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.tripcolor(*vv, F, Emax, cmap = 'gray', vmin=0)
    fig.colorbar(im, ax=ax, label='Maximum incident flux (W/m^2)')
    ax.set_aspect('equal')
    fig.savefig('Emax.png')
    plt.close(fig)
    print('- wrote Emax.png')
    
    # Make plot of maximum temperature
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.tripcolor(*vv, F, Tmax, cmap = 'jet')
    fig.colorbar(im, ax=ax, label='Maximum Temperature (K)')
    ax.set_aspect('equal')
    fig.savefig('Tmax.png')
    plt.close(fig)
    print('- wrote Tmax.png')

    
