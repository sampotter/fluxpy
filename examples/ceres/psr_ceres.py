#!/usr/bin/env python3

# finds areas in permanent shadow
# sun directions are from spice kernel for Ceres

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos, arctan2, arctan
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
    lonsun = arctan2(possun[:, 1], possun[:, 0])
    lonsun = np.mod(lonsun, 2*pi)
    radsun = np.sqrt(np.sum(possun[:, :2]**2, axis=1))
    latsun = arctan2(possun[:, 2], radsun)

    print('declination range',np.rad2deg(np.min(latsun)), np.rad2deg(np.max(latsun)) )
    print('lonsun range',np.rad2deg(np.min(lonsun)), np.rad2deg(np.max(lonsun)) )
    print('distance range',np.min(radsun),np.max(radsun))
    
    # load topo 
    with open('mesh.bin', 'rb') as f:
        shape_model = pickle.load(f)

    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    P = shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])
    print('V',V.shape,'F',F.shape,'P',P.shape)

    #V = V.astype(np.float32)
    #shape_model = MyTrimeshShapeModel(V, F)
    
    print('x_min:',np.min(V[:,0]), 'x_max:',np.max(V[:,0]) )
    print('y_min:',np.min(V[:,1]), 'y_max:',np.max(V[:,1]) )
    print('z_min:',np.min(V[:,2]), 'z_max:',np.max(V[:,2]) )
    
    # Define constants used in the simulation:
    F0 = 1365 # Solar constant
    # coordinates of location
    #latloc = np.deg2rad(-85.)
    #lonloc = 0.

    # Convert (x,y,z) to (lat,lon)
    # This assumes z=0 represents the equatorial plane
    x = P[:,0]; y = P[:,1]; z = P[:,2]
    latloc = arctan( z / np.sqrt(x**2 + y**2) )
    lonloc = arctan2(y,x)  # zero and direction might be arbitrary
    print('latloc range',np.rad2deg(np.min(latloc)), np.rad2deg(np.max(latloc)) )
    print('lonloc range',np.rad2deg(np.min(lonloc)), np.rad2deg(np.max(lonloc)) )
    
    Emax = np.zeros(F.shape[0])

    #print(latloc.shape, Emax.shape)
    print('')
    
    for i in range(0,len(et)):

        # set solar declination and hour angles
        decl = latsun[i]
        HA = lonsun[i] - lonloc  # or something like that
        HA = np.mod(HA,2*pi)
        k = np.where(HA>pi)
        HA[k] = HA[k]-2*pi

        #print('HA range',np.rad2deg(np.max(HA)), np.rad2deg(np.min(HA)) )

        # calculate local sun elevation and azimuth
        sinbeta = cos(latloc)*cos(decl)*cos(HA) + sin(latloc)*sin(decl)
        cosbeta = np.sqrt(1-sinbeta**2)
        buf = ( sin(decl)-sin(latloc)*sinbeta ) / (cos(latloc)*cosbeta)
        buf[np.where(buf>+1)]=+1.  # roundoff
        buf[np.where(buf<-1)]=-1.  # roundoff
        azSun = np.arccos(buf)
        k = np.where( sin(HA)>=0. )
        azSun[k] = 2*pi - azSun[k]
        #print('azSun range',np.rad2deg(np.min(azSun)), np.rad2deg(np.max(azSun)) )
        
        Rau = radsun[i]/149.59787e6  # km -> au
        
        # Direction of sun
        dir_sun = np.array([
            cos(azSun)*cosbeta, sin(azSun)*cosbeta, sinbeta 
        ])  # doesn't work
        #e0 = np.deg2rad(10.) # Solar elevation angle
        #dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # works
        dir_sun = dir_sun[:,0]  # already doesn't work
        
        # Compute the direct irradiance and find the elements which are in shadow.
        E = shape_model.get_direct_irradiance(F0/Rau**2, dir_sun)
        I_shadow = E == 0
        print(i,np.max(E))

        Emax = np.maximum.reduce([Emax, E])
        
        # output snapshots
        fn = 'E' + str(i) + '.png'
        fig, ax = tripcolor_vector(V, F, E, cmap='jet')
        fig.savefig(fn)
        plt.close(fig)

        
    print(np.min(Emax), np.max(Emax) )
        
    # Make plot of maximum direct irradiance
    fig, ax = tripcolor_vector(V, F, Emax, cmap='jet')
    fig.savefig('E.png')
    plt.close(fig)
    print('- wrote E.png')

    
