#!/usr/bin/env python3

# Calculates equilibrium temperature based on FF.bin using quasi-timesteps
# instead of radiosity equations
# Compares with exact solution for bowl-shaped crater

import matplotlib.pyplot as plt
import numpy as np
#import os

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import tripcolor_vector
from flux.model import update_incoming_radiances
from scipy.constants import Stefan_Boltzmann as sigSB
from equilbrT_from_FF import exact_solution_ingersoll

    
if __name__ == '__main__':
    
    # load FF.bin previously generated
    FF = CompressedFormFactorMatrix.from_file('FF.bin')
    print('- loaded FF.bin')
    shape_model = FF.shape_model
    V = FF.shape_model.V
    F = FF.shape_model.F
    N = FF.shape_model.N
    P = FF.shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])
    
    # Define constants used in the simulation:
    e0 = np.deg2rad(10) # Solar elevation angle
    F0 = 1365 # Solar constant
    albedo = 0.3
    emiss = 0.95
    
    dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # Direction of sun
    
    # Compute the direct irradiance and find the elements which are in shadow.
    E = shape_model.get_direct_irradiance(F0, dir_sun)
    print('Number of elements in E =',E.shape[0])
    
    # Make plot of direct irradiance
    fig, ax = tripcolor_vector(V, F, E, cmap='gray')
    fig.savefig('E.png')
    plt.close(fig)
    print('- wrote E.png')
   
    # calculate analytical solution for bowl-shaped crater
    Texact = exact_solution_ingersoll( F0, e0, albedo, emiss, 5., P, N, E, dir_sun)
    
    fig, ax = tripcolor_vector(V, F, Texact, cmap='inferno')
    fig.savefig('Texact.png')
    plt.close(fig)
    print('- wrote Texact.png')
    
    # set provisional values 
    Qabs = (1 - albedo) * E
    Tsurf = (Qabs / (sigSB * emiss)) ** 0.25
    Qrefl = np.zeros_like(E)
    QIR = np.zeros_like(E)
    
    for i in range(0,7):  # iterate (orders of reflections)
        Qrefl, QIR = update_incoming_radiances(FF, E, albedo, emiss, Qrefl, QIR, Tsurf)
        Qabs = (1 - albedo) * (E + Qrefl) + emiss * QIR
        Tsurf  = ( Qabs / (emiss*sigSB) )**0.25

        err = Tsurf - Texact
        max_error = np.linalg.norm(err, np.inf)
        abs_error = np.linalg.norm(err, 1) / err.size
        rms_error = np.linalg.norm(err, 2) / err.size
        
        print('iteration',i+1,
              'max_error',float(max_error),
              'abs_error',float(abs_error),
              'rms_error',float(rms_error) )

        
    fig, ax = tripcolor_vector(V, F, Tsurf, cmap='inferno')
    fig.savefig('T.png')
    plt.close(fig)
    print('- wrote T.png')
    
    vlim = ( np.percentile(err,95) + np.percentile(err,5) )/2
    fig, ax = tripcolor_vector(V, F, err, vmin=-vlim, vmax=+vlim, cmap='seismic')
    fig.savefig('error.png')
    plt.close(fig)
    print('- wrote error.png')
    


