#!/usr/bin/env python3

# Calculates equilibrium temperature based on FF.bin
# Compares with exact solution for bowl-shaped crater

cmap = dict()
try:
    import colorcet as cc
    cmap['jet'] = cc.cm.rainbow
    cmap['gray'] = cc.cm.gray
    cmap['fire'] = cc.cm.fire
except ImportError:
    print('failed to import colorcet: using matplotlib colormaps')
    cmap['jet'] = 'jet'
    cmap['gray'] = 'gray'
    cmap['fire'] = 'inferno'


import matplotlib.pyplot as plt
import numpy as np
import json
import pickle

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import tripcolor_vector
from flux.model import compute_steady_state_temp
from flux.solve import solve_radiosity
from flux.util import tic, toc
from scipy.constants import Stefan_Boltzmann as sigSB

import flux.form_factors as ff


def exact_solution_ingersoll(F0, e0, albedo, emiss, D2d, P, N, E, dir_sun):
    # calculate analytical solution for bowl-shaped crater
    # F0 ... solar constant [W/m^2]
    # e0 ... elevation of sun above horizontal surface [radians]
    # albedo ... (visible) albedo
    # emiss ... (infrared) emissivity
    # D2d ... diameter-to-depth ratio
    # P ... coordinates of triangle centers
    # N ... surface normals of length 1
    # E ... direct solar irradiance
    # dir_sun ... vector pointing toward sun

    f = 1 / (1 + 0.25*D2d**2)
    b = f*(emiss + albedo*(1-f)) / (1 - albedo*f)
    T = np.empty(P.shape[0])
    T[:] = np.nan
    Eexact = np.empty(P.shape[0])
    Eexact[:] = np.nan

    # plain around crater
    k = np.where( P[:,2]==0 )
    Eexact[k] = F0 * np.sin(e0)
    Q = (1-albedo) * F0 * np.sin(e0)
    T[k] = ( Q/(emiss*sigSB) )**0.25

    # shadowed portion of crater
    k = np.where( E==0 )
    Eexact[k] = 0.
    Q = (1-albedo) * F0 * b*np.sin(e0)
    T[k] = ( Q/(emiss*sigSB) )**0.25

    # sunlit portion of crater
    k = np.where( np.logical_and(E>0,  P[:,2]<0) )
    e = np.pi/2 - np.arccos( np.dot(N[k], dir_sun) )
    Eexact[k] = F0 * np.sin(e)
    Q = (1-albedo) * F0 * ( np.sin(e) + b*np.sin(e0) )
    T[k] = ( Q/(emiss*sigSB) )**0.25

    return T



compress = True


if __name__ == '__main__':

    if compress is False:
        with open('mesh.bin', 'rb') as f:
            shape_model = pickle.load(f)

        FF = ff.FormFactorMatrix.from_file('FF.bin')

    else:
        # load FF.bin previously generated
        FF = CompressedFormFactorMatrix.from_file('FF.bin')
        shape_model = FF.shape_model

    print('- loaded FF.bin')
    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    P = shape_model.P
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])

    # Define constants used in the simulation:
    e0 = np.deg2rad(10) # Solar elevation angle
    F0 = 1365 # Solar constant
    albedo = 0.3
    emiss = 0.95

    dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # Direction of sun

    # Compute the direct irradiance and find the elements which are in shadow.
    tic()
    E = shape_model.get_direct_irradiance(F0, dir_sun)
    t_E = toc()
    I_shadow = E == 0
    print('Number of elements in E =',E.shape[0])

    # Make plot of direct irradiance
    fig, ax = tripcolor_vector(V, F, E, cmap=cmap['gray'])
    fig.savefig('E.png')
    plt.close(fig)
    print('- wrote E.png')

    tic()
    B, niter_B = solve_radiosity(FF, E, rho=albedo)
    t_B = toc()
    print('- computed radiosity [%1.2f s]' % (t_B,))

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig('B.png')
    plt.close(fig)
    print('- wrote B.png')

    tic()
    T = compute_steady_state_temp(FF, E, albedo, emiss)
    t_T = toc()
    print('- computed T [%1.2f s]' % (t_T,))

    fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.fire)
    fig.savefig('T.png')
    plt.close(fig)
    print('- wrote T.png')

    stats = {
        'e0_deg': np.rad2deg(e0),
        'F0': F0,
        'albedo': albedo,
        'emiss': emiss,
        'num_faces': F.shape[0],
        't_E': float(t_E),
        't_B': float(t_B),
        't_T': float(t_T),
        'niter_B': niter_B
    }
    with open('stats.json', 'w') as f:
        json.dump(stats, f)
    print('- wrote stats.json')

    # IF the input was a bowl-shape crater, the solution can be compared with
    # the exact analytical solution. Otherwise
    # exit()

    # calculate analytical solution
    Texact = exact_solution_ingersoll( F0, e0, albedo, emiss, 5., \
                                              P, N, E, dir_sun)

    fig, ax = tripcolor_vector(V, F, Texact, cmap=cc.cm.fire)
    fig.savefig('Texact.png')
    plt.close(fig)
    print('- wrote Texact.png')

    err = T - Texact
    max_error = np.linalg.norm(err, np.inf)
    abs_error = np.linalg.norm(err, 1) / err.size
    rms_error = np.linalg.norm(err, 2) / err.size

    vlim = ( np.percentile(err,95) + np.percentile(err,5) )/2
    fig, ax = tripcolor_vector(V, F, err, vmin=-vlim, vmax=+vlim, cmap='seismic')
    fig.savefig('error.png')
    plt.close(fig)
    print('- wrote error.png')

    print('num_faces',F.shape[0],
        'max_error',float(max_error),
        'abs_error',float(abs_error),
        'rms_error',float(rms_error) )
