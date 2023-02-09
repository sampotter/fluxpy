# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:53:45 2023

@author: heath
"""

## This file will take in the Midpoint Form Factor Matrix and 
## the PyViewFactor Matrix and then compute the
## corresponding temperatures. Plots will be made aswell.

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Temperatures using PyViewFactor')
    #assigning values variables
    parser.add_argument(
        '-p', type=int, default=4,
        help='target edge length: h = (2/3)**p')
    parser.add_argument(
        '--e0', type=float, default=15,
        help='sun angle above horizon in [deg.]')
    parser.add_argument(
        '--F0', type=float, default=1000,
        help='solar constant [W/m^2]')
    parser.add_argument(
        '--rho', type=float, default=0.3,
        help='albedo')
    parser.add_argument(
        '--emiss', type=float, default=0.99,
        help='emissivity')
    parser.add_argument(
        '--tol', type=float, default=None,
        help='tolerance used when assembling compressed form factor matrix')
    parser.add_argument(
        '--beta', type=float, default=40,
        help='Ingersoll crater angle measured from vertical [deg.]')
    parser.add_argument(
        '--rc', type=float, default=0.8,
        help='Crater radius [m]')
    parser.add_argument(
        '--outdir', type=str, default='midpoint_6_data',
        help='Directory to write output files to')
    parser.add_argument(
        '--blocks', type=bool, default=False,
        help='Make a plot of the form factor matrix blocks')
    parser.add_argument(
        '--mprof', type=bool, default=False,
        help='Measure the memory use during assembly and save results')
    parser.add_argument(
        '--contour_rim', type=bool, default=True,
        help='Whether the rim of the crater should be contoured')
    parser.add_argument(
        '--contour_shadow', type=bool, default=False,
        help='Whether the shadow line in the crater should be contoured')
    args = parser.parse_args()

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.sparse

from flux.ingersoll import HemisphericalCrater
from flux.model import compute_steady_state_temp, sigSB
from flux.solve import solve_radiosity
from flux.util import tic, toc
from flux.plot import plot_blocks, tripcolor_vector

from make_ffm_midpoint import obj_to_shapemodel
    
if __name__ == '__main__':
    #if there is not a directory to write output files too, make a directory
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        
    #specifying some variables
    h = (2/3)**args.p
    e0 = np.deg2rad(args.e0)
    beta = np.deg2rad(args.beta)
    
    # Load the Form Factor Matrix from the given .npz file
    FF = scipy.sparse.load_npz('FF_midpoint_6.npz')
    
    # Import the Shape_Model
    shape_model = obj_to_shapemodel('hemispherical_mesh_6.obj', '.')
    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    
    # Create hemispherical crater
    hc = HemisphericalCrater(beta, args.rc, e0, args.F0, args.rho, args.emiss)
    
    # Array of direction sunlight
    dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])

    tic()
    # E: insolation, or exposure to the suns rays
    E = shape_model.get_direct_irradiance(args.F0, dir_sun, eps=1e-6)
    t_E = toc()
    print('- computed direct irradiance [%1.2f s]' % (t_E,))

    fig, ax = tripcolor_vector(V, F, E, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'E.png'))
    plt.close(fig)
    print('- wrote E.png to disk')

    tic()
    B, niter_B = solve_radiosity(FF, E, albedo_placement='right', rho=args.rho)
    t_B = toc()
    print('- computed radiosity [%1.2f s]' % (t_B,))

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'B.png'))
    plt.close(fig)
    print('- wrote B.png')

    Q_IR = FF@((1 - hc.rho)*B)
    Q, niter_Q = solve_radiosity(FF, Q_IR, 1, albedo_placement='right')
    Q *= hc.emiss
    Q += (1 - hc.rho)*B
    assert not np.isnan(Q).any()

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'Q.png'))
    plt.close(fig)
    print('- wrote Q.png')

    tic()
    T = compute_steady_state_temp(FF, E, args.rho, args.emiss, Fsurf=0)
    t_T = toc()
    print('- computed T [%1.2f s]' % (t_T,))

    fig, ax = tripcolor_vector(V, F, T, vmin = 125, vmax = 325, cmap=cc.cm.fire)
    fig.savefig(os.path.join(args.outdir, 'T.png'))
    plt.close(fig)
    print('- wrote T.png')

    # compute local elevation of sun for each facet
    e = np.maximum(0, np.pi/2 - np.arccos(N@dir_sun))

    b = hc.f*(hc.emiss + hc.rho*(1 - hc.f))/(1 - hc.rho*hc.f)

    Rc = np.sqrt(np.sum(shape_model.P[:, :2]**2, axis=1))

    mask_plane = Rc > hc.rc
    mask_crater = Rc <= hc.rc
    mask_shadow = mask_crater & (abs(E) < np.finfo(E.dtype).eps)
    mask_sun = mask_crater & (abs(E) >= np.finfo(E.dtype).eps)

    I_plane = np.where(mask_plane)[0]
    I_shadow = np.where(mask_shadow)[0]
    I_sun = np.where(mask_sun)[0]

    assert np.union1d(I_plane, np.union1d(I_shadow, I_sun)).size == mask_plane.size

    ## Setting up the Q groundtruth
    # Initializes an array of random numbers
    Q_gt = np.empty(shape_model.num_faces)
    # Doing groundtruth calculations
    Q_gt[I_plane] = (1 - hc.rho)*hc.F0*np.sin(hc.e0)
    Q_gt[I_shadow] = (1 - hc.rho)*hc.F0*b*np.sin(hc.e0)
    Q_gt[I_sun] = (1 - hc.rho)*hc.F0*(np.sin(e[I_sun]) + b*np.sin(hc.e0))
    
    # Saving files
    np.save(os.path.join(args.outdir, 'I_plane.npy'), I_plane)
    np.save(os.path.join(args.outdir, 'I_shadow.npy'), I_shadow)
    np.save(os.path.join(args.outdir, 'I_sun.npy'), I_sun)

    # Creating Q_gt.png figure and writing to disk
    
    fig, ax = tripcolor_vector(V, F, Q_gt, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'Q_gt.png'))
    plt.close(fig)
    print('- wrote Q_gt.png')

    # Plotting the pointwise difference between Q (approximation)
    # and Q_gt (ground truth)
    Q_error = Q - Q_gt
    vmax = abs(Q_error).max()
    vmin = -vmax
    fig, ax = tripcolor_vector(V, F, Q_error, vmin=vmin,vmax=vmax,cmap=cc.cm.coolwarm)
    fig.savefig(os.path.join(args.outdir, 'Q_error.png'))
    plt.close(fig)
    print('- wrote Q_error.png')

    # compute groundtruth temperature in crater
    T_gt = np.maximum(0, Q_gt/(sigSB*hc.emiss))**0.25

    # compute pointwise error
    T_error = T - T_gt

    np.save(os.path.join(args.outdir, 'T_error.npy'), T_error)

    # plot pointwise T_error
    vmax = abs(T_error).max()
    vmin = -vmax
    print(np.argmax(abs(T_error)))
    fig, ax = tripcolor_vector(V,F,T_error,vmin=vmin,vmax=vmax,cmap=cc.cm.coolwarm)
    plt.scatter(*shape_model.P[np.argmax(abs(T_error)), :2], s=30, color='red', zorder=2)
    fig.savefig(os.path.join(args.outdir, 'T_error.png'))
    plt.close(fig)
    print('- wrote T_error.png')
    
    # plot groundtruth temperature
    fig, ax = tripcolor_vector(V, F, T_gt, vmin = 125, vmax = 325, cmap=cc.cm.fire)
    fig.savefig(os.path.join(args.outdir, 'T_gt.png'))
    plt.close(fig)
    print('- wrote T_gt.png')

    max_T_error = abs(T_error).max()
    print('- max T_error: %1.2f K' % (max_T_error,))

    l2_T_error = np.linalg.norm(T_error)
    print('- l2 T_error: %1.2f K' % (l2_T_error,))

    l1_T_error = np.linalg.norm(T_error, ord=1)
    print('- l1 T_error: %1.2f K' % (l1_T_error,))

    rel_max_T_error = max_T_error/abs(T_gt).max()
    print(f'- relative max T_error: {100*rel_max_T_error:1.2f}%')

    rel_l2_T_error = l2_T_error/np.linalg.norm(T_gt)
    print(f'- relative l2 T_error: {100*rel_l2_T_error:1.2f}%')

    rel_l1_T_error = l1_T_error/np.linalg.norm(T_gt, ord=1)
    print(f'- relative l1 T_error: {100*rel_l1_T_error:1.2f}%')

    B.tofile(os.path.join(args.outdir, 'B.bin'))
    print('- wrote B.bin')

    T.tofile(os.path.join(args.outdir, 'T.bin'))
    print('- wrote T.bin')
    