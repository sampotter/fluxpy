#!/usr/bin/env python

'''This script uses SPICE to compute a trajectory for the sun, loads a
shape model discretizing a patch of the lunar south pole (made using
lsp_make_obj.py), and a compressed form factor matrix for that
shape model (computed using lsp_compress_form_factor_matrix.py).
It then proceeds to compute the steady state temperature at each sun
position, writing a plot of the temperature to disk for each sun
position.

'''
import logging
import pickle

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from pathlib import Path

import flux.compressed_form_factors as cff
import flux.form_factors as ff

from flux.model import compute_steady_state_temp, update_incoming_radiances, update_incoming_radiances_wsvd
from flux.plot import tripcolor_vector
from flux.thermal import PccThermalModel1D, setgrid
from scipy.constants import sigma as sigSB

# Use these temporary parameters...
from flux.shape import CgalTrimeshShapeModel, EmbreeTrimeshShapeModel

F0 = 1365  # Solar constant
emiss = 0.95  # Emissitivity
albedo = 0.12  # Visual (?) albedo
Fgeoth = 0.2
steady_state = False  # True

def illuminate_form_factor(FF_path = 'lsp_compressed_form_factors.bin', compressed=True, plot_fluxes=False,
                           use_svd=False, engine='cgal'):
    """
    Compute Sun position, illuminate FF and compute irradiance and temperature
    Args:
        FF_path: input form factor matrix
        compressed: bool, is the input FF compressed

    Returns:
        dict
    """
    # clktol = '10:000'

    spice.kclear()
    spice.furnsh('simple.furnsh')

    # Define time window

    et0 = spice.str2et('2011 MAR 01 00:00:00.00')
    et1 = spice.str2et('2011 APR 01 00:00:00.00')
    et = np.linspace(et0, et1, 10, endpoint=False)
    stepet = et[1]-et[0]

    # Sun positions over time period

    possun = spice.spkpos('SUN', et, 'MOON_ME', 'LT+S', 'MOON')[0]
    lonsun = np.arctan2(possun[:, 1], possun[:, 0])
    lonsun = np.mod(lonsun, 2*np.pi)
    radsun = np.sqrt(np.sum(possun[:, :2]**2, axis=1))
    latsun = np.arctan2(possun[:, 2], radsun)

    sun_dirs = np.array([
        np.cos(lonsun)*np.cos(latsun),
        np.sin(lonsun)*np.cos(latsun),
        np.sin(latsun)
    ]).T.copy(order='C')

    # Load compressed form factor matrix, including shape model, from disk
    if use_svd:
        V = np.load('lsp_V.npy')
        F = np.load('lsp_F.npy')
        N = np.load('lsp_N.npy')
        shape_model = CgalTrimeshShapeModel(V, F, N)
    elif compressed:
        logging.warning("Retrieving compressed FF block and shape_model...")
        FF = cff.CompressedFormFactorMatrix.from_file(FF_path)
        shape_model = FF.shape_model
        V = FF.shape_model.V
        F = FF.shape_model.F
    else:
        logging.warning("Retrieving full FF block and shape_model...")
        FF = ff.FormFactorMatrix.from_file(FF_path)
        V = np.load('lsp_V.npy')
        F = np.load('lsp_F.npy')
        N = np.load('lsp_N.npy')
        if engine == 'cgal':
            shape_model = CgalTrimeshShapeModel(V.copy(order='C'), F.copy(order='C'),
                                                N.copy(order='C'))  # TrimeshShapeModel(V, F, N, P)
        elif engine == 'embree':
            shape_model = EmbreeTrimeshShapeModel(V.copy(order='C'), F.copy(order='C'),
                                                  N.copy(order='C'))  # TrimeshShapeModel(V, F, N, P)
        else:
            logging.error("Please specify which ray tracing engine to use: cgal or embree.")

    # illuminate FF
    E_arr = []
    for i, sun_dir in enumerate(sun_dirs):
        E_arr.append(shape_model.get_direct_irradiance(F0, sun_dir, unit_Svec=True))

    E = np.vstack(E_arr).T
    Qrefl = []
    QIR = []
    T = []

    if steady_state:
        # Compute steady state temperature
        T_arr = compute_steady_state_temp(FF, E, albedo, emiss)
        T = np.vstack(T_arr).T
    elif False:
        # spin-up model until equilibrium is reached

        # generate 1D grid
        nz = 60
        zfac = 1.05
        zmax = 2.5
        z = setgrid(nz=nz, zfac=zfac, zmax=zmax)

        # compute provisional Tsurf and Q from direct illumination only
        Qabs0 = (1 - albedo) * E[:, 0] + Fgeoth
        Tsurf0 = (Qabs0 / (sigSB * emiss)) ** 0.25
        # set up model (with Qprev = Qabs0)
        model = PccThermalModel1D(nfaces=E.shape[0], z=z, T0=210, ti=120., rhoc=960000., # we have rho, but how do we get c?
                                  emissivity=emiss, Fgeotherm=Fgeoth, Qprev=Qabs0.astype(np.double), bcond='Q')
        # loop over time-steps/sun angles
        T = [model.T[:,0]]
        for i in range(len(sun_dirs) - 1):
            # get Q(np1) as in Eq. 17-19
            if i == 0:
                Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], albedo, emiss,
                                                               Qrefl=0, QIR=0, Tsurf=Tsurf0)
            else:
                Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], albedo, emiss,
                                                               Qrefl=Qrefl_np1, QIR=QIR_np1, Tsurf=model.T[:, 0])
            # compute Qabs, eq 19 radiosity paper
            Qnp1 = (1 - albedo) * (E[:, i + 1] + Qrefl_np1) + emiss * QIR_np1
            # extrapolate model at t_{i+1}, get model.T(i+1)
            model.step(stepet, Qnp1)
            # add check for steady-state (model.T(i) - model.T(i-1) < eps) to break loop
            T.append(model.T[:,0])
        # adapt shapes for plotting
        T = np.vstack(T).T
    else:
        # loop over time-steps/sun angles
        for i in range(len(sun_dirs) - 1):
            Qabs0 = (1 - albedo) * E[:, i] + Fgeoth
            Tsurf0 = (Qabs0 / (sigSB * emiss)) ** 0.25
            if not use_svd:
                # get Q(np1) as in Eq. 17-19
                if i == 0:
                    Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], albedo, emiss,
                                                                   Qrefl=0., QIR=0., Tsurf=Tsurf0)
                else:
                    Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], albedo, emiss,
                                                                   Qrefl=Qrefl_np1, QIR=0., Tsurf=Tsurf0)
            else:
                # read outputs of xSVDcomputation
                U = np.loadtxt('svd_U.dat');
                sigma = np.loadtxt('svd_sigma.dat')
                Vt = np.loadtxt('svd_V.dat')

                if i == 0:
                    Qrefl_np1, QIR = update_incoming_radiances_wsvd(E[:, i + 1], albedo, emiss, 0, 0,
                                                                Tsurf0, Vt, sigma, U)
                else:
                    Qrefl_np1, QIR = update_incoming_radiances_wsvd(E[:, i + 1], albedo, emiss, Qrefl_np1, 0,
                                                                Tsurf0, Vt, sigma, U)
            # save Qrefl
            Qrefl.append(Qrefl_np1)
            T.append(Tsurf0)
        # adapt shapes for plotting
        Qrefl = np.vstack(Qrefl).T
        T = np.vstack(T).T

    Path(f"./frames").mkdir(parents=True, exist_ok=True)

    if plot_fluxes:
        for i, sun_dir in enumerate(sun_dirs[:]):
            print('frame = %d' % i)

            fig, ax = tripcolor_vector(V, F, E[:,i], cmap=cc.cm.gray)
            fig.savefig(f"./frames/lsp_E1_%03d.png" % i)
            plt.close(fig)

            fig, ax = tripcolor_vector(V, F, T[:,i], cmap=cc.cm.fire)
            fig.savefig(f"./frames/lsp_T1_%03d.png" % i)
            plt.close(fig)

            I_shadow = E[:,i] == 0
            fig, ax = tripcolor_vector(V, F, T[:,i], I=I_shadow, cmap=cc.cm.rainbow, vmax=100)
            fig.savefig(f"./frames/lsp_T1_shadow_%03d.png" % i)
            plt.close(fig)

    # starting E from second element, to have same number of elements on all arrays
    return {"V":V,"F":F,"E":E[:,1:],'Qrefl':Qrefl,"QIR":QIR,"T":T}

if __name__ == '__main__':

    illuminate_form_factor(plot_fluxes=True)
