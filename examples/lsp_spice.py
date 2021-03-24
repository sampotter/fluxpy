#!/usr/bin/env python

'''This script uses SPICE to compute a trajectory for the sun, loads a
shape model discretizing a patch of the lunar south pole (made using
lsp_make_obj.py), and a compressed form factor matrix for that
shape model (computed using lsp_compress_form_factor_matrix.py).
It then proceeds to compute the steady state temperature at each sun
position, writing a plot of the temperature to disk for each sun
position.

'''

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from pathlib import Path

import flux.compressed_form_factors as cff

from flux.model import compute_steady_state_temp, update_incoming_radiances
from flux.plot import tripcolor_vector
from flux.shape import TrimeshShapeModel
from flux.thermal import PccThermalModel1D, setgrid
from scipy.constants import sigma

clktol = '10:000'

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
]).T

# Use these temporary parameters...

F0 = 1365 # Solar constant
emiss = 0.95 # Emissitivity
rho = 0.12 # Visual (?) albedo
Fgeoth = 0.2
subsurf_heat = True

# Load shape model

V = np.load('lsp_V.npy')
F = np.load('lsp_F.npy')
N = np.load('lsp_N.npy')

shape_model = TrimeshShapeModel(V, F, N)

# Load compressed form factor matrix from disk
FF_path = 'lsp_compressed_form_factors.bin'
FF = cff.CompressedFormFactorMatrix.from_file(FF_path)
# print("Getting FF block") # what's this?
# FF_gt = get_form_factor_block(shape_model)

E_arr = []
for i, sun_dir in enumerate(sun_dirs[:]):
    E_arr.append(shape_model.get_direct_irradiance(F0, sun_dir))

E = np.vstack(E_arr).T

if not subsurf_heat:
    # Compute steady state temperature
    T_arr = compute_steady_state_temp(FF, E, rho, emiss)
    T = np.vstack(T_arr).T
else:
    # spin-up model until equilibrium is reached

    # generate 1D grid
    nz = 60
    zfac = 1.05
    zmax = 2.5
    z = setgrid(nz=nz, zfac=zfac, zmax=zmax)

    # compute provisional Tsurf and Q from direct illumination only
    Qabs0 = (1 - rho) * E[:, 0] + Fgeoth
    Tsurf0 = (Qabs0 / (sigma * emiss)) ** 0.25
    # set up model (with Qprev = Qabs0)
    model = PccThermalModel1D(nfaces=E.shape[0], z=z, T0=210, ti=120., rhoc=960000., # we have rho, but how do we get c?
                              emissivity=emiss, Fgeotherm=Fgeoth, Qprev=Qabs0.astype(np.double), bcond='Q')
    # loop over time-steps/sun angles
    T = [model.T[:,0]]
    for i in range(len(sun_dirs) - 1):
        # get Q(np1) as in Eq. 17-19
        if i == 0:
            Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], rho, emiss,
                                                           Qrefl=0, QIR=0, Tsurf=Tsurf0)
        else:
            Qrefl_np1, QIR_np1 = update_incoming_radiances(FF, E[:, i + 1], rho, emiss,
                                                           Qrefl=Qrefl_np1, QIR=QIR_np1, Tsurf=model.T[:, 0])
        # compute Qabs, eq 19 radiosity paper
        Qnp1 = (1 - rho) * (E[:, i + 1] + Qrefl_np1) + emiss * QIR_np1
        # extrapolate model at t_{i+1}, get model.T(i+1)
        model.step(stepet, Qnp1)
        # add check for steady-state (model.T(i) - model.T(i-1) < eps) to break loop
        T.append(model.T[:,0])
    # adapt shapes for plotting
    T = np.vstack(T).T

Path('./frames').mkdir(parents=True, exist_ok=True)

for i, sun_dir in enumerate(sun_dirs[:]):
    print('frame = %d' % i)

    fig, ax = tripcolor_vector(V, F, E[:,i], cmap=cc.cm.gray)
    fig.savefig('./frames/lsp_E1_%03d.png' % i)
    plt.close(fig)

    fig, ax = tripcolor_vector(V, F, T[:,i], cmap=cc.cm.fire)
    fig.savefig('./frames/lsp_T1_%03d.png' % i)
    plt.close(fig)

    I_shadow = E[:,i] == 0
    fig, ax = tripcolor_vector(V, F, T[:,i], I=I_shadow, cmap=cc.cm.rainbow, vmax=100)
    fig.savefig('./frames/lsp_T1_shadow_%03d.png' % i)
    plt.close(fig)
