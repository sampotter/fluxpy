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
import pickle
import spiceypy as spice

import flux.compressed_form_factors as cff

from flux.model import compute_steady_state_temp
from flux.plot import tripcolor_vector
from flux.shape import TrimeshShapeModel

clktol = '10:000'

spice.kclear()
spice.furnsh('simple.furnsh')

# Define time window

utc0 = '2021 FEB 15 00:00:00.00'
utc1 = '2021 MAR 15 00:00:00.00'

et0 = spice.str2et(utc0)
et1 = spice.str2et(utc1)
stepet = 3*24.*3600
nbet = int(np.ceil((et1 - et0) / stepet))
et = np.linspace(et0, et1, nbet, endpoint=False)

# Sun positions over time period

possun = spice.spkpos('SUN', et, 'MOON_ME', 'LT+S', 'MOON')[0]

sun_dirs = possun/np.sqrt(np.sum(possun**2, axis=1)).reshape(possun.shape[0], 1)

# Use these temporary parameters...

F0 = 1365 # Solar constant
emiss = 0.95 # Emissitivity
rho = 0.12 # Visual (?) albedo

# Load shape model

V = np.load('lsp_V.npy')
F = np.load('lsp_F.npy')
N = np.load('lsp_N.npy')

shape_model = TrimeshShapeModel(V, F, N)

# Load compressed form factor matrix from disk

FF_path = 'lsp_compressed_form_factors.bin'
FF = cff.CompressedFormFactorMatrix.from_file(FF_path)

# Compute steady state temperature
E_arr = []
for i, sun_dir in enumerate(sun_dirs[:]):
    E_arr.append(shape_model.get_direct_irradiance(F0, sun_dir))

E = np.vstack(E_arr).T
T_arr = compute_steady_state_temp(FF, E, rho, emiss)
T = np.vstack(T_arr).T

for i, sun_dir in enumerate(sun_dirs[:]):
    print('frame = %d' % i)
    fig, ax = tripcolor_vector(V, F, E[:,i], cmap=cc.cm.gray)
    fig.savefig('lsp_E1_%03d.png' % i)
    plt.close(fig)

    fig, ax = tripcolor_vector(V, F, T[:,i], cmap=cc.cm.fire)
    fig.savefig('lsp_T1_%03d.png' % i)
    plt.close(fig)

    I_shadow = E[:,i] == 0
    fig, ax = tripcolor_vector(V, F, T[:,i], I=I_shadow, cmap=cc.cm.rainbow, vmax=100)
    fig.savefig('lsp_T1_shadow_%03d.png' % i)
    plt.close(fig)
