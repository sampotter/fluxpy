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

from flux.form_factors import get_form_factor_block
from flux.model import compute_steady_state_temp
from flux.plot import tripcolor_vector
from flux.shape import TrimeshShapeModel
from flux.util import tic, toc

clktol = '10:000'

spice.kclear()
spice.furnsh('simple.furnsh')

# Define time window

et0 = spice.str2et('2011 MAR 01 00:00:00.00')
et1 = spice.str2et('2011 APR 01 00:00:00.00')
et = np.linspace(et0, et1, 100, endpoint=False)

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

# Load shape model

V = np.load('lsp_V.npy')
F = np.load('lsp_F.npy')
N = np.load('lsp_N.npy')

shape_model = TrimeshShapeModel(V, F, N)

# Load compressed form factor matrix from disk

FF_path = 'lsp_compressed_form_factors.bin'
FF = cff.CompressedFormFactorMatrix.from_file(FF_path)

FF_gt = get_form_factor_block(shape_model)

# Compute steady state temperature

tic()
E = shape_model.get_direct_irradiance(F0, sun_dirs)
print(f'- got direct irradiances (batched) [{toc()}s]')

tic()
for i, sun_dir in enumerate(sun_dirs):
    E[:, i] = shape_model.get_direct_irradiance(F0, sun_dir)
print(f'- got direct irradiances (regular) [{toc()}s]')

tic()
T = np.empty_like(E)
for i, e in enumerate(E.T):
    T[:, i] = compute_steady_state_temp(FF, e, rho, emiss)
print(f'- computed steady state temps (compressed FF) [{toc()}s]')

tic()
vmin, vmax = T.min(), T.max()
print('saving plots')
for i, t in enumerate(T.T):
    print('- frame = %d' % i)
    plt.figure(figsize=(8, 8))
    plt.tripcolor(*V[:, :2].T, F, t, vmin=vmin, vmax=vmax, cmap=cc.cm.rainbow)
    plt.colorbar()
    plt.savefig('lsp_T_%03d.png' % i)
    plt.close()
print(f'made plots in {toc()}s')

tic()
T_gt = np.empty_like(E)
for i, e in enumerate(E.T):
    T_gt[:, i] = compute_steady_state_temp(FF_gt, e, rho, emiss)
print(f'- computed steady state temps (groundtruth FF) [{toc()}s]')

tic()
vmin_gt, vmax_gt = T_gt.min(), T_gt.max()
print('saving plots (groundtruth)')
for i, t_gt in enumerate(T_gt.T):
    print('- frame = %d' % i)
    plt.figure(figsize=(8, 8))
    plt.tripcolor(*V[:, :2].T, F, t_gt, vmin=vmin_gt, vmax=vmax_gt,
                  cmap=cc.cm.rainbow)
    plt.colorbar()
    plt.savefig('lsp_T_gt_%03d.png' % i)
    plt.close()
print(f'made plots in {toc()}s')

tic()
T_error = T - T_gt
vmax_error = abs(T_error).max()
print('saving plots (error)')
for i, t_error in enumerate(T_error.T):
    print('- frame = %d' % i)
    plt.figure(figsize=(8, 8))
    plt.tripcolor(*V[:, :2].T, F, t_error, vmin=-vmax_error, vmax=vmax_error,
                  cmap=cc.cm.coolwarm)
    plt.colorbar()
    plt.savefig('lsp_T_error_%03d.png' % i)
    plt.close()
print(f'made plots in {toc()}s')
