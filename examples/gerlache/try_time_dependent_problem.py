import os

import colorcet as cc
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from spice_util import get_sunvec
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import CompressedKernelMatrix
from flux.model import ThermalModel
from flux.solve import solve_radiosity
from flux.util import tic, toc

from PIL import Image

FF = CompressedFormFactorMatrix.from_file('FF.bin')
shape_model = FF.shape_model
V, F = shape_model.V, shape_model.F
dtype = V.dtype

F0 = 1365

frames_per_second = 0.3  # 30
animation_time = 86400*30 # s
num_frames = int(frames_per_second*animation_time + 1)
t = np.linspace(0, animation_time, num_frames)

use_spice = True
if use_spice:
    # Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)
    utc0 = '2011 MAR 01 00:00:00.00'
    # utc1 = '2011 MAR 02 00:00:00.00'
    # stepet = 3*3600
    # sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet)
    sun_vecs = get_sunvec(utc0=utc0, et_linspace=t)
    D = sun_vecs/np.linalg.norm(sun_vecs,axis=1)[:,np.newaxis]
    D = D.copy(order='C')
else:
    Az = np.linspace(0, 360, num_frames, endpoint=False)
    El = -10*np.ones_like(Az)

    Phi, Theta = np.deg2rad(Az), np.deg2rad(El)

    # D[i] = ith sun direction (num_frames x 3)
    D = np.array([
        (np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta))
        for phi, theta in zip(Phi, Theta)
    ])

z = np.linspace(0, 3e-3, 31)

thermal_model = ThermalModel(
    FF, t, D,
    F0=1365, rho=0.11, method='1mvp',
    z=z, T0=100, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q')

plot_layers = [0, 1, 2, 3]

Tmin, Tmax = np.inf, -np.inf
vmin, vmax = 90, 310

for frame_index, T in enumerate(thermal_model):
    Tmin = min(Tmin, T.min())
    Tmax = max(Tmax, T.max())

    print(f'frame {frame_index}: <T> = {T.mean()}, min(T) = {Tmin}, max(T) = {Tmax}')

    # plotting

    x0, y0, z0 = V.min(0)
    x1, y1, z1 = V.max(0)
    xc, yc = (x0 + x1)/2, (y0 + y1)/2
    dx, dy = x1 - x0, y1 - y0
    p = 0.25

    m, n = 512, 512

    xg = np.linspace(xc - p*dx, xc + p*dx, m).astype(dtype)
    yg = np.linspace(yc - p*dx, yc + p*dx, n).astype(dtype)
    z = z0 - max(abs(dx), abs(dy))
    d = np.array([0, 0, 1], dtype=dtype)

    value = np.empty((m, n, len(plot_layers)), dtype=dtype)
    value[...] = np.nan
    for i, j in it.product(range(m), range(n)):
        x = np.array([xg[i], yg[j], z], dtype=dtype)
        hit = shape_model.intersect1(x, d)
        if hit is not None:
            for layer_index, layer in enumerate(plot_layers):
                value[i, j, layer_index] = T[hit[0], layer]

    # save plot to disk
    if not os.path.exists('frame/'):
        os.mkdir('frame/')
    for layer_index, layer in enumerate(plot_layers[::3000]):
        plt.imsave(
            'frame/layer%d_frame%05d.png' % (layer, frame_index + 1),
            value[:, :, layer_index], vmin=vmin, vmax=vmax, cmap=cc.cm.bmy)
