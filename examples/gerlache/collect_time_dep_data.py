#!/usr/bin/env python

import os

import colorcet as cc
import itertools as it
import json_numpy as json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import sys

from spice_util import get_sunvec
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import CompressedKernelMatrix
from flux.model import ThermalModel
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.solve import solve_radiosity
from flux.util import tic, toc

from pathlib import Path

outdir = Path('T_frames')
if not outdir.exists():
    outdir.mkdir()

max_inner_area_str = sys.argv[1]
max_outer_area_str = sys.argv[2]
tol_str = sys.argv[3]

if tol_str == 'true':
    path = f'FF_{max_inner_area_str}_{max_outer_area_str}.npz'
    FF = scipy.sparse.load_npz(path)
    V = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
    F = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
else:
    path = f'FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.bin'
    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model

print('  * loaded form factor matrix and shape model')


with open('params.json') as f:
    params = json.load(f)

F0 = params['F0']

frames_per_second = 0.3  # 30
animation_time = 86400*30 # s
num_frames = int(frames_per_second*animation_time + 1)
t = np.linspace(0, animation_time, num_frames)

# Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)

utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2011 MAR 02 00:00:00.00'
num_frames = 100
stepet = 86400/100
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet)
t = np.linspace(0, 86400, num_frames + 1)

D = sun_vecs/np.linalg.norm(sun_vecs,axis=1)[:,np.newaxis]
D = D.copy(order='C')

print('  * got sun positions from SPICE')

z = np.linspace(0, 3e-3, 31)

thermal_model = ThermalModel(
    FF, t, D,
    F0=1365, rho=0.11, method='1mvp',
    z=z, T0=100, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q', shape_model=shape_model)

print('  * set up thermal model')

plot_layers = [0, 1, 2, 3]

Tmin, Tmax = np.inf, -np.inf
vmin, vmax = 90, 310

path_fmt = f'T%0{len(str(num_frames))}d.npy'

print('  * time stepping the thermal model:')
for frame_index, T in enumerate(thermal_model):
    print(f'    + {frame_index + 1}/{D.shape[0]}')
    path = outdir/f'{max_inner_area_str}_{max_outer_area_str}_{tol_str}'
    if not path.exists():
        path.mkdir()
    path = path/(path_fmt % frame_index)
    np.save(path, T)
