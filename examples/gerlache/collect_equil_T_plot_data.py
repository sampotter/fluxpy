#!/usr/bin/env python

import colorcet as cc
import glob
import itertools as it
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

plt.ion()

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.model import compute_steady_state_temp
from flux.util import tic, toc

from parula import parula_cmap

with open('params.json') as f:
    params = json.load(f)

F0 = params['F0']
rho = params['rho']
emiss = params['emiss']
az = params['az']
el = params['el']
phi, theta = np.deg2rad(az), np.deg2rad(el)
sundir = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

def get_data(shape_model, FF):
    def get_values_using_raytracing(field):
        assert field.ndim == 1 and field.size == shape_model.num_faces

        x0, y0, z0 = shape_model.V.min(0)
        x1, y1, z1 = shape_model.V.max(0)
        xc, yc = (x0 + x1)/2, (y0 + y1)/2
        dx, dy = x1 - x0, y1 - y0
        p = 0.25

        m, n = 512, 512

        dtype = field.dtype

        xg = np.linspace(xc - p*dx, xc + p*dx, m).astype(dtype)
        yg = np.linspace(yc - p*dx, yc + p*dx, n).astype(dtype)
        z = z0 - max(abs(dx), abs(dy))
        d = np.array([0, 0, 1], dtype=dtype)

        grid = np.empty((m, n), dtype=dtype)
        grid[...] = np.nan
        for i, j in it.product(range(m), range(n)):
            x = np.array([xg[i], yg[j], z], dtype=dtype)
            hit = shape_model.intersect1(x, d)
            if hit is not None:
                grid[i, j] = field[hit[0]]

        return grid

    tic()
    E = shape_model.get_direct_irradiance(F0, sundir)
    time_E = toc()

    tic()

    T = compute_steady_state_temp(FF, E, rho, emiss)
    time_T = toc()

    T_grid = get_values_using_raytracing(T)

    return {
        'E': E,
        'time_E': time_E,
        'T': T,
        'time_T': time_T,
        'T_grid': T_grid
    }

data = dict()

FF_paths = glob.glob('FF_*_*_*.bin')
for path in FF_paths:
    max_inner_area, max_outer_area, tol = map(float, path[:-4].split('_')[1:])
    print(max_inner_area, max_outer_area, tol)
    FF = CompressedFormFactorMatrix.from_file(path)
    data[path] = get_data(FF.shape_model, FF)

FF_true_paths = glob.glob('FF_*.npz')
for path in FF_true_paths:
    areas_str = path[3:-4]
    max_inner_area, max_outer_area = map(float, areas_str.split('_'))
    print(max_inner_area, max_outer_area)
    V = np.load(f'gerlache_verts_{areas_str}.npy')
    F = np.load(f'gerlache_faces_{areas_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
    FF_true = scipy.sparse.load_npz(path)
    data[path] = get_data(shape_model, FF_true)

T_grid_hat = data['FF_0.75_3.0_1e-2.bin']['T_grid']
T_grid = data['FF_0.4_3.0.npz']['T_grid']
error = (T_grid - T_grid_hat)/T_grid
perc_error = 100*abs(error)

plt.figure()
plt.imshow(perc_error, interpolation='none', vmax=10, cmap=cc.cm.bmw)
plt.colorbar()
plt.show()
