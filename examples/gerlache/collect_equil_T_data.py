#!/usr/bin/env python

import colorcet as cc
import glob
import itertools as it
import json_numpy as json
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

xmin = params['xmin']
xmax = params['xmax']
ymin = params['ymin']
ymax = params['ymax']
F0 = params['F0']
rho = params['rho']
emiss = params['emiss']
az = params['az']
el = params['el']
phi, theta = np.deg2rad(az), np.deg2rad(el)
sundir = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

def get_data(shape_model, shape_model_st, FF, xgrid, ygrid):
    def get_values_using_raytracing(field):
        assert field.ndim == 1 and field.size == shape_model_st.num_faces
        dtype = field.dtype
        d = np.array([0, 0, 1], dtype=np.float64)
        m, n = len(xgrid), len(ygrid)
        grid = np.empty((m, n), dtype=dtype)
        grid[...] = np.nan
        for i, j in it.product(range(m), range(n)):
            x = np.array([xgrid[i], ygrid[j], -1], dtype=dtype)
            hit = shape_model_st.intersect1(x, d)
            if hit is not None:
                grid[i, j] = field[hit[0]]
        return grid

    tic()
    E = shape_model.get_direct_irradiance(F0, sundir)
    time_E = toc()

    tic()

    T = compute_steady_state_temp(FF, E, rho, emiss)
    time_T = toc()
    print(time_T)

    T_grid = get_values_using_raytracing(T)

    return {
        'P_st': shape_model_st.P,
        'A': shape_model.A,
        'E': E,
        'time_E': time_E,
        'T': T,
        'time_T': time_T,
        'T_grid': T_grid
    }

data = dict()

# get data for compressed FF matrices

# FF_paths = glob.glob('FF_0.8_3.0_1e-2.bin')
FF_paths = glob.glob('FF_*_*_*.bin')

for path in FF_paths[:]:

    max_inner_area_str, max_outer_area_str, tol_str = path[3:-4].split('_')
    print(max_inner_area_str, max_outer_area_str, tol_str)

    areas_str = max_inner_area_str + '_' + max_outer_area_str

    # build shape model in stereographic projection
    V_st = np.load(f'gerlache_verts_stereo_{areas_str}.npy')
    V_st = np.concatenate([V_st, np.ones((V_st.shape[0], 1))], axis=1)
    F = np.load(f'gerlache_faces_{areas_str}.npy')
    N_st = get_surface_normals(V_st, F)
    N_st[N_st[:, 2] > 0] *= -1
    shape_model_st = CgalTrimeshShapeModel(V_st, F, N_st)

    # set up plot grid
    N = 512
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)

    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model
    data[path] = get_data(shape_model, shape_model_st, FF, x, y)

# get data from true FF matrices

# FF_true_paths = glob.glob('FF_0.8_3.0.npz')
FF_true_paths = glob.glob('FF_*_*.npz')

for path in FF_true_paths[:]:
    max_inner_area, max_outer_area = map(float, areas_str.split('_'))
    print(max_inner_area, max_outer_area)

    areas_str = path[3:-4]

    # build shape model in stereographic projection
    V_st = np.load(f'gerlache_verts_stereo_{areas_str}.npy')
    V_st = np.concatenate([V_st, np.ones((V_st.shape[0], 1))], axis=1)
    F = np.load(f'gerlache_faces_{areas_str}.npy')
    N_st = get_surface_normals(V_st, F)
    N_st[N_st[:, 2] > 0] *= -1
    shape_model_st = CgalTrimeshShapeModel(V_st, F, N_st)

    # set up plot grid
    N = 512
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)

    # build regular shape model
    V = np.load(f'gerlache_verts_{areas_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)

    FF_true = scipy.sparse.load_npz(path)
    data[path] = get_data(shape_model, shape_model_st, FF_true, x, y)

with open('equil_T_data.json', 'w') as f:
    json.dump(data, f)
