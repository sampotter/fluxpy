#!/usr/bin/env python

import itertools as it
import numpy as np
import sys

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.model import compute_steady_state_temp
from flux.util import tic, toc

max_inner_area_str = sys.argv[1]
max_outer_area_str = sys.argv[2]

tol_str = sys.argv[3]
tol = float(tol_str)

F0 = float(sys.argv[4])
rho = float(sys.argv[5])
emiss = float(sys.argv[6])

az = float(sys.argv[7])
el = float(sys.argv[8])
phi, theta = np.deg2rad(az), np.deg2rad(el)
sundir = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

V = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
F = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

N = get_surface_normals(V, F)
N[N[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(V, F, N)

def get_values_using_raytracing(field):
    assert field.ndim == 1 and field.size == F.shape[0]

    x0, y0, z0 = V.min(0)
    x1, y1, z1 = V.max(0)
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

if tol_str == 'true':
    FF_path = f'FF_{max_inner_area_str}_{max_outer_area_str}_true.npy'
    FF = scipy.sparse.load_npz(FF_path)
else:
    FF_path = f'FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.bin'
    FF = CompressedFormFactorMatrix.from_file(FF_path)

tic()
E = shape_model.get_direct_irradiance(F0, sundir)
time_E = toc()

tic()
T = compute_steady_state_temp(FF, E, rho, emiss)
time_T = toc()

T_grid = get_values_using_raytracing(T)
