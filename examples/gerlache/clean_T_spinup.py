#!/usr/bin/env python
import glob
import json
import os
import sys

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import get_surface_normals, CgalTrimeshShapeModel
import colorcet as cc

def compress_iter(path_to_remove, iter, nsteps):

    path_top = f"{path_to_remove}T_{iter}_top.npy"
    path_bottom = f"{path_to_remove}T_{iter}_bottom.npy"
    path_avg = f"{path_to_remove}T_{iter}_avg_per_layer.npy"

    topT = []
    bottomT = []
    avgT = []
    tframes = glob.glob(f"{path_to_remove}/T*_{iter}.npy")
    for t in tframes:
        arr = np.load(t)
        topT.append(arr[:, 0])
        bottomT.append(arr[:, -1])
        avgT.append({"meanT": np.mean(arr, axis=0), "maxT": np.max(arr, axis=0), "minT": np.min(arr, axis=0)})
    np.save(path_top, np.vstack(topT))
    np.save(path_bottom, np.vstack(bottomT))
    np.save(path_avg, np.vstack(avgT))

def extract_and_clean(path_to_remove):

    # count number of spin-up iterations
    niter = len(glob.glob(f"{path_to_remove}/T000_*.npy"))
    nsteps = len(glob.glob(f"{path_to_remove}/T*_0.npy"))

    print(f"- Found #{niter} spin-up iterations in {path_to_remove}, each with {nsteps} time steps")

    # for it in tqdm(range(niter)[:-1]):
    #     compress_iter(path_to_remove, it, nsteps)
    #
    # print("- Removing all iterations except last one")
    # for it in range(niter-1):
    #     if it < niter-1:
    #         [os.remove(f) for f in glob.glob(f"{path_to_remove}T*_{it}.npy")]

    return niter, nsteps


def get_data(field, areas_str, tol_str):
    import json
    import itertools as it

    with open('params.json') as f:
        params = json.load(f)

    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']

    # get data for compressed FF matrices
    print(max_inner_area_str, max_outer_area_str, tol_str)

    # build shape model in stereographic projection
    V_st = np.load(f'gerlache_verts_stereo_{areas_str}.npy')
    V_st = np.concatenate([V_st, np.ones((V_st.shape[0], 1))], axis=1)
    F = np.load(f'gerlache_faces_{areas_str}.npy')
    N_st = get_surface_normals(V_st, F)
    N_st[N_st[:, 2] > 0] *= -1
    shape_model_st = CgalTrimeshShapeModel(V_st, F, N_st)

    def get_values_using_raytracing(field):
        print(field.ndim, 1, field.size, shape_model_st.num_faces)
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

    # set up plot grid
    N = 512
    xgrid = np.linspace(xmin, xmax, N)
    ygrid = np.linspace(ymin, ymax, N)

    return shape_model_st, get_values_using_raytracing(field)


if __name__ == '__main__':

    max_inner_area_str = sys.argv[1]
    max_outer_area_str = sys.argv[2]
    tol_str = sys.argv[3]
    niter = 20
    # nsteps = 863

    path_to_remove = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/"
    # niter, nsteps = extract_and_clean(path_to_remove)

    # print(f"- Found #{niter} spin-up iterations in {path_to_remove}, each with {nsteps} time steps")

    arr = []
    for t in glob.glob(f"{path_to_remove}/T*_{niter-1}.npy"):
        T3d = np.load(t)
        arr.append(T3d[:,0]) # get surface temperature
    arr = np.vstack(arr)

    E0 = np.load(glob.glob(f"{path_to_remove}/E*_{niter - 1}.npy")[0])
    Qrefl0 = np.load(glob.glob(f"{path_to_remove}/Qrefl*_{niter - 1}.npy")[0])
    QIR0 = np.load(glob.glob(f"{path_to_remove}/QIR*_{niter - 1}.npy")[0])

    areas_str = max_inner_area_str + '_' + max_outer_area_str

    # average over epochs
    Tmax = np.max(arr,axis=0)
    Tavg = np.mean(arr,axis=0)

    _, E0_grid = get_data(E0, areas_str, tol_str)
    _, Qrefl0_grid = get_data(Qrefl0, areas_str, tol_str)
    _, QIR0_grid = get_data(QIR0, areas_str, tol_str)
    _, Tmax_grid = get_data(Tmax, areas_str, tol_str)
    shape_model_st, Tavg_grid = get_data(Tavg, areas_str, tol_str)

    if tol_str == 'true':
        # FF_path = f"FF_{areas_str}.npz"
        V = np.load(f'gerlache_verts_{areas_str}.npy')
        F = np.load(f'gerlache_faces_{areas_str}.npy')
        N = get_surface_normals(V, F)
        N[N[:, 2] > 0] *= -1
        shape_model = CgalTrimeshShapeModel(V, F, N)
    else:
        FF_path = f"FF_{areas_str}_{tol_str}.bin"
        FF = CompressedFormFactorMatrix.from_file(FF_path)
        shape_model = FF.shape_model

    dict = {
        'P_st': shape_model_st.P.tolist(),
        'A': shape_model.A.tolist(),
        'E0': E0.tolist(),
        'E0_grid': E0_grid.tolist(),
        'Qrefl0': Qrefl0.tolist(),
        'Qrefl0_grid': Qrefl0_grid.tolist(),
        'QIR0': QIR0.tolist(),
        'QIR0_grid': QIR0_grid.tolist(),
        'Tmax': Tmax.tolist(),
        'Tmax_grid': Tmax_grid.tolist(),
        'Tavg': Tavg.tolist(),
        'Tavg_grid': Tavg_grid.tolist()
    }

    with open(f'{path_to_remove}spinup_T_data.json', 'w') as f:
        json.dump(dict, f)

    # .astype(float)

    # vmax = max(Tmax_grid.max(), Tavg_grid.max())
    # with open('params.json') as f:
    #     params = json.load(f)
    #
    # xmin = params['xmin']
    # xmax = params['xmax']
    # ymin = params['ymin']
    # ymax = params['ymax']
    # extent = xmin, xmax, ymin, ymax
    #
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('$T$ (true)')
    # plt.imshow(Tmax_grid, interpolation='none', extent=extent,
    #            vmin=0, vmax=Tmax_grid.max(), cmap=cc.cm.fire)
    # plt.ylabel('$x$')
    # plt.xlabel('$y$')
    # plt.colorbar()
    # plt.gca().set_aspect('equal')
    # plt.subplot(1, 2, 2)
    # plt.title(r'$T$ (tol = %s - true)' % tol_str)
    # plt.imshow(Tavg_grid, interpolation='none', extent=extent,
    #            vmin=0, vmax=Tavg_grid.max(), cmap=cc.cm.fire, zorder=1)
    # plt.ylabel('$x$')
    # plt.xlabel('$y$')
    # plt.colorbar()
    # plt.gca().set_aspect('equal')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
