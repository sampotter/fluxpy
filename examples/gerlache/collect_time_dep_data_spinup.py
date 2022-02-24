#!/usr/bin/env python
import csv
import glob
import time

import json_numpy as json
import numpy as np
import scipy.sparse
import sys

# from tqdm import tqdm
from spice_util import get_sunvec

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.thermal import setgrid
from flux.model import ThermalModel
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

from pathlib import Path

outdir = Path('T_frames')
if not outdir.exists():
    outdir.mkdir()

max_inner_area_str = sys.argv[1]
max_outer_area_str = sys.argv[2]
tol_str = sys.argv[3]
niter = int(sys.argv[4])
from_iter = int(sys.argv[5])

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

# Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)
utc0 = '2001 JAN 01 00:00:00.00'
utc1 = '2001 JAN 30 00:01:00.00'
stepet = 34800 # 86400/3 #/100 # 60*30 #
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet)[:-1]
t = np.linspace(0, sun_vecs.shape[0]*stepet, sun_vecs.shape[0])
num_frames = len(t)

try:
    # get sub-points for extended Sun
    extsun_coord = f"../kernels/coordflux_100pts_outline33_centerlast_R1_F1_stdlimbdark.txt"
    extsun_ = []
    with open(extsun_coord) as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        for row in csvReader:
            extsun_.append([x for x in row if x != ''])
    extsun_ = np.vstack(extsun_)
    print(f"# Sun is an extended source (see {extsun_coord})...")

    sun_veccs = np.repeat(sun_vecs,extsun_.shape[0],axis=0)
    Zs = np.array([0, 0, 1])
    Us = sun_veccs/np.linalg.norm(sun_veccs,axis=1)[:,np.newaxis]
    Vs = np.cross(Zs, Us)
    Ws = np.cross(Us, Vs)
    Rs = 695700.  # Sun radius, km

    extsun_tiled = np.tile(extsun_,(sun_vecs.shape[0],1))
    sun_vecs = sun_veccs + Vs * extsun_tiled[:, 0][:, np.newaxis] * Rs + Ws * extsun_tiled[:, 1][:, np.newaxis] * Rs
except:
    print("# Sun is treated as a point source (directions file not found)...")

D = sun_vecs/np.linalg.norm(sun_vecs,axis=1)[:,np.newaxis]
D = D.copy(order='C')
print('  * got sun positions from SPICE')

# set up grid of subsurface layers
# z = np.linspace(0, 3e-3, 31)
nz = 60
zfac = 1.05
zmax = 2.5
z = setgrid(nz=nz, zfac=zfac, zmax=zmax)
z = np.hstack([0, z])  # add surface layer

# initialize T from previous run or from scratch
if from_iter == 0:
    T0 = 200
else:
    path_fmt = f'T%0{len(str(num_frames))}d_{from_iter-1}.npy'
    path = outdir / f'{max_inner_area_str}_{max_outer_area_str}_{tol_str}'
    num = num_frames-2
    path = path / (path_fmt % num)
    T0 = np.load(path)

# spin model up for niter iterations
iter_time = []
for it in range(niter)[from_iter:]:

    print(f'Iter #{it + 1}/{niter}')
    try:
        print(f"  * T0 reinitialized to an array T{T0.shape}")
        # print(T0)
    except:
        print(f"  * T0={T0}")

    thermal_model = ThermalModel(
        FF, t[:], D[:],
        F0=1365, rho=0.2, method='1mvp',
        z=z, T0=T0, ti=120, rhoc=9.6e5, emiss=0.95,
        Fgeotherm=0.005, bcond='Q', shape_model=shape_model)

    print('  * set up thermal model')

    plot_layers = [0, 1, 2, 3]

    Tmin, Tmax = np.inf, -np.inf
    vmin, vmax = 90, 310

    path_fmt = f'T%0{len(str(num_frames))}d_{it}.npy'

    print('  * time stepping the thermal model')
    start = time.time()
    # for frame_index, T in tqdm(enumerate(thermal_model), total=D.shape[0]):
    for frame_index, T in enumerate(thermal_model):
        # print(f'time step # + {frame_index + 1}/{D.shape[0]}')
        path_dir = outdir/f'{max_inner_area_str}_{max_outer_area_str}_{tol_str}'
        if not path_dir.exists():
            path_dir.mkdir()
        path = path_dir/(path_fmt % frame_index)
        np.save(path, T)

    if (it == 5) | (it == 10):
        print(f"  * reinitialize all layers to surface T mean over all epochs")
        Tf = glob.glob(f"{path_dir}/T*_{it-1}.npy")
        Tsurf = []
        for f in Tf:
            Tsurf.append(np.load(f)[:,0])
        T = np.vstack(Tsurf)
        T0 = np.repeat(np.mean(T,axis=0)[:, np.newaxis], nz + 1, axis=1)
    else:
        T0 = T
    end = time.time()
    print(f"  * spin-up iter ended after {end-start} seconds")
    iter_time.append(end-start)

stats = {
    'F0': F0,
    'num_faces_source': D.shape[0]/t.shape[0],
    'num_faces_mesh': shape_model.F.shape[0],
    'time_steps': D.shape[0],
    't_T': np.mean(np.vstack(iter_time)),
    'niter': niter,
    'from_iter': from_iter,
    'max_inner_area_str': max_inner_area_str,
    'max_outer_area_str': max_outer_area_str,
    'tol_str': tol_str
}
stats_json_path = f'stats_{max_inner_area_str}_{max_outer_area_str}_{tol_str}_{niter}_{from_iter}.json'
with open(stats_json_path, 'w') as f:
    json.dump(stats, f)
print(f'- wrote stats.json to {stats_json_path}')