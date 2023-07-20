#!/usr/bin/env python
import json_numpy as json
import scipy.sparse
import sys
import glob
import pyvista as pv
import os

import meshio
import numpy as np
from tqdm import tqdm

from spice_util import get_sunvec
from flux.compressed_form_factors_nmf import CompressedFormFactorMatrix
from flux.model import ThermalModel
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

from flux.thermal import setgrid

import argparse
import arrow

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "saca","sbrp","rand_sid",
    "stoch_radiosity",
    "true_model"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.add_argument('--cliques', action='store_true')
parser.add_argument('--n_cliques', type=int, default=25)

parser.set_defaults(feature=False)

args = parser.parse_args()


compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
tol_str = "{:.0e}".format(args.tol)

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "true_model":
    FF_dir = "true_{}_{}".format(max_area_str, outer_radius_str)

elif compression_type == "stoch_radiosity":
    FF_dir = "stoch_rad_{}_{}_{}k0".format(max_area_str, outer_radius_str,
        args.k0)

elif compression_type == "svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "rand_svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "rand_ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "nmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "rand_snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)

elif compression_type == "wsnmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "saca":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "sbrp":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "rand_sid":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)


if not (compression_type == "true_model" or compression_type == "stoch_radiosity") and args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if not (compression_type == "true_model" or compression_type == "stoch_radiosity") and max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)


if args.cliques:
    FF_dir = "results_cliques/"+FF_dir
    FF_dir = FF_dir+"_{}nc".format(args.n_cliques)
else:
    FF_dir = "results/"+FF_dir
if not os.path.exists(FF_dir):
    print("PATH DOES NOT EXIST "+FF_dir)
    assert False
savedir = FF_dir


# read shapemodel and form-factor matrix generated by make_compressed_form_factor_matrix.py
if compression_type == 'true_model' or compression_type == 'stoch_radiosity':
    path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}.npz'
    FF = scipy.sparse.load_npz(path)
    V = np.load(f'shackleton_verts_{max_area_str}_{outer_radius_str}.npy')
    F = np.load(f'shackleton_faces_{max_area_str}_{outer_radius_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
else:
    path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'
    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model

print('  * loaded form factor matrix and (cartesian) shape model')

# Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)

utc0 = '2000 JAN 01 12:00:00.00'
utc1 = '2000 JUL 26 00:20:00.00'
num_frames = 20161
stepet = 885
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet, path_to_furnsh="simple_long_spinup.furnsh",
                      target='SUN', observer='MOON', frame='MOON_ME')
t = np.linspace(0, num_frames*stepet, num_frames + 1)

print(t.shape, sun_vecs.shape)

D = sun_vecs/np.linalg.norm(sun_vecs, axis=1)[:, np.newaxis]
D = D.copy(order='C')

print('  * got sun positions from SPICE')

nz = 60
zfac = 1.05
zmax = 2.5
z = setgrid(nz=nz, zfac=zfac, zmax=zmax)
z = np.hstack([0, z])  # add surface layer

print('  * set up thermal model')
thermal_model = ThermalModel(
    FF, t, D,
    F0=np.repeat(1365, len(D)), rho=0.11, method='1mvp',
    z=z, T0=100, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q', shape_model=shape_model)

T_list = []
sim_start_time = arrow.now()
for frame_index, T in tqdm(enumerate(thermal_model), total=D.shape[0], desc='thermal models time-steps'):
    if frame_index > 2880*6:
        T_list.append(T)
    pass
path = savedir+"/T_long_spinup.npy"

T_list = np.array(T_list)
np.save(path, T_list.mean(axis=0))
sim_duration = (arrow.now()-sim_start_time).total_seconds()
print('  * thermal model spinup completed in {:.2f} seconds'.format(sim_duration))

np.save(savedir+f'/long_spinup_duration.npy', np.array(sim_duration))