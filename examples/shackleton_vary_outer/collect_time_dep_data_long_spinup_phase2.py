import scipy.sparse
import os

import numpy as np
from tqdm import tqdm

from spice_util import get_sunvec
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.model import ThermalModel
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

from flux.thermal import setgrid

import argparse
import arrow

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd", choices=["svd", "rand_svd", "aca", "rand_id", "paige", "sparse_tol", "sparse_k", "true_model"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)

parser.add_argument('--tol', type=float, default=1e-1)
parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)
parser.add_argument('--compress_sparse', action='store_true')

parser.add_argument('--add_residuals', action='store_true')
parser.add_argument('--k0', type=int, default=40)
parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--paige_mult', type=int, default=1)
parser.add_argument('--sparse_mult', type=int, default=1)

parser.add_argument('--cliques', action='store_true')
parser.add_argument('--n_cliques', type=int, default=25)
parser.add_argument('--obb', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()


compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
tol_str = "{:.0e}".format(args.tol)

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "true_model":
    FF_dir = "true_{}_{}".format(max_area_str, outer_radius_str)

elif compression_type == "paige":
    FF_dir = "paige_{}_{}_{}k".format(max_area_str, outer_radius_str,
        args.paige_mult)

elif compression_type == "sparse_tol":
    FF_dir = "sparse_{}_{}_{:.0e}".format(max_area_str, outer_radius_str,
        args.tol)

elif compression_type == "sparse_k":
    FF_dir = "sparse_{}_{}_{}k".format(max_area_str, outer_radius_str,
        args.sparse_mult)

elif compression_type == "svd" or compression_type == "aca":
    if args.add_residuals:
        FF_dir = "{}_resid_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)
    else:
        FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
            args.k0)

elif compression_type == "rand_svd" or compression_type == "rand_id":
    if args.add_residuals:
        FF_dir = "{}_resid_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
            args.p, args.q, args.k0)
    else:
        FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
            args.p, args.q, args.k0)


if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.compress_sparse:
    FF_dir += "_cs"

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.cliques:
    FF_dir = "results_cliques/"+FF_dir
    FF_dir = FF_dir+"_{}nc".format(args.n_cliques)
else:
    FF_dir = "results/"+FF_dir

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.cliques and args.obb:
    FF_dir = FF_dir + "_obb"

if not os.path.exists(FF_dir):
    print("PATH DOES NOT EXIST "+FF_dir)
    assert False
savedir = FF_dir + "/T_frames_long"
if not os.path.exists(savedir):
    os.mkdir(savedir)


# read shapemodel and form-factor matrix generated by make_compressed_form_factor_matrix.py
if compression_type == 'true_model' or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k":
    path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}.npz'
    FF = scipy.sparse.load_npz(path)
    V = np.load(f'shackleton_verts_{max_area_str}_{outer_radius_str}.npy')
    F = np.load(f'shackleton_faces_{max_area_str}_{outer_radius_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
else:
    if args.add_residuals:
        path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}_resid.bin'
    else:
        path = FF_dir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'
    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model

print('  * loaded form factor matrix and (cartesian) shape model')

# Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)

if not os.path.exists(FF_dir + "/T_long_spinup.npy"):
    raise RuntimeError("Need to run step 1 of spin up first!")
else:
    T_spinup = np.load(FF_dir + "/T_long_spinup.npy")
    T_surf_mean = T_spinup[:,0]
    T_init = np.repeat(T_surf_mean[:,np.newaxis], T_spinup.shape[1], axis=-1)

utc0 = '2000 JAN 01 12:00:00.00'
utc1 = '2000 OCT 22 12:20:00.00'
num_frames = 28801
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
    z=z, T0=T_init, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q', shape_model=shape_model, return_flux=True)

sim_start_time = arrow.now()
for frame_index, (T, E, _, _) in tqdm(enumerate(thermal_model), total=D.shape[0], desc='thermal models time-steps'):
    if (frame_index % 2880) == 0 or ((frame_index < 2880) and (frame_index % 20 == 0)):
        
        path = savedir+"/T{:03d}.npy".format(frame_index)
        np.save(path, T)
        
        path = savedir+"/E{:03d}.npy".format(frame_index)
        np.save(path, E)
sim_duration = (arrow.now()-sim_start_time).total_seconds()
print('  * thermal model run completed in {:.2f} seconds'.format(sim_duration))

np.save(savedir+f'/sim_duration_long.npy', np.array(sim_duration))