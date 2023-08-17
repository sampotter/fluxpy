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

import argparse
import arrow

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "true_model"])
parser.add_argument('--max_inner_area', type=float, default=0.8)
parser.add_argument('--max_outer_area', type=float, default=3.0)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.set_defaults(feature=False)

args = parser.parse_args()

# some useful routines
# transform cartesian to spherical (meters, radians)
def cart2sph(xyz):

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp

def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))

def project_stereographic(lon, lat, lon0, lat0, R=1):
    """
    project cylindrical coordinates to stereographic xy from central lon0/lat0
    :param lon: array of input longitudes (deg)
    :param lat: array of input latitudes (deg)
    :param lon0: center longitude for the projection (deg)
    :param lat0: center latitude for the projection (deg)
    :param R: planetary radius (km)
    :return: stereographic projection xy coord from center (km)
    """

    cosd_lat = cosd(lat)
    cosd_lon_lon0 = cosd(lon - lon0)
    sind_lat = sind(lat)

    k = (2. * R) / (1. + sind(lat0) * sind_lat + cosd(lat0) * cosd_lat * cosd_lon_lon0)
    x = k * cosd_lat * sind(lon - lon0)
    y = k * (cosd(lat0) * sind_lat - sind(lat0) * cosd_lat * cosd_lon_lon0)

    return x, y

# ============================================================
# main code

compression_type = args.compression_type
max_inner_area_str = str(args.max_inner_area)
max_outer_area_str = str(args.max_outer_area)
tol_str = "{:.0e}".format(args.tol)

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "true_model":
    FF_dir = "true_{}_{}".format(max_inner_area_str, max_outer_area_str)

elif compression_type == "svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, args.tol,
        args.k0)

elif compression_type == "ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, args.tol,
        args.k0)

elif compression_type == "rand_svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "rand_ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "nmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_inner_area_str, max_outer_area_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_inner_area_str, max_outer_area_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "rand_snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)

elif compression_type == "wsnmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_inner_area_str, max_outer_area_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


if not compression_type == "true_model" and args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if not compression_type == "true_model" and max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)


FF_dir = "results/"+FF_dir
if not os.path.exists(FF_dir):
    print("PATH DOES NOT EXIST "+FF_dir)
    assert False
savedir = FF_dir + "/T_frames"
if not os.path.exists(savedir):
    os.mkdir(savedir)


# read shapemodel and form-factor matrix generated by make_compressed_form_factor_matrix.py
if compression_type == 'true_model':
    path = FF_dir+f'/FF_{max_inner_area_str}_{max_outer_area_str}.npz'
    FF = scipy.sparse.load_npz(path)
    V = np.load(f'faustini_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
    F = np.load(f'faustini_faces_{max_inner_area_str}_{max_outer_area_str}.npy')
    N = get_surface_normals(V, F)
    N[N[:, 2] > 0] *= -1
    shape_model = CgalTrimeshShapeModel(V, F, N)
else:
    path = FF_dir+f'/FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}_{compression_type}.bin'
    FF = CompressedFormFactorMatrix.from_file(path)
    shape_model = FF.shape_model

print('  * loaded form factor matrix and (cartesian) shape model')

# choose simulation parameters
with open('params.json') as f:
    params = json.load(f)


# Define time window (it can be done either with dates or with utc0 - initial epoch - and np.linspace of epochs)

# utc0 = '2011 MAR 01 00:00:00.00'
# utc1 = '2011 MAR 02 00:00:00.00'
# num_frames = 100
# stepet = 86400/100
# sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet, path_to_furnsh="simple.furnsh",
#                       target='SUN', observer='MOON', frame='MOON_ME')
# t = np.linspace(0, 86400, num_frames + 1)

utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2011 MAR 31 00:00:00.00'
num_frames = 3000
stepet = 2592000/3000
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet, path_to_furnsh="simple.furnsh",
                      target='SUN', observer='MOON', frame='MOON_ME')
t = np.linspace(0, 2592000, num_frames + 1)

D = sun_vecs/np.linalg.norm(sun_vecs, axis=1)[:, np.newaxis]
D = D.copy(order='C')

print('  * got sun positions from SPICE')

z = np.linspace(0, 3e-3, 31)

print('  * set up thermal model')
thermal_model = ThermalModel(
    FF, t, D,
    F0=np.repeat(1365, len(D)), rho=0.11, method='1mvp',
    z=z, T0=100, ti=120, rhoc=9.6e5, emiss=0.95,
    Fgeotherm=0.2, bcond='Q', shape_model=shape_model)

Tmin, Tmax = np.inf, -np.inf
vmin, vmax = 90, 310

sim_start_time = arrow.now()
for frame_index, T in tqdm(enumerate(thermal_model), total=D.shape[0], desc='thermal models time-steps'):
    # print(f'    + {frame_index + 1}/{D.shape[0]}')
    path = savedir+"/T{:03d}.npy".format(frame_index)
    np.save(path, T)
sim_duration = (arrow.now()-sim_start_time).total_seconds()
print('  * thermal model run completed in {:.2f} seconds'.format(sim_duration))

np.save(savedir+f'/sim_duration.npy', np.array(sim_duration))

# retrieve T at surface for all epochs and compute max and mean along epochs
Tfiles = glob.glob(savedir+f"/T*.npy")

Tsurf = []
for f in Tfiles:
    Tsurf.append(np.load(f)[:, 0])
Tsurf = np.vstack(Tsurf)

Tsurf_max = np.max(Tsurf, axis=0) # max along epochs
Tsurf_mean = np.mean(Tsurf, axis=0) # mean along epochs

# plot
# generate stereographic counterpart to shape_model (for plotting)
Rp = 1737.4e3
r, lat, lon = cart2sph(shape_model.V)
x, y = project_stereographic(np.rad2deg(lon), np.rad2deg(lat), lon0=0, lat0=-90., R=1737.4e3)
V_st = np.vstack([x, y, r-Rp]).T
shape_model_st = CgalTrimeshShapeModel(V_st.copy(order='C'), shape_model.F)

# save mesh to file
mesh = meshio.Mesh(shape_model_st.V, [('triangle', shape_model_st.F)])
fout = f'faustini_{max_inner_area_str}_{max_outer_area_str}_st.ply'
mesh.write(fout)

# read with pyvista and plot Tmax and Tmean at surface
grid = pv.read(fout)
grid.cell_data['Tmax'] = Tsurf_max
grid.plot()

grid.cell_data['Tmean'] = Tsurf_mean
grid.plot()
