import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix, get_form_factor_paige, get_form_factor_sparsified
from flux.compressed_form_factors import CompressedFormFactorMatrix, FormFactorMinDepthQuadtreeBlock
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow
import os

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

parser.add_argument('--overwrite', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()

# produce compressed FF from shapemodel produced by make_mesh.py

compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
tol_str = "{:.0e}".format(args.tol)
tol = args.tol

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "true_model":
    compression_params = {}

    savedir = "true_{}_{}".format(max_area_str, outer_radius_str)


elif compression_type == "paige":
    compression_params = {}

    savedir = "paige_{}_{}_{}k".format(max_area_str, outer_radius_str, args.paige_mult)


elif compression_type == "sparse_tol":
    compression_params = {}

    savedir = "sparse_{}_{}_{:.0e}".format(max_area_str, outer_radius_str, args.tol)


elif compression_type == "sparse_k":
    compression_params = {}

    savedir = "sparse_{}_{}_{}k".format(max_area_str, outer_radius_str, args.sparse_mult)


elif compression_type == "svd" or compression_type == "aca":
    compression_params = {
        "k0": args.k0
    }

    prefix = compression_type + "_resid" if args.add_residuals else compression_type

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(prefix, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "rand_svd" or compression_type == "rand_id":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    prefix = compression_type + "_resid" if args.add_residuals else compression_type

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(prefix, max_area_str, outer_radius_str, tol,
        args.p, args.q, args.k0)




if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.min_depth != 1:
    savedir += "_{}mindepth".format(args.min_depth)

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and max_depth is not None:
    savedir += "_{}maxdepth".format(max_depth)

if not (compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.compress_sparse:
    savedir += "_cs"

savedir = "results/"+savedir
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists(savedir):
    os.mkdir(savedir)


if not args.overwrite:
    if compression_type == "true_model" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k":
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}.npz'):
            raise RuntimeError("Sparse FF already exists!")
    else:
        if args.add_residuals:
            if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}_resid.bin'):
                raise RuntimeError("Compressed FF already exists!")
        else:
            if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'):
                raise RuntimeError("Compressed FF already exists!")


verts = np.load(f'shackleton_verts_{max_area_str}_{outer_radius_str}.npy')
faces = np.load(f'shackleton_faces_{max_area_str}_{outer_radius_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)


start_assembly_time = arrow.now()

if args.compression_type == "true_model":
    FF = get_form_factor_matrix(shape_model)

elif args.compression_type == "paige":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_paige(shape_model, full_sparse_FF, args.paige_mult*np.sqrt(shape_model.F.shape[0]))

elif args.compression_type == "sparse_tol":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_sparsified(full_sparse_FF, tol=args.tol)

elif args.compression_type == "sparse_k":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_sparsified(full_sparse_FF, k=args.sparse_mult*shape_model.F.shape[0])

elif args.min_depth != 1:
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params,
        min_depth=args.min_depth, RootBlock=FormFactorMinDepthQuadtreeBlock, truncated_sparse=args.compress_sparse, add_residuals=args.add_residuals)

else:
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params,
        truncated_sparse=args.compress_sparse, add_residuals=args.add_residuals)

assembly_time = (arrow.now() - start_assembly_time).total_seconds()

np.save(savedir+f'/FF_assembly_time.npy', np.array(assembly_time))


if args.compression_type == "true_model" or args.compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k":
    scipy.sparse.save_npz(savedir+f'/FF_{max_area_str}_{outer_radius_str}', FF)
else:
    if args.add_residuals:
        FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}_resid.bin')
    else:
        FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin')