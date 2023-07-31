#!/usr/bin/env python

import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix, get_form_factor_stochastic_radiosity, get_form_factor_paige, get_form_factor_sparsified
from flux.compressed_form_factors_nmf import CompressedFormFactorMatrix, FormFactorMinDepthQuadtreeBlock
from flux.compressed_form_factors_truncated_sparse import TruncatedHierarchicalFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow
import os

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "aca", "brp", "rand_id","paca",
    "saca","sbrp","rand_sid","spaca",
    "stoch_radiosity","paige","sparse_tol","sparse_k",
    "sparse_hierarch",
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


elif compression_type == "stoch_radiosity":
    compression_params = {}

    savedir = "stoch_rad_{}_{}_{}k0".format(max_area_str, outer_radius_str, args.k0)


elif compression_type == "paige":
    compression_params = {}

    savedir = "paige_{}_{}_{}k".format(max_area_str, outer_radius_str, args.paige_mult)


elif compression_type == "sparse_tol":
    compression_params = {}

    savedir = "sparse_{}_{}_{:.0e}".format(max_area_str, outer_radius_str, args.tol)


elif compression_type == "sparse_hierarch":
    compression_params = {}

    savedir = "sparse_hier_{}_{}_{:.0e}".format(max_area_str, outer_radius_str, args.tol)


elif compression_type == "sparse_k":
    compression_params = {}

    savedir = "sparse_{}_{}_{}k".format(max_area_str, outer_radius_str, args.sparse_mult)


elif compression_type == "svd":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "ssvd":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "rand_svd":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "rand_ssvd":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "nmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_area_str, outer_radius_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "snmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_area_str, outer_radius_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "rand_snmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)


elif compression_type == "wsnmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_area_str, outer_radius_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "aca":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "brp":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "rand_id":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "saca":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "sbrp":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "rand_sid":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "paca":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


elif compression_type == "spaca":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, tol,
        args.k0)


if not (compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and args.min_depth != 1:
    savedir += "_{}mindepth".format(args.min_depth)

if not (compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k") and max_depth is not None:
    savedir += "_{}maxdepth".format(max_depth)


savedir = "results/"+savedir
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists(savedir):
    os.mkdir(savedir)


if (not args.overwrite):
    if compression_type == "true_model" or compression_type == "stoch_radiosity" or compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k":
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}'):
            raise RuntimeError("Sparse FF already exists!")
    else:
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'):
            raise RuntimeError("Compressed FF already exists!")


verts = np.load(f'blurred_pole_verts_{max_area_str}_{outer_radius_str}.npy')
faces = np.load(f'blurred_pole_faces_{max_area_str}_{outer_radius_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)


start_assembly_time = arrow.now()

if args.compression_type == "true_model":
    FF = get_form_factor_matrix(shape_model)

elif args.compression_type == "stoch_radiosity":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_stochastic_radiosity(full_sparse_FF, args.k0)

elif args.compression_type == "paige":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_paige(shape_model, full_sparse_FF, args.paige_mult*np.sqrt(shape_model.F.shape[0]))

elif args.compression_type == "sparse_tol":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_sparsified(full_sparse_FF, tol=args.tol)

elif args.compression_type == "sparse_hierarch":
    FF = TruncatedHierarchicalFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, max_depth=max_depth)

elif args.compression_type == "sparse_k":
    path = f'results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz'
    full_sparse_FF = scipy.sparse.load_npz(path)
    FF = get_form_factor_sparsified(full_sparse_FF, k=args.sparse_mult*shape_model.F.shape[0])

elif args.min_depth != 1:
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params,
        min_depth=args.min_depth, RootBlock=FormFactorMinDepthQuadtreeBlock)

else:
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params)

assembly_time = (arrow.now() - start_assembly_time).total_seconds()

np.save(savedir+f'/FF_assembly_time.npy', np.array(assembly_time))


if args.compression_type == "true_model" or args.compression_type == "stoch_radiosity" or args.compression_type == "paige" or compression_type == "sparse_tol" or compression_type == "sparse_k":
    scipy.sparse.save_npz(savedir+f'/FF_{max_area_str}_{outer_radius_str}', FF)
else:
    FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin')