#!/usr/bin/env python

import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors_cliques import CliquedFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow
import os

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "saca","sbrp","rand_sid"])
parser.add_argument('--max_inner_area', type=float, default=0.8)
parser.add_argument('--max_outer_area', type=float, default=3.0)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.add_argument('--load_ff', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()

# produce compressed FF from shapemodel produced by make_mesh.py

compression_type = args.compression_type
max_inner_area_str = str(args.max_inner_area)
max_outer_area_str = str(args.max_outer_area)
tol_str = "{:.0e}".format(args.tol)
tol = args.tol


if compression_type == "svd":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.k0)


elif compression_type == "ssvd":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.k0)


elif compression_type == "rand_svd":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "rand_ssvd":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.p, args.q, args.k0)


elif compression_type == "nmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_inner_area_str, max_outer_area_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "snmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_inner_area_str, max_outer_area_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "rand_snmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)


elif compression_type == "wsnmf":
    compression_params = {
        "max_iters": args.nmf_max_iters,
        "nmf_tol": args.nmf_tol,
        "k0": args.k0,
        "beta_loss": args.nmf_beta_loss
    }

    savedir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_inner_area_str, max_outer_area_str, tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)


elif compression_type == "saca":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.k0)


elif compression_type == "sbrp":
    compression_params = {
        "k0": args.k0
    }

    savedir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.k0)


elif compression_type == "rand_sid":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    savedir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_inner_area_str, max_outer_area_str, tol,
        args.p, args.q, args.k0)



savedir = "results_cliques/"+savedir
if not os.path.exists('results_cliques'):
    os.mkdir('results_cliques')
if not os.path.exists(savedir):
    os.mkdir(savedir)


verts = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
faces = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)


start_assembly_time = arrow.now()

if args.load_ff:
    path = f"results/true_{max_inner_area_str}_{max_outer_area_str}/FF_{max_inner_area_str}_{max_outer_area_str}.npz"
    FF_uncompressed = scipy.sparse.load_npz(path)
else:
    FF_uncompressed = get_form_factor_matrix(shape_model)

FF = CliquedFormFactorMatrix(
    shape_model, FF_uncompressed, tol=tol, min_size=16384, compression_type=compression_type, compression_params=compression_params)

assembly_time = (arrow.now() - start_assembly_time).total_seconds()

np.save(savedir+f'/FF_assembly_time.npy', np.array(assembly_time))

FF.save(savedir+f'/FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}_{compression_type}.bin')