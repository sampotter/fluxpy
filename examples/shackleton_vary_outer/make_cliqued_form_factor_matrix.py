#!/usr/bin/env python

import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors_nmf import CompressedFormFactorMatrix, FormFactorPartitionBlock, FormFactorQuadtreeBlock, FormFactorObbQuadtreeBlock
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow
import os

from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "aca", "brp", "rand_id",
    "saca","sbrp","rand_sid"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.add_argument('--n_cliques', type=int, default=25)

parser.add_argument('--obb', action='store_true')

parser.add_argument('--load_ff', action='store_true')

parser.add_argument('--overwrite', action='store_true')

parser.set_defaults(feature=False)

args = parser.parse_args()

# produce compressed FF from shapemodel produced by make_mesh.py

compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)
tol_str = "{:.0e}".format(args.tol)
tol = args.tol


if compression_type == "svd":
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



savedir = "results_cliques/"+savedir
savedir = savedir+"_{}nc".format(args.n_cliques)
if args.obb:
    savedir += "_obb"
if not os.path.exists('results_cliques'):
    os.mkdir('results_cliques')
if not os.path.exists(savedir):
    os.mkdir(savedir)


if (not args.overwrite):
    if args.compression_type == "true_model":
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}'):
            raise RuntimeError("Sparse FF already exists!")
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

if args.load_ff:
    path = f"results/true_{max_area_str}_{outer_radius_str}/FF_{max_area_str}_{outer_radius_str}.npz"
    FF_uncompressed = scipy.sparse.load_npz(path)
else:
    FF_uncompressed = get_form_factor_matrix(shape_model)


dmat = FF_uncompressed.A
FF_adj = np.zeros(FF_uncompressed.shape)
FF_adj[dmat >= np.quantile(dmat[dmat>0.], 0.0)] = 1.

all_pseudo_cliques = []
for i in np.random.permutation(FF_adj.shape[0]):
    
    # make sure the seed is not already in a clique
    if np.array([i in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
        continue
    
    nonzero_idx = list((FF_adj[i] > 0).nonzero()[0])
    nonzero_idx.append(i)
    nonzero_idx = np.array(nonzero_idx)
    
    _nonzero_idx = []
    for j in nonzero_idx:
        
        if j == i:
            _nonzero_idx.append(j)
        
        elif not np.array([j in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
            this_nonzero_idx = list((FF_adj[j] > 0).nonzero()[0])
            this_nonzero_idx.append(j)
            this_nonzero_idx = np.array(this_nonzero_idx)
            num_intersecting = np.intersect1d(nonzero_idx, this_nonzero_idx).shape[0]
            
            if num_intersecting >= 0.5*min(nonzero_idx.shape[0], this_nonzero_idx.shape[0]):
                _nonzero_idx.append(j)
    nonzero_idx = np.array(_nonzero_idx)
    
    pseudo_clique = set(list(np.copy(nonzero_idx)))    
    all_pseudo_cliques.append(pseudo_clique)
    
all_pseudo_clique_lists = []
for i in range(len(all_pseudo_cliques)):
    all_pseudo_clique_lists.append(np.array(list(all_pseudo_cliques[i])))
ordered_pseudo_clique_list = []
for new_idx in np.argsort([len(c) for c in all_pseudo_clique_lists])[::-1]:
    ordered_pseudo_clique_list.append(all_pseudo_clique_lists[new_idx])


culled_cliques = []

for i in range(args.n_cliques):    
    culled_cliques.append(np.copy(ordered_pseudo_clique_list[i]))
    
all_small_cliques = []
for i in range(args.n_cliques, len(ordered_pseudo_clique_list)):
    all_small_cliques += list(np.copy(ordered_pseudo_clique_list[i]))
culled_cliques.append(np.array(all_small_cliques))

current_clique_list = copy(culled_cliques)

if np.sum([len(this_clique) for this_clique in current_clique_list]) != FF_adj.shape[0]:
    raise RuntimeError("Cliques do not cover index set!")
else:
    print("Cliques cover index set!")

if args.obb:
    FF = CompressedFormFactorMatrix(
            shape_model, tol=tol, min_size=16384, compression_type=compression_type, compression_params=compression_params, parts=current_clique_list,
            RootBlock=FormFactorPartitionBlock, ChildBlock=FormFactorObbQuadtreeBlock)
else:
    FF = CompressedFormFactorMatrix(
            shape_model, tol=tol, min_size=16384, compression_type=compression_type, compression_params=compression_params, parts=current_clique_list,
            RootBlock=FormFactorPartitionBlock, ChildBlock=FormFactorQuadtreeBlock)

assembly_time = (arrow.now() - start_assembly_time).total_seconds()

np.save(savedir+f'/FF_assembly_time.npy', np.array(assembly_time))

FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin')