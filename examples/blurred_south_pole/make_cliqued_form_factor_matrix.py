import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import CompressedFormFactorMatrix, FormFactorPartitionBlock, FormFactorQuadtreeBlock, FormFactorObbQuadtreeBlock
from flux.shape import CgalTrimeshShapeModel, get_surface_normals

import argparse
import arrow
import os

from copy import copy

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd", choices=["svd", "rand_svd", "aca", "rand_id"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--sigma', type=float, default=5.0)

parser.add_argument('--tol', type=float, default=1e-1)
parser.add_argument('--max_depth', type=int, default=0)
parser.add_argument('--compress_sparse', action='store_true')

parser.add_argument('--add_residuals', action='store_true')
parser.add_argument('--k0', type=int, default=40)
parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

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
sigma_str = str(args.sigma)
tol_str = "{:.0e}".format(args.tol)
tol = args.tol

max_depth = args.max_depth if args.max_depth != 0 else None


if compression_type == "svd" or compression_type == "aca":
    compression_params = {
        "k0": args.k0
    }

    prefix = compression_type + "_resid" if args.add_residuals else compression_type

    savedir = "{}_{}_{}_{}_{:.0e}_{}k0".format(prefix, max_area_str, outer_radius_str, sigma_str, tol,
        args.k0)


elif compression_type == "rand_svd" or compression_type == "rand_id":
    compression_params = {
        "k0": args.k0,
        "p": args.p,
        "q": args.q
    }

    prefix = compression_type + "_resid" if args.add_residuals else compression_type

    savedir = "{}_{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(prefix, max_area_str, outer_radius_str, sigma_str, tol,
        args.p, args.q, args.k0)




if max_depth is not None:
    savedir += "_{}maxdepth".format(max_depth)

if args.compress_sparse:
    savedir += "_cs"

savedir = "results_cliques/"+savedir
savedir = savedir+"_{}nc".format(args.n_cliques)
if args.obb:
    savedir += "_obb"
if not os.path.exists('results_cliques'):
    os.mkdir('results_cliques')
if not os.path.exists(savedir):
    os.mkdir(savedir)


if not args.overwrite:
    if args.add_residuals:
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}_resid.bin'):
            raise RuntimeError("Compressed FF already exists!")
    else:
        if os.path.exists(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin'):
            raise RuntimeError("Compressed FF already exists!")


verts = np.load(f'blurred_pole_verts_{max_area_str}_{outer_radius_str}_{sigma_str}.npy')
faces = np.load(f'blurred_pole_faces_{max_area_str}_{outer_radius_str}_{sigma_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)


start_assembly_time = arrow.now()

if args.load_ff:
    path = f"results/true_{max_area_str}_{outer_radius_str}_{sigma_str}/FF_{max_area_str}_{outer_radius_str}.npz"
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
            shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params, parts=current_clique_list,
            RootBlock=FormFactorPartitionBlock, ChildBlock=FormFactorObbQuadtreeBlock, truncated_sparse=args.compress_sparse, add_residuals=args.add_residuals)
else:
    FF = CompressedFormFactorMatrix(
            shape_model, tol=tol, min_size=16384, max_depth=max_depth, compression_type=compression_type, compression_params=compression_params, parts=current_clique_list,
            RootBlock=FormFactorPartitionBlock, ChildBlock=FormFactorQuadtreeBlock, truncated_sparse=args.compress_sparse, add_residuals=args.add_residuals)

assembly_time = (arrow.now() - start_assembly_time).total_seconds()

np.save(savedir+f'/FF_assembly_time.npy', np.array(assembly_time))

if args.add_residuals:
    FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}_resid.bin')
else:
    FF.save(savedir+f'/FF_{max_area_str}_{outer_radius_str}_{tol_str}_{compression_type}.bin')