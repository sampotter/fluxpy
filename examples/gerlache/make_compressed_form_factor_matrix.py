#!/usr/bin/env python

import numpy as np
import scipy.sparse
import sys

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc

max_inner_area_str = sys.argv[1]
max_outer_area_str = sys.argv[2]

tol_str = sys.argv[3]
tol = float(tol_str)

verts = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
faces = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)

# use quadtree by default
FF = CompressedFormFactorMatrix(shape_model, tol=tol, min_size=1024)
FF.save(f'FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.bin')
