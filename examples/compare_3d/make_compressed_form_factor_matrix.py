#!/usr/bin/env python

import numpy as np
import scipy.sparse
import sys
import trimesh

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import \
    CompressedFormFactorMatrix, \
    FormFactorQuadtreeBlock, \
    FormFactorOctreeBlock
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc

obj_path = sys.argv[1]
FF_path = sys.argv[2]
tol_str = sys.argv[3]
tol = float(tol_str)
tree_type = sys.argv[4]

tm = trimesh.load(obj_path)

verts = np.array(tm.vertices).astype(np.float32)
faces = np.array(tm.faces).astype(np.uintp)
normals = get_surface_normals(verts, faces)
normals[np.sum(normals**2, axis=1) < 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)

if tree_type == 'octree':
    RootBlock = FormFactorOctreeBlock
elif tree_type == 'quadtree':
    RootBlock = FormFactorQuadtreeBlock
else:
    raise ValueError(f'unexpected tree type: {tree_type}')

# use quadtree by default
tic()
FF = CompressedFormFactorMatrix(shape_model, tol=tol, min_size=1024, RootBlock=RootBlock)
assembly_time = toc()

# just print the assembly time to a file for now
#
# not ideal...  we can figure a better solution out later
with open(FF_path[:-4] + '_assembly_time.txt', 'w') as f:
    print(assembly_time, file=f)

FF.save(FF_path)
