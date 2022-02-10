#!/usr/bin/env python

import numpy as np
import scipy.sparse
import sys

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import \
    CompressedFormFactorMatrix, \
    FormFactorQuadtreeBlock, \
    FormFactorOctreeBlock
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc

from util import shape_model_from_obj_file

obj_path = sys.argv[1]
FF_path = sys.argv[2]
tol_str = sys.argv[3]
tol = float(tol_str)
tree_type = sys.argv[4]

shape_model = shape_model_from_obj_file(obj_path)

# use quadtree by default
tic()
if tree_type == 'octree':
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, RootBlock=FormFactorOctreeBlock)
elif tree_type == 'quadtree':
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, min_size=16384, RootBlock=FormFactorQuadtreeBlock)
else:
    raise ValueError(f'unexpected tree type: {tree_type}')
assembly_time = toc()

FF.save(FF_path)

# just print the assembly time to a file for now
#
# not ideal...  we can figure a better solution out later
with open(FF_path[:-4] + '_assembly_time.txt', 'w') as f:
    print(assembly_time, file=f)
