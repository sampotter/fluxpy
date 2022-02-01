#!/usr/bin/env python

import numpy as np
import scipy.sparse
import sys
import trimesh

from flux.form_factors import get_form_factor_matrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc

obj_path = sys.argv[1]
FF_path = sys.argv[2]

tm = trimesh.load(obj_path)

verts = np.array(tm.vertices).astype(np.float32)
faces = np.array(tm.faces).astype(np.uintp)
normals = get_surface_normals(verts, faces)
normals[np.sum(normals**2, axis=1) < 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)

tic()
FF = get_form_factor_matrix(shape_model)
assembly_time = toc()

# just print the assembly time to a file for now
#
# not ideal...  we can figure a better solution out later
with open(FF_path[:-4] + '_assembly_time.txt', 'w') as f:
    print(assembly_time, file=f)

scipy.sparse.save_npz(FF_path, FF)
