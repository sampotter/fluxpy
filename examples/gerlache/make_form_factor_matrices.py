import numpy as np
import scipy.sparse

from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc

verts = np.load('gerlache_verts.npy')
faces = np.load('gerlache_faces.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)

# use quadtree by default
tic()
FF = CompressedFormFactorMatrix(shape_model, tol=1e-2, min_size=1024)
FF.save('FF.bin')
del FF
print('- assembled compressed form factor matrix [%1.2f]' % (toc(),))

tic()
FF_gt = get_form_factor_matrix(shape_model)
scipy.sparse.save_npz('FF_gt.npz', FF_gt)
del FF_gt
print('- assembled true form factor matrix [%1.2f]' % (toc(),))
