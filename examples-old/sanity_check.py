import itertools as it
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import pyvistaqt as pvqt

plt.ion()

np.seterr('raise')

import flux.compressed_form_factors as cff
import flux.config
import flux.form_factors as ff
import flux.shape

from flux.linalg import nbytes

flux.config.DEBUG = True

############################################################################
# Testing parameters:

tol = 1e-2

V, F = meshzoo.icosa_sphere(5)

# stretch out the sphere
V[:, 0] *= 1
V[:, 1] *= 1
V[:, 2] *= 1

shape_model = flux.shape.CgalTrimeshShapeModel(V, F)
# shape_model = flux.shape.EmbreeTrimeshShapeModel(V, F)

# make surface normals inward facing
outward = (shape_model.P*shape_model.N).sum(1) > 0
shape_model.N[outward] *= -1

# make surface normals outward facing
# shape_model.N *= -1

E = np.ones(shape_model.num_faces)

FF_gt = ff.get_form_factor_matrix(shape_model)
B1_gt = FF_gt@E # compute first order of scattered radiance

# import ipdb; ipdb.set_trace()
FF = cff.CompressedFormFactorMatrix(
    shape_model,
    tol=0,
    max_depth=2,
    force_max_depth=True,
    RootBlock=cff.FormFactorOctreeBlock)

B1 = FF@E

# grid = shape_model.get_pyvista_unstructured_grid()
# grid['B1 - B1_gt'] = B1 - B1_gt
# plotter = pvqt.BackgroundPlotter()
# plotter.add_mesh(grid, scalars='B1 - B1_gt')

print(f'depth: {FF.depth}')
print(f'error: ({(B1 - B1_gt).min()}, {(B1 - B1_gt).max()})')
print(f'FF size: {FF.nbytes/1024**2} MB')
print(f'FF gt size: {nbytes(FF_gt)/1024**2} MB')

########################################################################

# grid['inds'] = np.zeros(shape_model.num_faces)
# grid['inds'][291] = 1
# grid['inds'][[36, 38, 39]] = 2

# plotter = pvqt.BackgroundPlotter()
# plotter.add_mesh(grid, scalars='inds')

########################################################################
# testing a block

for i, j in it.product(range(8), repeat=2):
    I = FF._root._row_block_inds[i]
    J = FF._root._col_block_inds[j]

    block_gt = FF_gt[I, :][:, J].toarray()
    block = FF._root._blocks[i, j].toarray()

    I_bad, J_bad = np.where(block - block_gt)

    print(f'i = {i}, j = {j}, I bad: {I_bad}, J bad: {J_bad}')

    if len(I_bad) > 0:
        break

########################################################################
# test occlusion

D = np.random.randn(3)
D /= np.linalg.norm(D)

occluded = shape_model.is_occluded(np.arange(shape_model.num_faces), D)

grid = shape_model.get_pyvista_unstructured_grid()
grid['occluded'] = occluded

plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(grid, scalars='occluded')

########################################################################
# testing indices of 2nd level blocks...

i, j = 3, 3
I, J = FF._root._row_block_inds[i], FF._root._col_block_inds[j]
block = FF._root._blocks[i, j]
ii, jj = 2, 5
II, JJ = block._row_block_inds[ii], block._col_block_inds[jj]
subblock = block._blocks[ii, jj]
