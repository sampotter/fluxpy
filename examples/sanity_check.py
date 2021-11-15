import itertools as it
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import vtk

plt.ion()

np.seterr('raise')

import flux.form_factors as ff
import flux.compressed_form_factors as cff
import flux.config

from flux.linalg import nbytes
from flux.shape import TrimeshShapeModel

flux.config.DEBUG = True

############################################################################
# Testing parameters:

tol = 1e-2

V, F = meshzoo.icosa_sphere(5)

# stretch out the sphere
V[:, 0] *= 1
V[:, 1] *= 1
V[:, 2] *= 1

shape_model = TrimeshShapeModel(V, F)

# make surface normals inward facing
outward = (shape_model.P*shape_model.N).sum(1) > 0
shape_model.N[outward] *= -1

E = np.ones(shape_model.num_faces)

FF_gt = ff.get_form_factor_matrix(shape_model)
B1_gt = FF_gt@E # compute first order of scattered radiance

# import ipdb; ipdb.set_trace()
FF = cff.CompressedFormFactorMatrix(shape_model, tol=1e-2,
                                    min_size=400, max_depth=1,
                                    RootBlock=cff.FormFactorOctreeBlock)

B1 = FF@E

grid = shape_model.get_pyvista_unstructured_grid()
grid['B1 - B1_gt'] = B1 - B1_gt
plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(grid, scalars='B1 - B1_gt')

print(f'depth: {FF.depth}')
print(f'error: ({(B1 - B1_gt).min()}, {(B1 - B1_gt).max()})')
print(f'FF size: {FF.nbytes/1024**2} MB')
print(f'FF gt size: {nbytes(FF_gt)/1024**2} MB')

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
# vtk test

# import vtk

# verts = vtk.vtkPoints()
# verts.SetData(vtk.util.numpy_support.numpy_to_vtk(shape_model.V))

# faces = vtk.vtkCellArray()
# faces.SetCells(shape_model.num_faces,
#                vtk.util.numpy_support.numpy_to_vtkIdTypeArray(shape_model.F))

# poly_data = vtk.vtkPolyData()
# poly_data.SetPoints(verts)
# poly_data.SetPolys(faces)

# obb_tree = vtk.vtkOBBTree()
# obb_tree.SetDataSet(poly_data)
# obb_tree.BuildLocator()
