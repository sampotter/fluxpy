import itertools as it
import numpy as np

import flux.ingersoll
import flux.shape

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import FormFactorQuadtreeBlock

from flux.form_factors import get_form_factor_matrix

beta = np.deg2rad(25)
rc = 0.75
e0 = np.deg2rad(10)
F0 = 1000
rho = 0.3
emiss = 0.99

crater = flux.ingersoll.HemisphericalCrater(beta, rc, e0, F0, rho, emiss)

V, F = crater.make_trimesh(0.1, return_parts=False)

shape_model = flux.shape.CgalTrimeshShapeModel(V, F)

# make sure normals are pointing up
shape_model.N[shape_model.N[:, 2] < 0] *= -1

max_depth = 3

FF = CompressedFormFactorMatrix(
    shape_model,
    tol=0,
    max_depth=max_depth,
    force_max_depth=True,
    RootBlock=FormFactorQuadtreeBlock)

FF_gt = get_form_factor_matrix(shape_model).toarray()

dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])
E = shape_model.get_direct_irradiance(F0, dir_sun, unit_Svec=True)

B, B_gt = FF@E, FF_gt@E
mvp_rel_error = abs(B - B_gt).max()/abs(B_gt).max()
print(mvp_rel_error)

# check off-diagonal blocks on first level
for i0, j0 in it.product(range(4), repeat=2):
    if i0 == j0:
        continue
    I0 = FF._root._row_block_inds[i0]
    J0 = FF._root._col_block_inds[j0]
    block = FF._root._blocks[i0, j0]
    print(f'i0 = {i0}, j0 = {j0}: {abs(FF_gt[I0, :][:, J0] - block._mat).max()}')

# check off-diagonal blocks on second level
for i0, i1, j1 in it.product(range(4), repeat=3):
    if i1 == j1:
        continue
    j0 = i0
    I0 = FF._root._row_block_inds[i0]
    J0 = FF._root._col_block_inds[j0]
    block0 = FF._root._blocks[i0, j0]
    I1 = block0._row_block_inds[i1]
    J1 = block0._col_block_inds[j1]
    block1 = block0._blocks[i1, j1]
    diff = FF_gt[I0[I1], :][:, J0[J1]] - block1._mat
    print(f'i0 = {i0}: i1 = {i1}, j1 = {j1}: {abs(diff).max()}')

# check diagonal leaf blocks on second level
for i0, i1, j1 in it.product(range(4), repeat=3):
    if i1 != j1:
        continue
    j0 = i0
    I0 = FF._root._row_block_inds[i0]
    J0 = FF._root._col_block_inds[j0]
    block0 = FF._root._blocks[i0, j0]
    I1 = block0._row_block_inds[i1]
    J1 = block0._col_block_inds[j1]
    block1 = block0._blocks[i1, j1]
    if isinstance(block1, FormFactorQuadtreeBlock):
        continue
    diff = FF_gt[I0[I1], :][:, J0[J1]] - block1._mat
    print(f'{type(block1).__name__}: i0 = {i0}: i1 = {i1}, j1 = {j1}: {abs(diff).max()}')
    print(f'* nnz(block1) = {block1._mat.nonzero()[0].size}')
    print(f'* nnz(FF_gt[I0[I1], J0[J1]]) = {FF_gt[I0[I1], :][:, J0[J1]].nonzero()[0].size}')

# check off-diagonal blocks on third level
for i0, i1, i2, j2 in it.product(range(4), repeat=4):
    if i2 == j2:
        continue
    j0 = i0
    j1 = i1
    I0 = FF._root._row_block_inds[i0]
    J0 = FF._root._col_block_inds[j0]
    block0 = FF._root._blocks[i0, j0]
    I1 = block0._row_block_inds[i1]
    J1 = block0._col_block_inds[j1]
    block1 = block0._blocks[i1, j1]
    if not isinstance(block1, FormFactorQuadtreeBlock):
        continue
    I2 = block1._row_block_inds[i2]
    J2 = block1._col_block_inds[j2]
    block2 = block1._blocks[i2, j2]
    diff = FF_gt[I0[I1[I2]], :][:, J0[J1[J2]]] - block2._mat
    print(f'i0 = {i0}, i1 = {i1}: i2 = {i2}, j2 = {j2}: {abs(diff).max()}')
