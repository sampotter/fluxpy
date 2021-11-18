import itertools as it
import numpy as np
import unittest

from pathlib import Path

import flux.shape
import tests.common

from flux.compressed_form_factors import \
    CompressedFormFactorMatrix, \
    FormFactorDenseBlock, \
    FormFactorOctreeBlock, \
    FormFactorZeroBlock
from flux.form_factors import get_form_factor_matrix

class CompressedFormFactorMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_max_depth_1_for_stretched_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                tests.common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                # randomly stretch the sphere along each axis
                scale = np.random.random(3)
                V *= scale

                shape_model = TrimeshShapeModel(V, F)

                # Form factor matrix tests for the sphere mesh with
                # inward facing normals

                outward = (shape_model.P*shape_model.N).sum(1) > 0
                shape_model.N[outward] *= -1

                FF = CompressedFormFactorMatrix(
                    shape_model,
                    tol=0,
                    max_depth=1,
                    force_max_depth=True,
                    RootBlock=FormFactorOctreeBlock)

                FF_gt = get_form_factor_matrix(shape_model).toarray()

                # hierarchical matrix has correct structure
                self.assertEqual(FF.depth, 1)
                for i, j in it.product(range(8), repeat=2):
                    block = FF._root._blocks[i, j]
                    self.assertTrue(block.is_leaf)
                    self.assertIsInstance(block, FormFactorDenseBlock)

                # leaf blocks agree with groundtruth
                for i, j in it.product(range(8), repeat=2):
                    block = FF._root._blocks[i, j]
                    I = FF._root._row_block_inds[i]
                    J = FF._root._col_block_inds[j]
                    block_gt = FF_gt[I, :][:, J]
                    self.assertTrue((block._mat == block_gt).all())


    def test_max_depth_2_for_stretched_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                tests.common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                print(TrimeshShapeModel, npz_filename)

                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                # randomly stretch the sphere along each axis
                scale = np.random.random(3)
                V *= scale

                shape_model = TrimeshShapeModel(V, F)

                # Form factor matrix tests for the sphere mesh with
                # inward facing normals

                outward = (shape_model.P*shape_model.N).sum(1) > 0
                shape_model.N[outward] *= -1

                FF = CompressedFormFactorMatrix(
                    shape_model,
                    tol=0,
                    max_depth=2,
                    force_max_depth=True,
                    RootBlock=FormFactorOctreeBlock)

                FF_gt = get_form_factor_matrix(shape_model).toarray()

                # hierarchical matrix has correct structure
                self.assertEqual(FF.depth, 2)
                for i, j in it.product(range(8), repeat=2):
                    block = FF._root._blocks[i, j]
                    if i == j:
                        self.assertIsInstance(block, FormFactorOctreeBlock)
                        for ii, jj in it.product(range(8), repeat=2):
                            subblock = block._blocks[ii, jj]
                            self.assertTrue(subblock.is_leaf)
                            if subblock.shape == (1, 1):
                                self.assertIsInstance(
                                    subblock, FormFactorZeroBlock)
                            else:
                                self.assertTrue(subblock.is_dense)
                    else:
                        self.assertTrue(block.is_leaf)
                        self.assertTrue(block.is_dense)

                # leaf subblocks agree with groundtruth
                for i, j in it.product(range(8), repeat=2):
                    I = FF._root._row_block_inds[i]
                    J = FF._root._col_block_inds[j]
                    block = FF._root._blocks[i, j]
                    if i == j:
                        for ii, jj in it.product(range(8), repeat=2):
                            II = block._row_block_inds[ii]
                            JJ = block._col_block_inds[jj]
                            subblock = block._blocks[ii, jj]
                            diff = FF_gt[I[II], :][:, J[JJ]] - subblock._mat
                            if diff.shape[0] != 0 and diff.shape[1] != 0:
                                self.assertEqual(abs(diff).max(), 0)
                    else:
                        diff = FF_gt[I, :][:, J] - block._mat
                        if diff.shape[0] != 0 and diff.shape[1] != 0:
                            self.assertEqual(abs(diff).max(), 0)
