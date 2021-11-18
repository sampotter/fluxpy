import itertools as it
import numpy as np
import unittest

from pathlib import Path

import flux.shape
import tests.common

from flux.compressed_form_factors import \
    CompressedFormFactorMatrix, \
    FormFactorDenseBlock, \
    FormFactorOctreeBlock
from flux.form_factors import get_form_factor_matrix

class CompressedFormFactorMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_icosa_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                tests.common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                shape_model = TrimeshShapeModel(V, F)

                # Form factor matrix tests for the sphere mesh with
                # inward facing normals

                outward = (shape_model.P*shape_model.N).sum(1) > 0
                shape_model.N[outward] *= -1

                FF = CompressedFormFactorMatrix(
                    shape_model,
                    tol=0,
                    max_depth=1,
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
