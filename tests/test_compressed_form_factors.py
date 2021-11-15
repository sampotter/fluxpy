import numpy as np
import unittest

from pathlib import Path

import flux.shape

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import FormFactorOctreeBlock

class CompressedFormFactorMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_icosa_sphere(self):
        npz_file = np.load(self.data_path/'icosa_sphere.npz')
        shape_model = flux.shape.TrimeshShapeModel(npz_file['V'], npz_file['F'])

        FF = CompressedFormFactorMatrix(
            shape_model,
            tol=1e-2,
            max_depth=1,
            RootBlock=FormFactorOctreeBlock)

        self.assertEqual(FF.depth, 1)
