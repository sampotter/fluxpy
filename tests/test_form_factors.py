import itertools as it
import numpy as np
import unittest

from pathlib import Path

import flux.shape

from flux.form_factors import get_form_factor_matrix

class FormFactorsTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_get_form_factor_matrix_icosa_sphere_5(self):
        npz_file = np.load(self.data_path/'icosa_sphere_5.npz')
        shape_model = flux.shape.EmbreeTrimeshShapeModel(npz_file['V'], npz_file['F'])

        outward = (shape_model.P*shape_model.N).sum(1) > 0
        shape_model.N[outward] *= -1

        FF = get_form_factor_matrix(shape_model)

        for i, j in it.product(range(shape_model.num_faces), repeat=2):
            if i == j:
                self.assertEqual(FF[i, j], 0, f'i = {i}, j = {j}')
            else:
                self.assertNotEqual(FF[i, j], 0, f'i = {i}, j = {j}')

if __name__ == '__main__':
    unittest.main()
