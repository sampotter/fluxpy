import itertools as it
import numpy as np
import unittest

from pathlib import Path

import common
import flux.shape

from flux.form_factors import get_form_factor_matrix

class FormFactorsTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_get_form_factor_matrix_for_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                common.sphere_npz_filenames):
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

                FF = get_form_factor_matrix(shape_model)

                # diagonal => zero, off-diagonal => nonzero
                for i, j in it.product(range(shape_model.num_faces), repeat=2):
                    if i == j:
                        self.assertEqual(FF[i, j], 0, f'i = {i}, j = {j}')
                    else:
                        self.assertNotEqual(FF[i, j], 0, f'i = {i}, j = {j}')

                # Form factor matrix tests for the sphere mesh with
                # outward facing normals

                shape_model.N *= -1

                FF = get_form_factor_matrix(shape_model)

                I, J = np.where(FF.toarray() != np.zeros_like(FF))
                for i, j in zip(I, J):
                    print(f'FF[{i}, {j}] = {FF[i, j]}')

                # all entries should be zero
                self.assertTrue((FF.toarray() == np.zeros_like(FF)).all())

if __name__ == '__main__':
    unittest.main()
