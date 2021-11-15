import numpy as np
import unittest

from pathlib import Path

import flux.shape

class TrimeshShapeModelTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_get_visibility_matrix(self):
        npz_file = np.load(self.data_path/'icosa_sphere.npz')
        V, F = npz_file['V'], npz_file['F']

        shape_model = flux.shape.TrimeshShapeModel(V, F)

        # first, test the sphere with the normals pointing outward
        shape_model.N[(shape_model.N*shape_model.P).sum(1) < 0] *= -1

        num_faces = shape_model.num_faces
        shape = (num_faces, num_faces)

        # if we compute the un-oriented visibility matrix, then all
        # faces are mutually visible, the visibility matrix is all 1s
        # except with 0s on the diagonal
        vis = shape_model.get_visibility_matrix(oriented=False)
        vis_gt = (np.ones(shape) - np.eye(num_faces)).astype(bool)
        self.assertTrue((vis == vis_gt).all())

        # for the oriented visibility matrix with all normals pointed
        # out, we define the visibility matrix so that faces can't see
        # themselves, in which case the oriented visibility matrix
        # should be all zeros
        vis = shape_model.get_visibility_matrix(oriented=True)
        vis_gt = np.zeros(shape).astype(bool)
        self.assertTrue((vis == vis_gt).all())

        # second, test the sphere with normals pointing inward
        shape_model.N *= -1

        # visibility matrix should be all True in both cases

        vis_gt = (np.ones(shape) - np.eye(num_faces)).astype(bool)

        vis = shape_model.get_visibility_matrix(oriented=False)
        self.assertTrue((vis == vis_gt).all())

        vis = shape_model.get_visibility_matrix(oriented=True)
        self.assertTrue((vis == vis_gt).all())

if __name__ == '__main__':
    unittest.main()
