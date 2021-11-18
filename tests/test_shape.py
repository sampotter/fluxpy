import itertools as it
import numpy as np
import unittest

from pathlib import Path

import common
import flux.shape

class TrimeshShapeModelTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).parent.absolute()/'data'

        np.seterr('raise')

    def test_get_visibility_matrix_for_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                shape_model = TrimeshShapeModel(V, F)

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

    def test_is_occluded_for_sphere(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                shape_model = TrimeshShapeModel(V, F)

                # make sure all surface normals point outward
                shape_model.N[(shape_model.N*shape_model.P).sum(1) < 0] *= -1

                D = np.random.randn(3)
                D /= np.linalg.norm(D)

                face_inds = np.arange(shape_model.num_faces)
                occluded = shape_model.is_occluded(face_inds, D)
                occluded_gt = shape_model.N@D < 0

                self.assertTrue((occluded == occluded_gt).all())

    def test_get_visibility_and_get_visibility_1_to_N_are_equivalent(self):
        for TrimeshShapeModel, npz_filename in it.product(
                flux.shape.trimesh_shape_models,
                common.sphere_npz_filenames):
            with self.subTest(
                    trimesh_shape_model=TrimeshShapeModel.__name__,
                    npz_filename=npz_filename):
                npz_file = np.load(self.data_path/npz_filename)
                V, F = npz_file['V'], npz_file['F']

                shape_model = TrimeshShapeModel(V, F)

                # make sure all surface normals point inward
                shape_model.N[(shape_model.N*shape_model.P).sum(1) > 0] *= -1

                oriented = False

                I = np.arange(shape_model.num_faces)
                vis = shape_model.get_visibility(I, I, oriented=oriented)

                for i in I:
                    vis_i = shape_model.get_visibility_1_to_N(
                        i, I, oriented=oriented)
                    J_bad = np.where(vis[i] != vis_i)[0]
                    J_bad_str = ', '.join(str(j) for j in J_bad)
                    self.assertEqual(
                        J_bad.size, 0, f'i = {i}, bad j: {J_bad_str}')

if __name__ == '__main__':
    unittest.main()
