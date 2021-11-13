import itertools as it
import meshzoo
import pathlib
import unittest

import numpy as np

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import get_form_factor_matrix
from flux.shape import TrimeshShapeModel

class FormFactorOctreeBlockTestCase(unittest.TestCase):

    def setUp(self):
        self.data_path = pathlib.Path(__file__).parent.absolute()
        self.data_path /= 'data'
        self.data_path /= 'compressed_form_factors'
        self.data_path /= 'form_factor_quadtree_block'

        # Set up shape model
        self.V = np.load(self.data_path/'lsp_V.npy')
        self.F = np.load(self.data_path/'lsp_F.npy')
        self.N = np.load(self.data_path/'lsp_N.npy')
        self.shape_model = TrimeshShapeModel(self.V, self.F, self.N)

        # Assemble compressed form factor matrix
        self.FF = CompressedFormFactorMatrix.assemble_using_quadtree(
            self.shape_model,
            tol=1e-2 # aiming for 1% accuracy
        )

    def test_off_diag_blocks(self):
        L_plus_U = self.FF.get_off_diag_blocks()

        # Compute groundtruth form factor matrix
        L_plus_U_gt = get_form_factor_matrix(self.shape_model).toarray()
        for (i, I), (j, J) in it.product(
                enumerate(self.FF._root._row_block_inds),
                enumerate(self.FF._root._col_block_inds)):
            if i == j:
                continue
            L_plus_U_gt[I, :][:, J] = 0

        # Get random test vector and compute MVPs
        # x = np.random.randn(L_plus_U.shape[1])
        x = np.zeros(L_plus_U.shape[1])
        x[np.random.choice(L_plus_U.shape[1])] = 1
        y_gt = L_plus_U_gt@x
        y = L_plus_U@x


        import colorcet as cc
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        vmax = max(abs(y).max(), abs(y_gt).max())
        vmin = -vmax
        plt.subplot(2, 2, 1)
        plt.tripcolor(*self.V.T, triangles=self.F, facecolors=y,
                      vmin=vmin, vmax=vmax, cmap=cc.cm.coolwarm)
        plt.gca().set_aspect('equal')
        plt.title(r'$\hat{y}$ (approx)')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.tripcolor(*self.V.T, triangles=self.F, facecolors=y_gt,
                      vmin=vmin, vmax=vmax, cmap=cc.cm.coolwarm)
        plt.gca().set_aspect('equal')
        plt.title(r'$y$ (gt)')
        plt.colorbar()
        e = y - y_gt
        vmax = abs(e).max()
        vmin = -vmax
        plt.subplot(2, 2, 3)
        plt.tripcolor(*self.V.T, triangles=self.F, facecolors=e,
                      vmin=vmin, vmax=vmax, cmap=cc.cm.coolwarm)
        plt.gca().set_aspect('equal')
        plt.title(r'$\hat{y} - y$ (error)')
        plt.colorbar()
        plt.subplot(2, 2, 4)
        vmax = abs(x).max()
        plt.tripcolor(*self.V.T, triangles=self.F, facecolors=x,
                      vmin=-vmax, vmax=vmax, cmap=cc.cm.coolwarm)
        plt.gca().set_aspect('equal')
        plt.title(r'$x$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('debug.png')




        # Check that all
        for i in range(y.size):
            self.assertLessEqual(abs(y[i] - y_gt[i]), self.FF._tol)


if __name__ == '__main__':
    unittest.main()
