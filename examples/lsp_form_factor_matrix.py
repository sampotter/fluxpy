#!/usr/bin/env python
import logging
import pickle

import numpy as np

from flux.form_factors import get_form_factor_block
from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import TrimeshShapeModel


def setup_form_factor_matrix(compress=True, tol=1e-2):
    """
    Loads facets and produces FF
    Args:
        tol: max error for SVD truncation?
        compress: bool, get full or compressed FF
    """
    # Load mesh
    V = np.load('lsp_V.npy')
    F = np.load('lsp_F.npy')
    N = np.load('lsp_N.npy')

    # Set up shape model and build form factor matrix
    shape_model = TrimeshShapeModel(V, F, N)

    if compress:
        FF = CompressedFormFactorMatrix.assemble_using_quadtree(shape_model, tol=tol)
        FF.save('lsp_compressed_form_factors.bin')
    else:
        FF = get_form_factor_block(shape_model)
        with open('lsp_full_form_factors.bin', 'wb') as f:
            pickle.dump(FF,f)

if __name__ == '__main__':

    setup_form_factor_matrix()