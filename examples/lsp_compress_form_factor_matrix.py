#!/usr/bin/env python

import numpy as np

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.shape import TrimeshShapeModel

# Load mesh

V = np.load('lsp_V.npy')
F = np.load('lsp_F.npy')
N = np.load('lsp_N.npy')

# Set up shape model and build form factor matrix

shape_model = TrimeshShapeModel(V, F, N)
FF = CompressedFormFactorMatrix.assemble_using_quadtree(shape_model, tol=1e-5)
FF.save('lsp_compressed_form_factors.bin')
