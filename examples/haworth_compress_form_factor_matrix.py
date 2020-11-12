import numpy as np

from compressed_form_factors import CompressedFormFactorMatrix
from shape import TrimeshShapeModel

# Load mesh

V = np.load('haworth_V.npy')
F = np.load('haworth_F.npy')
N = np.load('haworth_N.npy')

# Set up shape model and build form factor matrix

shape_model = TrimeshShapeModel(V, F, N)
FF = CompressedFormFactorMatrix.assemble_using_quadtree(shape_model, tol=1e-5)
FF.save('haworth_compressed_form_factors.bin')
