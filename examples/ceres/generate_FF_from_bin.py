#!/usr/bin/env python

import matplotlib.pyplot as plt
#import os
import pickle
import scipy
import sys

from flux.plot import tripcolor_vector
from flux.form_factors import get_form_factor_matrix


if __name__ == '__main__':

    arg = sys.argv[1] # command line argument
    
    # Load a DEM stored as mesh.bin (e.g. use generate_FF_from_topo.py)
    with open(arg, 'rb') as f:   # enter file name here
        shape_model = pickle.load(f)

    V = shape_model.V
    F = shape_model.F
    print('- Number of vertices',V.shape[0],'Number of facets',F.shape[0])

    # Make plot of topography
    fig, ax = tripcolor_vector(V, F, V[:,2], cmap='jet')
    fig.savefig('topo2.png')
    plt.close(fig)
    print('wrote topo2.png')
    
    # Build the form factor matrix
    print('constructing form factor matrix')
    FF = get_form_factor_matrix( shape_model )
    print('assembled form factor matrix')

    # Save form factor matrix
    scipy.sparse.save_npz('FF.npz', FF)
    print('saved form factor matrix to FF.npz')

