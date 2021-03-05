#!/usr/bin/env python

import colorcet as cc
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np

from flux.compressed_form_factors import CompressedFormFactorMatrix

if __name__ == '__main__':
    # Load a compressed form factor matrix from FF.bin. This script
    # assumes that F stores the underlying shape model.
    FF = CompressedFormFactorMatrix.from_file('FF.bin')

    # Select a face by index. We'll pull out the corresponding column of
    # FF to plot.
    face_index = 0
    e = np.zeros(FF.shape[1])
    e[face_index] = 1
    f = FF@e # Multiply FF on the left by the "face_index"th standard
             # basis vector to get the column.

    # Since f[face_index] == 0, we set it to -f.max() so that we can
    # visualize its location in the plot below.
    f[face_index] = -f.max()

    # Make a pv.PolyData and set up a cell array on it containing the
    # values of the column (stored in f).
    tri_mesh = pv.make_tri_mesh(FF.shape_model.V, FF.shape_model.F)
    tri_mesh.cell_arrays['form factor'] = f

    # Plot it using a nicer colormap than PyVista's default. Also,
    # turn off lighting so colors are easier to interpret. In the
    # plot, the selected face will be the lone blue triangle, while
    # all other triangles should be somewhere between "whitish tan"
    # and red.
    #
    # (Note: we're using pvqt.BackgroundPlotter here since there are
    # some bugs with pv.Plotter on Mac at the moment.)
    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(tri_mesh, lighting=False, cmap=cc.cm.coolwarm)
    plotter.show()
