#!/usr/bin/env python

import colorcet as cc
import pyvista as pv
# import pyvistaqt as pvqt
import numpy as np

from flux.compressed_form_factors import CompressedFormFactorMatrix

if __name__ == '__main__':
    # Load a compressed form factor matrix from FF.bin. This script
    # assumes that F stores the underlying shape model.
    FF = CompressedFormFactorMatrix.from_file('lsp_fake_compressed_form_factors.bin') #'FF.bin')

    # Select a face by index, and make sure it's valid
    face_index = 500   # this will appear in blue/negative on the map!
    assert(face_index < FF.num_faces)

    # Pull out the column of FF corresponding to face_index
    e = np.zeros(FF.shape[1])
    e[face_index] = 1
    f = FF@e # Multiply FF on the left by the "face_index"th standard
             # basis vector to get the column.

    # Since f[face_index] == 0, we set it to -f.max() so that we can
    # visualize its location in the plot below.
    f[face_index] = -f.max()  # blue face is the selected one

    # Next, iterate over the indices corresponding to each row block,
    # and set the entries of f for each block that *doesn't* contain
    # face_index to NaN. Any NaN triangle will be colored gray, so
    # that we effectively only plot the block of the partition
    # containing face_index.
    #
    # (Note: if you wanted to plot finer blocks, you could reach into
    # FF._root._blocks to find the _row_block_inds variable for each
    # subblock.)
    for inds in FF._root._row_block_inds:
        if face_index in inds:
            continue
        f[inds] = np.nan

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
    plotter = pv.Plotter()
    plotter.add_mesh(tri_mesh, lighting=False, cmap=cc.cm.coolwarm)
    plotter.show()
