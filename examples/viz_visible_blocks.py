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

    # Select a face by index, and make sure it's valid
    face_index = 0
    assert(face_index < FF.num_faces)

    # Choose the block depth, make sure it's valid. Can be between 0
    # and FF.depth.
    depth = 3
    assert(0 <= depth and depth <= FF.depth)

    # Helper function for checking visibility
    def vis(inds):
        return FF.shape_model.check_vis_1_to_N(face_index, inds, eps=1e-4)

    # Label each face with the block index (at the specified depth)
    # *if this face and face_index can see each other*. Otherwise set
    # to NaN.
    block_inds = np.empty(FF.num_faces, dtype=np.float64)
    block_inds[...] = np.nan
    for i, row_inds in enumerate(FF.get_row_inds_at_depth(depth)):
        block_inds[row_inds[vis(row_inds)]] = i

    # The "face_index"th triangle gets -1 as a label to visually
    # distinguish it in the plot.
    block_inds[face_index] = -1

    # Get the number of blocks at this depth.
    num_blocks_at_depth = len(list(FF.get_blocks_at_depth(depth)))

    # Set up a trimesh with the block indices as a cell array for
    # plotting.
    tri_mesh = pv.make_tri_mesh(FF.shape_model.V, FF.shape_model.F)
    tri_mesh.cell_arrays['block_inds'] = block_inds

    plotter = pvqt.BackgroundPlotter()

    # We want to use the same camera for each plot. We set it up here.
    V_ptp = FF.shape_model.V.ptp(0)
    V_mean = FF.shape_model.V.mean(0)
    camera = pv.Camera()
    camera.position = V_mean + np.array([0, 0, 2.2*V_ptp[:2].mean()])
    camera.focal_point = V_mean
    camera.up = np.array([1, 0, 0])

    plotter.add_text(f'Block indices', font_size=12)
    plotter.add_mesh(tri_mesh, clim=(-1, num_blocks_at_depth),
                     lighting=False, cmap=cc.cm.glasbey)
    plotter.camera = camera

    plotter.show()
