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

    # Choose the block depth, make sure it's valid. Can be between 0
    # and FF.depth.
    depth = 3
    assert(0 <= depth and depth <= FF.depth)

    face_index = np.random.choice(FF.num_faces)
    print(f'face_index = {face_index}')

    check_visibility = True

    def vis(inds):
        return FF.shape_model.check_vis_1_to_N(face_index, inds, eps=1e-4)

    block_inds = np.empty(FF.num_faces)
    block_inds[...] = np.nan

    num_blocks = 0
    for block_ind, col_inds in enumerate(FF.get_col_inds_for_row(face_index)):
        if check_visibility:
            block_inds[col_inds[vis(col_inds)]] = block_ind
        else:
            block_inds[col_inds] = block_ind
        num_blocks += 1
    block_inds[face_index] = -1

    # Set up a trimesh with the block indices as a cell array for
    # plotting.
    tri_mesh = pv.make_tri_mesh(FF.shape_model.V, FF.shape_model.F)
    tri_mesh.cell_arrays['block_inds'] = block_inds

    # We want to use the same camera for each plot. We set it up here.
    V_ptp = FF.shape_model.V.ptp(0)
    V_mean = FF.shape_model.V.mean(0)
    camera = pv.Camera()
    camera.position = V_mean + np.array([0, 0, 2.2*V_ptp[:2].mean()])
    camera.focal_point = V_mean
    camera.up = np.array([1, 0, 0])

    # Make plot using PyVista
    plotter = pvqt.BackgroundPlotter()
    plotter.add_text('Block indices', font_size=12)
    plotter.add_mesh(tri_mesh, clim=(-1, num_blocks),
                     lighting=False, cmap=cc.cm.glasbey)
    plotter.remove_scalar_bar()
    plotter.camera = camera
    plotter.show()
