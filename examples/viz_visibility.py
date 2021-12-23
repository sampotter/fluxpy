#!/usr/bin/env python
import colorcet as cc
import pyvista as pv
# import pyvistaqt as pvqt
import numpy as np
from matplotlib import pyplot as plt

from flux.compressed_form_factors import CompressedFormFactorMatrix
import flux.form_factors as ff
from flux.plot import plot_blocks
from flux.shape import TrimeshShapeModel
from flux.util import tic

use_svd = False
use_full = False

if __name__ == '__main__':
    # Load a compressed form factor matrix from FF.bin. This script
    # assumes that F stores the underlying shape model.
    if use_full:
        FF = ff.FormFactorMatrix.from_file('lsp_full_form_factors.bin')  # FF.bin')
        V = np.load('lsp_V.npy')
        F = np.load('lsp_F.npy')
        N = np.load('lsp_N.npy')
        shape_model = TrimeshShapeModel(V, F, N)
    else:
        FF = CompressedFormFactorMatrix.from_file('lsp_quad_compressed_form_factors.bin') #FF.bin')
        print(f"FF.depth = {FF.depth}")
        shape_model = FF.shape_model
        tic()
        fig, ax = plot_blocks(FF._root)
        fig.savefig('blocks.png')
        plt.close(fig)
        print('- wrote blocks.png')

    # Select a face by index, and make sure it's valid
    face_index = 500 # {random:500,nice:3500,good:4120,bad:4128,4129}
    assert(face_index < shape_model.num_faces)

    # Pull out the column of FF corresponding to face_index
    e = np.zeros(FF.shape[1])
    e[face_index] = 1

    if use_svd:
        U = np.loadtxt('svd_U.dat')
        sigma = np.loadtxt('svd_sigma.dat')
        Vt = np.loadtxt('svd_V.dat')

        tmp1 = Vt @ e  # (NxT) * (1xN) = (1xT)
        tmp2 = np.multiply( sigma, tmp1)          # (T) * (1xT) = (1xT)
        f = U @ tmp2
    else:
        print("passing here")
        f = FF@e # Multiply FF on the left by the "face_index"th standard
                 # basis vector to get the column.

    # The form factor matrix only approximately stores the visibility
    # information. To retrieve a visibility mask for a specific face,
    # we need to threshold, i.e.: "element j is visible from element i
    # if |Fij| > tol".
    eps_f = 1.e-13
    print(np.unique(f))
    print(f[f<0], len(f[f<0]), len(f))
    # f1 = np.array([x for x in f if x>0.])
    # print(np.max(f1), np.min(f1), np.mean(f1))
    # fig = plt.figure()
    # hist = plt.hist(f1,1000, density=False)
    # plt.loglog()
    # plt.show() #"histo_vis.png")
    # exit()
    # f[f < 0] = 0.
    vis_f = (abs(f) > eps_f).astype(np.float32)

    # After getting the approximate visibility, we set
    # vis_f[face_index] to 0.5, which makes this triangle be colored
    # red in the plots below.
    vis_f[face_index] = 0.5

    # Now, do raytracing to get the groundtruth visibility for this
    # face. The parameter "eps" controls how much the start of each
    # ray is perturbed (in the ray+N direction) from the face center.
    eps = None #1e-5
    vis = shape_model.check_vis_1_to_N(face_index, np.arange(shape_model.num_faces), eps=eps)
    vis = vis.astype(np.float32)
    vis[face_index] = 0.5

    # Make a triangle mesh with the approximate visibility mask
    tm_vis_f = pv.make_tri_mesh(shape_model.V, shape_model.F)
    tm_vis_f.cell_arrays['vis_f'] = vis_f

    # ... one with the groundtruth visibility mask...
    tm_vis = pv.make_tri_mesh(shape_model.V, shape_model.F)
    tm_vis.cell_arrays['vis'] = vis

    # ... and one with their difference.
    tm_vis_diff = pv.make_tri_mesh(shape_model.V, shape_model.F)
    tm_vis_diff.cell_arrays['diff'] = vis_f - vis

    # The keyword argument "shape=(1, 3)" indicates that we'll have a
    # row with three columns of subplots.
    # plotter = pvqt.BackgroundPlotter(window_size=(900, 400), shape=(1, 3))
    plotter = pv.Plotter(window_size=(900, 400), shape=(1, 3))

    # We want to use the same camera for each plot. We set it up here.
    V_ptp = shape_model.V.ptp(0)
    V_mean = shape_model.V.mean(0)
    camera = pv.Camera()
    camera.position = V_mean + np.array([0, 0, 2.2*V_ptp[:2].mean()])
    camera.focal_point = V_mean
    camera.up = np.array([1, 0, 0])

    plotter.subplot(0, 0)
    plotter.add_text(f'From form factor matrix (eps = {eps_f})', font_size=12)
    plotter.add_mesh(tm_vis_f, lighting=False, cmap=cc.cm.fire)
    plotter.camera = camera

    plotter.subplot(0, 1)
    plotter.add_text(f'Raytracing (eps = {eps})', font_size=12)
    plotter.add_mesh(tm_vis, lighting=False, cmap=cc.cm.fire)
    plotter.camera = camera

    plotter.subplot(0, 2)
    plotter.add_text('Difference', font_size=12)
    plotter.add_mesh(tm_vis_diff, lighting=False, cmap=cc.cm.coolwarm)
    plotter.camera = camera

    plotter.show()
