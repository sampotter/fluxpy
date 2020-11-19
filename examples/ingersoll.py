#!/usr/bin/env python

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Ingersoll example')
    parser.add_argument(
        '-p', type=int, default=6,
        help='target edge length: h = (2/3)**p')
    parser.add_argument(
        '--e0', type=float, default=15,
        help='sun angle above horizon in [deg.]')
    parser.add_argument(
        '--F0', type=float, default=1000,
        help='solar constant [W/m^2]')
    parser.add_argument(
        '--rho', type=float, default=0.3,
        help='albedo')
    parser.add_argument(
        '--emiss', type=float, default=0.99,
        help='emissivity')
    parser.add_argument(
        '--tol', type=float, default=1e-5,
        help='tolerance used when assembling compressed form factor matrix')
    parser.add_argument(
        '--beta', type=float, default=40,
        help='Ingersoll crater angle measured from vertical [deg.]')
    parser.add_argument(
        '--rc', type=float, default=0.8,
        help='Crater radius [m]')
    args = parser.parse_args()

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import FormFactorMatrix
from flux.ingersoll import HemisphericalCrater
from flux.model import compute_steady_state_temp
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import TrimeshShapeModel, get_surface_normals

if __name__ == '__main__':
    h = (2/3)**args.p
    e0 = np.deg2rad(args.e0)
    beta = np.deg2rad(args.beta)

    hc = HemisphericalCrater(beta, args.rc, e0, args.F0, args.rho, args.emiss)
    print('- groundtruth temperature in shadow %1.2f K' % (hc.T_gt,))

    # Create the triangle mesh
    V, F, parts = hc.make_trimesh(h, return_parts=True)
    print('- created a Delaunay mesh with %d points and %d faces' % (
        V.shape[0], F.shape[0]))

    # Write the mesh to disk as an OBJ file
    trimesh.Trimesh(V, F).export('ingersoll.obj')
    print('- saved Wavefront OBJ file to ingersoll.obj')

    # Flip any upside down triangles
    N = get_surface_normals(V, F)
    N[N[:, 2] < 0] *= -1

    # Create a shape model from the triangle mesh (used for raytracing)
    shape_model = TrimeshShapeModel(V, F, N)

    # FF = CompressedFormFactorMatrix.assemble_using_quadtree(
    #     shape_model, tol=args.tol)
    FF = CompressedFormFactorMatrix.assemble_using_partition(
        shape_model, parts, tol=args.tol)
    print('- assembled compressed form factor matrix (%1.1f Mb)' %
          (FF.nbytes/1024**2))

    FF.save('ingersoll.bin')
    print('- saved compressed form factor matrix to disk')

    fig, ax = plot_blocks(FF._root)
    fig.savefig('ingersoll_blocks.png')
    plt.close(fig)

    # FF = FormFactorMatrix(shape_model)

    dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])
    E = shape_model.get_direct_irradiance(args.F0, dir_sun, eps=1e-6)

    fig, ax = tripcolor_vector(V, F, E, cmap=cc.cm.gray)
    fig.savefig('ingersoll_E.png')
    plt.close(fig)
    print('- wrote ingersoll_E.png to disk')

    T = compute_steady_state_temp(FF, E, args.rho, args.emiss)
    print('- computed T')

    fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.fire)
    fig.savefig('ingersoll_T.png')
    plt.close(fig)
    print('- wrote ingersoll_T.png to disk')

    error = T[parts[1]] - hc.T_gt
    vmax = abs(error).max()
    vmin = -vmax
    fig, ax = tripcolor_vector(
        V, F[parts[1]], error, vmin=vmin, vmax=vmax, cmap=cc.cm.fire)
    fig.savefig('error.png')
    plt.close(fig)
    print('- wrote error.png to disk')

    max_error = abs(error).max()
    print('- max error: %1.2f K' % (max_error,))

    rms_error = np.linalg.norm(error)/np.sqrt(error.size)
    print('- RMS error: %1.2f K' % (rms_error,))
