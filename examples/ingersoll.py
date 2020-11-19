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
    parser.add_argument(
        '--outdir', type=str, default='.',
        help='Directory to write output files to')
    args = parser.parse_args()

import colorcet as cc
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import FormFactorMatrix
from flux.ingersoll import HemisphericalCrater
from flux.model import compute_steady_state_temp
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import TrimeshShapeModel, get_surface_normals
from flux.solve import solve_radiosity
from flux.util import tic, toc

if __name__ == '__main__':
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    h = (2/3)**args.p
    e0 = np.deg2rad(args.e0)
    beta = np.deg2rad(args.beta)

    hc = HemisphericalCrater(beta, args.rc, e0, args.F0, args.rho, args.emiss)
    print('- groundtruth temperature in shadow: %1.2f K' % (hc.T_gt,))

    # Create the triangle mesh
    V, F, parts = hc.make_trimesh(h, return_parts=True)
    print('- created tri. mesh with %d faces' % (F.shape[0],))

    # Write the mesh to disk as an OBJ file
    trimesh.Trimesh(V, F).export(os.path.join(args.outdir, 'ingersoll.obj'))
    print('- wrote ingersoll.obj')

    # Flip any upside down triangles
    N = get_surface_normals(V, F)
    N[N[:, 2] < 0] *= -1

    # Create a shape model from the triangle mesh (used for raytracing)
    shape_model = TrimeshShapeModel(V, F, N)

    tic()
    FF = CompressedFormFactorMatrix.assemble_using_partition(
        shape_model, parts, tol=args.tol)
    t_FF = toc()
    print('- finished assembly (%1.1f Mb) [%1.2f s]' %
          (FF.nbytes/1024**2, t_FF))

    FF.save(os.path.join(args.outdir, 'FF.bin'))
    print('- wrote FF.bin')

    fig, ax = plot_blocks(FF._root)
    fig.savefig(os.path.join(args.outdir, 'blocks.png'))
    plt.close(fig)

    dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])

    tic()
    E = shape_model.get_direct_irradiance(args.F0, dir_sun, eps=1e-6)
    t_E = toc()
    print('- computed direct irradiance [%1.2f s]' % (t_E,))

    fig, ax = tripcolor_vector(V, F, E, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'E.png'))
    plt.close(fig)
    print('- wrote E to disk')

    tic()
    B, niter_B = solve_radiosity(FF, E, rho=args.rho)
    t_B = toc()
    print('- computed radiosity [%1.2f s]' % (t_B,))

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'B.png'))
    plt.close(fig)
    print('- wrote B.png')

    tic()
    T = compute_steady_state_temp(FF, E, args.rho, args.emiss)
    t_T = toc()
    print('- computed T [%1.2f s]' % (t_T,))

    fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.fire)
    fig.savefig(os.path.join(args.outdir, 'T.png'))
    plt.close(fig)
    print('- wrote T.png')

    error = T[parts[1]] - hc.T_gt
    vmax = abs(error).max()
    vmin = -vmax
    fig, ax = tripcolor_vector(
        V, F[parts[1]], error, vmin=vmin, vmax=vmax, cmap=cc.cm.fire)
    fig.savefig(os.path.join(args.outdir, 'error.png'))
    plt.close(fig)
    print('- wrote error.png')

    max_error = abs(error).max()
    print('- max error: %1.2f K' % (max_error,))

    rms_error = np.linalg.norm(error)/np.sqrt(error.size)
    print('- RMS error: %1.2f K' % (rms_error,))

    stats = {
        'p': args.p,
        'h': h,
        'num_faces': F.shape[0],
        'T_gt': float(hc.T_gt),
        'max_error': float(max_error),
        'rms_error': float(rms_error),
        'FF_size': float(FF.nbytes/1024**2),
        't_FF': float(t_FF),
        't_E': float(t_E),
        't_B': float(t_B),
        't_T': float(t_T),
        'niter_B': niter_B
    }
    with open(os.path.join(args.outdir, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    print('- wrote stats.json')
