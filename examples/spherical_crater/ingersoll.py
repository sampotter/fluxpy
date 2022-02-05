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
        '--tol', type=float, default=None,
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
    parser.add_argument(
        '--blocks', type=bool, default=False,
        help='Make a plot of the form factor matrix blocks')
    parser.add_argument(
        '--mprof', type=bool, default=False,
        help='Measure the memory use during assembly and save results')
    args = parser.parse_args()

import colorcet as cc
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path
import scipy.sparse
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import FormFactorPartitionBlock
from flux.form_factors import get_form_factor_matrix
from flux.ingersoll import HemisphericalCrater
from flux.model import compute_steady_state_temp, sigSB
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.solve import solve_radiosity
from flux.util import tic, toc

from memory_profiler import memory_usage

def assemble(args, shape_model, parts):
    if args.tol is None:
        tic()
        FF = get_form_factor_matrix(shape_model)
        t_FF = toc()
        FF_nbytes = FF.data.nbytes + FF.indptr.nbytes + FF.indices.nbytes
    else:
        tic()
        FF = CompressedFormFactorMatrix(
            shape_model, parts=parts, tol=args.tol,
            RootBlock=FormFactorPartitionBlock)
        t_FF = toc()
        FF_nbytes = FF.nbytes
    return FF, t_FF, FF_nbytes

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
    shape_model = CgalTrimeshShapeModel(V, F, N)

    if args.tol is None:
        print("- tol argument not passed: assembling sparse form factor matrix")

    if args.mprof:
        interval = 0.25
        mem_usage, (FF, t_FF, FF_nbytes) = memory_usage(
            (assemble, (args, shape_model, parts)),
            interval=interval,
            retval=True
        )
        mem_T = interval*np.arange(1, len(mem_usage) + 1)
        np.save(os.path.join(args.outdir, 'mem_usage.npy'), mem_usage)
        np.save(os.path.join(args.outdir, 'mem_T.npy'), mem_T)
        print("- saved mem_usage.npy and mem_T.npy")
    else:
        FF, t_FF, FF_nbytes = assemble(args, shape_model, parts)

    print('- finished assembly (%1.1f Mb) [%1.2f s]' %
          (FF_nbytes/1024**2, t_FF))

    if args.tol is None:
        scipy.sparse.save_npz(os.path.join(args.outdir, 'FF.npz'), FF)
        print('- wrote FF.npz')
    else:
        FF.save(os.path.join(args.outdir, 'FF.bin'))
        print('- wrote FF.bin')

    if args.blocks and args.tol:
        tic()
        fig, ax = plot_blocks(FF._root)
        fig.savefig(os.path.join(args.outdir, 'blocks.png'))
        plt.close(fig)
        print('- wrote blocks.png [%1.2f s]' % (toc(),))

    dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])

    tic()
    E = shape_model.get_direct_irradiance(args.F0, dir_sun, eps=1e-6)
    t_E = toc()
    print('- computed direct irradiance [%1.2f s]' % (t_E,))

    fig, ax = tripcolor_vector(V, F, E, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'E.png'))
    plt.close(fig)
    print('- wrote E.png to disk')

    tic()
    B, niter_B = solve_radiosity(FF, E, albedo_placement='right', rho=args.rho)
    t_B = toc()
    print('- computed radiosity [%1.2f s]' % (t_B,))

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'B.png'))
    plt.close(fig)
    print('- wrote B.png')

    Q_IR = FF@((1 - hc.rho)*B)
    Q, niter_Q = solve_radiosity(FF, Q_IR, 1, albedo_placement='right')
    Q *= hc.emiss
    Q += (1 - hc.rho)*B
    assert not np.isnan(Q).any()

    fig, ax = tripcolor_vector(V, F, B, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'Q.png'))
    plt.close(fig)
    print('- wrote Q.png')

    tic()
    T = compute_steady_state_temp(FF, E, args.rho, args.emiss, Fsurf=0)
    t_T = toc()
    print('- computed T [%1.2f s]' % (t_T,))

    fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.fire)
    fig.savefig(os.path.join(args.outdir, 'T.png'))
    plt.close(fig)
    print('- wrote T.png')

    # compute local elevation of sun for each facet
    e = np.maximum(0, np.pi/2 - np.arccos(N@dir_sun))

    b = hc.f*(hc.emiss + hc.rho*(1 - hc.f))/(1 - hc.rho*hc.f)

    Q_gt = np.empty(shape_model.num_faces)
    Q_gt[parts[0]] = (1 - hc.rho)*hc.F0*np.sin(hc.e0)
    Q_gt[parts[1]] = (1 - hc.rho)*hc.F0*b*np.sin(hc.e0)
    Q_gt[parts[2]] = (1 - hc.rho)*hc.F0*(np.sin(e[parts[2]]) + b*np.sin(hc.e0))

    np.save(os.path.join(args.outdir, 'I_plane.npy'), parts[0])
    np.save(os.path.join(args.outdir, 'I_shadow.npy'), parts[1])
    np.save(os.path.join(args.outdir, 'I_sun.npy'), parts[2])

    fig, ax = tripcolor_vector(V, F, Q_gt, cmap=cc.cm.gray)
    fig.savefig(os.path.join(args.outdir, 'Q_gt.png'))
    plt.close(fig)
    print('- wrote Q_gt.png')

    Q_error = Q - Q_gt
    vmax = abs(Q_error).max()
    vmin = -vmax
    fig, ax = tripcolor_vector(V, F, Q_error, vmin=vmin,vmax=vmax,cmap=cc.cm.coolwarm)
    fig.savefig(os.path.join(args.outdir, 'Q_error.png'))
    plt.close(fig)
    print('- wrote Q_error.png')

    # compute groundtruth temperature in crater
    T_gt = np.maximum(0, Q_gt/(sigSB*hc.emiss))**0.25

    # compute pointwise error
    T_error = T - T_gt

    np.save(os.path.join(args.outdir, 'T_error.npy'), T_error)

    # plot pointwise T_error
    vmax = abs(T_error).max()
    vmin = -vmax
    fig, ax = tripcolor_vector(V,F,T_error,vmin=vmin,vmax=vmax,cmap=cc.cm.coolwarm)
    fig.savefig(os.path.join(args.outdir, 'T_error.png'))
    plt.close(fig)
    print('- wrote T_error.png')

    max_T_error = abs(T_error).max()
    print('- max T_error: %1.2f K' % (max_T_error,))

    l2_T_error = np.linalg.norm(T_error)
    print('- l2 T_error: %1.2f K' % (l2_T_error,))

    l1_T_error = np.linalg.norm(T_error, ord=1)
    print('- l1 T_error: %1.2f K' % (l1_T_error,))

    rel_max_T_error = max_T_error/hc.T_gt
    print(f'- relative max T_error: {100*rel_max_T_error:1.2f}%')

    rel_l2_T_error = l2_T_error/(hc.T_gt*np.sqrt(F.shape[0]))
    print(f'- relative l2 T_error: {100*rel_l2_T_error:1.2f}%')

    rel_l1_T_error = l1_T_error/(hc.T_gt*F.shape[0])
    print(f'- relative l1 T_error: {100*rel_l1_T_error:1.2f}%')

    B.tofile(os.path.join(args.outdir, 'B.bin'))
    print('- wrote B.bin')

    T.tofile(os.path.join(args.outdir, 'T.bin'))
    print('- wrote T.bin')

    stats = {
        'p': args.p,
        'e0_deg': args.e0,
        'F0': args.F0,
        'rho': args.rho,
        'emiss': args.emiss,
        'tol': args.tol,
        'beta': args.beta,
        'rc': args.rc,
        'h': h,
        'num_faces': F.shape[0],
        'max_T_error': float(max_T_error),
        'l2_T_error': float(l2_T_error),
        'l1_T_error': float(l1_T_error),
        'rel_max_T_error': float(rel_max_T_error),
        'rel_l2_T_error': float(rel_l2_T_error),
        'rel_l1_T_error': float(rel_l1_T_error),
        'FF_size': float(FF_nbytes/1024**2),
        't_FF': float(t_FF),
        't_E': float(t_E),
        't_B': float(t_B),
        't_T': float(t_T),
        'niter_B': niter_B
    }
    with open(os.path.join(args.outdir, 'stats.json'), 'w') as f:
        json.dump(stats, f)
    print('- wrote stats.json')
