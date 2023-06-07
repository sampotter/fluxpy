import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import flux.compressed_form_factors_nmf as cff
from flux.compressed_form_factors_nmf import CompressedFormFactorMatrix

import scipy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="nmf",choices=["nmf","snmf","wsnmf","svd","true_model"])
parser.add_argument('--max_inner_area', type=float, default=0.8)
parser.add_argument('--max_outer_area', type=float, default=3.0)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.set_defaults(feature=False)

args = parser.parse_args()


def plot_blocks(block, fig, **kwargs):

    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
    else:
        fig.set_size_inches(12, 12)

    ax = fig.add_subplot()
    ax.axis('off')

    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k',
                             facecolor='none')
    ax.add_patch(rect)

    def get_ind_offsets(inds):
        return np.concatenate([[0], np.cumsum([len(I) for I in inds])])

    def add_rects(block, c0=(0, 0), w0=1, h0=1):
        row_offsets = get_ind_offsets(block._row_block_inds)
        col_offsets = get_ind_offsets(block._col_block_inds)
        for (i, row), (j, col) in it.product(
            enumerate(row_offsets[:-1]),
            enumerate(col_offsets[:-1])
        ):
            w = w0*(row_offsets[i + 1] - row)/row_offsets[-1]
            h = h0*(col_offsets[j + 1] - col)/col_offsets[-1]
            i0, j0 = c0
            c = (i0 + w0*row/row_offsets[-1], j0 + h0*col/col_offsets[-1])

            child = block._blocks[i, j]
            if child.is_leaf:
                if isinstance(child, cff.FormFactorSvdBlock) or isinstance(child, cff.FormFactorNmfBlock) or isinstance(child, cff.FormFactorSparseNmfBlock):
                    facecolor = 'cyan' if child.compressed else 'orange'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorZeroBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorSparseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='white')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorDenseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='magenta')
                    ax.add_patch(rect)
                elif isinstance(child, cff.FormFactorNullBlock):
                    continue
                else:
                    raise Exception('TODO: add %s to _plot_block' % type(child))
            else:
                add_rects(child, c, w, h)

            rect = patches.Rectangle(
                c, w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    add_rects(block)

    ax.invert_xaxis()

    return fig, ax




compression_type = args.compression_type
max_inner_area_str = str(args.max_inner_area)
max_outer_area_str = str(args.max_outer_area)

if compression_type == "true_model":
    FF_dir = "true_{:.1f}_{:.1f}".format(args.max_inner_area, args.max_outer_area)

elif compression_type == "svd":
    FF_dir = "{}_{:.1f}_{:.1f}_{:.0e}_{}k0".format(compression_type, args.max_inner_area, args.max_outer_area, args.tol,
        args.k0)

elif compression_type == "nmf":
    FF_dir = "{}_{:.1f}_{:.1f}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type, args.max_inner_area, args.max_outer_area, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "snmf":
    FF_dir = "{}_{:.1f}_{:.1f}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type, args.max_inner_area, args.max_outer_area, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "wsnmf":
    FF_dir = "{}_{:.1f}_{:.1f}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type, args.max_inner_area, args.max_outer_area, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

if compression_type == 'true_model':
    FF_path = 'results/' + FF_dir + f'/FF_{max_inner_area_str}_{max_outer_area_str}.npz'
    if not os.path.exists(FF_path):
        print("PATH DOES NOT EXIST " + FF_path)
        assert False
    FF = scipy.sparse.load_npz(FF_path)
else:
    FF_path =  "results/"+ FF_dir + "/FF_{:.1f}_{:.1f}_{:.0e}_{}.bin".format(args.max_inner_area, args.max_outer_area, args.tol, compression_type)
    if not os.path.exists(FF_path):
        print("PATH DOES NOT EXIST " + FF_path)
        assert False
    FF = CompressedFormFactorMatrix.from_file(FF_path)


fig = plt.figure(figsize=(18, 6))  # , figsize=(18, 6))
print(f'- {FF_path}')
fig, ax = plot_blocks(FF._root, fig)
fig.savefig("results/"+FF_dir+"/block_plot.png")
plt.close(fig)