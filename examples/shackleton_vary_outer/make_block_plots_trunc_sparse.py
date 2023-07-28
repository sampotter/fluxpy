import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import flux.compressed_form_factors_truncated_sparse as cff
from flux.compressed_form_factors_truncated_sparse import TruncatedHierarchicalFormFactorMatrix

import scipy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--min_depth', type=int, default=1)
parser.add_argument('--max_depth', type=int, default=0)

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
                if isinstance(child, cff.FormFactorTruncatedSparseBlock):
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




max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)

max_depth = args.max_depth if args.max_depth != 0 else None

FF_dir = "sparse_hier_{}_{}_{:.0e}".format(max_area_str, outer_radius_str, args.tol)


if args.min_depth != 1:
    FF_dir += "_{}mindepth".format(args.min_depth)

if max_depth is not None:
    FF_dir += "_{}maxdepth".format(max_depth)


FF_path =  "results/"+ FF_dir + "/FF_{}_{}_{:.0e}_sparse_hierarch.bin".format(max_area_str, outer_radius_str, args.tol)
if not os.path.exists(FF_path):
    print("PATH DOES NOT EXIST " + FF_path)
    assert False
FF = TruncatedHierarchicalFormFactorMatrix.from_file(FF_path)


fig = plt.figure(figsize=(18, 6))
print(f'- {FF_path}')
fig, ax = plot_blocks(FF._root, fig)
fig.savefig("results/"+FF_dir+"/block_plot.png")
plt.close(fig)