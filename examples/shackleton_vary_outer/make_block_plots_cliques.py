import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import flux.compressed_form_factors_nmf as cff
from flux.compressed_form_factors_cliques import CliquedFormFactorMatrix

import scipy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compression_type', type=str, default="svd",choices=["nmf","snmf","wsnmf",
    "svd","ssvd",
    "rand_svd","rand_ssvd","rand_snmf",
    "saca","sbrp","rand_sid"])
parser.add_argument('--max_area', type=float, default=3.0)
parser.add_argument('--outer_radius', type=int, default=80)
parser.add_argument('--tol', type=float, default=1e-1)

parser.add_argument('--nmf_max_iters', type=int, default=int(1e4))
parser.add_argument('--nmf_tol', type=float, default=1e-2)

parser.add_argument('--k0', type=int, default=40)

parser.add_argument('--p', type=int, default=5)
parser.add_argument('--q', type=int, default=1)

parser.add_argument('--nmf_beta_loss', type=int, default=2, choices=[1,2])

parser.add_argument('--n_cliques', type=int, default=25)

parser.add_argument('--plot_labels', action='store_true')

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
                if isinstance(child, cff.FormFactorSvdBlock) or isinstance(child, cff.FormFactorSparseSvdBlock) or \
                isinstance(child, cff.FormFactorNmfBlock) or isinstance(child, cff.FormFactorSparseNmfBlock) or \
                isinstance(child, cff.FormFactorSparseAcaBlock) or isinstance(child, cff.FormFactorSparseBrpBlock) or isinstance(child, cff.FormFactorSparseIdBlock):
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

                if args.plot_labels:
                    ax.text(c[0] + 0.5*w, c[1] + 0.5*h, "{:.0e}".format(np.prod(child.shape)), transform=ax.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')
            else:
                add_rects(child, c, w, h)

            rect = patches.Rectangle(
                c, w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    add_rects(block)

    ax.invert_xaxis()

    return fig, ax




compression_type = args.compression_type
max_area_str = str(args.max_area)
outer_radius_str = str(args.outer_radius)

if compression_type == "svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "rand_svd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "rand_ssvd":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)

elif compression_type == "nmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "klnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "sklnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "rand_snmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.p, args.q, args.k0)

elif compression_type == "wsnmf":
    FF_dir = "{}_{}_{}_{:.0e}_{:.0e}it_{:.0e}tol_{}k0".format(compression_type if args.nmf_beta_loss==2 else "wsklnmf", max_area_str, outer_radius_str, args.tol,
        args.nmf_max_iters, args.nmf_tol, args.k0)

elif compression_type == "saca":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "sbrp":
    FF_dir = "{}_{}_{}_{:.0e}_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.k0)

elif compression_type == "rand_sid":
    FF_dir = "{}_{}_{}_{:.0e}_{}p_{}q_{}k0".format(compression_type, max_area_str, outer_radius_str, args.tol,
        args.p, args.q, args.k0)

FF_dir = FF_dir + "_{}nc".format(args.n_cliques)

FF_path =  "results_cliques/"+ FF_dir + "/FF_{}_{}_{:.0e}_{}.bin".format(max_area_str, outer_radius_str, args.tol, compression_type)
if not os.path.exists(FF_path):
    print("PATH DOES NOT EXIST " + FF_path)
    assert False
FF = CliquedFormFactorMatrix.from_file(FF_path)


fig = plt.figure(figsize=(18, 6))  # , figsize=(18, 6))
print(f'- {FF_path}')
fig, ax = plot_blocks(FF._root, fig)
if args.plot_labels:
    fig.savefig("results_cliques/"+FF_dir+"/block_plot_labeled.png")
else:
    fig.savefig("results_cliques/"+FF_dir+"/block_plot.png")
plt.close(fig)