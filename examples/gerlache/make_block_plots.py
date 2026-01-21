#!/usr/bin/env python

import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks

levels = [2,3,5,9]
tols = [1e-1,1e-2]
num_mesh_res = len(glob('2lev/FF_*_1e-1.bin'))
fig, axes = plt.subplots(len(levels)*2, num_mesh_res)  # , figsize=(18, 6))
for i, lev in enumerate(levels):
    for j, FF_path in enumerate(glob(f'{lev}lev/FF_*_*_1e-1.bin')):
        print(f'- {FF_path}')
        FF = CompressedFormFactorMatrix.from_file(FF_path)
        fig, ax = plot_blocks(FF._root, fig)

plt.show()
exit()
plot_path = FF_path[:-4] + '.png'
fig.savefig(Path('gerlache_plots')/plot_path)
plt.close(fig)

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
                if isinstance(child, cff.FormFactorSvdBlock):
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