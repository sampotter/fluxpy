import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


import flux.compressed_form_factors as cff


# TODO: this function REALLY needs to be profiled and optimized!!!
def plot_blocks(block, **kwargs):
    fig = plt.figure()

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


def tripcolor_vector(V, F, v, I=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    if I is None:
        im = ax.tripcolor(*V[:, :2].T, F, v, **kwargs)
    else:
        im = ax.tripcolor(*V[:, :2].T, F[I], v[I], **kwargs)
    fig.colorbar(im, ax=ax)
    ax.set_aspect('equal')
    xmin, ymin = np.min(V[:, :2], axis=0)
    xmax, ymax = np.max(V[:, :2], axis=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    return fig, ax


def imray(shape_model, values, pos, look, up, shape, mode='ortho', **kwargs):
    '''Trace camera rays and return corresponding values.

    Arguments:
      shape_model -- The shape model to use to do the raytracing.
      values -- The array of values. Should be the same length as
                shape_model.num_faces.
      pos -- The camera position.
      look -- The direction the camera should look in.
      up -- The up vector for the camera.
      shape -- The size of the image to create.
      mode -- The type of camera to use (only "ortho" supported for now).

    Keyword Arguments (mode == 'ortho'):
      h -- The spacing between pixels in the image in world space.

    '''

    left = np.cross(up, look)

    if mode == 'ortho':
        if 'h' not in kwargs:
            raise Exception('for "ortho" mode, must pass h kwarg')
        h = kwargs['h']
        m, n = shape
        s = np.linspace(-h*m/2, h*m/2, m)
        t = np.linspace(-h*n/2, h*n/2, n)
        s, t = np.meshgrid(s, t)
        s, t = s.ravel(), t.ravel()
        orgs = pos + np.outer(s, left) + np.outer(t, up)
        dirs = np.outer(np.ones(m*n, dtype=look.dtype), look)
        im = np.empty(m*n, dtype=values.dtype)
        im[:] = np.nan
        for i, (o, d) in enumerate(zip(orgs, dirs)):
            hit = shape_model.intersect1(o, d)
            if hit is not None:
                hit_index, hit_point = hit # hit_point unused
                im[i] = values[hit_index]
    else:
        raise Exception('only mode == "ortho" currently supported')

    return im.reshape(shape)
