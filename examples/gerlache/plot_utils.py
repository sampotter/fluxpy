import colorcet as cc
import numpy as np
from matplotlib import pyplot as plt

from flux.shape import get_centroids


import numpy as np
import pyvista as pv

from plapp.import_obj import import_obj
from plapp.plot.contours import tricontourf_vector
from plapp.plot.fluxvars import cmap


# set up model (with whole mess from Sam's latest changes)
def promote_to_array_if_necessary(value, shape, dtype=None):
    if isinstance(value, (int, float, str)):
        return np.full(shape, value, dtype)
    elif isinstance(value, np.ndarray):
        if value.shape != shape:
            raise ValueError('invalid shape')
        return value
    else:
        raise TypeError(f'bad type: {type(value)}')

def plot_on_mesh(objin, array, varname='x', savefig=False):
    V, F, N = import_obj(obj_path=objin, get_normals=True)
    vertices = V.copy()
    faces = np.concatenate([3 * np.ones((F.shape[0], 1), dtype=F.dtype), F], axis=1)
    surf = pv.PolyData(vertices, faces)

    surf.cell_arrays[varname] = array
    surf.cell_arrays['opacity'] = 0.4  # np.logical_not(I_shadow).astype(T.dtype)

    this_cmap = cmap['jet']

    plotter = pv.Plotter()
    plotter.add_mesh(surf, scalars=varname, opacity='opacity',  # clim=[0,20],
                     use_transparency=True, cmap=this_cmap)

    cpos = plotter.show(auto_close=True)
    if savefig != False:
        plotter = pv.Plotter(off_screen=True)
        plotter.background_color = 'black'
        plotter.add_mesh(surf, scalars=varname, opacity='opacity',  # clim=[0,20],
                         use_transparency=True, cmap=this_cmap)
        plotter.camera_position = cpos
        plotter.screenshot(savefig)


def plot_on_grid(arr_to_plot, title='', save_to=None, suptitle='',
                 clim=None, objin=None, vertices=None, faces=None, xlim=None, ylim=None,
                 figsize=(10, 10), nrows=1, ncols=1, cmap=cc.cm.coolwarm,
                 sharex=False, sharey=False, **kwargs):

    from matplotlib import pyplot as plt
    from flux.shape import get_centroids
    import colorcet as cc

    if objin != None:
        vertices, faces, N = import_obj(obj_path=objin, get_normals=True)
    # elif (vertices == None) | (faces == None):
    #     print("** Function needs either valid objin or V, F.")
    #     exit()

    P = get_centroids(vertices, faces)
    # print("Applying mask to data (de Gerlache)")
    # mask = (P[:, 0] > -65) & (P[:, 0] < -30) & (P[:, 1] > -20) & (P[:, 1] < 20)
    # P = P[mask]
    # print(P)
    # if nrows * ncols > 1:
    #     arr_to_plot = [T[mask] for T in arr_to_plot]
    # else:
    #     arr_to_plot = arr_to_plot[mask]

    fig, axs = plt.subplots(nrows, ncols, sharey=sharey, sharex=sharex, figsize=figsize)

    if nrows * ncols > 1:
        try:
            for T, ax, t, cl, cm in zip(arr_to_plot, axs.ravel(), title, clim, cmap): #TODO check len/shape of clim and cmap
                if clim != None:
                    im = ax.tricontourf(*P[:, :2].T, T, extend='both', filled=True, cmap=cm,
                                        levels=np.linspace(cl[0], cl[1], 20), **kwargs)
                else:
                    im = ax.tricontourf(*P[:, :2].T, T, extend='both', filled=True,
                                        cmap=cm, clim=cl, **kwargs)  # , levels=np.linspace(0,clim[1],20))
                ax.set_title(t)
                fig.colorbar(im, ax=ax)
                ax.set_aspect('equal')
                if xlim != None:
                    ax.set_xlim(xlim[0], xlim[1])
                if ylim != None:
                    ax.set_ylim(ylim[0], ylim[1])

        except:
            for T, ax in zip(arr_to_plot, axs.ravel()): #TODO check len/shape of clim and cmap
                if clim != None:
                    im = ax.tricontourf(*P[:, :2].T, T, extend='both', filled=True, cmap=cmap,
                                        levels=np.linspace(clim[0], clim[1], 20), **kwargs)
                else:
                    im = ax.tricontourf(*P[:, :2].T, T, extend='both', filled=True,
                                        cmap=cmap, clim=clim, **kwargs)  # , levels=np.linspace(0,clim[1],20))
                ax.set_title(title)
                fig.colorbar(im, ax=ax)
                ax.set_aspect('equal')
                if xlim != None:
                    ax.set_xlim(xlim[0], xlim[1])
                if ylim != None:
                    ax.set_ylim(ylim[0], ylim[1])

        fig.suptitle(suptitle)
    else:
        if clim != None:
            im = axs.tricontourf(*P[:, :2].T, arr_to_plot, extend='both', filled=True,
                                 cmap=cc.cm.coolwarm, levels=np.linspace(clim[0], clim[1], 20))
        else:
            im = axs.tricontourf(*P[:, :2].T, arr_to_plot, extend='both', filled=True,
                                 cmap=cc.cm.coolwarm)
        axs.set_title(title)
        fig.colorbar(im, ax=axs)
        axs.set_aspect('equal')
        if xlim != None:
            axs.set_xlim(xlim[0], xlim[1])
        if ylim != None:
            axs.set_ylim(ylim[0], ylim[1])

    fig.tight_layout()
    if save_to != None:
        plt.savefig(save_to)
    else:
        plt.show()
    plt.close()


def plot_with_raster(arr_to_plot, ds_to_plot=None, title='', save_to=None,
                     clim=None, objin=None, V=None, F=None,
                     sharex=False, sharey=False):

    from matplotlib import pyplot as plt
    from flux.shape import get_centroids
    import colorcet as cc

    # if objin != None:
    #     V, F, N = import_obj(obj_path=objin, get_normals=True)
    # elif (V != None) & (F != None):
    #     pass
    # else:
    #     print("** Function needs either valid objin or V, F.")
    #     exit()

    P = get_centroids(V, F)

    fig, ax = plt.subplots(sharey=sharey, sharex=sharex)

    if ds_to_plot != None:
        ds_to_plot.band_data.plot(ax=ax)

    if clim != None:
        fig, ax = tricontourf_vector(P, F, arr_to_plot, filled=False,
                                     cmap=cc.cm.coolwarm, levels=np.linspace(clim[0], clim[1], 20), ax=ax, fig=fig)
    else:
        fig, ax = tricontourf_vector(P, F, arr_to_plot, filled=False,
                                     cmap=cc.cm.coolwarm, ax=ax, fig=fig)
    ax.set_title(title)
    ax.set_aspect('equal')

    fig.tight_layout()
    if save_to != None:
        plt.savefig(save_to)
    else:
        plt.show()
    plt.close()