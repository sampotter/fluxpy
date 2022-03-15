import glob
import sys

import numpy as np
import colorcet as cc

from flux.thermal import setgrid
from plot_utils import plot_on_grid
from flux.compressed_form_factors import CompressedFormFactorMatrix
from get_sublim_rate_at_depth import get_ice_depth

if __name__ == '__main__':

    max_inner_area_str = sys.argv[1]
    max_outer_area_str = sys.argv[2]
    tol_str = sys.argv[3]
    layer = int(sys.argv[4])

    # read SP mesh
    # path = "/home/sberton2/Lavoro/code/plapp-flux/examples/deGerlache/Moon/FF_Moon_quad_deGerlache_750.bin"
    # FF = CompressedFormFactorMatrix.from_file(path)
    # shape_model = FF.shape_model
    # V = shape_model.V
    # # V[:,0] = -1. * shape_model.V[:,1]
    # # V[:,1] = shape_model.V[:,0]
    #
    # F = shape_model.F
    vertices_path = f"gerlache_verts_stereo_{max_inner_area_str}_{max_outer_area_str}.npy"
    faces_path = f"gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy"
    V = np.load(vertices_path)
    F = np.load(faces_path)
    V = np.hstack([V,np.repeat(1737.4,V.shape[0])[:,np.newaxis]])

    # count number of spin-up iterations
    niter = len(glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T00_*.npy"))
    print(f"- Processing #{niter} spin-up iterations")

    # loop over all spin-up iterations
    Tmean_it = []
    for it in range(niter):
        Tfiles = glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T*_{it}.npy")

        # read T at surface and bottom layer for all epochs
        Tlayer = []
        for f in Tfiles:
            Tlayer.append(np.load(f)[:,layer])
        Tlayer = np.vstack(Tlayer)

        if it == niter - 1:
            # plot layerace max
            Tlayer_max = np.max(Tlayer,axis=0) # max along epochs
            save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T{layer}_max_{it}.png"
            plot_on_grid(arr_to_plot=Tlayer_max, vertices=V, faces=F,
                         title='Tlayer_max',  # clim=(-100,100),
                         save_to=save_to)
            print(f'- Tmax saved to {save_to}.')

        # save layerace layer mean T along epochs @ iter
        Tlayer_mean = np.mean(Tlayer,axis=0) # mean along epochs
        Tmean_it.append(Tlayer_mean)

    Tmean_it = np.vstack(Tmean_it)
    Tmean_it_diff = np.diff(Tmean_it,axis=0)

    save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T{layer}_mean_itdiff.png"
    plot_on_grid(arr_to_plot=Tmean_it_diff, vertices=V, faces=F,
                 title='', clim=(-1.,1.),
                 nrows=6, ncols= 5, figsize=(50,60), sharey=True, sharex= True,
                 save_to=save_to)
    print(f'- Tmean differences over iterations saved to {save_to}.')

    # plot z(T<110K)
    Tfiles = glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T*_{niter-1}.npy")
    # read T at surface and bottom layer for all epochs
    # plot z_T110
    nz = 60
    zfac = 1.05
    zmax = 2.5  # 2.5 for Moon
    print(f"- Considering subsurface layers with nz={nz}, zfac={zfac}, zmax={zmax}.")
    z = np.array([0] + [x for x in setgrid(nz=nz, zfac=zfac, zmax=zmax)])
    # zdict = dict(zip(np.arange(len(z)),z))
    #
    # zsub_epo = []
    # for f in Tfiles:
    #     T = np.load(f)[:,:]
    #     id, data = np.where(T < 110)
    #     _ndx = np.argsort(id)
    #     _id, _pos = np.unique(id[_ndx], return_index=True)
    #     g_min = np.minimum.reduceat(data[_ndx], _pos)
    #     z110 = np.vstack([_id,g_min]).T
    #
    #     fill = np.vstack([np.arange(T.shape[0]),np.repeat(np.nan,T.shape[0])]).T
    #     for row in z110:
    #         fill[row[0],-1] = row[1]
    #     zsub_epo.append(np.nan_to_num(fill[:,-1], nan=nz))
    #
    # zsub = np.vstack(zsub_epo)
    # zsub = np.max(zsub, axis=0)
    # zsub = [zdict[j] for j in zsub]

    zsub = get_ice_depth(filin_path=Tfiles, layers=z)

    save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/" \
              f"Ess_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.png"
    plot_on_grid(arr_to_plot=zsub, vertices=V, faces=F,
                 title=f'min_z(E<= 100 kg m-2 Gyr-1), m',
                 clim=(0, 0.5), cmap=cc.cm.bgyw,
                 save_to=save_to)
    print(f'- zice saved to {save_to}.')
