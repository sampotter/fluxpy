# https://doi.org/10.3847/PSJ/abb6ff , Schorghofer and Williams 2020, PSJ
# E(T) = exp (5.564214 - 5723.265/T + 3.03068 log(T) - 0.00728332 T) in kg m-2 s-1 , T = T(z_ice); eq.(8)
# Ess = ell/(z_ice + ell) E(T(z_ice)) with l = 100 um, E(T(z_ice)) the sublimation rate that; eq.(10)
# ice at T(z_ice) would have at the surface (can be based on the average T over a cycle), z_ice the ice depth.
# look for depth at which E(T)=100 kg/m2/Gyr

# compute E(T(z_ice)) at each layer, then compute Ess
import glob
import sys

import numpy as np
from plot_utils import plot_on_grid
from flux.thermal import setgrid
import colorcet as cc

def get_ice_depth(filin_path,layers):

    zdict = dict(zip(np.arange(len(layers)),layers))

    # read T at surface and bottom layer for all epochs
    Ess_z = []
    for layer, z in dict(list(zdict.items())[::]).items():
        Tlayer = []
        for f in filin_path:
            Tlayer.append(np.load(f)[:, layer])
        Tlayer = np.vstack(Tlayer)

        # save layer mean T along epochs
        T = np.mean(Tlayer, axis=0)  # mean along epochs

        E_T_surf = np.exp(5.564214 - 5723.265/T + 3.03068*np.log(T) - 0.00728332*T)
        E_T_surf *= 1.e9 * 365.25 * 86400.
        # check against Table 1 of Schorghofer and Williams, PSJ 2020
        # tst = np.vstack([T,E_T_surf]).T
        # print(tst[(tst[:,0]>110)&(tst[:,0]<110.1)])   # should give ~144
        # print(tst[(tst[:,0]>114)&(tst[:,0]<114.1)])   # should give ~1000

        Ess = (1.e-4 / (z + 1.e-4)) * E_T_surf
        Ess_z.append(Ess)

    Ess_z = np.vstack(Ess_z)
    icy_layers = np.where(Ess_z < 100, 1, 0)
    icy_layers = len(zdict)-np.sum(icy_layers,axis=0)
    z_100kg_m2Gyr = np.vectorize(zdict.get)(icy_layers)
    z_100kg_m2Gyr[np.isnan(z_100kg_m2Gyr.astype(np.float))] = zdict[len(zdict)-1]  # clamping no-ice layers to bottom layer

    return z_100kg_m2Gyr

if __name__ == '__main__':

    max_inner_area_str = sys.argv[1]
    max_outer_area_str = sys.argv[2]
    tol_str = sys.argv[3]

    # read SP mesh
    vertices_path = f"gerlache_verts_stereo_{max_inner_area_str}_{max_outer_area_str}.npy"
    faces_path = f"gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy"
    V = np.load(vertices_path)
    F = np.load(faces_path)
    V = np.hstack([V,np.repeat(1737.4,V.shape[0])[:,np.newaxis]])

    # count number of spin-up iterations
    niter = len(glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T00_*.npy"))
    print(f"- Getting last (#{niter}) spin-up iteration")

    # plot z(T<110K)
    Tfiles = glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/T*_{niter-1}.npy")
    # read T at surface and bottom layer for all epochs
    # plot z_T110
    nz = 60
    zfac = 1.05
    zmax = 2.5  # 2.5 for Moon
    print(f"- Considering subsurface layers with nz={nz}, zfac={zfac}, zmax={zmax}.")
    z = np.array([0] + [x for x in setgrid(nz=nz, zfac=zfac, zmax=zmax)])

    z_100kg_m2Gyr = get_ice_depth(filin_path=Tfiles, layers=z)

    save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_str}/" \
              f"Ess_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.png"
    plot_on_grid(arr_to_plot=z_100kg_m2Gyr, vertices=V, faces=F,
                 title=f'min_z(E<= 100 kg m-2 Gyr-1), m',
                 clim=(0, 0.5), cmap=cc.cm.bgyw,
                 savefig=save_to)
    print(f'- zice saved to {save_to}.')