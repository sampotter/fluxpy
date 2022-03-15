import glob
import sys

import numpy as np
from flux.thermal import setgrid
from matplotlib import pyplot as plt

from get_sublim_rate_at_depth import get_ice_depth
from plot_utils import plot_on_grid
from flux.shape import get_centroids
import colorcet as cc

if __name__ == '__main__':

    max_inner_area_str = sys.argv[1]
    max_outer_area_str = sys.argv[2]
    tol_strA = sys.argv[3]
    tol_strB = sys.argv[4]
    layer = int(sys.argv[5])

    # read SP mesh
    vertices_path = f"gerlache_verts_stereo_{max_inner_area_str}_{max_outer_area_str}.npy"
    faces_path = f"gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy"
    V = np.load(vertices_path)
    F = np.load(faces_path)
    V = np.hstack([V,np.repeat(1737.4,V.shape[0])[:,np.newaxis]])

    # count number of spin-up iterations
    niter = len(glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_strA}/T00_*.npy"))
    print(f"- Processing #{niter} spin-up iterations")

    TfilesA = glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_strA}/T*_{niter - 1}.npy")
    TfilesB = glob.glob(f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_strB}/T*_{niter - 1}.npy")

    # read T at surface and bottom layer for all epochs
    TlayerA = []
    for f in TfilesA:
        TlayerA.append(np.load(f)[:,layer])
    TlayerA = np.vstack(TlayerA)

    TlayerB = []
    for f in TfilesB:
        TlayerB.append(np.load(f)[:,layer])
    TlayerB = np.vstack(TlayerB)

    # plot layerace max
    TlayerA_max = np.max(TlayerA,axis=0) # max along epochs
    TlayerB_max = np.max(TlayerB,axis=0) # max along epochs

    max_diff = TlayerA_max - TlayerB_max

    P = get_centroids(V, F)
    print("Applying mask to data (de Gerlache)")
    mask = (P[:, 1] > -65) & (P[:, 1] < -30) & (P[:, 0] > -20) & (P[:, 0] < 20)
    F = F[mask]
    max_diff = max_diff[mask]

    save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_strA}/T{layer}_max_{tol_strA}_vs_{tol_strB}.png"
    plot_on_grid(arr_to_plot=max_diff, vertices=V, faces=F,
                 title=f'Tlayer max {tol_strA} vs {tol_strB}, K; max={round(np.max(max_diff),1)} K', clim=(-10,10),
                 save_to=save_to)
    print(f'- Tmax diff saved to {save_to}.')

    nz = 60
    zfac = 1.05
    zmax = 2.5  # 2.5 for Moon
    print(f"- Considering subsurface layers with nz={nz}, zfac={zfac}, zmax={zmax}.")
    z = np.array([0] + [x for x in setgrid(nz=nz, zfac=zfac, zmax=zmax)])

    z_100kg_m2Gyr_A = get_ice_depth(filin_path=TfilesA, layers=z)
    z_100kg_m2Gyr_B = get_ice_depth(filin_path=TfilesB, layers=z)

    ice_depth_diff = z_100kg_m2Gyr_A - z_100kg_m2Gyr_B

    save_to = f"T_frames/{max_inner_area_str}_{max_outer_area_str}_{tol_strA}/" \
              f"Ess_{max_inner_area_str}_{max_outer_area_str}_{tol_strA}_vs_{tol_strB}.png"
    plot_on_grid(arr_to_plot=ice_depth_diff, vertices=V, faces=F,
                 title=f'min_z(E<= 100 kg m-2 Gyr-1), m, diff {tol_strA} vs {tol_strB}; '
                       f'max={round(np.max(ice_depth_diff),1)} m',
                 clim=(-1, 1), # cmap=cc.cm.bgyw,
                 save_to=save_to)
    print(f'- zice diffs saved to {save_to}.')
