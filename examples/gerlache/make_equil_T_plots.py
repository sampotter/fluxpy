#!/usr/bin/env python

import colorcet as cc
import json_numpy as json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

plotdir = Path('gerlache_plots')
if not plotdir.exists():
    plotdir.mkdir()

with open('params.json') as f:
    params = json.load(f)

xmin = params['xmin']
xmax = params['xmax']
ymin = params['ymin']
ymax = params['ymax']
extent = [xmin, xmax, ymin, ymax]

r_ROI = params['ROI_radius']
x_ROI, y_ROI = (ymin + ymax)/2, (xmin + xmax)/2

with open('equil_T_data.json') as f:
    data = json.load(f)

FF_paths = [_ for _ in data.keys() if _[-4:] == '.bin']
FF_true_paths = [_ for _ in data.keys() if _[-4:] == '.npz']
assert len(FF_paths) + len(FF_true_paths) == len(data)

# get list of area strings sorted in decreasing order of max inner
# area (i.e., in increasing order of problem size and mesh fineness)
area_strs = [_[3:-4] for _ in FF_true_paths]
area_strs = sorted(area_strs, key=lambda _:float(_.split('_')[0]), reverse=True)

max_inner_areas = [float(_.split('_')[0]) for _ in area_strs]

# get list of tolerance strings in decreasing order (i.e., in
# increasing order of accuracy)
tol_strs = {_[3:-4].split('_')[-1] for _ in FF_paths}
tol_strs = sorted(tol_strs, key=float, reverse=True)

max_tol = max(map(float, tol_strs))

def get_FF_data(area_str, tol_str):
    FF_path = next(_ for _ in FF_paths if area_str in _ and tol_str in _)
    return data[FF_path]

def get_FF_true_data(area_str):
    FF_true_path = next(_ for _ in FF_true_paths if area_str in _)
    return data[FF_true_path]

def plot_ROI_circle(**kwargs):
    theta = np.linspace(0, 2*np.pi, 301)
    x, y = r_ROI*np.cos(theta) + y_ROI, r_ROI*np.sin(theta) + x_ROI
    plt.plot(x, y, **kwargs)

rel_l2_errors = dict()

for area_str in area_strs:
    max_inner_area_str, max_outer_area_str = area_str.split('_')
    print(f'max inner area: {max_inner_area_str}, max outer area: {max_outer_area_str}')

    FF_true_data = get_FF_true_data(area_str)

    rel_l2_errors[area_str] = dict()

    for tol_str in tol_strs:

        try:
            print(f'- tol: {tol_str}')

            FF_data = get_FF_data(area_str, tol_str)

            P_st, A = FF_data['P_st'], FF_data['A']

            I_ROI = (P_st[:, 0] - x_ROI)**2 + (P_st[:, 1] - y_ROI)**2 < r_ROI**2

            T = FF_data['T']
            T_true = FF_true_data['T']

            rel_l2_err = np.linalg.norm((T[I_ROI] - T_true[I_ROI])*A[I_ROI]) \
                / np.linalg.norm(T_true[I_ROI]*A[I_ROI])

            print(f'  * relative l2 error: {rel_l2_err}')

            rel_l2_errors[area_str][tol_str] = rel_l2_err

            T_grid = FF_data['T_grid'].T[::-1,:]
            T_true_grid = FF_true_data['T_grid'].T[::-1,:]

            vmax = max(T_grid.max(), T_true_grid.max())

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title('$T$ (true)')
            plt.imshow(T_true_grid, interpolation='none', extent=extent,
                       vmin=0, vmax=vmax, cmap=cc.cm.fire)
            plot_ROI_circle(c='cyan', linewidth=1, linestyle='--', zorder=2)
            plt.ylabel('$y$')
            plt.xlabel('$x$')
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.subplot(1, 2, 2)
            plt.title(f'$T$ (tol = {tol_str} - true), '
                      # f'max:{round(np.max(np.abs(T_grid-T_true_grid)),2)}, '
                      f'median:{round(np.median(np.abs(T_grid-T_true_grid)),2)}')
            plt.imshow(T_grid-T_true_grid, interpolation='none', extent=extent,
                       vmin=0, vmax=2, cmap=cc.cm.fire, zorder=1)
            plot_ROI_circle(c='cyan', linewidth=1, linestyle='--', zorder=2)
            plt.ylabel('$y$')
            plt.xlabel('$x$')
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(plotdir/f'{area_str}_{tol_str}_T.png')
            plt.close()

            T_error_grid = T_grid - T_true_grid
            T_max_grid = np.maximum(T_grid, T_true_grid)
            T_max_grid[T_max_grid == 0] = np.inf
            rel_error_grid = abs(T_error_grid)/T_max_grid

            plt.figure()
            plt.imshow(rel_error_grid, extent=extent, interpolation='none',
                       vmin=0, vmax=max_tol, cmap=cc.cm.bmw)
            plot_ROI_circle(c='cyan', linewidth=1, linestyle='--', zorder=2)
            plt.xlabel('$y$')
            plt.ylabel('$x$')
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(plotdir/f'{area_str}_{tol_str}_rel_error.png')
            plt.close()
        except:
            print(f"Failed for {tol_str}")

# make relative l2 errors plot
plt.figure()
for tol_str in tol_strs:
    try:
        tol = float(tol_str)
        errs = [rel_l2_errors[area_str][tol_str] for area_str in area_strs]
        plt.axhline(y=tol, c='k', linewidth=1, linestyle='--', zorder=1)
        plt.loglog(max_inner_areas, errs, label=tol_str, marker='*', zorder=2)
    except:
        print(f"Failed for {tol_str}")
plt.legend()
plt.savefig(plotdir/'rel_l2_errors_area_vs_tol.png')
