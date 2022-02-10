#!/usr/bin/env python

import json_numpy as json
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

from pathlib import Path

from util import *

# with open('params.json') as f:
#     params = json.load(f)

# xmin = params['xmin']
# xmax = params['xmax']
# ymin = params['ymin']
# ymax = params['ymax']
# xmin = -10
# xmax = 10
# ymin = -55
# ymax = -35
# extent = [xmin, xmax, ymin, ymax]

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

T_frames_path = Path('T_frames')
T_frame_dirs = list(T_frames_path.iterdir())
frame_names = [_.name for _ in T_frame_dirs[0].glob('T*.npy')]

rel_l2_errors = dict()
T_avgs = dict()

for i, frame_name in enumerate(frame_names):
    print(i, frame_name)

    for area_str in area_strs:

        if area_str not in rel_l2_errors:
            rel_l2_errors[area_str] = dict()

        if area_str not in T_avgs:
            T_avgs[area_str] = dict()

        if area_str not in T_stds:
            T_stds[area_str] = dict()

        shape_model_st = load_stereographic_shape_model(area_str)

        T_frame_dir_true = next(_ for _ in T_frame_dirs
                                if area_str in _.name and 'true' in _.name)
        T_frame_path_true = T_frame_dir_true/frame_name
        T_frame_true = np.load(T_frame_path_true)

        for tol_str in tol_strs:
            T_frame_dir = next(_ for _ in T_frame_dirs
                               if area_str in _.name and tol_str in _.name)
            T_frame_path = T_frame_dir/frame_name
            T_frame = np.load(T_frame_path)

            rel_l2_error = np.linalg.norm(T_frame - T_frame_true)
            rel_l2_error /= np.linalg.norm(T_frame_true)

            if tol_str not in rel_l2_errors[area_str]:
                rel_l2_errors[area_str][tol_str] = []
            rel_l2_errors[area_str][tol_str].append(rel_l2_error)

            if tol_str not in T_avgs[area_str]:
                T_avgs[area_str][tol_str] = []
            T_avgs[area_str][tol_str].append(T_frame.mean())

            # if i == 90 and tol_str == '1e-1':
            #     xgrid = np.linspace(xmin, xmax, 256)
            #     ygrid = np.linspace(ymin, ymax, 256)

            #     depth = 25
            #     diff = abs(T_frame[:, depth] - T_frame_true[:, depth])
            #     diff /= abs(T_frame[:, depth])
            #     diff_grid = raytrace_values(shape_model_st, diff, xgrid, ygrid)

            #     T_frame_grid = raytrace_values(shape_model_st, T_frame[:, depth], xgrid, ygrid)

            #     plt.figure()
            #     plt.imshow(diff_grid, interpolation='none', extent=extent)
            #     # plt.imshow(T_frame_grid, interpolation='none', extent=extent)
            #     plt.colorbar()
            #     plt.show()

linestyles = ['-', '--', ':']
colors = ['k', 'b', 'r']

plt.figure()
for linestyle, tol_str in zip(linestyles, tol_strs):
    for color, area_str in zip(colors, area_strs):
        plt.semilogy(rel_l2_errors[area_str][tol_str],
                     linestyle=linestyle, linewidth=1,
                     c=color,
                     label=f'tol = {tol_str}, area = {area_str}')
plt.legend()
plt.title(f'Relative $\ell_2$ error in temperature field')
plt.ylabel(r'$\|\hat{T}_n - T_n\|_2/\|T_n\|$')
plt.xlabel('$n$ (Iteration)')
plt.savefig('gerlache_plots/time_dep_error.png')

plt.figure()
for linestyle, tol_str in zip(linestyles, tol_strs):
    for color, area_str in zip(colors, area_strs):
        plt.plot(T_avgs[area_str][tol_str],
                 linestyle=linestyle, linewidth=1,
                 c=color,
                 label=f'tol = {tol_str}, area = {area_str}')
plt.legend()
plt.title(f'Average temperature [K]')
plt.ylabel(r'$<T_n>$')
plt.xlabel('$n$ (Iteration)')
plt.savefig('gerlache_plots/time_dep_T_avg.png')
