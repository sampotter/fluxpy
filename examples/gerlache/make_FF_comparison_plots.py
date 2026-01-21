#!/usr/bin/env python

import json_numpy as json
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.sparse

plt.ion()

from pathlib import Path

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.util import nbytes

plt.figure()

plotdir = Path('gerlache_plots')

if not plotdir.exists():
    plotdir.mkdir()

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

# figure out the number of faces for each triangle mesh
num_faces = []
for area_str in area_strs:
    F = np.load(f'gerlache_faces_{area_str}.npy')
    num_faces.append(F.shape[0])

# PLOT: SIZE OF FORM FACTOR MATRIX

FF_nbytes = dict()
for area_str in area_strs:
    FF_nbytes[area_str] = dict()

    for tol_str in tol_strs:
        try:
            path = next(_ for _ in FF_paths if area_str in _ and tol_str in _)
            FF = CompressedFormFactorMatrix.from_file(path)
            FF_nbytes[area_str][tol_str] = FF.nbytes
        except:
            print(f"Failed for {tol_str}")

    path = next(_ for _ in FF_true_paths if area_str in _)
    FF = scipy.sparse.load_npz(path)
    FF_nbytes[area_str]['true'] = nbytes(FF)

df = pandas.DataFrame(FF_nbytes).T

plt.clf()
for tol_str in tol_strs + ['true']:
    plt.loglog(num_faces, df[tol_str]/1024**2, marker='*', label=tol_str)
plt.ylabel('Size [MB]')
plt.xlabel('$N$')
plt.legend()
plt.savefig(plotdir/'FF_sizes.png')

# PLOT: FORM FACTOR MATRIX TIMINGS

FF_assembly_times = dict()
with open('FF_assembly_times.txt', 'r') as f:
    for line in f:
        tol_str, max_inner_area_str, max_outer_area_str, time_str = line.split()
        area_str = max_inner_area_str + '_' + max_outer_area_str
        if area_str not in FF_assembly_times:
            FF_assembly_times[area_str] = dict()
        FF_assembly_times[area_str][tol_str] = float(time_str)
df_assembly_times = pandas.DataFrame(FF_assembly_times).T

FF_T_times = dict()
for area_str in area_strs:
    FF_T_times[area_str] = dict()
    for tol_str in tol_strs:
        try:
            path = next(_ for _ in FF_paths if area_str in _ and tol_str in _)
            FF_T_times[area_str][tol_str] = data[path]['time_T']
        except:
            print(f"Failed on FFtim {tol_str}")
    path = next(_ for _ in FF_true_paths if area_str in _)
    FF_T_times[area_str]['true'] = data[path]['time_T']
df_T_times = pandas.DataFrame(FF_T_times).T

plt.clf()
for tol_str in tol_strs + ['true']:
    try:
        plt.loglog(num_faces, df_assembly_times[tol_str], marker='*',
                label=f'assembly ({tol_str})')
    except:
        print(f"Failed on FFtim plot {tol_str}")
for tol_str in tol_strs + ['true']:
    try:
        print(tol_str, num_faces, df_T_times[tol_str])
        plt.loglog(num_faces, df_T_times[tol_str], marker='*', linestyle='--',
               label=f'compute T ({tol_str})')
    except:
        print(f"Failed on Ttim plot {tol_str}")
plt.ylabel('Time [s]')
plt.xlabel('$N$')
plt.legend()
plt.savefig(plotdir/'FF_times.png')

plt.close()
