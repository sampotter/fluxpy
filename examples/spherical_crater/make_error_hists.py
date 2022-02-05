#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

stats_path = Path('stats')

vmax = 0
for path in stats_path.glob('gt/ingersoll_p*'):
    T_error = np.load(path/'T_error.npy')
    vmax = max(vmax, abs(T_error).max())
vmin = -vmax

for path in stats_path.glob('gt/ingersoll_p*'):
    print(path)

    T_error = np.load(path/'T_error.npy')
    I_shadow = np.load(path/'I_shadow.npy')
    I_sun = np.load(path/'I_sun.npy')

    with open(path/'stats.json', 'r') as f:
        stats = json.load(f)

    num_faces = stats['num_faces']

    T_error_shadow_mean = T_error[I_shadow].mean()
    T_error_sun_mean = T_error[I_sun].mean()

    print(f'- mean error in shadow: {T_error_shadow_mean:1.2f} K')
    print(f'- mean error in sun: {T_error_sun_mean:1.2f} K')

    plt.figure()
    plt.hist(T_error[I_shadow], density=True, histtype='step',
             label=f'shadow ({num_faces})', bins=65, color='r')
    plt.hist(T_error[I_sun], density=True, histtype='step',
             label=f'sunlit ({num_faces})', bins=65, color='b')
    plt.axvline(x=T_error_shadow_mean, c='r', linestyle='--')
    plt.axvline(x=T_error_sun_mean, c='b', linestyle='--')
    plt.xlim(vmin, vmax)
    plt.legend()
    plt.savefig(f'spherical_crater_plots/hist_{num_faces}.png')
    plt.close()
