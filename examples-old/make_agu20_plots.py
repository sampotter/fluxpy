#!/usr/bin/env python

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    stats = dict()
    for dirname in glob.glob('eps_*'):
        eps_key = dirname.split('_')[1]
        stats[eps_key] = dict()

    for dirname in glob.glob('eps_*/ingersoll_*'):
        path = os.path.join(dirname, 'stats.json')
        if not os.path.exists(path):
            continue
        eps_key, p_key = dirname.split('/')
        eps_key = eps_key.split('_')[1]
        p_key = int(p_key.split('_')[1][1:])
        with open(path, 'r') as f:
            stats[eps_key][p_key] = json.load(f)

    plt.rcParams.update({
        "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_rms_error = np.array(
            [(v['num_faces'], v['rms_error'])
             for k, v in stats[eps_key].items()
             if v['rms_error'] < v['T_gt']/2]
        )
        I = np.argsort(N_vs_rms_error[:, 0])
        N_vs_rms_error = N_vs_rms_error[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_rms_error.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('RMS Error')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_rms_error.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_max_error = np.array(
            [(v['num_faces'], v['max_error'])
             for k, v in stats[eps_key].items()
             if v['max_error'] < v['T_gt']/2]
        )
        I = np.argsort(N_vs_max_error[:, 0])
        N_vs_max_error = N_vs_max_error[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_max_error.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('Max Error')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_max_error.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_T_assemble = np.array(
            [(v['num_faces'], v['t_FF'])
             for k, v in stats[eps_key].items()]
        )
        I = np.argsort(N_vs_T_assemble[:, 0])
        N_vs_T_assemble = N_vs_T_assemble[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_T_assemble.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('Assembly Time [s.]')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_T_assemble.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_T_solve = np.array(
            [(v['num_faces'], v['t_T'])
             for k, v in stats[eps_key].items()]
        )
        I = np.argsort(N_vs_T_solve[:, 0])
        N_vs_T_solve = N_vs_T_solve[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_T_solve.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('Compute $T$ time [s.]')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_T_solve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_B_solve = np.array(
            [(v['num_faces'], v['t_B'])
             for k, v in stats[eps_key].items()]
        )
        I = np.argsort(N_vs_B_solve[:, 0])
        N_vs_B_solve = N_vs_B_solve[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_B_solve.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('Compute $B$ time [s.]')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_B_solve.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for eps_key in stats.keys():
        N_vs_FF_size = np.array(
            [(v['num_faces'], v['FF_size'])
             for k, v in stats[eps_key].items()]
        )
        I = np.argsort(N_vs_FF_size[:, 0])
        N_vs_FF_size = N_vs_FF_size[I, :]
        eps_pow = int(eps_key[2:])
        plt.loglog(*N_vs_FF_size.T, label=r'$\epsilon = 10^{%d}$' % eps_pow,
                   marker='*')
    plt.legend()
    plt.ylabel('Size of compressed $F$ [Mb]')
    plt.xlabel('Faces')
    plt.savefig('plots/N_vs_FF_size.png')
    plt.close()
