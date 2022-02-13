#!/usr/bin/env python

import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from matplotlib.ticker import AutoMinorLocator
from pathlib import Path

from plot_style import linewidth, dpi, marker, colors, linestyles, square_figsize

PAPER_PLOT_DIR = sys.argv[1]
SAVE_PDF_PLOTS = False

def load_stats(path):
    with open(path, 'r') as f:
        return json.load(f)

def print_stats(stats):
    print(json.dumps(stats, sort_keys=True, indent=4))

def read_all_stats_files_to_dicts(glob_path, verbose=False):
    Stats = dict()
    for test_path in glob.glob(glob_path):
        p = int(test_path.split('_')[-1][1:])
        stats_path = os.path.join(test_path, 'stats.json')
        stats = load_stats(stats_path)
        if verbose:
            print_stats(stats)
        Stats[p] = stats
    return Stats

def tol_from_path(path):
    return path.split('_')[-1]

def tol_to_tex(tol):
    return '10^{%d}' % int(tol[2:])

def load_direct_comparison_data_to_dict(path):
    Data = dict()
    with open(path/'B_rel_l2_errors.pickle', 'rb') as f:
        Data['B_rel_l2_errors'] = pickle.load(f)
    with open(path/'T_rel_l2_errors.pickle', 'rb') as f:
        Data['T_rel_l2_errors'] = pickle.load(f)
    return Data

def get_values_by_key(dicts, key):
    lst = []
    for k in sorted(dicts):
        d = dicts[k]
        lst.append(d[key])
    return lst

stats_gt_path = 'stats/gt'
stats_path_pattern = 'stats/eps_*'
comparison_path_pattern = 'stats/gt_vs_*'

################################################################################
# MAKE INGERSOLL PLOTS
#

# plot parameters

matplotlib.rcParams.update({'font.size': 18})

# Load statistics
StatsGt = read_all_stats_files_to_dicts(os.path.join(stats_gt_path, 'ingersoll_p*'))
Stats = {
    tol_from_path(path): read_all_stats_files_to_dicts(
        os.path.join(path, 'ingersoll_p*'))
    for path in glob.glob(stats_path_pattern)
}

# Load direct comparison data
GtVsTol = {
    tol_from_path(path): load_direct_comparison_data_to_dict(
        Path(f'./stats/gt_vs_{tol_from_path(path)}'))
    for path in glob.glob(comparison_path_pattern)
}

Tols = list(GtVsTol.keys())
Tols = sorted(Tols, key=lambda tol: float(tol))

# Get values of H
H = get_values_by_key(StatsGt, 'h')
N = get_values_by_key(StatsGt, 'num_faces')

# Make loglog N vs T rel errors plot_blocks
for j, order in enumerate(['max', 'l2', 'l1']):
    plt.figure(figsize=square_figsize)
    plt.loglog(N, get_values_by_key(StatsGt, f'rel_{order}_T_error'),
               linewidth=linewidth, marker='o', c=colors[0],
               label='Dense $F$', zorder=1, linestyle=linestyles[0])
    for i, tol in enumerate(Tols):
        plt.loglog(N, get_values_by_key(Stats[tol], f'rel_{order}_T_error'),
                   linewidth=linewidth, marker=marker, c=colors[i + 1],
                   label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
                   zorder=2, linestyle=linestyles[j + 1])
    plt.legend()
    plt.xlabel('$N$')
    order_tex = {
        'max': '$\ell_\infty$',
        'l2': '$\ell_2$',
        'l1': '$\ell_1$'
    }[order]
    plt.ylabel(r'Relative {order_tex} error in $T$ (shadow)')
    plt.tight_layout()
    if SAVE_PDF_PLOTS:
        plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_{order}.pdf', dpi=dpi)
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_{order}.png', dpi=dpi)
    plt.close()

# Make loglog N vs T_rel_l2_errors and B_rel_l2_errors plot
plt.figure(figsize=square_figsize)
for i, tol in enumerate(Tols):
    plt.loglog(
        N, GtVsTol[tol]['T_rel_l2_errors'],
        linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='-',
        label=r'$\|T_{gt} - T\|_2/\|T_{gt}\|_2$ ($\epsilon = %s$)' % (
            tol_to_tex(tol),))
    plt.loglog(
        N, GtVsTol[tol]['B_rel_l2_errors'],
        linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
        label=r'$\|B_{gt} - B\|_2/\|B_{gt}\|_2$ ($\epsilon = %s$)' % (
            tol_to_tex(tol),))
plt.legend()
plt.xlabel('$N$')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_ptwise_errors.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_ptwise_errors.png', dpi=dpi)
plt.close()

# Make loglog N vs size plot
plt.figure(figsize=square_figsize)
plt.axline((N[0], get_values_by_key(StatsGt, 'FF_size')[0]), slope=1,
           linewidth=2, c='palegoldenrod', linestyle='--', label=r'$O(N)$', zorder=0)
plt.axline((N[0], get_values_by_key(StatsGt, 'FF_size')[0]), slope=2,
           linewidth=2, c='pink', linestyle='--', label=r'$O(N^2)$', zorder=0)
plt.loglog(N, get_values_by_key(StatsGt, 'FF_size'),
           linewidth=linewidth, marker=marker, c=colors[0], label='True $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(N, get_values_by_key(Stats[tol], 'FF_size'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Size of $F$ [MB]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_size.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_size.png', dpi=dpi)
plt.close()

# Make loglog N vs compute T time plot
plt.figure(figsize=square_figsize)
plt.axline((N[0], get_values_by_key(StatsGt, 't_T')[0]), slope=1,
           linewidth=2, c='palegoldenrod', linestyle='--', label=r'$O(N)$', zorder=0)
plt.axline((N[0], get_values_by_key(StatsGt, 't_T')[0]), slope=2,
           linewidth=2, c='pink', linestyle='--', label=r'$O(N^2)$', zorder=0)
plt.loglog(N, get_values_by_key(StatsGt, 't_T'),
           linewidth=linewidth, marker=marker, c=colors[0], label='True $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(N, get_values_by_key(Stats[tol], 't_T'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Time to compute $T$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_T_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_T_time.png', dpi=dpi)
plt.close()

# Make loglog N vs compute B and E time plot
plt.figure(figsize=square_figsize)
plt.axline((N[0], get_values_by_key(StatsGt, 't_E')[0]), slope=1,
           linewidth=2, c='palegoldenrod', linestyle='--', label=r'$O(N)$', zorder=0)
plt.axline((N[0], get_values_by_key(StatsGt, 't_E')[0]), slope=2,
           linewidth=2, c='pink', linestyle='--', label=r'$O(N^2)$', zorder=0)
plt.loglog(N, get_values_by_key(StatsGt, 't_B'),
           linewidth=linewidth, marker=marker, c=colors[0], label='True $F$', zorder=1)
plt.loglog(N, get_values_by_key(StatsGt, 't_E'), linewidth=linewidth, marker=marker,
           c=colors[0], linestyle='--',
           label='Compute $E$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(N, get_values_by_key(Stats[tol], 't_B'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Time to compute $B$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_B_and_E_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_B_and_E_time.png', dpi=dpi)
plt.close()

# Make loglog N vs assembly time plot
plt.figure(figsize=square_figsize)
plt.axline((N[0], get_values_by_key(StatsGt, 't_FF')[0]), slope=2,
           linewidth=2, c='pink', linestyle='--', label=r'$O(N^2)$', zorder=0)
plt.axline((N[0], get_values_by_key(StatsGt, 't_FF')[0]), slope=3,
           linewidth=2, c='lightsteelblue', linestyle='--', label=r'$O(N^3)$', zorder=0)
plt.loglog(N, get_values_by_key(StatsGt, 't_FF'),
           linewidth=linewidth, marker=marker, c=colors[0], label='True $F$', zorder=1)
for i, tol in enumerate(Tols):
    plt.loglog(N, get_values_by_key(Stats[tol], 't_FF'),
               linewidth=linewidth, marker=marker, c=colors[i + 1], linestyle='--',
               label=r'Compressed $F$ ($\epsilon = %s$)' % (tol_to_tex(tol),),
               zorder=2)
plt.legend()
plt.xlabel('$N$')
plt.ylabel('Time to assemble $F$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_assembly_time.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/n_vs_assembly_time.png', dpi=dpi)
plt.close()
