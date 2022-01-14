#!/usr/bin/env python

import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from pathlib import Path

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

def load_direct_comparison_data_to_dict(path):
    Data = dict()
    with open(path/'B_rel_l2_errors.pickle', 'rb') as f:
        Data['B_rel_l2_errors'] = pickle.load(f)
    with open(path/'T_rel_l2_errors.pickle', 'rb') as f:
        Data['T_rel_l2_errors'] = pickle.load(f)
    with open(path/'FF_rel_fro_errors.pickle', 'rb') as f:
        Data['FF_rel_fro_errors'] = pickle.load(f)
    return Data

def get_values_by_key(dicts, key):
    lst = []
    for k in sorted(dicts):
        d = dicts[k]
        lst.append(d[key])
    return lst

stats_gt_path = 'stats/gt'
stats_path = 'stats/eps_1e-2'

################################################################################
# MAKE INGERSOLL PLOTS
#

# plot parameters
linewidth = 3
dpi = 100
marker = 'o'

matplotlib.rcParams.update({'font.size': 18})

# Load statistics
StatsGt = read_all_stats_files_to_dicts(os.path.join(stats_gt_path, 'ingersoll_p*'))
Stats = read_all_stats_files_to_dicts(os.path.join(stats_path, 'ingersoll_p*'))

# Load direct comparison data
GtVs1em2 = load_direct_comparison_data_to_dict(Path('./stats/gt_vs_1e-2'))

# Get values of H
H = get_values_by_key(StatsGt, 'h')

# Make loglog h vs T_rms plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 'rms_error'),
           linewidth=linewidth, marker='o', c='black', label='Sparse $F$', zorder=1)
plt.loglog(H, get_values_by_key(Stats, 'rms_error'),
           linewidth=linewidth, marker=marker, c='magenta', linestyle='--',
           label='Compressed $F$', zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('RMS error in $T$ (shadow)')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_rms.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_rms.png', dpi=dpi)
plt.close()

# Make loglog h vs T_rel_l2_errors and B_rel_l2_errors plot
plt.figure(figsize=(6, 6))
plt.loglog(H, GtVs1em2['T_rel_l2_errors'], linewidth=linewidth, marker=marker,
           c='black', linestyle='-', label=r'$\|T_{gt} - T\|_2/\|T_{gt}\|_2$')
plt.loglog(H, GtVs1em2['B_rel_l2_errors'], linewidth=linewidth, marker=marker,
           c='black', linestyle='--', label=r'$\|B_{gt} - B\|_2/\|B_{gt}\|_2$')
plt.legend()
plt.xlabel('$h$')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_ptwise_errors.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_ptwise_errors.png', dpi=dpi)
plt.close()

# Make loglog h vs FF_rel_fro_errors
plt.figure(figsize=(6, 6))
plt.loglog(H, GtVs1em2['FF_rel_fro_errors'], linewidth=linewidth, marker=marker,
           c='black', linestyle='-')
plt.ylabel(r'$\|\mathbf{F}_{gt} - \mathbf{F}\|_{F}/\|\mathbf{F}_{gt}\|_{F}$')
plt.xlabel(r'$h$')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/FF_rel_fro_errors.pdf', dpi=dpi)
plt.savefig('paper_plots/FF_rel_fro_errors.png', dpi=dpi)
plt.close()

# Make loglog h vs size plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 'FF_size'),
           linewidth=linewidth, marker=marker, c='black', label='Sparse $F$', zorder=1)
plt.loglog(H, get_values_by_key(Stats, 'FF_size'),
           linewidth=linewidth, marker=marker, c='magenta', linestyle='--',
           label='Compressed $F$', zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Size of $F$ [MB]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_size.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_size.png', dpi=dpi)
plt.close()

# Make loglog h vs compute T time plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_T'),
           linewidth=linewidth, marker=marker, c='black', label='Sparse $F$', zorder=1)
plt.loglog(H, get_values_by_key(Stats, 't_T'),
           linewidth=linewidth, marker=marker, c='magenta', linestyle='--',
           label='Compressed $F$', zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to compute $T$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_T_time.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_T_time.png', dpi=dpi)
plt.close()

# Make loglog h vs compute B and E time plot
E_time = [
    min(t, t_gt) for t, t_gt in zip(
        get_values_by_key(Stats, 't_E'), get_values_by_key(StatsGt, 't_E'))]
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_B'),
           linewidth=linewidth, marker=marker, c='black', label='Sparse $F$', zorder=1)
plt.loglog(H, E_time, linewidth=linewidth, marker=marker,
           c='black', linestyle='--',
           label='Compute $E$', zorder=1)
plt.loglog(get_values_by_key(Stats, 'h'), get_values_by_key(Stats, 't_B'),
           linewidth=linewidth, marker=marker, c='magenta', linestyle='--',
           label='Compressed $F$', zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to compute $B$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_B_and_E_time.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_B_and_E_time.png', dpi=dpi)
plt.close()

# Make loglog h vs assembly time plot
plt.figure(figsize=(6, 6))
plt.loglog(H, get_values_by_key(StatsGt, 't_FF'),
           linewidth=linewidth, marker=marker, c='black', label='Sparse $F$', zorder=1)
plt.loglog(H, get_values_by_key(Stats, 't_FF'),
           linewidth=linewidth, marker=marker, c='magenta', linestyle='--',
           label='Compressed $F$', zorder=2)
plt.legend()
plt.xlabel('$h$')
plt.ylabel('Time to assemble $F$ [s]')
plt.tight_layout()
if SAVE_PDF_PLOTS:
    plt.savefig('paper_plots/h_vs_assembly_time.pdf', dpi=dpi)
plt.savefig('paper_plots/h_vs_assembly_time.png', dpi=dpi)
plt.close()
