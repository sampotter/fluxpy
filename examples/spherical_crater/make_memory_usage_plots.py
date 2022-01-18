#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

PAPER_PLOT_DIR = sys.argv[1]

from plot_style import linewidth, dpi

mem_T_gt = np.load('mprof_stats/gt/mem_T.npy')
mem_usage_gt = np.load('mprof_stats/gt/mem_usage.npy')

mem_T_1em2 = np.load('mprof_stats/1e-2/mem_T.npy')
mem_usage_1em2 = np.load('mprof_stats/1e-2/mem_usage.npy')

plt.figure(figsize=(6.5, 3))
plt.plot(mem_T_gt, mem_usage_gt, label='True $\mathbf{F}$',
         linewidth=linewidth, c='k', linestyle='-')
plt.plot(mem_T_1em2, mem_usage_1em2, label='HODLR $\mathbf{F}$',
         linewidth=linewidth, c='k', linestyle='--')
plt.legend()
plt.ylabel('Memory use [MB]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(f'{PAPER_PLOT_DIR}/memory_usage.pdf', dpi=dpi)
plt.savefig(f'{PAPER_PLOT_DIR}/memory_usage.png', dpi=dpi)
