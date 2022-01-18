#!/usr/bin/env python

import sys
import glob
import numpy as np
import os
import pickle
import scipy.sparse

from pathlib import Path

def read_pickled(pickle_path, path, pattern):
    objs = dict()
    for test_path in path.glob(pattern):
        p = int(str(test_path).split('_')[-1][1:])
        with open(os.path.join(test_path, pickle_path), 'rb') as f:
            objs[p] = pickle.load(f)
    return objs

def read_bins(bin_path, path, pattern):
    bins = dict()
    for test_path in path.glob(pattern):
        p = int(str(test_path).split('_')[-1][1:])
        bins[p] = np.fromfile(
            os.path.join(test_path, bin_path),
            dtype=np.float32)
    return bins

def read_spmats(spmat_npz_path, path, pattern):
    spmats = dict()
    for test_path in path.glob(pattern):
        p = int(str(test_path).split('_')[-1][1:])
        spmats[p] = scipy.sparse.load_npz(os.path.join(test_path, spmat_npz_path))
    return spmats

tol = sys.argv[1]

stats_path = Path('stats')
path_gt = stats_path/'gt'
path = stats_path/f'eps_{tol}'
comparison_path = stats_path/f'gt_vs_{tol}'

# Load binary files for direct comparisons

pattern = 'ingersoll_p*'

T_gt = read_bins('T.bin',  path_gt, pattern)
T =    read_bins('T.bin',  path,    pattern)

B_gt = read_bins('B.bin',  path_gt, pattern)
B =    read_bins('B.bin',  path,    pattern)

FF_gt = read_spmats('FF.npz', path_gt, pattern)
FF = read_pickled('FF.bin', path, pattern)

# Get p values
P = list(T.keys())

# Do direct comparisons

T_rel_l2_errors = np.array([
    np.linalg.norm(T_gt[p] - T[p])/np.linalg.norm(T_gt[p])
    for p in P
])

B_rel_l2_errors = np.array([
    np.linalg.norm(B_gt[p] - B[p])/np.linalg.norm(B_gt[p])
    for p in P
])

FF_rel_fro_errors = np.array([
    np.divide(
        np.linalg.norm((FF[p].tocsr() - FF_gt[p]).toarray(), ord='fro'),
        np.linalg.norm(FF_gt[p].toarray(), ord='fro'))
    for p in P
])

# Save results

comparison_path.mkdir(parents=True, exist_ok=True)

with open(comparison_path/'T_rel_l2_errors.pickle', 'wb') as f:
    pickle.dump(T_rel_l2_errors, f)

with open(comparison_path/'B_rel_l2_errors.pickle', 'wb') as f:
    pickle.dump(B_rel_l2_errors, f)

with open(comparison_path/'FF_rel_fro_errors.pickle', 'wb') as f:
    pickle.dump(FF_rel_fro_errors, f)
