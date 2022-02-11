#!/usr/bin/env python

import itertools as it
import numpy as np
import scipy.sparse
import sys

from numpy.linalg import norm

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.util import nbytes

stats = dict()

planets, sizes, tols = set(), set(), set()
Ts = dict()

for i in range(len(sys.argv)//5):
    form_factor_matrix_path = sys.argv[5*i + 1]
    if '.npz' in form_factor_matrix_path:
        FF = scipy.sparse.load_npz(form_factor_matrix_path)
        planet, size = form_factor_matrix_path.split('.')[0].split('_')
        tol = 'true'
    else:
        FF = CompressedFormFactorMatrix.from_file(form_factor_matrix_path)
        planet, size, tol = form_factor_matrix_path.split('.')[0].split('_')
    FF_nbytes = nbytes(FF)

    planets.add(planet)
    sizes.add(size)
    tols.add(tol)

    assembly_time_path = sys.argv[5*i + 2]
    with open(assembly_time_path, 'r') as f:
        assembly_time = f.readline().strip()

    T_path = sys.argv[5*i + 3]
    T = np.load(T_path)
    Ts[planet, size, tol] = T

    T_time_path = sys.argv[5*i + 4]
    with open(T_time_path, 'r') as f:
        T_time = f.readline().strip()

    p_path = sys.argv[5*i + 5]
    p = np.load(p_path)

    stats[planet, size, tol] = {
        'FF_nbytes': FF_nbytes,
        'assembly_time': assembly_time,
        'T_time': T_time,
        'T_max': T.max(),
        'T_mean': T.mean(),
        'T_min': T.min(),
        'p_max': p.max(),
        'p_mean': p.mean(),
    }

for planet, size, tol in it.product(planets, sizes, tols):
    if tol == 'true':
        continue

    T = Ts[planet, size, tol]
    T_true = Ts[planet, size, 'true']
    E = T - T_true

    stats[planet, size, tol]['l1'] = norm(E, ord=1)/norm(T_true, ord=1)
    stats[planet, size, tol]['l2'] = norm(E, ord=2)/norm(T_true, ord=2)
    stats[planet, size, tol]['linf'] = norm(E, ord=np.inf)/norm(T_true, ord=np.inf)
