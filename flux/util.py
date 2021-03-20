import time

from functools import singledispatch

import scipy.sparse

t0 = None

def tic():
    global t0
    t0 = time.perf_counter()

def toc():
    global t0
    t = time.perf_counter()
    return t - t0

@singledispatch
def nbytes(arg):
    return arg.nbytes

@nbytes.register
def _(arg: scipy.sparse.csr_matrix):
    return arg.data.nbytes + arg.indices.nbytes + arg.indptr.nbytes
