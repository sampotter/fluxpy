import pickle

import flux.config
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from cached_property import cached_property

from flux.debug import IndentedPrinter

def get_form_factor_matrix(shape_model, I=None, J=None, eps=None):
    P = shape_model.P
    N = shape_model.N
    A = shape_model.A

    if eps is None:
        eps = flux.config.DEFAULT_EPS
    if I is None:
        I = np.arange(P.shape[0], dtype=np.uintp)
    if J is None:
        J = np.arange(P.shape[0], dtype=np.uintp)

    m, n = len(I), len(J)

    NJ_PJ = np.sum(N[J]*P[J], axis=1)
    AJ = A[J]

    # NOTE: we're deliberately using Python's built-in lists here
    # to accumulate the data for the sparse matrix instead of
    # numpy arrays, since using np.concatenate in the loop below
    # ends up making _compute_FF_block O(N^3).
    size = m
    data = np.empty(m, dtype=shape_model.dtype)
    indices = np.empty(m, dtype=np.uintp)
    indptr = [0]

    # Current index into data and indices
    i0 = 0

    for r, i in enumerate(I):
        row_data = np.maximum(0, N[i]@(P[J] - P[i]).T) \
            * np.maximum(0, P[i]@N[J].T - NJ_PJ)

        # Set diagonal entries to zero.
        row_data[i == J] = 0

        row_indices = np.where(abs(row_data) > eps)[0]
        if row_indices.size == 0:
            indptr.append(indptr[-1])
            continue
        vis = shape_model.get_visibility_1_to_N(i, J[row_indices].astype(np.uintp))
        row_indices = row_indices[vis]

        s = np.pi*np.sum((P[i] - P[J[row_indices]])**2, axis=1)**2
        s[s == 0] = np.inf
        row_data = row_data[row_indices]*AJ[row_indices]/s

        assert row_data.size == row_indices.size

        if size < row_data.size + i0:
            size = max(2*size, row_data.size + i0)

            tmp = np.empty(size, dtype=shape_model.dtype)
            tmp[:i0] = data[:i0]
            data = tmp

            tmp = np.empty(size, dtype=np.intp)
            tmp[:i0] = indices[:i0]
            indices = tmp

        data[i0:(row_data.size + i0)] = row_data
        indices[i0:(row_indices.size + i0)] = row_indices

        i0 += row_data.size

        indptr.append(indptr[-1] + row_indices.size)

    data = data[:i0]
    indices = indices[:i0]
    indptr = np.array(indptr, dtype=np.intp)

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))
