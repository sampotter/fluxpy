import numpy as np
import scipy.sparse.linalg

from flux.debug import DebugLinearOperator, IndentedPrinter
from flux.util import nbytes


def sparse_svd(spmat, k):
    with IndentedPrinter() as _:
        _.print('svds(%d x %d, %d)' % (*spmat.shape, k))
        wrapped_spmat = DebugLinearOperator(spmat)
        U, S, Vt = scipy.sparse.linalg.svds(wrapped_spmat, k)
        wrapped_spmat.debug_print()
    U = scipy.sparse.csr_matrix(U[:, ::-1])
    S = S[::-1]
    Vt = scipy.sparse.csr_matrix(Vt[::-1, :])
    return U, S, Vt


def estimate_rank(spmat, tol, max_nbytes=None, k0=40):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    m, k = min(spmat.shape), k0
    dtype_eps = np.finfo(spmat.dtype).eps
    while True:
        k = min(k, m - 1)
        if not k >= 1:
            raise RuntimeError('bad value of k')
        U, S, Vt = sparse_svd(spmat, k)
        svd_nbytes = nbytes(U) + nbytes(S) + nbytes(Vt)
        if max_nbytes is not None and svd_nbytes >= max_nbytes:
            return None
        try:
            thresh = S/S[0]
        except:
            import pdb; pdb.set_trace()
            pass
        if k == m:
            return U, S, Vt, thresh[m]
        assert thresh[0] >= 1
        below_thresh = np.where(thresh <= tol)[0]
        if below_thresh.size > 0:
            r = below_thresh[0]
            return U[:, :r], S[:r], Vt[:r, :], thresh[r]
        k *= 2


def nnz(mat, tol=None):
    if tol is None:
        return (mat != 0).sum()
    else:
        return (abs(mat) < tol).sum()


def sparsity(mat, tol=None):
    nnz = _get_nnz(mat, tol)
    size = mat.size
    return 0 if size == 0 else nnz/size


def sparsify(spmat, tol=1e-2):
    spmat_ = np.zeros(spmat.shape)
    for i, row in enumerate(spmat):
        if row.nnz == 0:
            continue
        row = row.toarray().ravel()
        abs_row = abs(row)
        index_array = np.argsort(abs_row)
        cumsum = np.cumsum(abs_row[index_array])
        k = np.where(normalized_cumsum >= tol)[0][0]
        # print(abs_row[index_array[:i]].sum()/abs_row.max())
        spmat_[i, index_array[k:]] = row[index_array[k:]]
    return scipy.sparse.csr_matrix(spmat_)

def winnow(spmat):
    I, J = spmat.nonzero()
    I, J = np.unique(I), np.unique(J)
    return spmat[I, :][:, J], I, J
