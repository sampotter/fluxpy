import numpy as np
import scipy.sparse.linalg


# from sparsesvd import sparsesvd


from flux.debug import DebugLinearOperator, IndentedPrinter
from flux.util import nbytes


# def sparse_svd(spmat, k):
#     dtype, dtype_eps = spmat.dtype, np.finfo(spmat.dtype).eps

#     Ut, S, Vt = sparsesvd(spmat.tocsc(), k)

#     Ut = Ut.astype(dtype)
#     Ut_abs = abs(Ut)
#     Ut[Ut_abs < dtype_eps*Ut_abs.max()] = 0
#     U = scipy.sparse.csr_matrix(Ut.T)

#     S = S.astype(dtype)

#     Vt = Vt.astype(dtype)
#     Vt_abs = abs(Vt)
#     Vt[Vt_abs < dtype_eps*Vt_abs.max()] = 0
#     Vt = scipy.sparse.csr_matrix(Vt)

#     return U, S, Vt


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
    m, k, sqrt_n = min(spmat.shape), k0, np.sqrt(spmat.shape[1])
    dtype_eps = np.finfo(spmat.dtype).eps
    while True:
        k = min(k, m - 1)
        U, S, Vt = sparse_svd(spmat, k)
        svd_nbytes = nbytes(U) + nbytes(S) + nbytes(Vt)
        if max_nbytes is not None and svd_nbytes >= max_nbytes:
            return None
        thresh = sqrt_n*S/S[0]
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
