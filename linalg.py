import numpy as np
import scipy.sparse.linalg


from debug import DebugLinearOperator, IndentedPrinter


def estimate_rank(spmat, tol, k0=40):
    k = k0
    while True:
        k = min(k, min(spmat.shape) - 1)
        with IndentedPrinter() as _:
            _.print('svds(%d x %d, %d)' % (*spmat.shape, k))
            wrapped_spmat = DebugLinearOperator(spmat)
            S = scipy.sparse.linalg.svds(wrapped_spmat, k,
                                         return_singular_vectors=False)
            wrapped_spmat.debug_print()
        sv_thresh = S[-1]*max(spmat.shape)*tol
        if S[0] < sv_thresh:
            return np.where((S < sv_thresh)[::-1])[0][0]
        k *= 2
        if k > min(spmat.shape):
            return min(spmat.shape)


def nnz(mat, tol=None):
    if tol is None:
        return (mat != 0).sum()
    else:
        return (abs(mat) < tol).sum()


def sparsity(mat, tol=None):
    nnz = _get_nnz(mat, tol)
    size = mat.size
    return 0 if size == 0 else nnz/size
