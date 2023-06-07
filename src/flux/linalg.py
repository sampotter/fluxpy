import numpy as np
import scipy.sparse.linalg
from sklearn.decomposition import NMF

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
    while True:
        k = min(k, m - 1)
        if not k >= 1:
            raise RuntimeError('bad value of k')
        U, S, Vt = sparse_svd(spmat, k)
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
            svd_nbytes = nbytes(U[:, :r]) + nbytes(S[:r]) + nbytes(Vt[:r, :])
            if max_nbytes is not None and svd_nbytes >= max_nbytes:
                return None
            return U[:, :r], S[:r], Vt[:r, :], thresh[r]
        k *= 2


def fit_nmf(spmat, k, max_iters=int(1e3), tol=1e-4):
    # with IndentedPrinter() as _:
    #     _.print('nmf(%d x %d, %d)' % (*spmat.shape, k))
    #     wrapped_spmat = DebugLinearOperator(spmat)
    #     print(wrapped_spmat.shape)
    #     nmf = NMF(k,init='nndsvd', solver='cd', tol=1e-4, max_iter=int(1e4))
    #     W = nmf.fit_transform(wrapped_spmat)
    #     H = nmf.components_
    #     wrapped_spmat.debug_print()

    nmf = NMF(k,init='nndsvd', solver='cd', tol=tol, max_iter=max_iters)
    W = nmf.fit_transform(spmat)
    H = nmf.components_
    
    W = scipy.sparse.csr_matrix(W)
    H = scipy.sparse.csr_matrix(H)
    return W, H

# def fit_nmf(spmat, k, max_iters=int(1e3), tol=1e-2, eps=1e-5):

#     W_0 = np.random.uniform(low=0,high=10, size=(spmat.shape[0],k))
#     H_0 = np.random.uniform(low=0,high=10, size=(k,spmat.shape[1]))

#     diff_W = np.inf
#     diff_W_list = []

#     W = np.copy(W_0)
#     H = np.copy(H_0)
#     n_iter = 0
#     while diff_W > tol and n_iter < max_iters:

#         _H = np.multiply(H, np.divide(W.T @ spmat , (W.T @ W @ H) + eps))
#         H = _H

#         _W = np.multiply(W, np.divide(spmat @ (H.T), (W @ H @ (H.T)) + eps))
#         diff_W = np.linalg.norm(_W - W, ord='fro') / np.linalg.norm(W, ord='fro')
#         diff_W_list.append(diff_W)
#         W = _W

#         n_iter += 1

#     if n_iter == max_iters:
#         print("NMF exceeded max_iters.")
    
#     W = scipy.sparse.csr_matrix(W)
#     H = scipy.sparse.csr_matrix(H)
#     return W, H


def estimate_rank_nmf(spmat, tol, max_nbytes=None, k0=40, max_iters=int(1e3), nmf_tol=1e-2):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    m, k = min(spmat.shape), k0
    num_iters = 0
    while True:
        num_iters += 1
        k = min(k, m)
        if not k >= 1:
            raise RuntimeError('bad value of k')
        W, H = fit_nmf(spmat, k, max_iters=max_iters, tol=nmf_tol)
        nmf_nbytes = nbytes(W) + nbytes(H)
        if max_nbytes is not None and nmf_nbytes >= max_nbytes:
            return None
        thresh = scipy.sparse.linalg.norm((W@H) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
        if k == m or thresh <= tol:
            return W, H, thresh
        k *= 2



def estimate_sparsity_nmf(spmat, tol, max_nbytes=None, k0=5, max_iters=int(1e3), nmf_tol=1e-2):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    W, H = fit_nmf(spmat, k0, max_iters=max_iters, tol=nmf_tol)
    resid = (spmat - (W@H)).A

    num_resids = resid.shape[0] * resid.shape[1]
    nnz_resid = np.count_nonzero(resid)

    sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

    target = np.power(scipy.sparse.linalg.norm((W@H) - spmat, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

    if target <= 0:
        Sr = np.zeros_like(resid)
        Sr = scipy.sparse.csr_matrix(Sr)

    else:
        cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
        keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

        Sr = np.copy(resid)
        Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
        Sr = scipy.sparse.csr_matrix(Sr)

    sparse_nmf_nbytes = nbytes(W) + nbytes(H) + nbytes(Sr)
    if max_nbytes is not None and sparse_nmf_nbytes >= max_nbytes:
        return None

    return W, H, Sr

    # m, s = min(spmat.shape), s0
    # num_iters = 0
    # while True:
    #     num_iters += 1
    #     s = min(s, nnz_resid)
    #     if not s >= 1:
    #         raise RuntimeError('bad value of s')

    #     Sr = np.copy(resid)
    #     Sr[sorted_resid_idx[0][:num_resids-s], sorted_resid_idx[1][:num_resids-s]] = 0.
    #     Sr = scipy.sparse.csr_matrix(Sr)
        
    #     sparse_nmf_nbytes = nbytes(W) + nbytes(H) + nbytes(Sr)
    #     if max_nbytes is not None and sparse_nmf_nbytes >= max_nbytes:
    #         return None
    #     thresh = scipy.sparse.linalg.norm(((W@H) + Sr) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
    #     if s == nnz_resid or thresh <= tol:
    #         return W, H, Sr, thresh
    #     s *= 2



def estimate_sparsity_nmf_weighted(spmat, FF_weights, tol, max_nbytes=None, k0=5, max_iters=int(1e3), nmf_tol=1e-2):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    W, H = fit_nmf(spmat, k, max_iters=max_iters, tol=nmf_tol)
    resid = (spmat - (W@H)).A

    num_resids = resid.shape[0] * resid.shape[1]
    nnz_resid = np.count_nonzero(resid)

    weighted_resid = np.multiply(FF_weights, resid)
    sorted_weighted_resid_idx = np.unravel_index(np.argsort(abs(weighted_resid), axis=None), resid.shape)

    m, s = min(spmat.shape), s0
    num_iters = 0
    while True:
        num_iters += 1
        s = min(s, nnz_resid)
        if not s >= 1:
            raise RuntimeError('bad value of s')

        Sr = np.copy(resid)
        Sr[sorted_weighted_resid_idx[0][:num_resids-s], sorted_weighted_resid_idx[1][:num_resids-s]] = 0.
        Sr = scipy.sparse.csr_matrix(Sr)
        
        sparse_nmf_nbytes = nbytes(W) + nbytes(H) + nbytes(Sr)
        if max_nbytes is not None and sparse_nmf_nbytes >= max_nbytes:
            return None
        thresh = scipy.sparse.linalg.norm(((W@H) + Sr) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
        if s == nnz_resid or thresh <= tol:
            return W, H, Sr, thresh
        s *= 2



def nnz(mat, tol=None):
    if tol is None:
        return (mat != 0).sum()
    else:
        return (abs(mat) < tol).sum()


def sparsity(mat, tol=None):
    size = mat.size
    return 0 if size == 0 else nnz(mat, tol)/size
