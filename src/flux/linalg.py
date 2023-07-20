import numpy as np
import scipy.sparse.linalg
from sklearn.decomposition import NMF

from flux.debug import DebugLinearOperator, IndentedPrinter
from flux.util import nbytes




# SVD ALGORITHMS

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
        if k == m-1:
            return U, S, Vt, thresh[k-1]
        assert thresh[0] >= 1
        below_thresh = np.where(thresh <= tol)[0]
        if below_thresh.size > 0:
            r = below_thresh[0]
            svd_nbytes = nbytes(U[:, :r]) + nbytes(S[:r]) + nbytes(Vt[:r, :])
            if max_nbytes is not None and svd_nbytes >= max_nbytes:
                return None
            return U[:, :r], S[:r], Vt[:r, :], thresh[r]
        k *= 2


def estimate_sparsity_svd(spmat, tol, max_nbytes=None, k0=40):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    prev_nbytes = np.inf
    Sr_prev = np.zeros(spmat.shape)
    
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m - 1)
        
        if not k >= 1:
            raise RuntimeError('bad value of k')
        
        U, S, Vt = sparse_svd(spmat, k)
        
        for kk in range(prev_k+1,k+1):
            
            Uk, Sk, Vtk = U[:, :kk], S[:kk], Vt[:kk, :]
            resid = (spmat - (Uk @ (Vtk.T * np.diag(Sk)).T)).A

            if kk == m-1:
                return U[:, :kk], S[:kk], Vt[:kk, :], scipy.sparse.csr_matrix(np.zeros_like(resid))
                
            num_resids = resid.shape[0] * resid.shape[1]
            nnz_resid = np.count_nonzero(resid)

            sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

            target = np.power(np.linalg.norm(resid, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

            if target <= 0:
                Sr = np.zeros_like(resid)
                Sr = scipy.sparse.csr_matrix(Sr)

            else:
                cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
                keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

                Sr = np.copy(resid)
                Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
                Sr = scipy.sparse.csr_matrix(Sr)

            sparse_svd_nbytes = nbytes(Uk) + nbytes(Sk) + nbytes(Vtk) + nbytes(Sr)

            if sparse_svd_nbytes >= prev_nbytes:
                return U[:, :kk-1], S[:kk-1], Vt[:kk-1, :], Sr_prev

            if max_nbytes is not None and sparse_svd_nbytes >= max_nbytes:
                return None
            

            prev_nbytes = sparse_svd_nbytes
            Sr_prev = Sr

        prev_k = k
        k *= 2




# NMF ALGORITHMS

def fit_nmf(spmat, k, max_iters=int(1e3), tol=1e-4, beta_loss=2, init='svd'):
    if beta_loss == 2:
        nmf = NMF(k,init='nndsvd' if init=='svd' else 'random', solver='cd', tol=tol, max_iter=max_iters, beta_loss=beta_loss)
    elif beta_loss == 1:
        nmf = NMF(k,init='nndsvdar' if init=='svd' else 'random', solver='mu', tol=tol, max_iter=max_iters, beta_loss=beta_loss)
    W = nmf.fit_transform(spmat)
    H = nmf.components_

    if np.isnan(W).any() or np.isnan(H).any():
        return None, None
    
    W = scipy.sparse.csr_matrix(W)
    H = scipy.sparse.csr_matrix(H)
    return W, H


def estimate_rank_nmf(spmat, tol, max_nbytes=None, k0=40, max_iters=int(1e3), nmf_tol=1e-2, beta_loss=2):
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
        W, H = fit_nmf(spmat, k, max_iters=max_iters, tol=nmf_tol, beta_loss=beta_loss)
        nmf_nbytes = nbytes(W) + nbytes(H)
        if max_nbytes is not None and nmf_nbytes >= max_nbytes:
            return None
        thresh = scipy.sparse.linalg.norm((W@H) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
        if k == m or thresh <= tol:
            return W, H, thresh
        k *= 2


def estimate_sparsity_nmf(spmat, tol, max_nbytes=None, k0=5, max_iters=int(1e3), nmf_tol=1e-2, beta_loss=2):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    W, H = fit_nmf(spmat, k0, max_iters=max_iters, tol=nmf_tol, beta_loss=beta_loss, init='random')

    if W is None:
        return None
    
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


def estimate_sparsity_nmf_weighted(spmat, FF_weights, tol, max_nbytes=None, k0=5, max_iters=int(1e3), nmf_tol=1e-2, beta_loss=2):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    W, H = fit_nmf(spmat, k, max_iters=max_iters, tol=nmf_tol, beta_loss=beta_loss)
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




# ACA ALGORITHMS

def cross_approximation_full(M, k):
    A = np.empty((M.shape[0], 0))
    B = np.empty((0, M.shape[1]))
    F_hat = A @ B
    
    nu = 1
    while nu <= k:
        i_star, j_star = np.unravel_index(np.argmax(abs(M - F_hat)), M.shape)
        delta = M[i_star,j_star] - F_hat[i_star,j_star]
        
        if delta == 0:
            return scipy.sparse.csr_matrix(A), scipy.sparse.csr_matrix(B)
        
        a_nu = (M[:, j_star] - F_hat[:,j_star]).reshape(-1,1)
        b_nu = ((M[i_star, :] - F_hat[i_star,:]) / delta).reshape(-1,1)
        
        A = np.concatenate([A, a_nu], axis=1)
        B = np.concatenate([B, b_nu.T], axis=0)
        F_hat = A @ B
                   
        nu += 1
    
    return scipy.sparse.csr_matrix(A), scipy.sparse.csr_matrix(B)


def estimate_rank_aca(spmat, tol, max_nbytes=None, k0=40):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        if not k >= 1:
            raise RuntimeError('bad value of k')
        A, B = cross_approximation_full(spmat.A, k)

        for kk in range(prev_k+1,k+1):
            
            Ak, Bk = A[:, :kk], B[:kk, :]

            aca_nbytes = nbytes(Ak) + nbytes(Bk)
            if max_nbytes is not None and aca_nbytes >= max_nbytes:
                return None
            thresh = scipy.sparse.linalg.norm((Ak @ Bk) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
            if kk == m or thresh <= tol:
                return Ak, Bk, thresh
        
        prev_k = k
        k *= 2
        

def estimate_sparsity_aca(spmat, tol, max_nbytes=None, k0=40):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    prev_nbytes = np.inf
    Sr_prev = np.zeros(spmat.shape)
    
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        
        if not k >= 1:
            raise RuntimeError('bad value of k')

        A, B = cross_approximation_full(spmat.A, k)
        
        for kk in range(prev_k+1,k+1):
            
            Ak, Bk = A[:, :kk], B[:kk, :]
            resid = (spmat - (Ak @ Bk)).A

            if kk == m:
                return A[:, :kk], B[:kk, :], scipy.sparse.csr_matrix(np.zeros_like(resid))

            num_resids = resid.shape[0] * resid.shape[1]
            nnz_resid = np.count_nonzero(resid)

            sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

            target = np.power(np.linalg.norm(resid, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

            if target <= 0:
                Sr = np.zeros_like(resid)
                Sr = scipy.sparse.csr_matrix(Sr)

            else:
                cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
                keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

                Sr = np.copy(resid)
                Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
                Sr = scipy.sparse.csr_matrix(Sr)

            sparse_aca_nbytes = nbytes(Ak) + nbytes(Bk) + nbytes(Sr)

            if sparse_aca_nbytes >= prev_nbytes:
                return A[:, :kk-1], B[:kk-1, :], Sr_prev

            if max_nbytes is not None and sparse_aca_nbytes >= max_nbytes:
                return None
            

            prev_nbytes = sparse_aca_nbytes
            Sr_prev = Sr

        prev_k = k
        k *= 2




# BRP ALGORITHMS

def bilateral_random_projection(X, k):
    
    m, n = X.shape
    
    A1 = np.random.normal(loc=0.0, scale=1.0, size=(n,k))
    Y1 = X @ A1
    
    A2 = np.copy(Y1)
    Y2 = X.T @ A2
    
    A1 = np.copy(Y2)
    Y1 = X @ A1

    return scipy.sparse.csr_matrix(Y1), scipy.sparse.csr_matrix(np.linalg.inv(A2.T @ Y1)), scipy.sparse.csr_matrix(Y2)


def estimate_rank_brp(spmat, tol, max_nbytes=None, k0=40):
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
        Y1, D, Y2 = bilateral_random_projection(spmat.A, k)
        brp_nbytes = nbytes(Y1) + nbytes(D) + nbytes(Y2)
        if max_nbytes is not None and brp_nbytes >= max_nbytes:
            return None
        thresh = scipy.sparse.linalg.norm((Y1 @ D @ Y2.T) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
        if k == m or thresh <= tol:
            return Y1, D, Y2, thresh
        k += 5


def estimate_sparsity_brp(spmat, tol, max_nbytes=None, k0=5):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    prev_nbytes = np.inf
    Sr_prev = np.zeros(spmat.shape)
    
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        
        if not k >= 1:
            raise RuntimeError('bad value of k')

        Y1, D, Y2 = bilateral_random_projection(spmat.A, k)
        
        resid = (spmat - (Y1 @ D @ Y2.T)).A
        
        if k == m:
            return Y1, D, Y2, scipy.sparse.csr_matrix(np.zeros_like(resid))

        num_resids = resid.shape[0] * resid.shape[1]
        nnz_resid = np.count_nonzero(resid)

        sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

        target = np.power(np.linalg.norm(resid, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

        if target <= 0:
            Sr = np.zeros_like(resid)
            Sr = scipy.sparse.csr_matrix(Sr)

        else:
            cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
            keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

            Sr = np.copy(resid)
            Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
            Sr = scipy.sparse.csr_matrix(Sr)

        sparse_brp_nbytes = nbytes(Y1) + nbytes(D) + nbytes(Y2) + nbytes(Sr)

        if sparse_brp_nbytes >= prev_nbytes:
            return Y1_prev, D_prev, Y2_prev, Sr_prev

        if max_nbytes is not None and sparse_brp_nbytes >= max_nbytes:
            return None

        prev_nbytes = sparse_brp_nbytes
        Y1_prev = Y1
        D_prev = D
        Y2_prev = Y2
        Sr_prev = Sr
        prev_k = k
        k += 5




# RANDOM ID ALGORITHMS

'''
Adapted from Ristretto: https://github.com/erichson/ristretto
'''

def interpolative_decomp_index_set(A, k):

    m, n = A.shape

    Q, R, P = scipy.linalg.qr(A, mode='economic', overwrite_a=False, pivoting=True,
                        check_finite=False)

    C = A[:, P[:k]]

    T =  scipy.linalg.pinv(R[:k, :k]).dot(R[:k, k:n])
    V = np.bmat([[np.eye(k), T]])
    V = V[:, np.argsort(P)]

    return P[:k], V


def random_interpolative_decomp(A, k, p=5, q=5):

    Q = simple_range_finder(A, k, p=p, q=q)
    B = Q.T @ A

    J, V = interpolative_decomp_index_set(B, k)
    J = J[:k]

    return scipy.sparse.csr_matrix(A[:, J]), scipy.sparse.csr_matrix(V)


def estimate_rank_random_id(spmat, tol, max_nbytes=None, k0=40, p=5, q=5):
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
        C, V = random_interpolative_decomp(spmat.A, k, p=p, q=q)
        id_nbytes = nbytes(C) + nbytes(V)
        if max_nbytes is not None and id_nbytes >= max_nbytes:
            return None
        thresh = scipy.sparse.linalg.norm((C@V) - spmat, ord='fro') / scipy.sparse.linalg.norm(spmat, ord='fro')
        if k == m or thresh <= tol:
            return C, V, thresh
        k += 5


def estimate_sparsity_random_id(spmat, tol, max_nbytes=None, k0=40, p=5, q=5):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    prev_nbytes = np.inf
    Sr_prev = np.zeros(spmat.shape)
    
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        
        if not k >= 1:
            raise RuntimeError('bad value of k')

        C, V = random_interpolative_decomp(spmat.A, k, p=p, q=q)
        
        resid = (spmat - (C @ V)).A

        if k == m:
            return C, V, scipy.sparse.csr_matrix(np.zeros_like(resid))

        num_resids = resid.shape[0] * resid.shape[1]
        nnz_resid = np.count_nonzero(resid)

        sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

        target = np.power(np.linalg.norm(resid, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

        if target <= 0:
            Sr = np.zeros_like(resid)
            Sr = scipy.sparse.csr_matrix(Sr)

        else:
            cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
            keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

            Sr = np.copy(resid)
            Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
            Sr = scipy.sparse.csr_matrix(Sr)

        sparse_id_nbytes = nbytes(C) + nbytes(V) + nbytes(Sr)

        if sparse_id_nbytes >= prev_nbytes:
            return C_prev, V_prev, Sr_prev

        if max_nbytes is not None and sparse_id_nbytes >= max_nbytes:
            return None
        

        prev_nbytes = sparse_id_nbytes
        C_prev = C
        V_prev = V
        Sr_prev = Sr
        prev_k = k
        k += 5




# OTHER RANDOM ALGORITHMS

def randomized_HALS(X, Q, k, tol=1e-4, max_iters=1e4):
    m,n = X.shape
    
    l = Q.shape[1]
    
    W = np.random.uniform(low=0.0, high=1.0, size=(m,k))
    _W = np.random.uniform(low=0.0, high=1.0, size=(l,k))
    H = np.random.uniform(low=0.0, high=1.0, size=(k,n))
    
    B = Q.T @ X        # l x n
    
    criterion = True
    num_iters = 0
    H_norm = np.linalg.norm(H, ord='fro')
    while criterion:
        
        R = B.T @ _W       # n x k
        S = _W.T @ _W      # k x k
        for j in range(k):
            if S[j,j] != 0:
                H[j,:] = H[j,:] + ((R[:,j] - (H.T @ S[:,j])) / S[j,j])
                H[j,:] = np.maximum(H[j,:], 0.)
            
        T = B @ H.T        # l x k
        V = H @ H.T        # k x k
        for j in range(k):
            if V[j,j] != 0:
                _W[:,j] = _W[:,j] + ((T[:,j] - (_W @ V[:,j])) / V[j,j])
                W[:,j] = np.maximum(0, Q @ _W[:,j])
                _W[:,j] = Q.T @ W[:,j]
            
        num_iters += 1
        if num_iters >= max_iters:
            print('Exceeded max_iters')
            criterion = False
           
        new_H_norm = np.linalg.norm(H, ord='fro')
        if (abs(H_norm - new_H_norm) / H_norm)  < tol:
            criterion = False
        
        H_norm = new_H_norm

        if np.isnan(W).any() or np.isnan(H).any():
            return None, None
            
    W = scipy.sparse.csr_matrix(W)
    H = scipy.sparse.csr_matrix(H)
    return W, H


def randomized_svd(B, Q, k):
    _U, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ _U

    U = scipy.sparse.csr_matrix(U)
    Vt = scipy.sparse.csr_matrix(Vt)
    return U, S, Vt


def simple_range_finder(X, k, p=5, q=1, rand_dist='uniform'):
    m,n = X.shape
    
    l = k + p
    
    if rand_dist == 'uniform':
        Omega = np.random.uniform(low=0.0, high=1.0, size=(n,l))
    elif rand_dist == 'normal':
        Omega = np.random.normal(loc=0.0, scale=1.0, size=(n,l))
        
    Y = X @ Omega
    
    for j in range(q):
        Q = np.linalg.qr(Y, mode='reduced')[0]
        Q = np.linalg.qr(X.T @ Q, mode='reduced')[0]
        Y = X @ Q
        
    Q = np.linalg.qr(Y, mode='reduced')[0]
    
    return Q


def estimate_sparsity_random_nmf(spmat, tol, max_nbytes=None, k0=5, max_iters=int(1e3), nmf_tol=1e-2, p=5, q=1):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    Q = simple_range_finder(spmat, k0, p=p, q=q)
    W, H = randomized_HALS(spmat, Q, k0, tol=nmf_tol, max_iters=max_iters)

    if W is None:
        return None
    
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


def estimate_rank_random_svd(spmat, tol, max_nbytes=None, k0=40, p=5, q=1):
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

        Q = simple_range_finder(spmat, k, p=p, q=q)
        B = Q.T @ spmat
        U, S, Vt = randomized_svd(B, Q, k)
        try:
            thresh = S/S[0]
        except:
            import pdb; pdb.set_trace()
            pass
        if k == m-1:
            return U, S, Vt, thresh[k-1]
        assert thresh[0] >= 1
        below_thresh = np.where(thresh <= tol)[0]
        if below_thresh.size > 0:
            r = below_thresh[0]
            svd_nbytes = nbytes(U[:, :r]) + nbytes(S[:r]) + nbytes(Vt[:r, :])
            if max_nbytes is not None and svd_nbytes >= max_nbytes:
                return None
            return U[:, :r], S[:r], Vt[:r, :], thresh[r]
        k *= 2


def estimate_sparsity_random_svd(spmat, tol, max_nbytes=None, k0=40, p=5, q=1):
    assert tol < 1

    if spmat.shape[0] == 0 or spmat.shape[1] == 0:
        return 0

    if spmat.shape == (1, 1):
        return 1

    prev_k = 0
    prev_nbytes = np.inf
    Sr_prev = np.zeros(spmat.shape)
    
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        
        if not k >= 1:
            raise RuntimeError('bad value of k')

        Q = simple_range_finder(spmat, k, p=p, q=q)
        B = Q.T @ spmat
        U, S, Vt = randomized_svd(B, Q, k)
        
        for kk in range(prev_k+1,k+1):
            
            Uk, Sk, Vtk = U[:, :kk], S[:kk], Vt[:kk, :]
            resid = (spmat - (Uk @ (Vtk.T * np.diag(Sk)).T)).A

            if kk == m:
                return U[:, :kk], S[:kk], Vt[:kk, :], scipy.sparse.csr_matrix(np.zeros_like(resid))

            num_resids = resid.shape[0] * resid.shape[1]
            nnz_resid = np.count_nonzero(resid)

            sorted_resid_idx = np.unravel_index(np.argsort(abs(resid), axis=None), resid.shape)

            target = np.power(np.linalg.norm(resid, ord='fro'), 2) - np.power(tol*scipy.sparse.linalg.norm(spmat, ord='fro'), 2)

            if target <= 0:
                Sr = np.zeros_like(resid)
                Sr = scipy.sparse.csr_matrix(Sr)

            else:
                cumulative_residual = np.cumsum(np.power(resid[sorted_resid_idx[0], sorted_resid_idx[1]][::-1], 2))
                keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

                Sr = np.copy(resid)
                Sr[sorted_resid_idx[0][:num_resids-keep_resids], sorted_resid_idx[1][:num_resids-keep_resids]] = 0.
                Sr = scipy.sparse.csr_matrix(Sr)

            sparse_svd_nbytes = nbytes(Uk) + nbytes(Sk) + nbytes(Vtk) + nbytes(Sr)

            if sparse_svd_nbytes >= prev_nbytes:
                return U[:, :kk-1], S[:kk-1], Vt[:kk-1, :], Sr_prev

            if max_nbytes is not None and sparse_svd_nbytes >= max_nbytes:
                return None
            

            prev_nbytes = sparse_svd_nbytes
            Sr_prev = Sr

        prev_k = k
        k *= 2




def nnz(mat, tol=None):
    if tol is None:
        return (mat != 0).sum()
    else:
        return (abs(mat) < tol).sum()


def sparsity(mat, tol=None):
    size = mat.size
    return 0 if size == 0 else nnz(mat, tol)/size
