import numpy as np
import scipy.sparse.linalg

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




# ACA ALGORITHMS

def cross_approximation_partial(M, k):
    
    for i in range(M.shape[0]):
        if not (M[i,:] == np.zeros_like(M[i,:])).all():
            i_star = i
            break
    
    row_pivot_idx = []
    A = np.empty((M.shape[0], 0))
    B = np.empty((0, M.shape[1]))
    F_hat = A @ B
    
    nu = 1
    while nu <= k:
        j_star = np.argmax(abs(M[i_star,:] - F_hat[i_star,:]))
        delta = M[i_star,j_star] - F_hat[i_star,j_star]
        
        if delta == 0:
            return scipy.sparse.csr_matrix(A), scipy.sparse.csr_matrix(B)
        
        a_nu = (M[:, j_star] - F_hat[:,j_star]).reshape(-1,1)
        b_nu = ((M[i_star, :] - F_hat[i_star,:]) / delta).reshape(-1,1)
        
        A = np.concatenate([A, a_nu], axis=1)
        B = np.concatenate([B, b_nu.T], axis=0)
        F_hat = A @ B
        
        row_pivot_idx.append(i_star)
        
        # get the next row pivot index
        i_star = None
        i_sorted = np.argsort(abs(M[:, j_star] - F_hat[:, j_star]))[::-1]
        for i in i_sorted:
            if i not in row_pivot_idx and not (M[i,:] - F_hat[i, :] == np.zeros_like(M[i,:])).all():
                i_star = i
                break
        
        if i_star is None:
            raise RuntimeError("Could not find unused row pivot index")
            
        nu += 1
    
    return scipy.sparse.csr_matrix(A), scipy.sparse.csr_matrix(B)


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


def estimate_rank_partial_aca(spmat, tol, max_nbytes=None, k0=40):
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
        A, B = cross_approximation_partial(spmat.A, k)

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


def estimate_sparsity_partial_aca(spmat, tol, max_nbytes=None, k0=40):
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

        A, B = cross_approximation_partial(spmat.A, k)
        
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

    dmat = spmat.A
    m, k = min(spmat.shape), k0
    while True:
        k = min(k, m)
        if not k >= 1:
            raise RuntimeError('bad value of k')

        Q = simple_range_finder(dmat, k, p=p, q=q)
        B = Q.T @ dmat
        _, n = B.shape

        _, R, P = scipy.linalg.qr(B, mode='economic', overwrite_a=False, pivoting=True, check_finite=False)
        qr_spectrum = abs(R[np.arange(R.shape[0]), np.arange(R.shape[0])])

        thresh = qr_spectrum/qr_spectrum[0]
        assert thresh[0] >= 1
        below_thresh = np.where(thresh <= tol)[0]
        if below_thresh.size > 0:
            r = below_thresh[0]

            T =  scipy.linalg.pinv(R[:r, :r]).dot(R[:r, r:n])
            V = np.bmat([[np.eye(r), T]])
            V = V[:, np.argsort(P)]
            V = scipy.sparse.csr_matrix(V)

            J = P[:r]
            C = scipy.sparse.csr_matrix(dmat[:, J])

            id_nbytes = nbytes(C) + nbytes(V)
            if max_nbytes is not None and id_nbytes >= max_nbytes:
                return None
            return C, V, thresh[r]

        if k == m:
            return None
        
        k *= 2


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
