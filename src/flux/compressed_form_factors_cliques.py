import copy
import itertools as it
import logging


import numpy as np
import pickle
import scipy.sparse.linalg


import flux.linalg

from flux.form_factors import get_form_factor_matrix
from flux.util import nbytes, get_sunvec


import flux.compressed_form_factors_nmf as cff


@np.vectorize
def _is_dense(block):
    return isinstance(block, FormFactorDenseBlock)


@np.vectorize
def _is_sparse(block):
    return isinstance(block, FormFactorSparseBlock) \
        or isinstance(block, FormFactorZeroBlock)


class FormFactorCliqueMatrix(cff.CompressedFormFactorBlock,
                            scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, shape):
        super().__init__(root, shape)

    def make_block(self, shape_model, I, J, spmat, compression_type="svd", compression_params={}):
        
        # First, check for degenerate cases: zero blocks and blocks
        # which have no rows or columns
        nnz, shape = spmat.nnz, spmat.shape
        if nnz == 0:
            return self.root.make_zero_block(shape)
        if shape[0] == 0 or shape[1] == 0:
            return self.root.make_null_block(shape)

        size = np.product(shape)
        sparsity = nnz/size
        sparse_block = self.root.make_sparse_block(spmat)
        nbytes_sparse = nbytes(spmat)

        # First, if the matrix is small enough, we don't want to
        # bother with trying to compress it or recursively descending
        # further.
        #
        # NOTE: I've also observed that occasionally ARPACK will choke
        # on tiny form factor matrices. Not sure why. But that's
        # another thing to be careful about.
        if size < self._min_size:
            # See if we can save a few bytes by storing a dense matrix
            # instead...
            #
            # TODO: can compute the dense nbytes without actually
            # forming the dense block! And should do this check even
            # if we aren't in this clause!
            dense_block = self.root.make_dense_block(spmat)
            if dense_block.nbytes < nbytes_sparse:
                return dense_block
            else:
                return sparse_block


        # Next, since the block is "big enough", we go ahead and
        # attempt to compress it.
        if compression_type == "svd":
            
            k0 = compression_params["k0"]

            compressed_block = self._get_svd_block(spmat, k0=k0)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "ssvd":
            
            k0 = compression_params["k0"]

            compressed_block = self._get_sparse_svd_block(spmat, k0=k0)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "rand_svd":
            
            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            compressed_block = self._get_random_svd_block(spmat, k0=k0, p=p, q=q)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "rand_ssvd":
            
            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            compressed_block = self._get_sparse_random_svd_block(spmat, k0=k0, p=p, q=q)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "nmf":
            
            max_iters = compression_params["max_iters"]
            nmf_tol = compression_params["nmf_tol"]
            k0 = compression_params["k0"]
            beta_loss = compression_params["beta_loss"]
            
            compressed_block = self._get_nmf_block(spmat, max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "snmf":

            max_iters = compression_params["max_iters"]
            nmf_tol = compression_params["nmf_tol"]
            k0 = compression_params["k0"]
            beta_loss = compression_params["beta_loss"]

            compressed_block = self._get_sparse_nmf_block(spmat, max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "rand_snmf":

            max_iters = compression_params["max_iters"]
            nmf_tol = compression_params["nmf_tol"]
            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            compressed_block = self._get_sparse_random_nmf_block(spmat, max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, p=p, q=q)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "wsnmf":

            max_iters = compression_params["max_iters"]
            nmf_tol = compression_params["nmf_tol"]
            k0 = compression_params["k0"]
            beta_loss = compression_params["beta_loss"]

            FF_weights = self._root._FF_weights[I][:,J]

            compressed_block = self._get_weighted_sparse_nmf_block(spmat, FF_weights, max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "saca":
            
            k0 = compression_params["k0"]

            compressed_block = self._get_sparse_aca_block(spmat, k0=k0)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "sbrp":
            
            k0 = compression_params["k0"]

            compressed_block = self._get_sparse_brp_block(spmat, k0=k0)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        elif compression_type == "rand_sid":
            
            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            compressed_block = self._get_sparse_random_id_block(spmat, k0=k0, p=p, q=q)
            nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)

        else:
            raise RuntimeError('invalid compression_type: %d' % compression_type)


        nbytes_min = min(nbytes_sparse, nbytes_compressed)

        # Select the block with the smallest size
        if nbytes_sparse == nbytes_min:
            block = sparse_block
        else:
            block = compressed_block


        # Finally, do a little post-processing: if all of the child
        # blocks are dense blocks, then collapse them into a single
        # block and return it...
        if block.is_dense():
            return self.root.make_dense_block(spmat)

        # ... ditto if all the child blocks are sparse blocks.
        if block.is_sparse():
            return self.root.make_sparse_block(spmat)

        if isinstance(block, type(self)):
            assert all(_ is not None for _ in block._blocks.ravel())

        # Finally, we return whatever we have at this point. This
        # should either be an instance of ChildBlock or an SVD block.
        assert isinstance(block, type(self)) \
            or isinstance(block, cff.FormFactorSvdBlock) \
            or isinstance(block, cff.FormFactorSparseSvdBlock) \
            or isinstance(block, cff.FormFactorNmfBlock) \
            or isinstance(block, cff.FormFactorSparseNmfBlock) \
            or isinstance(block, cff.FormFactorSparseAcaBlock) \
            or isinstance(block, cff.FormFactorSparseBrpBlock) \
            or isinstance(block, cff.FormFactorSparseIdBlock)
        return block

    def _get_svd_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_rank(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        U, S, Vt, tol = ret
        svd_block = self.root.make_svd_block(U, S, Vt)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return svd_block

        logging.warning("""computed a really inaccurate SVD, using
        a larger sparse block instead...""")
        return None

    def _get_random_svd_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_rank_random_svd(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        U, S, Vt, tol = ret
        svd_block = self.root.make_svd_block(U, S, Vt)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return svd_block

        logging.warning("""computed a really inaccurate SVD, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_svd_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_sparsity_svd(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        U, S, Vt, Sr = ret
        s_svd_block = self.root.make_sparse_svd_block(U, S, Vt, Sr)

        return s_svd_block

    def _get_sparse_random_svd_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_sparsity_random_svd(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        U, S, Vt, Sr = ret
        s_svd_block = self.root.make_sparse_svd_block(U, S, Vt, Sr)

        return s_svd_block

    def _get_nmf_block(self, spmat, max_iters=int(1e3), nmf_tol=1e-2, k0=40, beta_loss=2):
        ret = flux.linalg.estimate_rank_nmf(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
        if ret is None:
            return None

        W, H, tol = ret
        nmf_block = self.root.make_nmf_block(W, H)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return nmf_block

        logging.warning("""computed a really inaccurate NMF, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_nmf_block(self, spmat, max_iters=int(1e3), nmf_tol=1e-2, k0=5, beta_loss=2):
        ret = flux.linalg.estimate_sparsity_nmf(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
        if ret is None:
            return None

        W, H, Sr = ret
        s_nmf_block = self.root.make_sparse_nmf_block(W, H, Sr)

        return s_nmf_block

    def _get_sparse_random_nmf_block(self, spmat, max_iters=int(1e3), nmf_tol=1e-2, k0=5, p=5, q=1):
        ret = flux.linalg.estimate_sparsity_random_nmf(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, p=p, q=q)
        if ret is None:
            return None

        W, H, Sr = ret
        s_nmf_block = self.root.make_sparse_nmf_block(W, H, Sr)

        return s_nmf_block

    def _get_weighted_sparse_nmf_block(self, spmat, FF_weights, max_iters=int(1e3), nmf_tol=1e-2, k0=5, beta_loss=2):
        ret = flux.linalg.estimate_sparsity_nmf_weighted(
            spmat, FF_weights, self._tol, max_nbytes=nbytes(spmat),
            max_iters=max_iters, nmf_tol=nmf_tol, k0=k0, beta_loss=beta_loss)
        if ret is None:
            return None

        W, H, Sr, tol = ret
        s_nmf_block = self.root.make_sparse_nmf_block(W, H, Sr)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return s_nmf_block

        logging.warning("""computed a really inaccurate weighted sparse NMF, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_aca_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_sparsity_aca(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        A, B, Sr = ret
        s_aca_block = self.root.make_sparse_aca_block(A, B, Sr)

        return s_aca_block

    def _get_sparse_brp_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_sparsity_brp(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        Y1, D, Y2, Sr = ret
        s_brp_block = self.root.make_sparse_brp_block(Y1, D, Y2, Sr)

        return s_brp_block

    def _get_sparse_random_id_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_sparsity_random_id(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        C, V, Sr = ret
        s_id_block = self.root.make_sparse_id_block(C, V, Sr)

        return s_id_block

    def _matmat(self, x):
        y = np.zeros((self.shape[0], x.shape[1]), dtype=self.dtype)
        for i, row_inds in enumerate(self._row_block_inds):
            for j, col_inds in enumerate(self._col_block_inds):
                block = self._blocks[i, j]
                if block.is_empty_leaf: continue
                y[row_inds] += block@x[col_inds]
        return y

    def __add__(self, x):
        if x.shape != self.shape:
            raise ValueError('cannot add %r and %r shape object: shape mismatch' % (self, x.shape))
        y = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        for i, row_inds in enumerate(self._row_block_inds):
            for j, col_inds in enumerate(self._col_block_inds):
                block = self._blocks[i, j]
                if block.is_empty_leaf: continue
                update_inds = np.ix_(row_inds, col_inds)
                y[update_inds] = block + x[update_inds]
        return y

    @property
    def nbytes(self):
        return sum(I.nbytes for I in self._row_block_inds) \
            + sum(J.nbytes for J in self._col_block_inds) \
            + sum(block.nbytes for block in self._blocks.flatten())

    @property
    def is_leaf(self):
        return False

    def is_dense(self):
        return _is_dense(self._blocks).all()

    def is_sparse(self):
        return _is_sparse(self._blocks).all()




class FormFactorCliqueWrapper(FormFactorCliqueMatrix):

    def __init__(self, root, shape_model, spmat):
        super().__init__(
            root,
            root.shape
        )
        
        dmat = spmat.A
        FF_adj = np.zeros(spmat.shape)
        FF_adj[dmat >= np.quantile(dmat[dmat>0.], 0.0)] = 1.
        self._set_block_inds(FF_adj)

        blocks = []
        for i, row_inds in enumerate(self._row_block_inds):
            I = row_inds
            row = []
            for j, col_inds in enumerate(self._col_block_inds):
                J = col_inds
                spmat_block = spmat[row_inds, :][:, col_inds]
                block = self.make_block(shape_model, I, J, spmat_block,
                                        compression_type=self._root._compression_type,
                                        compression_params=self._root._compression_params)
                assert block is not None
                row.append(block)
            blocks.append(row)
        self._blocks = np.array(blocks, dtype=cff.CompressedFormFactorBlock)

    @property
    def is_empty_leaf(self):
        return False

    @property
    def bshape(self):
        """The number of blocks along each dimension."""
        return (len(self._row_block_inds), len(self._col_block_inds))
    
    def _set_block_inds(self, FF_adj):
        # all_pseudo_cliques = []
        # for i in np.random.permutation(FF_adj.shape[0]):

        #     # make sure the seed is not already in a clique
        #     if np.array([i in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
        #         continue

        #     nonzero_idx = list((FF_adj[i] > 0).nonzero()[0])
        #     nonzero_idx.append(i)
        #     nonzero_idx = np.array(nonzero_idx)

        #     _nonzero_idx = []
        #     for j in nonzero_idx:
        #         if not np.array([j in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
        #             _nonzero_idx.append(j)
        #     nonzero_idx = np.array(_nonzero_idx)

        #     all_nonzero_idx = [nonzero_idx]
        #     for j in nonzero_idx:
        #         this_nonzero_idx = list((FF_adj[j] > 0).nonzero()[0])
        #         this_nonzero_idx.append(j)
        #         this_nonzero_idx = np.array(this_nonzero_idx)
        #         all_nonzero_idx.append(this_nonzero_idx)

        #     pseudo_clique = set(list(np.copy(all_nonzero_idx[0])))
        #     for j in range(1, len(all_nonzero_idx)):
        #         if not np.array([j in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
        #             if np.intersect1d(all_nonzero_idx[0], all_nonzero_idx[j]).shape[0] > 500:
        #                 pseudo_clique = pseudo_clique.union(set(list(all_nonzero_idx[j])))

        #     all_pseudo_cliques.append(pseudo_clique)

        all_pseudo_cliques = []
        for i in np.random.permutation(FF_adj.shape[0]):
            
            # make sure the seed is not already in a clique
            if np.array([i in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
                continue
            
            nonzero_idx = list((FF_adj[i] > 0).nonzero()[0])
            nonzero_idx.append(i)
            nonzero_idx = np.array(nonzero_idx)
            
            _nonzero_idx = []
            for j in nonzero_idx:
                
                if j == i:
                    _nonzero_idx.append(j)
                
                elif not np.array([j in _pseudo_clique for _pseudo_clique in all_pseudo_cliques]).any():
                    this_nonzero_idx = list((FF_adj[j] > 0).nonzero()[0])
                    this_nonzero_idx.append(j)
                    this_nonzero_idx = np.array(this_nonzero_idx)
                    num_intersecting = np.intersect1d(nonzero_idx, this_nonzero_idx).shape[0]
                    
                    if num_intersecting >= 0.5*min(nonzero_idx.shape[0], this_nonzero_idx.shape[0]):
                        _nonzero_idx.append(j)
            nonzero_idx = np.array(_nonzero_idx)
            
            pseudo_clique = set(list(np.copy(nonzero_idx)))    
            all_pseudo_cliques.append(pseudo_clique)
            
        all_pseudo_clique_lists = []
        for i in range(len(all_pseudo_cliques)):
            all_pseudo_clique_lists.append(np.array(list(all_pseudo_cliques[i])))
            
        self._row_block_inds = all_pseudo_clique_lists
        self._col_block_inds = all_pseudo_clique_lists

        if np.sum([len(this_clique) for this_clique in all_pseudo_cliques]) != FF_adj.shape[0]:
            raise RuntimeError("Cliques do not cover index set!")
        else:
            print("Cliques cover index set!")




class CliquedFormFactorMatrix(scipy.sparse.linalg.LinearOperator):
    def __init__(self, shape_model, spmat, tol=1e-5,
                 min_size=16384, compression_type="svd", compression_params={},
                 RootBlock=FormFactorCliqueWrapper, **kwargs):

        self.shape_model = shape_model
        self._tol = tol
        self._min_size = min_size
        self._compression_type = compression_type
        self._compression_params = compression_params

        self._root = RootBlock(self, shape_model, spmat, **kwargs)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def dtype(self):
        return self.shape_model.dtype

    @property
    def num_faces(self):
        return self.shape_model.num_faces

    @property
    def shape(self):
        return self.shape_model.num_faces, self.shape_model.num_faces

    @property
    def nbytes(self):
        return self._root.nbytes

    @property
    def sparsity_threshold(self):
        return 2.0/3.0

    def make_null_block(self, *args):
        return cff.FormFactorNullBlock(self, *args)

    def make_zero_block(self, *args):
        return cff.FormFactorZeroBlock(self, *args)

    def make_dense_block(self, *args):
        return cff.FormFactorDenseBlock(self, *args)

    def make_sparse_block(self, *args, fmt='csr'):
        if fmt == 'csr':
            return cff.FormFactorCsrBlock(self, *args)
        else:
            raise Exception('unknown sparse matrix format "%s"' % fmt)

    def make_svd_block(self, *args):
        return cff.FormFactorSvdBlock(self, *args)

    def make_sparse_svd_block(self, *args):
        return cff.FormFactorSparseSvdBlock(self, *args)

    def make_nmf_block(self, *args):
        return cff.FormFactorNmfBlock(self, *args)

    def make_sparse_nmf_block(self, *args):
        return cff.FormFactorSparseNmfBlock(self, *args)

    def make_sparse_aca_block(self, *args):
        return cff.FormFactorSparseAcaBlock(self, *args)

    def make_sparse_brp_block(self, *args):
        return cff.FormFactorSparseBrpBlock(self, *args)

    def make_sparse_id_block(self, *args):
        return cff.FormFactorSparseIdBlock(self, *args)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _matmat(self, x):
        return self._root@x

    def __add__(self, x):
        return self._root+x

    def toarray(self):
        def std_basis_vec(i):
            e = np.zeros(self.num_faces, dtype=self.dtype)
            e[i] = 1
            return e
        arr = np.array([self@std_basis_vec(i) for i in range(self.num_faces)])
        return arr.T

    def tocsr(self):
        return scipy.sparse.csr_matrix(self.toarray())