import copy
import itertools as it
import logging

from abc import ABC

import numpy as np
import pickle
import scipy.sparse.linalg

import flux.linalg

from flux.debug import IndentedPrinter
from flux.form_factors import get_form_factor_matrix
from flux.octree import get_octant_order
from flux.quadtree import get_quadrant_order
from flux.obbtree import get_obb_partition_2d
from flux.util import nbytes, get_sunvec


@np.vectorize
def _is_dense(block):
    return isinstance(block, FormFactorDenseBlock)


@np.vectorize
def _is_sparse(block):
    return isinstance(block, FormFactorSparseBlock) \
        or isinstance(block, FormFactorZeroBlock)


class CompressedFormFactorBlock(ABC):
    """A block of a CompressedFormFactorMatrix."""

    def __init__(self, root, shape):
        self._root = root
        self.shape = shape

    @property
    def dtype(self):
        return self._root.dtype

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def root(self):
        return self._root

    @property
    def _min_size(self):
        return self._root._min_size

    @property
    def _tol(self):
        return self._root._tol

    @property
    def _sparsity_threshold(self):
        return self._root.sparsity_threshold


class FormFactorLeafBlock(CompressedFormFactorBlock):

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def depth(self):
        return 0

    @property
    def is_leaf(self):
        return True

    def get_blocks_at_depth(self, depth):
        if depth == 0:
            yield self


class FormFactorNullBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, shape):
        if shape[0] != 0 and shape[1] != 0:
            raise RuntimeError('a null block must have a degenerate shape')
        super().__init__(root, shape)

        self.partial_norm = 0.

    def _matmat(self, x):
        if self.shape[1] != x.shape[0]:
            raise ValueError(
                'multiplying null block with matrix with wrong shape')
        new_shape = (self.shape[0], x.shape[1])
        return np.empty(new_shape, dtype=self.dtype)

    def __add__(self, x):
        return np.empty(self.shape, dtype=self.dtype)

    def is_dense(self):
        return True

    def is_sparse(self):
        return True

    @property
    def _mat(self):
        return np.empty(self.shape, dtype=self.dtype)

    @property
    def nbytes(self):
        return 0

    @property
    def is_empty_leaf(self):
        return True

    @property
    def sq_resid_sum(self):
        return 0.


class FormFactorZeroBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):
    def __init__(self, root, shape):
        super().__init__(root, shape)

        self.partial_norm = 0.

    def _matmat(self, x):
        m = self.shape[0]
        y_shape = (m,) if x.ndim == 1 else (m, x.shape[1])
        return np.zeros(y_shape, dtype=self.dtype)

    def __add__(self, x):
        return x

    def is_dense(self):
        return False

    def is_sparse(self):
        return True

    @property
    def _mat(self):
        return np.zeros(self.shape, self.dtype)

    @property
    def nbytes(self):
        return 0

    def tocsr(self):
        return scipy.sparse.csr_matrix(self.shape, dtype=self.dtype)

    @property
    def is_empty_leaf(self):
        return True

    @property
    def sq_resid_sum(self):
        return 0.


class FormFactorDenseBlock(FormFactorLeafBlock,
                           scipy.sparse.linalg.LinearOperator):
    def __init__(self, root, mat):
        if isinstance(mat, scipy.sparse.spmatrix):
            mat = mat.toarray()
        super().__init__(root, mat.shape)
        self._mat = mat

        self.partial_norm = np.power(mat.flatten(), 2).sum()

    @property
    def nbytes(self):
        try:
            return self._mat.nbytes
        except:
            import pdb; pdb.set_trace()

    def toarray(self):
        return self._mat

    def is_dense(self):
        return True

    def is_sparse(self):
        return False

    def _matmat(self, x):
        return self._mat@x

    def __add__(self, x):
        return self._mat+x

    def _get_sparsity(self, tol=None):
        nnz = flux.linalg.nnz(self._mat, tol)
        size = self._mat.size
        return 0 if size == 0 else nnz/size

    @property
    def is_empty_leaf(self):
        return False

    @property
    def sq_resid_sum(self):
        return 0.


class FormFactorSparseBlock(FormFactorLeafBlock,
                            scipy.sparse.linalg.LinearOperator):
    def __init__(self, *args):
        super().__init__(*args)

    def _matmat(self, x):
        return np.array(self._spmat@x)

    def __add__(self, x):
        return np.array(self._spmat+x)

    def is_dense(self):
        return False

    def is_sparse(self):
        return True

    @property
    def is_empty_leaf(self):
        return False

    @property
    def sq_resid_sum(self):
        return 0.


class FormFactorCsrBlock(FormFactorSparseBlock):

    def __init__(self, root, mat):
        super().__init__(root, mat.shape)
        if isinstance(mat, np.ndarray):
            self.partial_norm = np.power(mat.flatten(), 2).sum()
            spmat = scipy.sparse.csr_matrix(mat)
        elif isinstance(mat, scipy.sparse.spmatrix):
            self.partial_norm = np.power(mat.A.flatten(), 2).sum()
            spmat = mat
        else:
            raise Exception('invalid class for mat: %s' % type(mat))
        self._spmat = spmat

    @property
    def nbytes(self):
        return self._spmat.data.nbytes

    def tocsr(self):
        return self._spmat

    @property
    def _mat(self):
        return self._spmat.toarray()

    @property
    def sq_resid_sum(self):
        return 0.


class FormFactorTruncatedCsrBlock(FormFactorSparseBlock):

    def __init__(self, root, mat):
        super().__init__(root, mat.shape)
        if isinstance(mat, np.ndarray):
            dmat = mat
        elif isinstance(mat, scipy.sparse.spmatrix):
            dmat = mat.A
        else:
            raise Exception('invalid class for mat: %s' % type(mat))

        Sr = self.get_form_factor_truncated(dmat)
        self._spmat = Sr

        self.partial_norm = np.power(dmat.flatten(), 2).sum()
        self._sq_resid_sum = np.power((self._spmat - dmat).flatten(), 2).sum()

    def get_form_factor_truncated(self, mat):

        num_elements = mat.shape[0] * mat.shape[1]

        sorted_ff_idx = np.unravel_index(np.argsort(abs(mat), axis=None), mat.shape)

        target = np.power(np.linalg.norm(mat, ord='fro'), 2) - np.power(0.1*self._tol*np.linalg.norm(mat, ord='fro'), 2)

        cumulative_residual = np.cumsum(np.power(mat[sorted_ff_idx[0], sorted_ff_idx[1]][::-1], 2))
        keep_resids = (cumulative_residual > target).nonzero()[0][0] + 1

        Sr = np.copy(mat)
        Sr[sorted_ff_idx[0][:num_elements-keep_resids], sorted_ff_idx[1][:num_elements-keep_resids]] = 0.

        Sr = scipy.sparse.csr_matrix(Sr)

        return Sr

    @property
    def nbytes(self):
        return self._spmat.data.nbytes

    def tocsr(self):
        return self._spmat

    @property
    def _mat(self):
        return self._spmat.toarray()

    @property
    def sq_resid_sum(self):
        return self._sq_resid_sum


class FormFactorSvdBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, u, s, vt, mat_block):
        shape = (u.shape[0], vt.shape[1])
        super().__init__(root, shape)

        self._k = s.size
        self._s = s

        u_csr = scipy.sparse.csr_matrix(u)
        self._u = u if nbytes(u) < nbytes(u_csr) else u_csr

        vt_csr = scipy.sparse.csr_matrix(vt)
        self._vt = vt if nbytes(vt) < nbytes(vt_csr) else vt_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power(((self._u @ np.diag(self._s) @ self._vt) - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._vt@x
        y = (y.T*self._s).T
        y = self._u@y
        return y

    def __add__(self, x):
        return (self._u @ np.diag(self._s) @ self._vt) + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._u) + nbytes(self._s) + nbytes(self._vt)

    @property
    def compressed(self):
        return isinstance(self._u, scipy.sparse.spmatrix) \
            or isinstance(self._vt, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        u_nnz = flux.linalg.nnz(self._u, tol)
        v_nnz = flux.linalg.nnz(self._vt, tol)
        size = self._u.size + self._vt.size + self._k
        return 0 if size == 0 else (u_nnz + v_nnz + self._k)/size


class FormFactorSparseSvdBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, u, s, vt, sr, mat_block):
        shape = (u.shape[0], vt.shape[1])
        super().__init__(root, shape)

        self._k = s.size
        self._s = s

        u_csr = scipy.sparse.csr_matrix(u)
        self._u = u if nbytes(u) < nbytes(u_csr) else u_csr

        vt_csr = scipy.sparse.csr_matrix(vt)
        self._vt = vt if nbytes(vt) < nbytes(vt_csr) else vt_csr

        sr_csr = scipy.sparse.csr_matrix(sr)
        self._sr = sr if nbytes(sr) < nbytes(sr_csr) else sr_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power(((self._u @ np.diag(self._s) @ self._vt) + self._sr - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._vt@x
        y = (y.T*self._s).T
        y = self._u@y
        y = y + (self._sr@x)
        return y

    def __add__(self, x):
        return (self._u @ np.diag(self._s) @ self._vt) + self._sr + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._u) + nbytes(self._s) + nbytes(self._vt) + nbytes(self._sr)

    @property
    def compressed(self):
        return isinstance(self._u, scipy.sparse.spmatrix) \
            or isinstance(self._vt, scipy.sparse.spmatrix) \
            or isinstance(self._sr, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        u_nnz = flux.linalg.nnz(self._u, tol)
        v_nnz = flux.linalg.nnz(self._vt, tol)
        sr_nnz = flux.linalg.nnz(self._sr, tol)
        size = self._u.size + self._vt.size + self._k + self._sr.size
        return 0 if size == 0 else (u_nnz + v_nnz + self._k + sr_nnz)/size


class FormFactorAcaBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, a, b, mat_block):
        shape = (a.shape[0], b.shape[1])
        super().__init__(root, shape)

        a_csr = scipy.sparse.csr_matrix(a)
        self._a = a if nbytes(a) < nbytes(a_csr) else a_csr

        b_csr = scipy.sparse.csr_matrix(b)
        self._b = b if nbytes(b) < nbytes(b_csr) else b_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power((self._a@self._b - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._b@x
        y = self._a@y
        return y

    def __add__(self, x):
        return self._a@self._b + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._a) + nbytes(self._b)

    @property
    def compressed(self):
        return isinstance(self._a, scipy.sparse.spmatrix) \
            or isinstance(self._b, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        a_nnz = flux.linalg.nnz(self._a, tol)
        b_nnz = flux.linalg.nnz(self._b, tol)
        size = self._a.size + self._b.size
        return 0 if size == 0 else (a_nnz + b_nnz)/size


class FormFactorSparseAcaBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, a, b, sr, mat_block):
        shape = (a.shape[0], b.shape[1])
        super().__init__(root, shape)

        a_csr = scipy.sparse.csr_matrix(a)
        self._a = a if nbytes(a) < nbytes(a_csr) else a_csr

        b_csr = scipy.sparse.csr_matrix(b)
        self._b = b if nbytes(b) < nbytes(b_csr) else b_csr

        sr_csr = scipy.sparse.csr_matrix(sr)
        self._sr = sr if nbytes(sr) < nbytes(sr_csr) else sr_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power((self._a@self._b + self._sr - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._b@x
        y = self._a@y
        y = y + (self._sr@x)
        return y

    def __add__(self, x):
        return self._a@self._b + self._sr + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._a) + nbytes(self._b) + nbytes(self._sr)

    @property
    def compressed(self):
        return isinstance(self._a, scipy.sparse.spmatrix) \
            or isinstance(self._b, scipy.sparse.spmatrix) \
            or isinstance(self._sr, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        a_nnz = flux.linalg.nnz(self._a, tol)
        b_nnz = flux.linalg.nnz(self._b, tol)
        sr_nnz = flux.linalg.nnz(self._sr, tol)
        size = self._a.size + self._b.size + self._sr.size
        return 0 if size == 0 else (a_nnz + b_nnz + sr_nnz)/size


class FormFactorIdBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, c, v, mat_block):
        shape = (c.shape[0], v.shape[1])
        super().__init__(root, shape)

        c_csr = scipy.sparse.csr_matrix(c)
        self._c = c if nbytes(c) < nbytes(c_csr) else c_csr

        v_csr = scipy.sparse.csr_matrix(v)
        self._v = v if nbytes(v) < nbytes(v_csr) else v_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power((self._c@self._v - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._v@x
        y = self._c@y
        return y

    def __add__(self, x):
        return self._c@self._v + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._c) + nbytes(self._v)

    @property
    def compressed(self):
        return isinstance(self._c, scipy.sparse.spmatrix) \
            or isinstance(self._v, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        c_nnz = flux.linalg.nnz(self._c, tol)
        v_nnz = flux.linalg.nnz(self._v, tol)
        size = self._c.size + self._v.size
        return 0 if size == 0 else (c_nnz + v_nnz)/size


class FormFactorSparseIdBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, c, v, sr, mat_block):
        shape = (c.shape[0], v.shape[1])
        super().__init__(root, shape)

        c_csr = scipy.sparse.csr_matrix(c)
        self._c = c if nbytes(c) < nbytes(c_csr) else c_csr

        v_csr = scipy.sparse.csr_matrix(v)
        self._v = v if nbytes(v) < nbytes(v_csr) else v_csr

        sr_csr = scipy.sparse.csr_matrix(sr)
        self._sr = sr if nbytes(sr) < nbytes(sr_csr) else sr_csr

        self.partial_norm = np.power(mat_block.flatten(), 2).sum()
        self.sq_resid_sum = np.power((self._c@self._v + self._sr - mat_block).flatten(), 2).sum()

    def _matmat(self, x):
        y = self._v@x
        y = self._c@y
        y = y + (self._sr@x)
        return y

    def __add__(self, x):
        return self._c@self._v + self._sr + x

    def is_dense(self):
        return False

    def is_sparse(self):
        return False

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        return nbytes(self._c) + nbytes(self._v) + nbytes(self._sr)

    @property
    def compressed(self):
        return isinstance(self._c, scipy.sparse.spmatrix) \
            or isinstance(self._v, scipy.sparse.spmatrix) \
            or isinstance(self._sr, scipy.sparse.spmatrix)

    def _get_sparsity(self, tol=None):
        c_nnz = flux.linalg.nnz(self._c, tol)
        v_nnz = flux.linalg.nnz(self._v, tol)
        sr_nnz = flux.linalg.nnz(self._sr, tol)
        size = self._c.size + self._v.size + self._sr.size
        return 0 if size == 0 else (c_nnz + v_nnz + sr_nnz)/size


class FormFactorBlockMatrix(CompressedFormFactorBlock,
                            scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, shape):
        super().__init__(root, shape)

    def make_block(self, shape_model, I, J, spmat,
                   max_depth=None, force_max_depth=False, compression_type="svd", compression_params={}):
        if max_depth is not None and not isinstance(max_depth, int):
            raise RuntimeError(
                'invalid max_depth type: %s' % str(type(max_depth)))

        if isinstance(max_depth, int) and max_depth <= 0:
            raise RuntimeError('invalid max_depth value: %d' % max_depth)

        if force_max_depth and max_depth is None:
            raise RuntimeError(
                'force_max_depth is True, but max_depth not specified')

        # If force_max_depth is True, we use the following simple
        # recursion when building the hierarchical block matrix
        if force_max_depth and max_depth > 1:
            assert False # this is wrong---fix
            block = self.make_child_block(
                shape_model, spmat, I, J, max_depth - 1, force_max_depth)
            if block.is_dense():
                block = self.root.make_dense_block(spmat.toarray())
            elif block.is_sparse():
                block = self.root.make_sparse_block(spmat)
            return block

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

            if not 'k0' in compression_params.keys():
                raise RuntimeError('compression_params must include key: k0')

            k0 = compression_params["k0"]
            if self.root._add_residuals:
                compressed_block = self._get_sparse_svd_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)
            else:
                compressed_block = self._get_svd_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)


        elif compression_type == "rand_svd":

            if not np.array([key in compression_params.keys() for key in ['k0', 'p', 'q']]).all():
                raise RuntimeError('compression_params must include keys: %s' % ['k0', 'p', 'q'])

            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            if self.root._add_residuals:
                compressed_block = self._get_sparse_random_svd_block(spmat, k0=k0, p=p, q=q)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)
            else:
                compressed_block = self._get_random_svd_block(spmat, k0=k0, p=p, q=q)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)


        elif compression_type == "aca":

            if not 'k0' in compression_params.keys():
                raise RuntimeError('compression_params must include key: k0')

            k0 = compression_params["k0"]

            if self.root._add_residuals:
                compressed_block = self._get_sparse_aca_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)
            else:
                compressed_block = self._get_aca_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)


        elif compression_type == "paca":

            if not 'k0' in compression_params.keys():
                raise RuntimeError('compression_params must include key: k0')

            k0 = compression_params["k0"]

            if self.root._add_residuals:
                compressed_block = self._get_sparse_paca_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)
            else:
                compressed_block = self._get_paca_block(spmat, k0=k0)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)


        elif compression_type == "rand_id":

            if not np.array([key in compression_params.keys() for key in ['k0', 'p', 'q']]).all():
                raise RuntimeError('compression_params must include keys: %s' % ['k0', 'p', 'q'])

            k0 = compression_params["k0"]
            p = compression_params["p"]
            q = compression_params["q"]

            if self.root._add_residuals:
                compressed_block = self._get_sparse_random_id_block(spmat, k0=k0, p=p, q=q)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)
            else:
                compressed_block = self._get_random_id_block(spmat, k0=k0, p=p, q=q)
                nbytes_compressed = np.inf if compressed_block is None else nbytes(compressed_block)


        else:
            raise RuntimeError('invalid compression_type: %d' % compression_type)


        # If we haven't specified a max depth, or if we haven't
        # bottomed out yet, then we attempt to descend another
        # level.
        if max_depth is None or max_depth > 1:
            child_block = self.make_child_block(
                shape_model, spmat, I, J,
                None if max_depth is None else max_depth - 1)
        else:
            child_block = None
        nbytes_child = np.inf if child_block is None else nbytes(child_block)

        nbytes_min = min(nbytes_sparse, nbytes_compressed, nbytes_child)

        # Select the block with the smallest size
        if nbytes_sparse == nbytes_min:
            block = sparse_block
        elif nbytes_compressed == nbytes_min:
            block = compressed_block
        else:
            block = child_block


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
            or isinstance(block, FormFactorSvdBlock) \
            or isinstance(block, FormFactorSparseSvdBlock) \
            or isinstance(block, FormFactorAcaBlock) \
            or isinstance(block, FormFactorSparseAcaBlock) \
            or isinstance(block, FormFactorIdBlock) \
            or isinstance(block, FormFactorSparseIdBlock) \
            or isinstance(block, FormFactorQuadtreeBlock) \
            or isinstance(block, FormFactorObbQuadtreeBlock)
        return block

    def _get_svd_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_rank(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        U, S, Vt, tol = ret
        svd_block = self.root.make_svd_block(U, S, Vt, spmat.A)

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
        svd_block = self.root.make_svd_block(U, S, Vt, spmat.A)

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
        s_svd_block = self.root.make_sparse_svd_block(U, S, Vt, Sr, spmat.A)

        return s_svd_block

    def _get_sparse_random_svd_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_sparsity_random_svd(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        U, S, Vt, Sr = ret
        s_svd_block = self.root.make_sparse_svd_block(U, S, Vt, Sr, spmat.A)

        return s_svd_block

    def _get_aca_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_rank_aca(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        A, B, tol = ret
        aca_block = self.root.make_aca_block(A, B, spmat.A)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return aca_block

        logging.warning("""computed a really inaccurate ACA, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_aca_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_sparsity_aca(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        A, B, Sr = ret
        s_aca_block = self.root.make_sparse_aca_block(A, B, Sr, spmat.A)

        return s_aca_block

    def _get_paca_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_rank_partial_aca(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        A, B, tol = ret
        aca_block = self.root.make_aca_block(A, B, spmat.A)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return aca_block

        logging.warning("""computed a really inaccurate ACA, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_paca_block(self, spmat, k0=40):
        ret = flux.linalg.estimate_sparsity_partial_aca(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0)
        if ret is None:
            return None

        A, B, Sr = ret
        s_aca_block = self.root.make_sparse_aca_block(A, B, Sr, spmat.A)

        return s_aca_block

    def _get_random_id_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_rank_random_id(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        C, V, tol = ret
        id_block = self.root.make_id_block(C, V, spmat.A)

        # If the tolerance estimated this way doesn't satisfy
        # the requested tolerance, return the sparse block
        # assert tol != 0
        if tol <= self._tol:
            return id_block

        logging.warning("""computed a really inaccurate ID, using
        a larger sparse block instead...""")
        return None

    def _get_sparse_random_id_block(self, spmat, k0=40, p=5, q=1):
        ret = flux.linalg.estimate_sparsity_random_id(
            spmat, self._tol, max_nbytes=nbytes(spmat),
            k0=k0, p=p, q=q)
        if ret is None:
            return None

        C, V, Sr = ret
        s_id_block = self.root.make_sparse_id_block(C, V, Sr, spmat.A)

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
        if isinstance(x, scipy.sparse.csr_matrix):
            y = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        else:
            y = np.zeros((self.shape[0], self.shape[1]), dtype=self.dtype)
        for i, row_inds in enumerate(self._row_block_inds):
            for j, col_inds in enumerate(self._col_block_inds):
                block = self._blocks[i, j]
                if block.is_empty_leaf: continue
                update_inds = np.ix_(row_inds, col_inds)
                y[update_inds] = block + x[update_inds]
        return y

    @property
    def partial_norm(self):
        total = 0.
        for i in range(len(self._row_block_inds)):
            for j in range(len(self._col_block_inds)):
                total += self._blocks[i, j].partial_norm
        return total

    @property
    def sq_resid_sum(self):
        total = 0.
        for i in range(len(self._row_block_inds)):
            for j in range(len(self._col_block_inds)):
                total += self._blocks[i, j].sq_resid_sum
        return total

    @property
    def nbytes(self):
        return sum(I.nbytes for I in self._row_block_inds) \
            + sum(J.nbytes for J in self._col_block_inds) \
            + sum(block.nbytes for block in self._blocks.flatten())

    @property
    def depth(self):
        return 1 + max(block.depth for block in self._blocks.flatten())

    @property
    def is_leaf(self):
        return False

    def is_dense(self):
        return _is_dense(self._blocks).all()

    def is_sparse(self):
        return _is_sparse(self._blocks).all()

    # TODO: disabling toarray and tocsr for now. These can cause
    # confusion because there are two ways of concatenating the
    # subblocks together: either we undo the row and column
    # permutations before concatenating, or we don't. The user needs
    # to choose, and it's probably best that they just do this sort of
    # thing manually.

    # def toarray(self):
    #     row = []
    #     for row_blocks in self._blocks:
    #         col = []
    #         for block in row_blocks:
    #             col.append(block.toarray())
    #         row.append(np.hstack(col))
    #     return np.vstack(row)

    # def tocsr(self):
    #     row = []
    #     for row_blocks in self._blocks:
    #         col = []
    #         for block in row_blocks:
    #             col.append(block.tocsr())
    #         row.append(scipy.sparse.hstack(col))
    #     return scipy.sparse.vstack(row)

    def _get_blocks_at_depth(self, depth):
        if depth == 0:
            yield self
        else:
            for block in self._blocks.ravel():
                yield from block._get_blocks_at_depth(depth - 1)

    def _get_row_inds_at_depth(self, depth, parent_inds):
        assert depth >= 1
        if depth == 1:
            for row_block_inds in self._row_block_inds:
                yield parent_inds[row_block_inds]
        else:
            for row_block_inds, block in zip(
                    self._row_block_inds, np.diag(self._blocks)):
                if block.is_leaf:
                    yield parent_inds[row_block_inds]
                else:
                    yield from block._get_row_inds_at_depth(
                        depth - 1, parent_inds[row_block_inds])

    def _get_row_blocks(self, row_ind, parent_inds):
        for row_block_inds, row_blocks in zip(
                self._row_block_inds, self._blocks):
            row_inds = parent_inds[row_block_inds]
            if row_ind in row_inds:
                for block in row_blocks:
                    if block.is_leaf:
                        yield block
                    else:
                        yield from block._get_row_blocks(row_ind, row_inds)
                break # Can break here since self._row_block_inds
                      # partitions parent_inds (could assert this if
                      # we get paranoid)

    def _get_col_inds_for_row(self, row_ind, parent_row_inds, parent_col_inds):
        for row_block_inds, row_blocks in zip(
                self._row_block_inds, self._blocks):
            row_inds = parent_row_inds[row_block_inds]
            if row_ind in row_inds:
                for block, col_block_inds in zip(
                        row_blocks, self._col_block_inds):
                    col_inds = parent_col_inds[col_block_inds]
                    if block.is_leaf:
                        yield col_inds
                    else:
                        yield from block._get_col_inds_for_row(
                            row_ind, row_inds, col_inds)
                break # See comment in _get_row_blocks above


class FormFactor2dTreeBlock(FormFactorBlockMatrix):

    def __init__(self, root, shape_model, spmat_par=None, I_par=None,
                 J_par=None, max_depth=None, force_max_depth=False):
        """Initializes a 2d-tree block.

        Parameters
        ----------
        root : FormFactorBlockMatrix
            The containing instance of the hierarchical block matrix.
        shape_model : ShapeModel
            The underlying geometry providing the form factors.
        spmat_par : sparse matrix, optional
            The uncompressed sparse form factor matrix for this block's parent.
            Its rows and columns correspond to the indices in I_par and J_par.
        I_par : array_like, optional
            Row indices for the ambient space. If not passed, assumed to span
            [0, root.shape[0]).
        J_par : array_like, optional
            Column indices for the ambient space. See explanation for I_par.
        max_depth : positive integer or None
            The maximum depth to which to recursively expand this
            block (i.e., the tree height of the tree below this block
            will be at most max_depth). If max_depth is None, then the
            recursion will terminate naturally when one of the other
            conditions are met.
        force_max_depth : boolean
            Whether to build the tree to the maximum depth level. If True,
            each leaf node will have the same height. Defaullt: False.

        """

        super().__init__(
            root,
            root.shape if I_par is None else (len(I_par), len(J_par))
        )
        self._set_block_inds(shape_model, I_par, J_par)

        blocks = []
        for i, row_inds in enumerate(self._row_block_inds):
            I = row_inds if I_par is None else I_par[row_inds]
            row = []
            for j, col_inds in enumerate(self._col_block_inds):
                J = col_inds if J_par is None else J_par[col_inds]
                if spmat_par is None:
                    IndentedPrinter().print(
                        'get_form_factor_matrix(|I%d| = %d, |J%d| = %d)' % (
                            i, len(row_inds), j, len(col_inds)))
                    spmat = get_form_factor_matrix(shape_model, I, J)
                else:
                    spmat = spmat_par[row_inds, :][:, col_inds]
                block = self.make_block(shape_model, I, J, spmat,
                                        max_depth, force_max_depth,
                                        compression_type=self._root._compression_type,
                                        compression_params=self._root._compression_params)
                assert block is not None
                row.append(block)
            blocks.append(row)
        self._blocks = np.array(blocks, dtype=CompressedFormFactorBlock)

    @property
    def is_empty_leaf(self):
        return False

    @property
    def bshape(self):
        """The number of blocks along each dimension."""
        return (len(self._row_block_inds), len(self._col_block_inds))

    def get_individual_block(self, i, j):
        """Return a shallow copy of the block matrix with all blocks other
        than (i, j)th block set to zero.

        """
        tmp = copy.copy(self)
        tmp._blocks = copy.copy(tmp._blocks)
        for i_, j_ in it.product(*(range(_) for _ in self.bshape)):
            if i_ != i or j_ != j:
                continue
            block = self.root.make_zero_block(tmp._blocks[i, j].shape)
            tmp._blocks[i, j] = block
        return tmp

    def get_diag_blocks(self):
        """Return a shallow copy of the block matrix with the off-diagonal
        blocks set to zero.

        """
        tmp = copy.copy(self)
        tmp._blocks = copy.copy(tmp._blocks)
        for i, j in it.product(*(range(_) for _ in self.bshape)):
            if i == j:
                continue
            block = self.root.make_zero_block(tmp._blocks[i, j].shape)
            tmp._blocks[i, j] = block
        return tmp

    def get_off_diag_blocks(self):
        """Return a shallow copy of the block matrix with the diagonal blocks
        set to zero.

        """
        tmp = copy.copy(self)
        tmp._blocks = copy.copy(tmp._blocks)
        for i, j in it.product(*(range(_) for _ in self.bshape)):
            if i != j:
                continue
            block = self.root.make_zero_block(tmp._blocks[i, j].shape)
            tmp._blocks[i, j] = block
        return tmp


class FormFactorQuadtreeBlock(FormFactor2dTreeBlock):
    """A form factor matrix block corresponding to one level of a quadtree
    partition.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_child_block(self, *args):
        return self.root.make_quadtree_block(*args)

    def _set_block_inds(self, shape_model, I, J):
        P = shape_model.P

        PI = P[:, :2] if I is None else P[I, :2]
        self._row_block_inds = [I for I in get_quadrant_order(PI)]

        PJ = P[:, :2] if J is None else P[J, :2]
        self._col_block_inds = [J for J in get_quadrant_order(PJ)]

class FormFactorMinDepthQuadtreeBlock(FormFactorQuadtreeBlock):
    """A form factor matrix block corresponding to a quadtree
    partition which goes down at least min_depth levels. Only
    for use as a root block.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_block_inds(self, shape_model, I, J):
        P = shape_model.P

        PI = P[:, :2]
        prev_level_block_inds = [I for I in get_quadrant_order(PI)]

        for _ in range(1, self._root._min_depth):
            current_level_block_inds = []
            for i in range(len(prev_level_block_inds)):
                PI = P[prev_level_block_inds[i], :2]
                current_level_block_inds += [prev_level_block_inds[i][I] for I in get_quadrant_order(PI)]
            prev_level_block_inds = current_level_block_inds

        self._row_block_inds = prev_level_block_inds
        self._col_block_inds = prev_level_block_inds

class FormFactorObbQuadtreeBlock(FormFactor2dTreeBlock):
    """A form factor matrix block corresponding to one level of a quadtree
    via oriented bounding box partitioning.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_child_block(self, *args):
        return self.root.make_obb_quadtree_block(*args)

    def _set_block_inds(self, shape_model, I, J):

        self._row_block_inds = get_obb_partition_2d(np.arange(shape_model.F.shape[0]) if I is None else I, shape_model)

        self._col_block_inds = get_obb_partition_2d(np.arange(shape_model.F.shape[0]) if J is None else J, shape_model)

class FormFactorOctreeBlock(FormFactor2dTreeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_child_block(self, *args):
        return self.root.make_octree_block(*args)

    def _set_block_inds(self, shape_model, I, J):
        P = shape_model.P

        PI = P[:] if I is None else P[I]
        self._row_block_inds = [I for I in get_octant_order(PI)]

        PJ = P[:] if J is None else P[J]
        self._col_block_inds = [J for J in get_octant_order(PJ)]

class FormFactorPartitionBlock(FormFactorBlockMatrix):

    def __init__(self, root, shape_model, parts=None,
                 max_depth=None, force_max_depth=False,
                 ChildBlock=FormFactorQuadtreeBlock, slurm=None):

        if parts is None:
            parts = [np.arange(shape_model.num_faces)]

        I_union = []
        for I in parts:
            I_union = np.union1d(I_union, I)
        if I_union.size != sum(I.size for I in parts):
            raise Exception('parts must be disjoint')
        del I_union

        self.ChildBlock = ChildBlock

        super().__init__(root, root.shape)
        self._row_block_inds = parts
        self._col_block_inds = parts

        # blocks = []
        # # can send each of the items of this thread to a different node to parallelize whole mesh (see on github how to call)
        # for i, I in tqdm(enumerate(parts), total=len(parts), desc="FormFactorPartitionBlock i"):
        #     row_blocks = []
        #     for j, J in tqdm(enumerate(parts), total=len(parts), desc="FormFactorPartitionBlock j"):
        #         spmat = get_form_factor_matrix(shape_model, I, J)
        #         block = self.make_block(shape_model, I, J, spmat,
        #                                 max_depth, force_max_depth)
        #         assert block is not None
        #         row_blocks.append(block)
        #     blocks.append(row_blocks)
        #
        # self._blocks = np.array(blocks, dtype=CompressedFormFactorBlock)

        def compute_block(I, J):
            spmat = get_form_factor_matrix(shape_model, I, J)
            block = self.make_block(shape_model, I, J, spmat,
                                    max_depth, force_max_depth,
                                    compression_type=self._root._compression_type,
                                    compression_params=self._root._compression_params)
            return block

        # slurm = None
        blocks = []
        if slurm != None:
            slurm.update_parameters(slurm_name="flux_FF")
            jobs = slurm.map_array(compute_block,[I for (I,J) in it.product(parts, repeat=2)],[J for (I,J) in it.product(parts, repeat=2)])
                                      # [(f"in/{zip_file.split('/')[-1].split('.')[0]}.stl", 1e-1,
                                      #   f"in/{zip_file.split('/')[-1].split('.')[0]}_1e-1.bin") for zip_file in zip_files])
            for job in jobs:
                print(job.result())
            exit()
        else:
            for I, J in it.product(parts, repeat=2):
                blocks.append(compute_block(I, J))
            blocks = np.reshape(blocks, (len(parts), len(parts)))
            self._blocks = np.array(blocks, dtype=CompressedFormFactorBlock)

            # check indexing and reshape
            # idx = []
            # for I, J in it.product(parts, repeat=2):
                # idx.append(f"{I[0]}_{J[0]}")
            # print(np.reshape(idx, (len(parts), len(parts))))
            # exit()

    def make_child_block(self, *args):
        return self.ChildBlock(self.root, *args)

    @property
    def is_empty_leaf(self):
        return True


class CompressedFormFactorMatrix(scipy.sparse.linalg.LinearOperator):
    """A compressed form factor (view factor) matrix. This provides an
    approximate version of the radiosity kernel matrix in an
    HODLR-style format, which strives to use O(N log N) space and
    provide an O(N log N) matrix-vector product for use in solving the
    radiosity integral equation.

    """

    def __init__(self, shape_model, tol=1e-5,
                 min_size=16384, min_depth=None, max_depth=None, force_max_depth=False, RootBlock=FormFactorQuadtreeBlock,
                 compression_type="svd", compression_params={}, add_residuals=False, truncated_sparse=False, **kwargs):
        """Create a new CompressedFormFactorMatrix.

        Parameters
        ----------
        shape_model : ShapeModel
            A discretization of the surface over which to solve the
            radiosity integral equation.
        tol : nonnegative float
            The compression tolerance (TODO: this is broken---update this
            once it's fixed to explain semantics)
        min_size : positive integer
            The minimum number of elements needed in a sub-block required
            before compression is attempted.
        min_depth : None of positive integer
            The minimum depth of the hieararchical matrix. If None is passed,
            then the compression begins at the top level of the tree.
        max_depth : None or positive integer
            The maximum depth of the hierarchical matrix. If None is passed,
            then there is no maximum depth.
        force_max_depth : boolean
            Whether to enforce the maximum depth of the hierarchical matrix.
            If True is passed, then each leaf node in the tree will have the
            same height. This is mostly useful for debugging. Default: False.
        RootBlock : class
            The class to use for the root node the hierarchical matrix.
        compression_type : string
            Indicator for the type of low-rank matrix decomposition to be used
            in the algebraic compression.
        compression_params: dictionary
            Dictionary elements correspond to the parmaeters of the low-rank
            decomposition, such as initial rank and number of power iterations.
        add_residuals: boolean
            Whether to add sparse residuals to the low-rank compression in
            matrix blocks.
        truncated_sparse: boolean
            Whether to truncate the elements contained in the sparse matrix
            blocks to meet the target error tolerance.

        """
        if tol < 0:
            raise RuntimeError('tol should be a nonnegative float')

        self.shape_model = shape_model
        self._tol = tol
        self._min_size = min_size
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._force_max_depth = force_max_depth
        self._compression_type = compression_type
        self._compression_params = compression_params
        self._add_residuals = add_residuals
        self._truncated_sparse = truncated_sparse

        self._root = RootBlock(self, shape_model, max_depth=max_depth,
                               force_max_depth=force_max_depth, **kwargs)

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
    def depth(self):
        return self._root.depth

    @property
    def sparsity_threshold(self):
        return 2.0/3.0

    def make_null_block(self, *args):
        return FormFactorNullBlock(self, *args)

    def make_zero_block(self, *args):
        return FormFactorZeroBlock(self, *args)

    def make_dense_block(self, *args):
        return FormFactorDenseBlock(self, *args)

    def make_sparse_block(self, *args, fmt='csr'):
        if fmt == 'csr':
            if self._truncated_sparse:
                return FormFactorTruncatedCsrBlock(self, *args)
            else:
                return FormFactorCsrBlock(self, *args)
        else:
            raise Exception('unknown sparse matrix format "%s"' % fmt)

    def make_svd_block(self, *args):
        return FormFactorSvdBlock(self, *args)

    def make_sparse_svd_block(self, *args):
        return FormFactorSparseSvdBlock(self, *args)

    def make_aca_block(self, *args):
        return FormFactorAcaBlock(self, *args)

    def make_sparse_aca_block(self, *args):
        return FormFactorSparseAcaBlock(self, *args)

    def make_id_block(self, *args):
        return FormFactorIdBlock(self, *args)

    def make_sparse_id_block(self, *args):
        return FormFactorSparseIdBlock(self, *args)

    def make_quadtree_block(self, *args):
        return FormFactorQuadtreeBlock(self, *args)

    def make_obb_quadtree_block(self, *args):
        return FormFactorObbQuadtreeBlock(self, *args)

    def make_octree_block(self, *args):
        return FormFactorOctreeBlock(self, *args)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _matmat(self, x):
        return self._root@x

    def __add__(self, x):
        return self._root+x

    def resid_frobenius_norm(self):
        return np.power(self._root.sq_resid_sum, 0.5)

    def full_mat_frobenius_norm(self):
        return np.power(self._root.partial_norm, 0.5)

    def get_blocks_at_depth(self, depth):
        if depth > self.depth:
            raise Exception('specified depth (%d) exceeds tree depth (%d)' % (
                depth, self.depth))
        if depth == 0 or self._root.is_leaf:
            yield self._root
        else:
            yield from self._root._get_blocks_at_depth(depth)

    def get_row_inds_at_depth(self, depth):
        if depth > self.depth:
            raise Exception('specified depth (%d) exceeds tree depth (%d)' % (
                depth, self.depth))
        row_inds = np.arange(self.num_faces)
        if depth == 0:
            yield row_inds
        else:
            yield from self._root._get_row_inds_at_depth(depth, row_inds)

    def get_row_blocks(self, row_ind):
        if not (0 <= row_ind and row_ind < self.num_faces):
            raise Exception('row_ind (== %d) should be in range [0, %d)' % (
                row_ind, self.num_faces))
        block, row_inds = self._root, np.arange(self.num_faces)
        if block.is_leaf:
            yield block
        else:
            yield from block._get_row_blocks(row_ind, row_inds)

    def get_col_inds_for_row(self, row_ind):
        if not (0 <= row_ind and row_ind < self.num_faces):
            raise Exception('row_ind (== %d) should be in range [0, %d)' % (
                row_ind, self.num_faces))
        block = self._root
        row_inds = np.arange(self.num_faces)
        col_inds = np.arange(self.num_faces)
        if self._root.is_leaf:
            yield col_inds
        else:
            yield from block._get_col_inds_for_row(row_ind, row_inds, col_inds)

    def get_individual_block(self, i, j):
        tmp = copy.copy(self) # shallow copy
        tmp._root = tmp._root.get_block(i, j)
        return tmp

    def get_diag_blocks(self):
        tmp = copy.copy(self) # shallow copy
        tmp._root = tmp._root.get_diag_blocks()
        return tmp

    def get_off_diag_blocks(self):
        tmp = copy.copy(self) # shallow copy
        tmp._root = tmp._root.get_off_diag_blocks()
        return tmp

    def toarray(self):
        def std_basis_vec(i):
            e = np.zeros(self.num_faces, dtype=self.dtype)
            e[i] = 1
            return e
        arr = np.array([self@std_basis_vec(i) for i in range(self.num_faces)])
        return arr.T

    def tocsr(self):
        return scipy.sparse.csr_matrix(self.toarray())

class CompressedKernelMatrix(scipy.sparse.linalg.LinearOperator):
    def __init__(self, form_factor_matrix, rho, albedo_placement='right'):
        self.form_factor_matrix = form_factor_matrix

        if not (0 < rho <= 1):
            raise ValueError('need 0 < rho <= 1')
        self.rho = rho

        if albedo_placement not in {'left', 'right'}:
            raise ValueError('albedo_placement must be "left" or "right"')
        self.albedo_placement = albedo_placement

    @property
    def shape(self):
        return self.form_factor_matrix.shape

    @property
    def dtype(self):
        return self.form_factor_matrix.dtype

    def _matvec(self, x):
        y = x.copy()
        if self.albedo_placement == 'left':
            y -= self.rho*(self.form_factor_matrix@x)
        elif self.albedo_placement == 'right':
            y -= self.form_factor_matrix@(self.rho*x)
        else:
            raise RuntimeError('albedo_placement must be "left" or "right"')
        return y

    def _rmatvec(self, x):
        y = x.copy()
        if self.albedo_placement == 'left':
            y -= (self.rho*x)@self.form_factor_matrix
        elif self.albedo_placement == 'right':
            y -= self.rho*(x@self.form_factor_matrix)
        else:
            raise RuntimeError('albedo_placement must be "left" or "right"')
        return y
