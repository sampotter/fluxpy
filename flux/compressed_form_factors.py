import numpy as np
import pickle
import scipy.sparse.linalg


import flux.linalg


from flux.debug import DebugLinearOperator, IndentedPrinter
from flux.form_factors import get_form_factor_block
from flux.quadtree import get_quadrant_order


@np.vectorize
def _is_dense(block):
    return isinstance(block, FormFactorDenseBlock)


@np.vectorize
def _is_sparse(block):
    return isinstance(block, FormFactorSparseBlock) \
        or isinstance(block, FormFactorZeroBlock)


class CompressedFormFactorBlock:

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
    def _max_rank(self):
        return self._root._max_rank

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


class FormFactorNullBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):

    def __init__(self, root):
        super().__init__(root, (0, 0))

    def _matmat(self, x):
        return np.array([], dtype=self.dtype)

    @property
    def nbytes(self):
        return 0

    @property
    def is_empty_leaf(self):
        return True


class FormFactorZeroBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, shape):
        super().__init__(root, shape)

    def _matmat(self, x):
        m = self.shape[0]
        y_shape = (m,) if x.ndim == 1 else (m, x.shape[1])
        return np.zeros(y_shape, dtype=self.dtype)

    @property
    def nbytes(self):
        return 0

    def tocsr(self):
        return scipy.sparse.csr_matrix(self.shape, dtype=self.dtype)

    @property
    def is_empty_leaf(self):
        return True


class FormFactorDenseBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, mat):
        if isinstance(mat, scipy.sparse.spmatrix):
            mat = mat.toarray()
        super().__init__(root, mat.shape)
        self._mat = mat

    @property
    def nbytes(self):
        try:
            return self._mat.nbytes
        except:
            import pdb; pdb.set_trace()

    def toarray(self):
        return self._mat

    def _matmat(self, x):
        return self._mat@x

    def _get_sparsity(self, tol=None):
        nnz = flux.linalg.nnz(self._mat, tol)
        size = self._mat.size
        return 0 if size == 0 else nnz/size

    @property
    def is_empty_leaf(self):
        return False


class FormFactorSparseBlock(FormFactorLeafBlock,
                          scipy.sparse.linalg.LinearOperator):

    def __init__(self, *args):
        super().__init__(*args)

    def _matmat(self, x):
        return np.array(self._spmat@x)

    @property
    def is_empty_leaf(self):
        return False


class FormFactorCsrBlock(FormFactorSparseBlock):

    def __init__(self, linop, mat):
        super().__init__(linop, mat.shape)
        if isinstance(mat, np.ndarray):
            spmat = scipy.sparse.csr_matrix(mat)
        elif isinstance(mat, scipy.sparse.spmatrix):
            spmat = mat
        else:
            raise Exception('invalid class for mat: %s' % type(mat))
        self._spmat = spmat

    @property
    def nbytes(self):
        return self._spmat.data.nbytes

    def tocsr(self):
        return self._spmat


class FormFactorSvdBlock(FormFactorLeafBlock,
                         scipy.sparse.linalg.LinearOperator):

    def __init__(self, linop, mat, k):
        super().__init__(linop, mat.shape)
        self._k = k
        with IndentedPrinter() as _:
            _.print('svds(%d x %d, %d)' % (*mat.shape, k))
            wrapped_mat = DebugLinearOperator(mat)
            [u, s, vt] = scipy.sparse.linalg.svds(wrapped_mat, k)
            wrapped_mat.debug_print()
        self._u = u
        self._s = s
        self._vt = vt
        self._try_to_compress()

    def _try_to_compress(self):
        self._compressed = \
            self._get_sparsity(self._tol) > self._sparsity_threshold
        if self._compressed:
            self._I = np.where(np.any(abs(self._u) > self._tol, axis=1))[0]
            self._J = np.where(np.any(abs(self._vt) > self._tol, axis=0))[0]
            self._u = self._u[self._I, :]
            self._vt = self._vt[:, self._J]

    def _matmat(self, x):
        if self._compressed:
            y_ = self._vt@x[self._J, :]
            y_ = (y_.T*self._s).T
            y_ = self._u@y_
            y = np.zeros((self.shape[0], x.shape[1]), dtype=self.dtype)
            y[self._I, :] = y_
            return y
        else:
            y = self._vt@x
            y = (y.T*self._s).T
            y = self._u@y
            return y

    @property
    def is_empty_leaf(self):
        return False

    @property
    def nbytes(self):
        nbytes = self._u.nbytes + self._s.nbytes + self._vt.nbytes
        if self._compressed:
            nbytes += self._I.nbytes + self._J.nbytes
        return nbytes

    def _get_sparsity(self, tol=None):
        u_nnz = flux.linalg.nnz(self._u, tol)
        v_nnz = flux.linalg.nnz(self._vt, tol)
        size = self._u.size + self._vt.size + self._k
        return 0 if size == 0 else (u_nnz + v_nnz + self._k)/size

    @property
    def compressed(self):
        return self._compressed


class FormFactorBlockMatrix(CompressedFormFactorBlock,
                            scipy.sparse.linalg.LinearOperator):

    def __init__(self, root, shape):
        super().__init__(root, shape)

    def make_block(self, shape_model, I, J, is_diag=False, spmat=None):
        nnz, shape = spmat.nnz, spmat.shape
        size = np.product(shape)
        sparsity = nnz/size

        if shape[0] == 0 and shape[1] == 0:
            return self.root.make_null_block()

        if nnz == 0:
            return self.root.make_zero_block(shape)

        if size < self._min_size:
            if sparsity < self._sparsity_threshold:
                return self.root.make_sparse_block(spmat, fmt='csr')
            else:
                return self.root.make_dense_block(spmat.toarray())
        else:
            if not is_diag:
                # TODO: how inefficient is this now that we're using our
                # new implementation of estimate_rank? at the very least,
                # we should save the SVD computed in the process of
                # determing the rank and pass it to make_svd_block if
                # necessary to avoid recomputing the SVD
                #
                # TODO: need to check how much time passing
                # "return_singular_vectors=False" saves
                with IndentedPrinter() as _:
                    rank = flux.linalg.estimate_rank(spmat, self._tol)
                    _.print('estimate_rank(tol = %g) = %d' % (self._tol, rank))

                if rank == 0:
                    return self.root.make_zero_block(shape)

                # Compute the size of the SVD and compare with the size
                # of the sparse matrix (already computed) to determine
                # which to use
                nbytes_svd = \
                    rank*np.dtype(spmat.dtype).itemsize*(sum(shape) + 1)
                nbytes_sparse = spmat.data.nbytes \
                    + spmat.indices.nbytes + spmat.indptr.nbytes
                if nbytes_svd < nbytes_sparse:
                    return self.root.make_svd_block(spmat, rank)
                else:
                    return self.root.make_sparse_block(spmat)
            else:
                # Since we descend depth-first through the
                # hierarchical block matrix, we want to free this here
                # to keep our memory footprint small
                block = self.make_child_block(shape_model, spmat, I, J)
                if block.is_dense():
                    block = self.root.make_dense_block(block.toarray())
                elif block.is_sparse():
                    block = self.root.make_sparse_block(block.tocsr())
                return block

    def _matmat(self, x):
        ys = []
        n = x.shape[1]
        for i, row_block_inds in enumerate(self._row_block_inds):
            m = len(row_block_inds)
            terms = (
                self._blocks[i, j]@x[J]
                for j, J in enumerate(self._col_block_inds)
                if not self._blocks[i, j].is_empty_leaf
            )
            col_block = sum(terms, np.zeros((m, n), dtype=self.dtype))
            ys.append(col_block)
        try:
            return np.concatenate(ys)[self._row_rev_perm]
        except:
            import pdb; pdb.set_trace()

    @property
    def nbytes(self):
        return sum(I.nbytes for I in self._row_block_inds) \
            + sum(J.nbytes for J in self._col_block_inds) \
            + self._row_rev_perm.nbytes \
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

    def toarray(self):
        row = []
        for row_blocks in self._blocks:
            col = []
            for block in row_blocks:
                col.append(block.toarray())
            row.append(np.hstack(col))
        return np.vstack(row)

    def tocsr(self):
        row = []
        for row_blocks in self._blocks:
            col = []
            for block in row_blocks:
                col.append(block.tocsr())
            row.append(scipy.sparse.hstack(col))
        return scipy.sparse.vstack(row)


class FormFactor2dTreeBlock(FormFactorBlockMatrix):

    def __init__(self, root, shape_model, parent_spmat=None, I0=None, J0=None):
        # TODO: it would be helpful to come up with a way to
        # distinguish more carefully between the two different
        # permutations used here. The arrays I and J (stored in
        # _row_block_inds and _col_block_inds) index into the current
        # block, while I0 and J0 index into the "ambient" index space,
        # so that I0[I] indexes the original form factor matrix.

        super().__init__(
            root,
            root.shape if I0 is None else (len(I0), len(J0))
        )
        self._set_block_inds(shape_model, I0, J0)
        self._row_rev_perm = np.argsort(np.concatenate(self._row_block_inds))

        blocks = []
        for i, I in enumerate(self._row_block_inds):
            I_ = I if I0 is None else I0[I]
            row = []
            for j, J in enumerate(self._col_block_inds):
                J_ = J if J0 is None else J0[J]
                with IndentedPrinter() as _:
                    _.print(
                        '_get_form_factor_block(|I_%d| = %d, |J_%d| = %d)' % (
                            i, len(I), j, len(J)))
                    if parent_spmat is None:
                        spmat = get_form_factor_block(shape_model, I_, J_)
                    else:
                        try:
                            spmat = parent_spmat[I, :][:, J]
                        except:
                            import pdb; pdb.set_trace()
                            print()
                    is_diag = i == j
                    block = self.make_block(shape_model, I_, J_, is_diag, spmat)
                    if block is None:
                        import pdb; pdb.set_trace()
                row.append(block)
            blocks.append(row)
        self._blocks = np.array(blocks, dtype=CompressedFormFactorBlock)

    @property
    def is_empty_leaf(self):
        return False


class FormFactorQuadtreeBlock(FormFactor2dTreeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_child_block(self, *args):
        return self.root.make_quadtree_block(*args)

    def _set_block_inds(self, shape_model, I, J):
        P = shape_model.P
        PI = P[:, :2] if I is None else P[I, :2]
        PJ = P[:, :2] if J is None else P[J, :2]
        self._row_block_inds = get_quadrant_order(PI)
        self._col_block_inds = get_quadrant_order(PJ)


class FormFactorOctreeBlock(FormFactor2dTreeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_child_block(self, *args):
        return self.root.make_octree_block(*args)

    def _set_block_inds(self, shape_model, I, J):
        P = shape_model.P
        PI = P[:] if I is None else P[I]
        PJ = P[:] if J is None else P[J]
        self._row_block_inds = _octant_order(PI)
        self._col_block_inds = _octant_order(PJ)


class FormFactorPartitionBlock(FormFactorBlockMatrix):

    def __init__(self, root, shape_model, parts,
                 ChildBlock=FormFactorQuadtreeBlock):
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
        self._row_rev_perm = np.argsort(np.concatenate(self._row_block_inds))

        blocks = []
        for i, I in enumerate(parts):
            row_blocks = []
            for j, J in enumerate(parts):
                is_diag = i == j
                spmat = get_form_factor_block(shape_model, I, J)
                block = self.make_block(shape_model, I, J, is_diag, spmat)
                row_blocks.append(block)
            blocks.append(row_blocks)
        self._blocks = np.array(blocks, dtype=CompressedFormFactorBlock)

    def make_child_block(self, *args):
        return self.ChildBlock(self.root, *args)

    @property
    def is_empty_leaf(self):
        return True


class CompressedFormFactorMatrix(scipy.sparse.linalg.LinearOperator):

    def __init__(self, shape_model, *args, tol=1e-5, max_rank=60,
                 min_size=16384, RootBlock=FormFactorQuadtreeBlock,
                 **kwargs):
        self.dtype = shape_model.dtype
        self.num_faces = shape_model.num_faces

        self._tol = tol
        self._max_rank = max_rank
        self._min_size = min_size
        self._root = RootBlock(self, shape_model, *args, **kwargs)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def assemble_using_quadtree(cls, *args, **kwargs):
        return CompressedFormFactorMatrix(
            *args, **kwargs, RootBlock=FormFactorQuadtreeBlock)

    @classmethod
    def assemble_using_octree(cls, *args, **kwargs):
        return CompressedFormFactorMatrix(
            *args, **kwargs, RootBlock=FormFactorOctreeBlock)

    @classmethod
    def assemble_using_partition(cls, *args, **kwargs):
        return CompressedFormFactorMatrix(
            *args, **kwargs, RootBlock=FormFactorPartitionBlock)

    @property
    def shape(self):
        return self.num_faces, self.num_faces

    @property
    def nbytes(self):
        return self._root.nbytes

    @property
    def depth(self):
        return self._root.depth

    @property
    def sparsity_threshold(self):
        return 2/3

    def make_null_block(self):
        return FormFactorNullBlock(self)

    def make_zero_block(self, *args):
        return FormFactorZeroBlock(self, *args)

    def make_dense_block(self, *args):
        return FormFactorDenseBlock(self, *args)

    def make_sparse_block(self, *args, fmt='csr'):
        if fmt == 'csr':
            return FormFactorCsrBlock(self, *args)
        else:
            raise Exception('unknown sparse matrix format "%s"' % fmt)

    def make_svd_block(self, *args):
        return FormFactorSvdBlock(self, *args)

    def make_quadtree_block(self, *args):
        return FormFactorQuadtreeBlock(self, *args)

    def make_octree_block(self, *args):
        return FormFactorOctreeBlock(self, *args)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def _matmat(self, x):
        return self._root@x
