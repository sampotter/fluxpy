import embree
import itertools as it
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.sparse
import scipy.sparse.linalg


DEFAULT_EPS = 1e-5


def _get_centroids(V, F):
    return V[F].mean(axis=1)


def _get_surface_normals_and_areas(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A


def _estimate_rank(spmat, tol, k0=40):
    k = k0
    while True:
        k = min(k, min(spmat.shape) - 1)
        S = scipy.sparse.linalg.svds(spmat, k, return_singular_vectors=False)
        sv_thresh = S[-1]*max(spmat.shape)*tol
        if S[0] < sv_thresh:
            return np.where((S < sv_thresh)[::-1])[0][0]
        k *= 2
        if k > min(spmat.shape):
            return min(spmat.shape)


def _compute_FF_block(P, N, A, I=None, J=None, scene=None, eps=None):
    if eps is None:
        eps = DEFAULT_EPS

    if I is None:
        I = np.arange(P.shape[0])
    if J is None:
        J = np.arange(P.shape[0])
    m, n = len(I), len(J)

    def check_vis(i, J):
        assert J.size > 0
        rayhit = embree.RayHit1M(len(J))
        rayhit.org[:] = P[i]
        rayhit.dir[:] = P[J] - P[i]
        rayhit.tnear[:] = eps
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
        context = embree.IntersectContext()
        scene.intersect1M(context, rayhit)
        return np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == J
        )

    NJ_PJ = np.sum(N[J]*P[J], axis=1)
    AJ = A[J]

    data = np.array([], dtype=P.dtype)
    indices = np.array([], dtype=int)
    indptr = np.array([0], dtype=int)
    for r, i in enumerate(I):
        row_data = np.maximum(0, N[i]@(P[J] - P[i]).T) \
            * np.maximum(0, P[i]@N[J].T - NJ_PJ)
        row_data[r] = 0
        row_indices = np.where(abs(row_data) > 0)[0]
        if row_indices.size == 0:
            indptr = np.concatenate([indptr, [indptr[-1]]])
            continue
        vis = check_vis(i, row_indices)
        row_indices = row_indices[vis]
        s = np.pi*np.sum((P[i] - P[J[row_indices]])**2, axis=1)**2
        s[s == 0] = np.inf
        row_data = row_data[row_indices]*AJ[row_indices]/s
        data = np.concatenate([data, row_data])
        indices = np.concatenate([indices, row_indices])
        indptr = np.concatenate([indptr, [indptr[-1] + row_indices.size]])

    vis = scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

    if indices.size == 0:
        return vis

    return vis


def _quadrant_order(X, bbox=None):
    if bbox is not None:
        (xmin, xmax), (ymin, ymax) = bbox
    else:
        xmin, ymin = np.min(X, axis=0)
        xmax, ymax = np.max(X, axis=0)
    xc, yc = (xmin + xmax)/2, (ymin + ymax)/2
    x, y = X.T
    Is = []
    for xop, yop in it.product([np.less_equal, np.greater], repeat=2):
        B = np.column_stack([xop(x, xc), yop(y, yc)])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
    return Is

def _octant_order(X, bbox=None):
    if bbox is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    else:
        xmin, ymin, zmin = np.min(X, axis=0)
        xmax, ymax, zmax = np.max(X, axis=0)
    xc, yc, zc = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
    x, y, z = X.T
    Is = []
    for xop, yop, zop in it.product([np.less_equal, np.greater], repeat=3):
        B = np.column_stack([xop(x, xc), yop(y, yc), zop(z, zc)])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
    return Is


# TODO: this function REALLY needs to be profiled and optimized!!!
def _plot_block(block, **kwargs):
    fig = plt.figure()

    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
    else:
        fig.set_size_inches(12, 12)

    ax = fig.add_subplot()
    ax.axis('off')

    ax.set_xlim(-0.001, 1.001)
    ax.set_ylim(-0.001, 1.001)

    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='k',
                             facecolor='none')
    ax.add_patch(rect)

    def get_ind_offsets(inds):
        return np.concatenate([[0], np.cumsum([len(I) for I in inds])])

    def add_rects(block, c0=(0, 0), w0=1, h0=1):
        row_offsets = get_ind_offsets(block._row_block_inds)
        col_offsets = get_ind_offsets(block._col_block_inds)
        for (i, row), (j, col) in it.product(
            enumerate(row_offsets[:-1]),
            enumerate(col_offsets[:-1])
        ):
            w = w0*(row_offsets[i + 1] - row)/row_offsets[-1]
            h = h0*(col_offsets[j + 1] - col)/col_offsets[-1]
            i0, j0 = c0
            c = (i0 + w0*row/row_offsets[-1], j0 + h0*col/col_offsets[-1])

            child = block._blocks[i, j]
            if child.is_leaf:
                if isinstance(child, FormFactorSvdBlock):
                    facecolor = 'cyan' if child.compressed else 'orange'
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor=facecolor)
                    ax.add_patch(rect)
                elif isinstance(child, FormFactorZeroBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif isinstance(child, FormFactorSparseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='red')
                    ax.add_patch(rect)
                elif isinstance(child, FormFactorDenseBlock):
                    rect = patches.Rectangle(
                        c, w, h, edgecolor='none', facecolor='magenta')
                    ax.add_patch(rect)
                else:
                    raise Exception('TODO: add %s to _plot_block' % type(child))
            else:
                add_rects(child, c, w, h)

            rect = patches.Rectangle(
                c, w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax.add_patch(rect)

    add_rects(block)

    ax.invert_xaxis()

    return fig, ax


def _get_nnz(mat, tol=None):
    if tol is None:
        return (mat != 0).sum()
    else:
        return (abs(mat) < tol).sum()


def _get_sparsity(mat, tol=None):
    nnz = _get_nnz(mat, tol)
    size = mat.size
    return 0 if size == 0 else nnz/size


@np.vectorize
def _is_dense(block):
    return isinstance(block, FormFactorDenseBlock)


@np.vectorize
def _is_sparse(block):
    return isinstance(block, FormFactorSparseBlock) \
        or isinstance(block, FormFactorZeroBlock)


class FormFactorBlock:

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
    def _P(self):
        return self._root.P

    @property
    def _N(self):
        return self._root.N

    @property
    def _A(self):
        return self._root.A

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

    def show(self, **kwargs):
        return _plot_block(self, **kwargs)


class FormFactorLeafBlock(FormFactorBlock):

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def depth(self):
        return 0

    @property
    def is_leaf(self):
        return True


class FormFactorNullBlock(FormFactorLeafBlock):

    def __init__(self, root):
        super().__init__(root, (0, 0))

    def __matmul__(self, x):
        return np.array([], dtype=self.dtype)

    @property
    def nbytes(self):
        return 0


class FormFactorZeroBlock(FormFactorLeafBlock):

    def __init__(self, root, shape):
        super().__init__(root, shape)

    def __matmul__(self, x):
        m = self.shape[0]
        y_shape = (m,) if x.ndim == 1 else (m, x.shape[1])
        return np.zeros(y_shape, dtype=self.dtype)

    @property
    def nbytes(self):
        return 0


class FormFactorDenseBlock(FormFactorLeafBlock):

    def __init__(self, root, mat):
        if not isinstance(mat, np.ndarray):
            import pdb; pdb.set_trace()
            raise Exception('`mat` must be a numpy ndarray')
        super().__init__(root, mat.shape)
        self._mat = mat

    @property
    def nbytes(self):
        try:
            return self._mat.nbytes
        except:
            import pdb; pdb.set_trace()

    def __matmul__(self, x):
        return self._mat@x

    def _get_sparsity(self, tol=None):
        nnz = _get_nnz(self._mat, tol)
        size = self._mat.size
        return 0 if size == 0 else nnz/size


class FormFactorSparseBlock(FormFactorLeafBlock):

    def __init__(self, *args):
        super().__init__(*args)

    def __matmul__(self, x):
        return np.array(self._spmat@x)


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


class FormFactorSvdBlock(FormFactorLeafBlock):

    def __init__(self, linop, mat, k):
        super().__init__(linop, mat.shape)
        self._k = k
        [u, s, vt] = scipy.sparse.linalg.svds(mat, k)
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

    def __matmul__(self, x):
        if self._compressed:
            y_ = self._vt@x[self._J]
            y_ *= self._s
            y_ = self._u@y_
            y = np.zeros(self.shape[0], dtype=self.dtype)
            y[self._I] = y_
            return y
        else:
            y = self._vt@x
            y *= self._s
            y = self._u@y
            return y

    @property
    def nbytes(self):
        nbytes = self._u.nbytes + self._s.nbytes + self._vt.nbytes
        if self._compressed:
            nbytes += self._I.nbytes + self._J.nbytes
        return nbytes

    def _get_sparsity(self, tol=None):
        u_nnz, v_nnz = _get_nnz(self._u, tol), _get_nnz(self._vt, tol)
        size = self._u.size + self._vt.size + self._k
        return 0 if size == 0 else (u_nnz + v_nnz + self._k)/size

    @property
    def compressed(self):
        return self._compressed


class FormFactor2dTreeBlock(FormFactorBlock):

    def __init__(self, root, I0=None, J0=None):
        super().__init__(
            root,
            root.shape if I0 is None else (len(I0), len(J0))
        )
        self._set_block_inds(I0, J0)
        self._row_rev_perm = np.argsort(np.concatenate(self._row_block_inds))

        blocks = []
        for i, I in enumerate(self._row_block_inds):
            I_ = I if I0 is None else I0[I]
            row = []
            for j, J in enumerate(self._col_block_inds):
                J_ = J if J0 is None else J0[J]
                block = self._get_form_factor_block(I_, J_)
                row.append(block)
            blocks.append(row)
        self._blocks = np.array(blocks, dtype=FormFactorBlock)

    def _get_form_factor_block(self, I, J):
        # TODO: according to some policy, we will occasionally want to
        # replace this with a linear operator version to save
        # memory... to get the best effect from this, we're going to
        # need to come up with a good policy to decide how to do this
        #
        # maybe there's some way to query numpy and figure out what
        # the biggest matrix it can allocate is? or maybe we should
        # just pass a "max_bytes" parameter as some kind of
        # threshold...
        #
        if len(I) == 0 and len(J) == 0:
            return self.root.make_null_block()
        spmat = self.root._compute_FF_block(I, J)
        nnz, shape = spmat.nnz, spmat.shape
        size = np.product(shape)
        sparsity = nnz/size
        if size < self._min_size:
            # TODO: _sparse_threshold should be O(log(n)/n), not constant
            if sparsity < self._sparsity_threshold:
                return self.root.make_sparse_block(spmat, fmt='csr')
            else:
                return self.root.make_dense_block(spmat.toarray())
        else:
            # TODO: how inefficient is this now that we're using our
            # new implementation of _estimate_rank? at the very least,
            # we should save the SVD computed in the process of
            # determing the rank and pass it to make_svd_block if
            # necessary to avoid recomputing the SVD
            rank = _estimate_rank(spmat, self._tol)
            if rank == 0:
                if nnz > 0:
                    return self.root.make_sparse_block(spmat)
                else:
                    return self.root.make_zero_block(shape)
            elif rank < min(len(I), len(J)) and rank <= self._max_rank:
                return self.root.make_svd_block(spmat, rank)
            else:
                block = self._make_tree_block(I, J)
                if _is_dense(block._blocks).all():
                    block = self.root.make_dense_block(spmat)
                elif _is_sparse(block._blocks).all():
                    block = self.root.make_sparse_block(spmat)
                return block

    def __matmul__(self, x):
        ys = []
        for i in range(len(self._row_block_inds)):
            ys.append(sum(
                self._blocks[i, j]@x[J]
                for j, J in enumerate(self._col_block_inds)
            ))
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


class FormFactorQuadtreeBlock(FormFactor2dTreeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_tree_block(self, *args):
        return self.root.make_quadtree_block(*args)

    def _set_block_inds(self, I, J):
        PI = self._P[:, :2] if I is None else self._P[I, :2]
        PJ = self._P[:, :2] if J is None else self._P[J, :2]
        self._row_block_inds = _quadrant_order(PI)
        self._col_block_inds = _quadrant_order(PJ)


class FormFactorOctreeBlock(FormFactor2dTreeBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_tree_block(self, *args):
        return self.root.make_octree_block(*args)

    def _set_block_inds(self, I, J):
        PI = self._P[:] if I is None else self._P[I]
        PJ = self._P[:] if J is None else self._P[J]
        self._row_block_inds = _octant_order(PI)
        self._col_block_inds = _octant_order(PJ)


class FormFactorMatrix:

    def __init__(self, scene, V, F, tol=1e-5, max_rank=60, min_size=1024,
                 RootBlock=FormFactorQuadtreeBlock):
        self.V = V
        self.F = F
        self.P = _get_centroids(V, F)

        N, A = _get_surface_normals_and_areas(V, F)
        self.N = N
        self.A = A

        self._tol = tol
        self._max_rank = max_rank
        self._min_size = min_size

        self._scene = scene
        self._root = RootBlock(self)

        # Delete self._scene here, since we only need it when
        # constructing the form factor matrix, and since storing it in
        # a FormFactorMatrix instance prevents us from pickling
        del self._scene

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def assemble_using_quadtree(cls, *args, **kwargs):
        return FormFactorMatrix(
            *args, **kwargs, RootBlock=FormFactorQuadtreeBlock)

    @classmethod
    def assemble_using_octree(cls, *args, **kwargs):
        return FormFactorMatrix(
            *args, **kwargs, RootBlock=FormFactorOctreeBlock)

    @property
    def num_faces(self):
        return self.P.shape[0]

    @property
    def shape(self):
        return self.num_faces, self.num_faces

    @property
    def dtype(self):
        return self.P.dtype

    @property
    def nbytes(self):
        return self.P.nbytes + self.F.nbytes + self.N.nbytes \
            + self.A.nbytes + self._root.nbytes

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

    def show(self, **kwargs):
        return self._root.show(**kwargs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __matmul__(self, x):
        return self._root@x

    def _compute_FF_block(self, I=None, J=None):
        return _compute_FF_block(
            self.P, self.N, self.A, I, J, self._scene, self._tol)
