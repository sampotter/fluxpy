import flux.config
import embree
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


from cached_property import cached_property


from flux.debug import IndentedPrinter


def _get_form_factor_block(scene, P, N, A, I, J, eps):
    m, n = len(I), len(J)

    def check_vis(i, J):
        assert J.size > 0
        rayhit = embree.RayHit1M(len(J))
        context = embree.IntersectContext()
        rayhit.org[:] = P[i]
        rayhit.dir[:] = P[J] - P[i]
        rayhit.tnear[:] = eps
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
        scene.intersect1M(context, rayhit)
        return np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == J
        )

    NJ_PJ = np.sum(N[J]*P[J], axis=1)
    AJ = A[J]

    # NOTE: we're deliberately using Python's built-in lists here
    # to accumulate the data for the sparse matrix instead of
    # numpy arrays, since using np.concatenate in the loop below
    # ends up making _compute_FF_block O(N^3).
    size = m
    data = np.empty(m, dtype=np.float32)
    indices = np.empty(m, dtype=np.intc)
    indptr = [0]

    # Current index into data and indices
    i0 = 0

    for r, i in enumerate(I):
        row_data = np.maximum(0, N[i]@(P[J] - P[i]).T) \
            * np.maximum(0, P[i]@N[J].T - NJ_PJ)

        # Set diagonal entries to zero.
        row_data[i == J] = 0

        row_indices = np.where(abs(row_data) > eps)[0]
        if row_indices.size == 0:
            indptr.append(indptr[-1])
            continue
        vis = check_vis(i, J[row_indices])
        row_indices = row_indices[vis]

        s = np.pi*np.sum((P[i] - P[J[row_indices]])**2, axis=1)**2
        s[s == 0] = np.inf
        row_data = row_data[row_indices]*AJ[row_indices]/s

        assert row_data.size == row_indices.size

        if size < row_data.size + i0:
            size *= 2

            tmp = np.empty(size, dtype=np.float32)
            tmp[:i0] = data[:i0]
            data = tmp

            tmp = np.empty(size, dtype=np.intc)
            tmp[:i0] = indices[:i0]
            indices = tmp

        data[i0:(row_data.size + i0)] = row_data
        indices[i0:(row_indices.size + i0)] = row_indices

        i0 += row_data.size

        indptr.append(indptr[-1] + row_indices.size)

    data = data[:i0]
    indices = indices[:i0]
    indptr = np.array(indptr, dtype=np.intc)

    nbytes = data.nbytes + indices.nbytes + indptr.nbytes

    vis = scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

    return vis


def get_form_factor_block(shape_model, I=None, J=None, eps=None):
    P = shape_model.P
    N = shape_model.N
    A = shape_model.A

    scene = shape_model.scene

    if eps is None:
        eps = flux.config.DEFAULT_EPS
    if I is None:
        I = np.arange(P.shape[0])
    if J is None:
        J = np.arange(P.shape[0])

    with IndentedPrinter() as _:
        _.print('_get_form_factor_block()')
        return _get_form_factor_block(scene, P, N, A, I, J, eps)


class FormFactorMatrix(scipy.sparse.linalg.LinearOperator):

    def __init__(self, shape_model, I, J, eps=None):
        self.shape_model = shape_model

        self.I = I
        self.J = J

        if eps is None:
            self.eps = flux.config.DEFAULT_EPS

        self._col_vis = np.empty(len(self.J), dtype=object)

    @property
    def dtype(self):
        return self.P.dtype

    @property
    def shape(self):
        return (len(self.I), len(self.J))

    @cached_property
    def NI(self):
        return self.N[self.I]

    @cached_property
    def PI(self):
        return self.P[self.I]

    @cached_property
    def NI_PI(self):
        return np.sum(self.NI*self.PI, axis=1)

    @cached_property
    def NJ(self):
        return self.N[self.J]

    @cached_property
    def PJ(self):
        return self.P[self.J]

    @cached_property
    def NJ_PJ(self):
        return np.sum(self.NJ*self.PJ, axis=1)

    def _get_row_vis(self, row, i):
        nonzero = np.where(abs(row) > self.eps)[0]

        # TODO: should make a template rayhit below to avoid setting
        # things that don't change as i varies
        rayhit = embree.RayHit1M(len(nonzero))
        rayhit.org[:] = self.P[i]
        rayhit.dir[:] = self.PJ[nonzero] - self.P[i]
        rayhit.tnear[:] = self.eps
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
        context = embree.IntersectContext()
        self.scene.intersect1M(context, rayhit)

        vis = np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == self.J[nonzero]
        )
        return nonzero[vis]

    def _get_col_vis(self, col, j):
        nonzero = np.where(abs(col) > self.eps)[0]

        nnz = len(nonzero)
        if nnz == 0:
            return np.array([], dtype=nonzero.dtype)

        # TODO: should make a template rayhit below to avoid setting
        # things that don't change as i varies
        rayhit = embree.RayHit1M(nnz)
        rayhit.org[:] = self.P[j]
        rayhit.dir[:] = self.PI[nonzero] - self.P[j]
        rayhit.tnear[:] = self.eps
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        context = embree.IntersectContext()
        context.flags = embree.IntersectContextFlags.Coherent

        self.scene.intersect1M(context, rayhit)

        vis = np.logical_and(
            rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
            rayhit.prim_id == self.I[nonzero]
        )
        vis = nonzero[vis]

    def _matvec(self, x):
        return self._matmat(x.reshape(x.size, 1)).ravel()

    def _matmat(self, X):
        # TODO: any intermediate variables that are O(N) in size below
        # and can be reused should be stored as cached properties

        # TODO: we want to think about the "Jv" optimization...

        # Create matrix used to store result
        Y = np.empty((self.shape[0], X.shape[1]), dtype=X.dtype)

        for r, i in enumerate(self.I):
            row = np.maximum(0, self.N[i]@(self.PJ - self.P[i]).T) \
                * np.maximum(0, self.P[i]@self.NJ.T - self.NJ_PJ)

            # Set diagonal entries to zero.
            row[i == self.J] = 0

            # Find visible indices for current element
            vis = self._get_row_vis(row, i)
            Jv = self.J[vis]

            # Normalize row
            s = np.pi*np.sum((self.P[i] - self.P[Jv])**2, axis=1)**2
            s[s == 0] = np.inf

            row = row[vis]*self.A[Jv]/s

            # Compute row of Y
            Y[r, :] = row@X[vis, :]

        return Y

    def _rmatvec(self, x):
        return self._rmatmat(x.reshape(x.size, 1)).ravel()

    def _rmatmat(self, X):
        # TODO: see comments above for _matmat

        # Create matrix used to store result
        Y = np.empty((self.shape[1], X.shape[1]), dtype=X.dtype)

        for c, j in enumerate(self.J):
            col = np.maximum(0, (self.PI - self.P[j])@self.N[j]) \
                * np.maximum(0, self.NI@self.P[j] - self.NI_PI)

            # Set diagonal entries to zero.
            col[self.I == j] = 0

            # Find visible indices for current element
            vis = self._get_col_vis(col, j)
            Iv = self.I[vis]

            # Normalize row
            s = np.pi*np.sum((self.P[Iv] - self.P[j])**2, axis=1)**2
            s[s == 0] = np.inf

            # Compute row of Y
            Y[c, :] = (col[vis]*self.A[j]/s)@X[vis, :]

        return Y
