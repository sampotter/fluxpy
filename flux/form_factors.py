import pickle

import flux.config
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


from cached_property import cached_property


from flux.debug import IndentedPrinter


def _get_form_factor_block(shape_model, P, N, A, I, J, eps):
    m, n = len(I), len(J)

    NJ_PJ = np.sum(N[J]*P[J], axis=1)
    AJ = A[J]

    # NOTE: we're deliberately using Python's built-in lists here
    # to accumulate the data for the sparse matrix instead of
    # numpy arrays, since using np.concatenate in the loop below
    # ends up making _compute_FF_block O(N^3).
    size = m
    data = np.empty(m, dtype=shape_model.dtype)
    indices = np.empty(m, dtype=np.intp)
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
        vis = shape_model.check_vis_1_to_N(i, J[row_indices])
        row_indices = row_indices[vis]

        s = np.pi*np.sum((P[i] - P[J[row_indices]])**2, axis=1)**2
        s[s == 0] = np.inf
        row_data = row_data[row_indices]*AJ[row_indices]/s

        assert row_data.size == row_indices.size

        if size < row_data.size + i0:
            size = max(2*size, row_data.size + i0)

            tmp = np.empty(size, dtype=shape_model.dtype)
            tmp[:i0] = data[:i0]
            data = tmp

            tmp = np.empty(size, dtype=np.intp)
            tmp[:i0] = indices[:i0]
            indices = tmp

        data[i0:(row_data.size + i0)] = row_data
        indices[i0:(row_indices.size + i0)] = row_indices

        i0 += row_data.size

        indptr.append(indptr[-1] + row_indices.size)

    data = data[:i0]
    indices = indices[:i0]
    indptr = np.array(indptr, dtype=np.intp)

    nbytes = data.nbytes + indices.nbytes + indptr.nbytes

    spmat = scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

    return spmat


def get_form_factor_block(shape_model, I=None, J=None, eps=None):
    P = shape_model.P
    N = shape_model.N
    A = shape_model.A

    if eps is None:
        eps = flux.config.DEFAULT_EPS
    if I is None:
        I = np.arange(P.shape[0])
    if J is None:
        J = np.arange(P.shape[0])

    with IndentedPrinter() as _:
        _.print('_get_form_factor_block()')
        return _get_form_factor_block(shape_model, P, N, A, I, J, eps)


def get_vis_block(shape_model, I=None, J=None, eps=None):
    if eps is None:
        eps = flux.config.DEFAULT_EPS
    if I is None:
        I = np.arange(shape_model.num_faces)
    if J is None:
        J = np.arange(shape_model.num_faces)

    m, n = len(I), len(J)

    vis = np.zeros((m, n), dtype=np.bool8)

    for _, i in enumerate(I):
        vis[_] = shape_model.check_vis_1_to_N(i, J)

    return vis


class FormFactorMatrix(scipy.sparse.linalg.LinearOperator):

    def __init__(self, shape_model, I=None, J=None, eps=None):
        self.shape_model = shape_model

        if I is None:
            self.I = np.arange(self.num_faces)
        else:
            self.I = I

        if J is None:
            self.J = np.arange(self.num_faces)
        else:
            self.J = J

        if eps is None:
            self.eps = flux.config.DEFAULT_EPS

        self._col_vis = np.empty(len(self.J), dtype=object)

    @property
    def num_faces(self):
        return self.shape_model.num_faces

    @property
    def dtype(self):
        return self.P.dtype

    @property
    def shape(self):
        return (len(self.I), len(self.J))

    @property
    def N(self):
        return self.shape_model.N

    @property
    def P(self):
        return self.shape_model.P

    @property
    def A(self):
        return self.shape_model.A

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

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # def _get_col_vis(self, col, j):
    #     nonzero = np.where(abs(col) > self.eps)[0]

    #     nnz = len(nonzero)
    #     if nnz == 0:
    #         return np.array([], dtype=nonzero.dtype)

    #     # TODO: should make a template rayhit below to avoid setting
    #     # things that don't change as i varies
    #     rayhit = embree.RayHit1M(nnz)
    #     rayhit.org[:] = self.P[j]
    #     rayhit.dir[:] = self.PI[nonzero] - self.P[j]
    #     rayhit.tnear[:] = self.eps
    #     rayhit.tfar[:] = np.inf
    #     rayhit.flags[:] = 0
    #     rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

    #     context = embree.IntersectContext()
    #     context.flags = embree.IntersectContextFlags.Coherent

    #     self.scene.intersect1M(context, rayhit)

    #     vis = np.logical_and(
    #         rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
    #         rayhit.prim_id == self.I[nonzero]
    #     )
    #     vis = nonzero[vis]

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
            vis = self.shape_model.check_vis_1_to_N(i, self.J)
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
