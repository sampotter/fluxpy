import pickle

import array
import flux.config
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from cached_property import cached_property

def get_form_factor_matrix(shape_model, I=None, J=None, eps=None):
    P = shape_model.P
    N = shape_model.N
    A = shape_model.A

    if eps is None:
        eps = flux.config.DEFAULT_EPS
    if I is None:
        I = np.arange(P.shape[0], dtype=np.uintp)
    if J is None:
        J = np.arange(P.shape[0], dtype=np.uintp)

    m, n = len(I), len(J)

    try:
        NJ_PJ = np.sum(N[J]*P[J], axis=1)
    except:
        import ipdb; ipdb.set_trace()

    AJ = A[J]

    if shape_model.dtype == np.float32:
        typecode = 'f'
    elif shape_model.dtype == np.float64:
        typecode = 'd'
    else:
        raise RuntimeError(f'unsupported dtype {shape_model.dtype}')

    size = m
    data = array.array(typecode)
    indices = array.array('Q')
    indptr = array.array('Q')
    indptr.append(0)

    for r, i in enumerate(I):
        row_data = np.maximum(0, N[i]@(P[J] - P[i]).T) \
            * np.maximum(0, P[i]@N[J].T - NJ_PJ)

        # Set diagonal entries to zero.
        row_data[i == J] = 0

        row_indices = np.where(abs(row_data) > eps)[0]
        if row_indices.size == 0:
            indptr.append(indptr[-1])
            continue
        try:
            vis = shape_model.get_visibility_1_to_N(i, J[row_indices].astype(np.uintp))
        except:
            import ipdb; ipdb.set_trace()
        row_indices = row_indices[vis]

        s = np.pi*np.sum((P[i] - P[J[row_indices]])**2, axis=1)**2
        s[s == 0] = np.inf
        row_data = row_data[row_indices]*AJ[row_indices]/s

        assert row_data.size == row_indices.size

        data.frombytes(row_data.tobytes())
        indices.frombytes(row_indices.astype(np.uintp).tobytes())
        indptr.append(indptr[-1] + row_indices.size)

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

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
