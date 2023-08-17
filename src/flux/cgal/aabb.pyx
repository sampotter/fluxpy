import numpy as np

from cython.parallel import prange
from libcpp cimport bool

cimport cython

cdef extern from "aabb_wrapper.h":
    cdef struct cgal_aabb:
        pass
    void cgal_aabb_alloc(cgal_aabb **aabb) except +
    void cgal_aabb_init_from_trimesh(cgal_aabb *aabb,
				     size_t num_points, double (*points)[3],
				     size_t num_faces, size_t (*faces)[3]) except +
    void cgal_aabb_deinit(cgal_aabb *aabb) except +
    void cgal_aabb_dealloc(cgal_aabb **aabb) except +
    bool cgal_aabb_test_face_to_face_vis(const cgal_aabb *aabb,
                                         size_t i, size_t j) nogil except +
    bool cgal_aabb_ray_from_centroid_is_occluded(const cgal_aabb *aabb,
                                                 size_t i, double d[3]) nogil except +
    bool cgal_aabb_intersect1(const cgal_aabb *aabb, const double x[3], const double d[3],
                              size_t *i, double xt[3]) nogil except +

cdef class AABB:
    cdef cgal_aabb *aabb

    def __init__(self, *args):
        if len(args) > 0:
            raise RuntimeError('initialize CgalAABB using factory functions')

    def __cinit__(self, *args):
        if len(args) > 0:
            raise RuntimeError('initialize CgalAABB using factory functions')

    @staticmethod
    def from_trimesh(double[:, ::1] points, size_t[:, ::1] faces):
        aabb = AABB()
        cgal_aabb_alloc(&aabb.aabb)
        cgal_aabb_init_from_trimesh(
            aabb.aabb,
            points.shape[0], <double(*)[3]>&points[0, 0],
            faces.shape[0], <size_t(*)[3]>&faces[0, 0])
        return aabb

    def __dealloc__(self):
        cgal_aabb_dealloc(&self.aabb)

    def intersect1(self, double[::1] x, double[::1] d):
        cdef size_t i
        cdef double xt[3]
        if cgal_aabb_intersect1(self.aabb, &x[0], &d[0], &i, &xt[0]):
            return i, np.array([xt[0], xt[1], xt[2]])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def intersect1_2d_with_coords(self, double[:,::1] X, double[:,::1] D):
        cdef size_t m = X.shape[0]
        cdef size_t n = D.shape[0]
        cdef size_t l
        cdef double xt[3]

        if m != n:
            raise RuntimeError('intersect1_2d expects input centers and directions with the same length')

        cdef long[:] fint = np.zeros((m), dtype=np.int_)
        cdef double[:, :] xta = np.zeros((m, 3), dtype=np.float_)
        cdef size_t[:] i = np.zeros((m), dtype=np.uint)

        for l in prange(m, nogil=True): # run outer loop in parallel
                if cgal_aabb_intersect1(self.aabb, &X[l,0], &D[l,0], &i[l], &xt[0]):
                    fint[l] = i[l]
                    xta[l, 0] = xt[0]
                    xta[l, 1] = xt[1]
                    xta[l, 2] = xt[2]
                else:
                    fint[l] = -1
                    xta[l] = -1

        # return intersected faces and their centroid coordinates
        return np.asarray(fint), np.asarray(xta)

    def intersect1_2d(self, X, D):

        fint, xta = self.intersect1_2d_with_coords(X, D)

        return fint

    def test_face_to_face_vis(self, size_t i, size_t j):
        return cgal_aabb_test_face_to_face_vis(self.aabb, i, j)

    @cython.boundscheck(False)
    def test_face_to_face_vis_MN(self, size_t[::1] I, size_t[::1] J):
        cdef size_t m = len(I)
        cdef size_t n = len(J)
        cdef bool[:, ::1] vis = np.zeros((m, n), dtype=np.bool_)
        cdef size_t i, j, p, q
        for p in prange(m, nogil=True): # run outer loop in parallel
            i = I[p]
            for q in range(n):
                j = J[q]
                if i == j:
                    continue
                vis[p, q] = cgal_aabb_test_face_to_face_vis(self.aabb, i, j)
        return np.asarray(vis)

    def ray_from_centroid_is_occluded(self, size_t i, double[::1] d):
        return cgal_aabb_ray_from_centroid_is_occluded(self.aabb, i, &d[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def ray_from_centroid_is_occluded_2d(self, long[::1] I, double[:,::1] D):
        cdef size_t m = len(I)
        cdef size_t n = D.shape[0]
        #cdef double[::1] d
        cdef bool[:, ::1] occ = np.zeros((m, n), dtype=np.bool_)
        cdef size_t i, j

        for i in prange(m, nogil=True): # run outer loop in parallel
            for j in range(n):
                #d = D[j,:]
                occ[i,j] = cgal_aabb_ray_from_centroid_is_occluded(self.aabb, i, &D[j,0])
        return occ