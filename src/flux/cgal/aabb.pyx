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
                                                 size_t i, double d[3]) except +

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