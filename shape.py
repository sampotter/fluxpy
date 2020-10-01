import embree
import numpy as np


def get_centroids(V, F):
    return V[F].mean(axis=1)


def get_cross_products(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C


def get_surface_normals_and_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A


class TrimeshShapeModel:

    def __init__(self, V, F):
        self.dtype = V.dtype

        self.V = V
        self.F = F

        P = get_centroids(V, F)
        N, A = get_surface_normals_and_areas(V, F)

        self.P = P
        self.N = N
        self.A = A

        assert self.P.dtype == self.dtype
        assert self.N.dtype == self.dtype
        assert self.A.dtype == self.dtype

        # Next, we need to set up Embree. The lines below allocate some
        # memory that Embree manages, and loads our vertices and index
        # lists for the faces. In Embree parlance, we create a "device",
        # which manages a "scene", which has one "geometry" in it, which
        # is our mesh.
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        scene = device.make_scene()
        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            V.shape[0], # item_count
        )
        vertex_buffer[:] = V[:]
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            F.shape[0]
        )
        index_buffer[:] = F[:]
        geometry.commit()
        scene.attach_geometry(geometry)
        geometry.release()
        scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)
        self.scene = scene

    @property
    def num_faces(self):
        return self.P.shape[0]

    def get_direct_irradiance(self, F0, dir_sun, eps=None):
        if eps is None:
            eps = 1e3*np.finfo(np.float32).resolution

        # Here, we use Embree directly to find the indices of triangles
        # which are directly illuminated (I_sun) or not (I_shadow).
        ray = embree.Ray1M(self.num_faces)
        ray.org[:] = self.P + eps*self.N
        ray.dir[:] = dir_sun
        ray.tnear[:] = 0
        ray.tfar[:] = np.inf
        ray.flags[:] = 0
        context = embree.IntersectContext()
        self.scene.occluded1M(context, ray)

        # Determine which rays escaped (i.e., can see the sun)
        I = np.isposinf(ray.tfar)

        # Compute the direct irradiance
        E = np.zeros(self.num_faces, dtype=self.dtype)
        E[I] = F0*np.maximum(0, self.N[I]@dir_sun)

        return E
