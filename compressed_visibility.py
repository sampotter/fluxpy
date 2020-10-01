import zlib

I = get_quadtree_inds(*P[:, :2].T)

nvis = []
nbytes = []

for i in range(num_faces):

    tmp1 = np.maximum(0, (P[I] - P[i])@N[i])
    tmp2 = np.maximum(0, np.sum((P[i] - P[I])*N[I], axis=1))
    V = (tmp1 > 0) & (tmp2 > 0)

    rayhit = embree.RayHit1M(num_faces)
    rayhit.org[:] = P[i]
    rayhit.dir[:] = P[I] - P[i]
    rayhit.tnear[:] = 1e-5
    rayhit.tfar[:] = np.inf
    rayhit.flags[:] = 0
    rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
    context = embree.IntersectContext()
    scene.intersect1M(context, rayhit)
    H = (rayhit.geom_id != embree.INVALID_GEOMETRY_ID) & (rayhit.prim_id == I)

    V &= H

    nvis.append(V.sum())
    nbytes.append(len(zlib.compress(bytes(V), level=9)))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(nvis, density=True)
plt.subplot(1, 2, 2)
plt.hist(nbytes, density=True)
plt.tight_layout()
plt.show()
