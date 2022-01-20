import sys
import numpy as np
import meshio

from coord_tools import unproject_stereographic, sph2cart
from flux.shape import get_surface_normals, get_centroids

BODY_RADIUS=1737.4
LAT0=-90

mesh_path = sys.argv[1]

mesh = meshio.read(mesh_path)

V = mesh.points
V = V.astype(np.float32)  # embree is anyway single precision
V = V[:, :3]
F = mesh.cells[0].data

lon, lat = unproject_stereographic(V[:,0], V[:,1], 0, LAT0,  R=BODY_RADIUS*1.e3) # + V[:,2])
x, y, z = sph2cart(BODY_RADIUS*1.e3 + V[:,2], lat, lon)
Vcart = np.vstack([x,y,z]).T

N = get_surface_normals(Vcart, F)
P = get_centroids(Vcart, F)
N[(N * P).sum(1) < 0] *= -1

# save the triangle mesh as an OBJ file.
num_faces = F.shape[0]
points = V
cells = [('triangle', F)]
mesh = meshio.Mesh(points, cells)
mesh_path = 'deGerlache%d.obj' % num_faces
mesh.write(mesh_path)
print('wrote %s to disk' % mesh_path)