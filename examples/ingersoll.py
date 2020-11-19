import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import FormFactorMatrix
from flux.ingersoll import HemisphericalCrater
from flux.model import compute_steady_state_temp
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import TrimeshShapeModel, get_surface_normals

# Parameters related to the spatial discretization (triangle mesh)
p = 6
h = (2/3)**p

# Form factor matrix parameters
FF_tol = 1e-7

# Physical parameters related to the problem
e0 = np.deg2rad(15)
F0 = 1000
rho = 0.3
emiss = 0.99

# Geometric parameters controlling the shape of the hemispherical crater
beta = np.deg2rad(40)
rc = 0.8

hc = HemisphericalCrater(beta, rc, e0, F0, rho, emiss)
print('- groundtruth temperature in shadow %1.2f K' % (hc.T_gt,))

# Create the triangle mesh
V, F, parts = hc.make_trimesh(h, return_parts=True)
print('- created a Delaunay mesh with %d points and %d faces' % (
    V.shape[0], F.shape[0]))

# Write the mesh to disk as an OBJ file
trimesh.Trimesh(V, F).export('ingersoll.obj')
print('- saved Wavefront OBJ file to ingersoll.obj')

# Flip any upside down triangles
N = get_surface_normals(V, F)
N[N[:, 2] < 0] *= -1

# Create a shape model from the triangle mesh (used for raytracing)
shape_model = TrimeshShapeModel(V, F, N)

# FF = CompressedFormFactorMatrix.assemble_using_quadtree(
#     shape_model, tol=FF_tol)
FF = CompressedFormFactorMatrix.assemble_using_partition(
    shape_model, parts, tol=FF_tol)
print('- assembled compressed form factor matrix (tol = %g, %1.1f Mb)' % (
    FF_tol, FF.nbytes/1024**2))

FF.save('ingersoll.bin')
print('- saved compressed form factor matrix to disk')

fig, ax = plot_blocks(FF._root)
fig.savefig('ingersoll_blocks.png')
plt.close(fig)

# FF = FormFactorMatrix(shape_model)

dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])
E = shape_model.get_direct_irradiance(F0, dir_sun, eps=1e-6)

fig, ax = tripcolor_vector(V, F, E, cmap=cc.cm.gray)
fig.savefig('ingersoll_E.png')
plt.close(fig)
print('- wrote ingersoll_E.png to disk')

T = compute_steady_state_temp(FF, E, rho, emiss)
print('- computed T')

fig, ax = tripcolor_vector(V, F, T, cmap=cc.cm.fire)
fig.savefig('ingersoll_T.png')
plt.close(fig)
print('- wrote ingersoll_T.png to disk')

error = T[parts[1]] - hc.T_gt
vmax = abs(error).max()
vmin = -vmax
fig, ax = tripcolor_vector(
    V, F[parts[1]], error, vmin=vmin, vmax=vmax, cmap=cc.cm.fire)
fig.savefig('error.png')
plt.close(fig)
print('- wrote error.png to disk')

max_error = abs(error).max()
print('- max error: %1.2f K' % (max_error,))

rms_error = np.linalg.norm(error)/np.sqrt(error.size)
print('- RMS error: %1.2f K' % (rms_error,))
