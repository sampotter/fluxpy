import cmocean
import matplotlib.pyplot as plt
import meshpy.triangle as triangle
import numpy as np
import scipy.integrate
import scipy.optimize
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.form_factors import FormFactorMatrix
from flux.model import get_T
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import TrimeshShapeModel, get_surface_normals

from scipy.interpolate import interp1d

# Parameters related to the spatial discretization (triangle mesh)
p = 5
h = (2/3)**p
min_area = (2/3)*h**2

# Physical parameters related to the problem
e0 = np.deg2rad(15)
FF_tol = 1e-7
F0 = 1000
rho = 0.3
emiss = 0.99

# Geometric parameters controlling the shape of the hemispherical crater
beta = np.deg2rad(40)
rc = 0.8
H = rc/np.tan(beta)
r = rc/np.sin(beta)

################################################################################
# Build the triangulation

# We want our triangle mesh to conform to the shadow line. This takes
# a little doing.

# First, define a function that computes the x coordinate of a point
# on the shadow line for a given y coordinate. There's probably a
# smarter and more robust way to do this...
def x_silhouette(y):
    if abs(y) > rc:
        return None
    x0 = np.sqrt(rc**2 - y**2)
    def f(x):
        z = H - np.sqrt(r**2 - x**2 - y**2)
        dx = x0 - x
        return z + np.tan(e0)*dx
    X = np.linspace(-x0, x0, 21)
    F = f(X)
    I = np.argsort(abs(F))[:2]
    i = np.argsort(abs(X[I] - x0))[1]
    xm = X[I[i]]
    dxm = min(abs(xm - x0), abs(xm + x0))/2
    if np.sign(f(xm - dxm)) == np.sign(f(xm + dxm)):
        return None
    return scipy.optimize.brentq(f, xm - dxm, xm + dxm)

# First, find the location of the points where the shadow line and the
# rim of the crater intersects, (xp, yp) and (xp, -yp).
t0 = np.tan(e0)
yp = np.sqrt((t0**2 + 1)*rc**2 - t0**2*r**2)
xp = np.sqrt(rc**2 - yp**2)

# Next, discretize the shadow line using by uniformly sampling the y
# coordinate. This will be nonuniform with respect to the arc length
# of the shadow line, which we will fix.
Yp = np.linspace(-yp, yp, int(np.ceil(2*yp/h)))
Xp = np.empty_like(Yp)
Xp[0] = xp
Xp[-1] = xp
Xp[1:-1] = np.array([x_silhouette(y) for y in Yp[1:-1]])

while True:
    # The method I'm using right now for finding the x coordinates of
    # the silhouette doesn't work very well near the crater rim, and
    # will set values of Xp to NaN when it fails. If this happens,
    # just compute these values by interpolation.
    isnan = np.isnan(Xp)
    if isnan.any():
        Xp[isnan] = interp1d(Yp[~isnan], Xp[~isnan], kind='cubic')(Yp[isnan])

    # Integrate along the parameter of the piecewise linear curve
    # discretizing the shadow line and find the number of points that we
    # would need to use to discretize it with roughly O(h) fineness.
    dL = np.sqrt((Xp[1:] - Xp[:-1])**2 + (Yp[1:] - Yp[:-1])**2)
    dL = np.concatenate([[0], dL])
    L = np.cumsum(dL)
    n = int(np.ceil(L[-1]/h))

    param_error = dL[1:].max() - dL[1:].mean()
    if param_error < np.finfo(np.float32).resolution:
        break

    # Reparametrize the curve uniformly. The resulting segments will not
    # all have exactly the same length, but they should be close enough
    # for our purposes.
    Yl = np.empty((n,), dtype=np.float64)
    Xl = np.empty_like(Yl)
    j = 1
    for i, l in enumerate(np.linspace(0, L[-1], n)):
        while l < L[j - 1] or L[j] < l:
            j += 1
        s = (l - L[j - 1])/dL[j]
        Yl[i] = (1 - s)*Yp[j - 1] + s*Yp[j]
        Xl[i] = x_silhouette(Yl[i])
    Xl[0] = xp
    Xl[-1] = xp

    Xp = Xl.copy()
    Yp = Yl.copy()

# Next, we need to discretize the two parts of the crater. We do this
# by finding the angle to (xp, yp), and sampling the arc on either
# side of (xp, +/- yp).

phi = np.arctan(yp/xp)

theta = np.linspace(-phi, phi, int(np.ceil(2*rc*phi/h)))
Xcirc1 = rc*np.cos(theta)
Ycirc1 = rc*np.sin(theta)

theta = np.linspace(phi, 2*np.pi - phi, int(np.ceil(2*rc*(np.pi - phi)/h)))
Xcirc2 = rc*np.cos(theta)
Ycirc2 = rc*np.sin(theta)

def should_refine(verts, area):
    return area > min_area

# Make a plot of the meshing procedure.

fig = plt.figure(figsize=(8.4, 4.9))

# First, mesh the upper horizontal plane. Start by adding a square
# boundary.
points = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.float32)
facets = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], np.intc)

ax = fig.add_subplot(2, 3, 1)
ax.plot(*np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
        linewidth=1, c='k', zorder=1)
ax.scatter(*points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
ax.plot(
    np.concatenate([Xcirc1, Xcirc2[1:]]),
    np.concatenate([Ycirc1, Ycirc2[1:]]),
    linewidth=1, c='k', zorder=1)
ax.scatter(
    np.concatenate([Xcirc1, Xcirc2[1:-1]]),
    np.concatenate([Ycirc1, Ycirc2[1:-1]]),
    s=15, edgecolors='k', facecolors='r', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

points = np.concatenate([
    points,
    np.array([Xcirc1, Ycirc1]).T,
    np.array([Xcirc2[1:-1], Ycirc2[1:-1]]).T,
], axis=0)
ncirc = Xcirc1.size + Xcirc2.size - 2
facets = np.concatenate([
    facets,
    facets.shape[0] + np.array([
        np.arange(ncirc),
        np.mod(np.arange(ncirc) + 1, ncirc)
    ]).T
], axis=0)

info = triangle.MeshInfo()
info.set_points(points)
info.set_holes([(0, 0)])
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=should_refine)
xy = np.array(mesh.points)
F = np.array(mesh.elements)

ax = fig.add_subplot(2, 3, 4)
ax.triplot(*xy.T, triangles=F, linewidth=1, c='k', zorder=1)
ax.scatter(*xy.T, s=3, c='k', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

# Next, mesh the shadowed part of the crater
points = np.concatenate([
    np.array([Xp[::-1], Yp[::-1]]).T,
    np.array([Xcirc1[1:-1], Ycirc1[1:-1]]).T
], axis=0)
npts = Xp.size + Xcirc1.size - 2
facets = np.array([
    np.arange(npts),
    np.mod(np.arange(npts) + 1, npts)
]).T

ax = fig.add_subplot(2, 3, 2)
ax.plot(*np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
        linewidth=1, c='k', zorder=1)
ax.scatter(*points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

info = triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=should_refine)
F = np.concatenate([F, xy.shape[0] + np.array(mesh.elements)], axis=0)
xy = np.concatenate([xy, np.array(mesh.points)], axis=0)

ax = fig.add_subplot(2, 3, 5)
ax.triplot(*np.array(mesh.points).T,
           triangles=np.array(mesh.elements), linewidth=1, c='k',
           zorder=1)
ax.scatter(*np.array(mesh.points).T, s=3, c='k', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

# Now, mesh the other side
points = np.concatenate([
    np.array([Xp, Yp]).T,
    np.array([Xcirc2[1:-1], Ycirc2[1:-1]]).T
], axis=0)
npts = Xp.size + Xcirc2.size - 2
facets = np.array([
    np.arange(npts),
    np.mod(np.arange(npts) + 1, npts)
]).T

ax = fig.add_subplot(2, 3, 3)
ax.plot(*np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
        linewidth=1, c='k', zorder=1)
ax.scatter(*points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

info = triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=should_refine)
F = np.concatenate([F, xy.shape[0] + np.array(mesh.elements)], axis=0)
xy = np.concatenate([xy, np.array(mesh.points)], axis=0)

mesh = triangle.build(info, refinement_func=should_refine)
F = np.concatenate([F, xy.shape[0] + np.array(mesh.elements)], axis=0)
xy = np.concatenate([xy, np.array(mesh.points)], axis=0)

ax = fig.add_subplot(2, 3, 6)
ax.triplot(*np.array(mesh.points).T,
           triangles=np.array(mesh.elements), linewidth=1, c='k',
           zorder=1)
ax.scatter(*np.array(mesh.points).T, s=3, c='k', zorder=2)
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

fig.savefig('ingersoll-meshing.pdf')
plt.close(fig)

# Compute the z component of the crater vertices
Rc = np.sqrt(np.sum(xy**2, axis=1))
I = Rc <= rc + np.finfo(np.float32).eps
Z = -(np.sqrt(r**2 - Rc**2) - H)[I]
assert Z.max() < np.finfo(np.float32).eps
Z = np.minimum(0, Z)

# Assemble the vertex array
V = np.empty((xy.shape[0], 3), dtype=np.float32)
V[:, :2] = xy
V[:, 2] = 0
V[I, 2] = Z

print('- created a Delaunay mesh with %d points and %d faces' % (
    V.shape[0], F.shape[0]))

plt.figure(figsize=(11, 10))
plt.triplot(*V[:, :2].T, triangles=F, linewidth=1, color='k', zorder=1)
plt.scatter(*V[I, :2].T, facecolors='red', edgecolors='k', s=30,
            linewidth=1, zorder=2)
plt.gca().set_aspect('equal')
plt.savefig('tris.pdf')
plt.close()
print('- wrote triplot to tris.pdf')

trimesh.Trimesh(V, F).export('ingersoll.obj')
print('- saved Wavefront OBJ file to ingersoll.obj')

N = get_surface_normals(V, F)
N[N[:, 2] < 0] *= -1

shape_model = TrimeshShapeModel(V, F, N)

FF = CompressedFormFactorMatrix.assemble_using_quadtree(shape_model, tol=FF_tol)
print('- assembled compressed form factor matrix (tol = %g, %1.1f Mb)' % (
    FF_tol, FF.nbytes/1024**2))

FF.save('ingersoll.bin')
print('- saved compressed form factor matrix to disk')

fig, ax = plot_blocks(FF._root)
fig.savefig('ingersoll_blocks.png')
plt.close(fig)

# FF = FormFactorMatrix(shape_model)

dir_sun = np.array([np.cos(e0), 0, np.sin(e0)])
E = shape_model.get_direct_irradiance(F0, dir_sun)

fig, ax = tripcolor_vector(V, F, E, cmap=cmocean.cm.gray)
fig.savefig('ingersoll_E.png')
plt.close(fig)
print('- wrote ingersoll_E.png to disk')

T, nmul = get_T(FF, E, rho, emiss)
print('- computed T (%d matrix multiplications)' % (nmul,))

fig, ax = tripcolor_vector(V, F, T, cmap=cmocean.cm.solar)
# ax.plot(Xp, Yp, linewidth=1, c='cyan', marker='.', zorder=1)
# ax.plot(Xl, Yl, linewidth=1, c='orange', marker='.', linestyle='--', zorder=2)
# ax.plot(np.sqrt(t0**2*(r**2 - Ys**2)/(t0**2 + 1)), Ys, zorder=3)
# ax.scatter([xp, xp], [yp, -yp], s=30, facecolor='red', edgecolors='white', zorder=4)
fig.savefig('ingersoll_T.png')
plt.close(fig)
print('- wrote ingersoll_T.png to disk')
