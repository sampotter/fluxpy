import colorcet as cc
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import vtk

from numpy.linalg import norm
from pathlib import Path
from scipy.integrate import dblquad

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.view_factor import ff_narayanaswamy

experiment_path = Path('./stats/eps_1e-1/ingersoll_p8')

FF = CompressedFormFactorMatrix.from_file(experiment_path/'FF.bin')

shape_model = FF.shape_model

V, F, N, P, A = shape_model.V, shape_model.F, shape_model.N, shape_model.P, shape_model.A

I_sun = np.load(experiment_path/'I_sun.npy')
I_shadow = np.load(experiment_path/'I_shadow.npy')
I = np.concatenate([I_sun, I_shadow])

i = I[np.argmin(np.sqrt(np.sum(P[I, :2]**2, axis=1)))]

res = np.empty(I.size, dtype=float)
FF = np.empty(I.size, dtype=float)
FF_refine = np.empty(I.size, dtype=float)

def get_F(x, y, nx, ny):
    r = norm(x - y)
    theta_x = max(0, nx@(y - x)/r)
    theta_y = max(0, ny@(x - y)/r)
    return theta_x*theta_y/(np.pi*r**2)

def get_F_uv(u, v, x, x0, t1, t2, nx, ny):
    # dx = x - x0
    # fac = ny@dx/np.pi
    # a, b, c = nx@t1, nx@t2, -nx@dx
    # numer = a*u + b*v + c
    # A, B, C, D, E, F = dx@dx, -2*dx@t1, -2*dx@t2, t1@t1, 2*t1@t2, t2@t2
    # denom = (A + B*u + C*v + D*u**2 + E*u*v + F*v**2)**2
    # return fac*numer/denom
    y = x0 + u*t1 + v*t2
    return max(0, nx@(y - x))*max(0, ny@(x - y))/(np.pi*norm(y - x)**4)

def get_ff_midpoint(x, y, nx, ny, Ay):
    return get_F(x, y, nx, ny)*Ay

def get_ff_SAI(i, j):
    n, p = N[i], P[i]
    x = V[F[j]]
    tot = 0
    for k in range(3):
        k1 = (k + 1) % 3
        a, b = x[k] - p, x[k1] - p
        c = np.cross(a, b)
        e = np.linalg.norm(c)
        d = a@b
        tot += (n@c)*np.arctan(e/d)/e
    return abs(tot)/(2*np.pi)

def get_ff_dblquad(i, j):
    x = V[F[j]]
    x0, t1, t2 = x[0], x[1] - x[0], x[2] - x[0]
    _, error = dblquad(
        lambda u, v: get_F_uv(u, v, P[i], x0, t1, t2, N[i], N[j]),
        0, 1, # u range
        0, lambda u: 1 - u, # v range
        epsabs=1e-7,
        epsrel=1e-7)
    return 2*_*A[j], error

def plot_ff(i):
    x0, x1, x2 = V[F[i]]
    nbad = N[i]
    U = np.linspace(0, 1)
    V = np.linspace(0, 1)
    F_tri_grid = np.empty((S.size, T.size))
    F_tri_grid[:] = np.nan
    for (a, s), (b, t) in it.product(enumerate(S), enumerate(T)):
        if s + t <= 1:
            p = (1 - s - t)*x0 + s*x1 + t*x2
            F_tri_grid[a, b] = get_F(P[i], p, N[i], nbad)
            F_tri_grid[a, b] = get_F_uv(s, t, P[i], x0, x1 - x0, x2 - x0, N[i], nbad)
    plt.figure()
    plt.imshow(F_tri_grid, cmap=cc.cm.fire)
    plt.colorbar()
    plt.show()

# try to do something smarter with the bad triangles

# j0, j1, j2 = I[[k for k, f in enumerate(F[I]) if np.intersect1d(F[i], f).size == 2]]

# p = P[i]
# x = V[F[j1]]

# term = []
# tot = 0
# for k in range(3):
#     k1 = (k + 1) % 3
#     a, b = x[k] - p, x[k1] - p
#     c = np.cross(a, b)
#     e = np.linalg.norm(c)
#     d = a@b
#     term = (n@c)*np.arctan(e/d)/e
#     print(term)
#     tot += term
# tot = abs(tot)/(2*np.pi)

############################################################################
# select triangles for computation and plotting

# find triangles immediately adjacent to target triangle (index i)
j0, j1, j2 = I[[k for k, f in enumerate(F[I]) if np.intersect1d(F[i], f).size == 2]]
I_plot = np.setdiff1d(I, [i, j0, j1, j2])

############################################################################
# compute form factors using different methods

# midpoint rule

print('computing form factors using midpoint rule')
ff_midpoint = np.zeros(F.shape[0])
for j in I_plot:
    if j == i:
        continue
    ff_midpoint[j] = get_ff_midpoint(P[i], P[j], N[i], N[j], A[j])

# reference integration using scipy

print('computing form factors using scipy.integrate.dblquad')
ff_true, ff_errors_true = np.zeros(F.shape[0]), np.zeros(F.shape[0])
for j in I_plot:
    if j == i:
        continue
    ff, error = get_ff_dblquad(i, j)
    ff_true[j] = ff
    ff_errors_true[j] = error

# "single area integration"

print('computing form factors using SAI')
ff_SAI = np.zeros(F.shape[0])
for j in I_plot:
    if j == i:
        continue
    ff_SAI[j] = get_ff_SAI(i, j)

############################################################################
# plotting

grid = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: F[I_plot]}, V)
grid['values'] = (ff_true[I_plot] - ff_midpoint[I_plot])/(ff_midpoint[I_plot])
plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(grid, scalars='values', cmap=cc.cm.coolwarm,
                 # clim=(0, 1e-7),
                 # clim=(0, max(ff_midpoint)),
                 nan_color='magenta')
plotter.add_mesh(pv.Sphere(0.01, P[i]), color='cyan')
for j in [j0, j1, j2]:
    plotter.add_mesh(pv.Sphere(0.01, P[j]), color='pink')

assert False

# plot all the form factors on the triangle mesh

nx, ny = 1024, 1024

xmin, ymin = V.min(0)[:2]
xmax, ymax = V.max(0)[:2]

# xmin, xmax = -0.5, 0.5
# ymin, ymax = -0.5, 0.5

xgrid = np.linspace(xmin, xmax, nx)
ygrid = np.linspace(ymin, ymax, ny)

F_grid = np.empty((nx, ny))
F_grid[...] = np.nan

i_target = i
# i_target = np.random.choice(I)

d = np.array([0, 0, -1], dtype=np.float64)
for (a, x), (b, y) in it.product(enumerate(xgrid), enumerate(ygrid)):
    o = np.array([x, y, 1], dtype=np.float64)
    hit = shape_model.intersect1(o, d)
    if hit is not None:
        hit_index, hit_x = hit
        F_grid[a, b] = get_F(P[i_target], hit_x, N[i_target], N[hit_index])

plt.figure()
plt.imshow(F_grid, extent=[xmin, xmax, ymin, ymax], cmap=cc.cm.bmy)
plt.colorbar()
plt.show()

############################################################################
# misc

def print_verts_matlab(i):
    print('V = [', end='')
    for k, v in enumerate(V[F[i]]):
        print('[%0.16g, %0.16g, %0.16g]' % (v[0], v[1], v[2]), end='')
        if k < 2:
            print(';')
        else:
            print(']')
