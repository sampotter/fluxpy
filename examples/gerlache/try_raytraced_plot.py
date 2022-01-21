import colorcet as cc
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.compressed_form_factors import CompressedKernelMatrix
from flux.model import compute_steady_state_temp
from flux.solve import solve_radiosity
from flux.util import tic, toc

FF = CompressedFormFactorMatrix.from_file('FF.bin')
shape_model = FF.shape_model
V, F = shape_model.V, shape_model.F
dtype = V.dtype

# FF_gt = scipy.sparse.load_npz('FF_gt.npz')

az, el = 45, -10
phi, theta = np.deg2rad(az), np.deg2rad(el)

F0 = 1365
E = shape_model.get_direct_irradiance(
    F0,
    np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)])
)

rho, emiss, method = 0.11, 0.999, 'jacobi'

tic()
T = compute_steady_state_temp(FF, E, rho, emiss)
print(toc())

tic()
B = solve_radiosity(FF, E, rho, method=method, albedo_placement='left')[0]
print(toc())

# T_gt = compute_steady_state_temp(FF_gt, E, rho, emiss)
# B_gt = solve_radiosity(FF_gt, E, rho, method=method='left')[0]

# KK = CompressedKernelMatrix(FF, 0.11, albedo_placement='left')
# B = scipy.sparse.linalg.cg(KK, E)[0]

x0, y0, z0 = V.min(0)
x1, y1, z1 = V.max(0)
xc, yc = (x0 + x1)/2, (y0 + y1)/2
dx, dy = x1 - x0, y1 - y0
p = 0.3

# make plot using raytracing

m, n = 1024, 1024

xg = np.linspace(xc - p*dx, xc + p*dx, m).astype(dtype)
yg = np.linspace(yc - p*dx, yc + p*dx, n).astype(dtype)
z = z0 - max(abs(dx), abs(dy))
d = np.array([0, 0, 1], dtype=dtype)

value = np.empty((m, n), dtype=dtype)
value[...] = np.nan

for i, j in it.product(range(m), range(n)):
    x = np.array([xg[i], yg[j], z], dtype=dtype)
    hit = shape_model.intersect1(x, d)
    if hit is not None:
        # value[i, j] = B[hit[0]] - E[hit[0]]
        value[i, j] = T[hit[0]]
        # value[i, j] = (B[hit[0]] - B_gt[hit[0]])/B.max()

plt.figure()
plt.imshow(value_, interpolation='none', cmap=cc.cm.bmw)
plt.colorbar()
plt.show()
