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

from PIL import Image

FF = CompressedFormFactorMatrix.from_file('FF.bin')
shape_model = FF.shape_model
V, F = shape_model.V, shape_model.F
dtype = V.dtype

F0 = 1365

frames_per_second = 30
animation_time = 30 # s
num_frames = frames_per_second*animation_time

Az = np.linspace(0, 360, num_frames, endpoint=False)
El = -10*np.ones_like(Az)

Phi, Theta = np.deg2rad(Az), np.deg2rad(El)

# D[i] = ith sun direction (num_frames x 3)
D = np.array([
    (np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta))
    for phi, theta in zip(Phi, Theta)
])

F0, rho, emiss = 1365, 0.11, 0.999

for frame_index, d in enumerate(D):
    print(frame_index)

    tic()
    E = shape_model.get_direct_irradiance(F0, d)
    print(toc())

    tic()
    T = compute_steady_state_temp(FF, E, rho, emiss)
    print(toc())

    # plotting

    tic()

    x0, y0, z0 = V.min(0)
    x1, y1, z1 = V.max(0)
    xc, yc = (x0 + x1)/2, (y0 + y1)/2
    dx, dy = x1 - x0, y1 - y0
    p = 0.25

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
            value[i, j] = T[hit[0]]

    print(toc())

    # save plot to disk

    tic()

    plt.imsave('frame/frame%05d.png' % (frame_index + 1), value, cmap=cc.cm.bmy)

    print(toc())
