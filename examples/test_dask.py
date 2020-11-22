#!/usr/bin/env python

import dask.array as da
import dask.delayed
import itertools as it
import numpy as np

from dask.distributed import Client
from flux.form_factors import get_form_factor_block
from flux.ingersoll import HemisphericalCrater
from flux.shape import TrimeshShapeModel, get_surface_normals

if __name__ == '__main__':
    client = Client()

    p = 10
    h = (2/3)**p
    e0 = np.deg2rad(15)
    beta = np.deg2rad(20)
    F0 = 1000
    rc = 0.9
    rho = 0.3
    emiss = 0.99
    hc = HemisphericalCrater(beta, rc, e0, F0, rho, emiss)

    V, F, parts = hc.make_trimesh(h, return_parts=True)
    N = get_surface_normals(V, F)
    N[N[:, 2] < 0] *= -1
    shape_model = TrimeshShapeModel(V, F, N)
    num_faces = shape_model.num_faces

    eps = 1e-7

    bsize = 512
    bshape = (num_faces//bsize, num_faces//bsize)

    @dask.delayed
    def block(i, j):
        i0, i1 = bsize*i, min(num_faces, bsize*(i + 1))
        j0, j1 = bsize*j, min(num_faces, bsize*(j + 1))
        I = np.arange(i0, i1, dtype=np.intp)
        J = np.arange(j0, j1, dtype=np.intp)
        spmat = get_form_factor_block(shape_model, I, J, eps)
        return spmat

    print(block(50, 50))

    # TODO: we need to implement pickling in python-embree to get
    # the next line to run
    block(50, 50).compute()

    # TODO: try to call block for each pair (i, j) and see if we can
    # create a task graph that will assemble the full form factor
    # matrix...
