#!/usr/bin/env python

import colorcet as cc
import matplotlib
import numpy as np
import sys

from flux.plot import imray

from util import shape_model_from_obj_file

shape_model = shape_model_from_obj_file(sys.argv[1])
p = np.load(sys.argv[2])

xmin, ymin, zmin = shape_model.V.min(0)
xmax, ymax, zmax = shape_model.V.max(0)
xc, yc, zc = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

pos = np.array([xc, yc, zc - dx])
look = np.array([0, 0, 1], dtype=pos.dtype)
up = np.array([0, 1, 0], dtype=pos.dtype)
shape = (512, 512)
h = 1.1*max(dx, dy)/max(shape)

im = imray(shape_model, p, pos, look, up, shape, h=h)

im[np.isnan(im)] = 0

with open(sys.argv[3], 'r') as f:
    p_max = float(f.readline())

matplotlib.image.imsave(sys.argv[4], im, vmin=0, vmax=p_max, cmap=cc.cm.gouldian)
