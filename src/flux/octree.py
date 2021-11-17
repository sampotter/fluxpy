import itertools as it
import numpy as np


def get_octant_order(X, bbox=None):
    if bbox is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    else:
        xmin, ymin, zmin = np.min(X, axis=0)
        xmax, ymax, zmax = np.max(X, axis=0)
    xc, yc, zc = (xmin + xmax)/2, (ymin + ymax)/2, (zmin + zmax)/2
    x, y, z = X.T
    Is = []
    for xop, yop, zop in it.product([np.less_equal, np.greater], repeat=3):
        B = np.column_stack([xop(x, xc), yop(y, yc), zop(z, zc)])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
    return Is
