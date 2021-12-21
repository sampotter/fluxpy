import itertools as it
import numpy as np


def get_quadrant_order(X, bbox=None):
    if bbox is not None:
        (xmin, xmax), (ymin, ymax) = bbox
    else:
        xmin, ymin = np.min(X, axis=0)
        xmax, ymax = np.max(X, axis=0)
    xc, yc = (xmin + xmax)/2, (ymin + ymax)/2
    x, y = X.T
    Is = []
    for xop, yop in it.product([np.less_equal, np.greater], repeat=2):
        B = np.column_stack([xop(x, xc), yop(y, yc)])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
    return Is
