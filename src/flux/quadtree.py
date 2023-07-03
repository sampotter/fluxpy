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

def get_quadrant_order_roi(X, roi_c, roi_r):
    
    roi_idx = (np.power(X - roi_c, 2).sum(axis=1) < np.power(roi_r, 2))
    
    X_roi = X[roi_idx]
    xmin_roi, ymin_roi = np.min(X_roi, axis=0)
    xmax_roi, ymax_roi = np.max(X_roi, axis=0)
    xc_roi, yc_roi = (xmin_roi + xmax_roi)/2, (ymin_roi + ymax_roi)/2
    
    X_non_roi = X[~roi_idx]
    xmin_non_roi, ymin_non_roi = np.min(X_non_roi, axis=0)
    xmax_non_roi, ymax_non_roi = np.max(X_non_roi, axis=0)
    xc_non_roi, yc_non_roi = (xmin_non_roi + xmax_non_roi)/2, (ymin_non_roi + ymax_non_roi)/2
        
    x, y = X.T
    Is = []
    for xop, yop in it.product([np.less_equal, np.greater], repeat=2):
        B = np.column_stack([xop(x, xc_roi), yop(y, yc_roi), roi_idx])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
        
        B = np.column_stack([xop(x, xc_roi), yop(y, yc_roi), ~roi_idx])
        I = np.where(np.all(B, axis=1))[0]
        Is.append(I)
    return Is