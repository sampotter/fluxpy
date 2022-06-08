import itertools as it
import numpy as np

from flux.spatial_tree import SpatialTree
from flux.error import ASSERT

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

class Quadtree(SpatialTree):
    def __init__(self, points=None, root=None, parent=None, bbox=None, i0=None,
                 i1=None, K=2, depth=0, copy_points=True, max_leaf_size=16):
        if points is None and root is None:
            raise ValueError('must pass points or root to build quadtree')

        if points is not None and not isinstance(points, np.ndarray):
            if copy_points:
                points = np.array(points).copy()
            else:
                points = np.array(points)

        self._i0 = 0 if i0 is None else i0
        self._i1 = points.shape[0] if i1 is None else i1
        self._parent = parent
        self._points = points
        self._root = self if root is None else root
        self._bbox = self._get_bbox() if bbox is None else bbox
        self._K = K
        self._depth = depth
        self._max_leaf_size = max_leaf_size
        self._is_copy = False

        self._perm = np.arange(self.num_points) if self.is_root else None

        # Verify that all of the current node's points are contained
        # in this node's bounding box
        ASSERT((self._bbox[0] <= self.points).all())
        ASSERT((self.points <= self._bbox[1]).all())

        # Build children in breadth-first order
        self._children = []
        if self.num_points > self._max_leaf_size:
            # Partially sort current node's points so that they're in
            # the current quadtree order for the current level only
            points_ = np.empty_like(self.points)
            points_[:] = np.nan

            perm_ = np.empty_like(self.perm, dtype=int)
            perm_[:] = -1

            # The `j` indices are relative to the current node's
            # points, while the `i` indices are are relative to the
            # root node's indices
            Bbox, J0, J1 = [], [0], []
            for inds, bbox in self._get_inds():
                j0 = J0[-1]
                j1 = j0 + inds.size
                points_[j0:j1] = self.points[inds]
                perm_[j0:j1] = self.perm[inds]
                Bbox.append(bbox)
                J0.append(j1)
                J1.append(j1)
            del J0[-1]

            # Make sure we didn't accidentally skip any nodes...
            ASSERT(np.isfinite(points_).all())

            # ... before permuting them. This permutes the points
            # stored by the root of the tree.
            self.points[:] = points_
            self.perm[:] = perm_

            # Construct each quadtree child
            for bbox, j0, j1 in zip(Bbox, J0, J1):
                # If there are no points, keep going---we don't want
                # to add empty nodes to the quadtree
                if j0 == j1:
                    continue

                # Continue building the quadtree depth-first
                child = Quadtree(root=self._root, parent=self, bbox=bbox,
                                 i0=self._i0 + j0, i1=self._i0 + j1,
                                 K=K, depth=depth + 1, copy_points=copy_points,
                                 max_leaf_size=self._max_leaf_size)
                self._children.append(child)

    def __repr__(self):
        quadtree_name = {
            1: f'{self._K}-ary tree node',
            2: 'quadtree node',
            3: 'octree node'
        }[self.dim]
        return f'<{quadtree_name} in R{self.dim} with {self.num_points} points>'

    @property
    def points(self):
        return self._root._points[self._i0:self._i1]

    @property
    def perm(self):
        return self._root._perm[self._i0:self._i1]

    @property
    def dtype(self):
        return self.points.dtype

    @property
    def dim(self):
        return 1 if self.points.ndim == 1 else self.points.shape[1]

    @property
    def inds(self):
        return np.arange(self.i0, self.i1)

    @property
    def is_empty(self):
        return self.num_points == 0

    @property
    def is_copy(self):
        return self._is_copy

    def _get_bbox(self, force_cube=True):
        a = self.points.min(0)
        b = self.points.max(0)

        # add a small margin for safety
        eps = np.finfo(self.dtype).resolution*np.maximum(abs(a), abs(b))
        a -= eps
        b += eps

        if force_cube:
            c = (a + b)/2
            d = b - a
            h = abs(d)
            hmax = np.max(h)
            a = c - (hmax/(2*h))*d
            b = c + (hmax/(2*h))*d
        return a, b

    def _get_inds(self):
        a, b = self._bbox
        d = b - a
        K = self._K
        for k in it.product(range(K), repeat=self.dim):
            k = np.array(k, dtype=int)
            margin = (k + 1 == K).astype(self.dtype)
            bbox = (a + k*d/K, a + (k + 1)*d/K)
            mask = (bbox[0] <= self.points) & (self.points < bbox[1] + margin)
            if self.dim > 1:
                mask = mask.all(1)
            inds = np.where(mask)[0]
            yield inds, bbox
