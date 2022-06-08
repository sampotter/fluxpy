import numpy as np

from abc import ABC
from copy import copy

from flux.error import ASSERT

class SpatialTree(ABC):
    @property
    def num_points(self):
        return self._i1 - self._i0

    @property
    def i0(self):
        return self._i0

    @property
    def i1(self):
        return self._i1

    @property
    def root(self):
        return self._root

    @property
    def perm(self):
        return self._perm

    @property
    def mesh(self):
        return self._mesh

    @property
    def depth(self):
        return self._depth

    def get_max_depth(self):
        return max(_.depth for _ in self.get_nodes())

    @property
    def children(self):
        return self._children

    @property
    def is_root(self):
        return self._root is self

    @property
    def parent(self):
        return self._parent

    @property
    def is_leaf(self):
        return not self._children

    @property
    def depth(self):
        return self._depth

    @property
    def children(self):
        return self._children

    def get_nodes(self):
        yield self
        for child in self.children:
            yield from child.get_nodes()

    def get_levels(self):
        levels = dict()
        for node in self.get_nodes():
            if node.depth not in levels:
                levels[node.depth] = []
            levels[node.depth].append(node)
        return levels

    @property
    def rev_perm(self):
        rev_perm = np.empty_like(self.perm)
        for i, j in enumerate(self.perm):
            rev_perm[j] = i
        return rev_perm

    def show(self):
        if self.dim == 3:
            import pyvista as pv
            poly_data = pv.PolyData(self.points)
            poly_data['i'] = np.arange(self._i0, self._i1)
            plotter = pv.Plotter()
            plotter.add_mesh(poly_data)
            plotter.show()
        else:
            raise RuntimeError(f'show not implemented for dim == {self.dim}')

    def _get_dot_label(self):
        return f'[{self.i0}, {self.i1})'

    def _add_dot_nodes(self, dot, parent_label):
        for node in self.children:
            if node.num_points > 0:
                label = node._get_dot_label()
                dot.node(label)
                dot.edge(parent_label, label)
                node._add_dot_nodes(dot, label)

    def todot(self):
        import graphviz
        dot = graphviz.Digraph()
        label = self._get_dot_label()
        dot.node(label)
        self._add_dot_nodes(dot, label)
        return dot

    def copy_below(self):
        '''Create a shallow copy of the current quadtree node beneath this
        node. The purpose of this is to replicate the current node at
        the next level below. This assumes that this node is a leaf
        (otherwise the operation makes little sense). The copy is
        returned after being added to this node's children (at which
        point it becomes this node's only child).

        '''
        ASSERT(self.is_leaf)
        node = copy(self)
        node._depth += 1
        node._parent = self
        node._is_copy = True
        node._children = [] # because we're making a shallow copy!
        self._children.append(node)
        return node
