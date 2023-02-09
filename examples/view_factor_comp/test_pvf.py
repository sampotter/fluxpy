# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:07:54 2023

@author: heath
"""

# Testing PyViewFactor integration
from make_ffm_pyviewfactor import make_ffm_pvf
from flux.form_factors import get_form_factor_matrix
from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from make_ffm_midpoint import obj_to_shapemodel
import matplotlib as plt
import scipy.sparse
import numpy as np

# Loading the OBJ File
outdir = '.'
filename_mesh = 'hemispherical_mesh_4.obj'
shape_model = obj_to_shapemodel(filename_mesh, outdir)

FF_midpoint1 = scipy.sparse.load_npz('FF_4.npz')

def test_shape_model(shape_model):
    V = shape_model.V
    F = shape_model.F
    N = shape_model.N
    up_norms = []
    for i in range(len(F)):
        norm = N[i]
        print(i)
        print(norm)
        if norm[2] < 0:
            up_norms.append(i)
    return up_norms

print(test_shape_model(shape_model))

# Plot #2 - PVF Form Factor Plot
verts = shape_model.V[:, :2]
faces = shape_model.F
plt.tripcolor(*verts.T, FF_midpoint1.toarray()[177], triangles=faces)
plt.colorbar()
plt.scatter(*shape_model.P[256, :2], s=30, color='red', zorder=2)
plt.scatter(*shape_model.P[244, :2], s=30, color='green', zorder=2)
plt.scatter(*shape_model.P[254, :2], s=30, color='orange', zorder=2)
plt.scatter(*shape_model.P[177, :2], s=30, color='blue', zorder=2)
plt.scatter(*shape_model.P[210, :2], s=30, color='yellow', zorder=2)
#plt.scatter(*shape_model.P[255, :2], s=30, color='red', zorder=2)
plt.show()
            