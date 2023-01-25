# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:52:33 2023

@author: heath
"""
# This file will create a Hemishperical Crater Mesh, create a shape model, 
# and calculate the form factor matrix using PyViewFactor

import os.path
import numpy as np
import scipy.sparse
import array

from flux.util import tic, toc
from make_ffm_midpoint import obj_to_shapemodel

import pyvista as pv
import pyviewfactor as pvf

def make_ffm_pvf(shape_model):
    """ Function that takes in a shape_model and calculates
    the Form Factor Matrix using the PyViewFactor python
    module.
    
    Parameters
    ----------
    shape_model : CgalTrimeshShapeModel
    """
    # Extracting the array of verticies and surfaces
    V = shape_model.V 
    F = shape_model.F
                      
    if shape_model.dtype == np.float32:
        typecode = 'f'
    elif shape_model.dtype == np.float64:
        typecode = 'd'
    else:
        raise RuntimeError(f'unsupported dtype {shape_model.dtype}')

    data = array.array(typecode)
    indices = array.array('Q')
    ind_ptr = array.array('Q')
    
    num_faces = len(F)
    indptr = 0
    tic()
    for i in range(num_faces):
        ind_ptr.append(indptr)
        c_face = V[F][i]
        current_triangle = pv.Triangle([c_face[0], c_face[1], c_face[2]])
        for j in range(num_faces):
            o_face = V[F][j]
            other_triangle = pv.Triangle([o_face[0], o_face[1], o_face[2]])
            if pvf.get_visibility(current_triangle, other_triangle):
                vf = pvf.compute_viewfactor(current_triangle, other_triangle)
                data.append(vf)
                indices.append(j)
                indptr +=1
    t_FF = toc()
    ind_ptr.append(indptr)
    FF = scipy.sparse.csr_matrix((data, indices, ind_ptr), shape = (num_faces, num_faces))
    return FF, t_FF
            
if __name__ == '__main__':
    #if there is not a directory to write output files too, make a directory
    outdir = '.'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Loading the OBJ File
    filename_mesh = 'hemispherical_mesh_5.obj'
    shape_model = obj_to_shapemodel(filename_mesh, outdir)
    
    # Compute Form Factor Matrix
    FF, t_FF = make_ffm_pvf(shape_model)

    print('- finished assembly [%1.2f s]' %(t_FF))
    
    # When using compressed matrices, tolarance will come into play
    # Ensure to readd the args section from collect_data.py
    tol = None
    filename_FF = 'FF_pvf_5.npz'
    if tol is None:
        scipy.sparse.save_npz(os.path.join(outdir, filename_FF), FF)
        print('- wrote ' + filename_FF)
    else:
        FF.save(os.path.join(outdir, 'FF_pvf.bin'))
        print('- wrote FF_pvf.bin')
    
    
    
    