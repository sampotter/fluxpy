# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:29:13 2023

@author: heath
"""

# This file will convert an OBJ file to a CgalTrimeshShapModel
# and calculate the form factor matrix using the midpoint approximation

import os.path
import trimesh
import scipy.sparse

from flux.shape import CgalTrimeshShapeModel, get_surface_normals
from flux.util import tic, toc
from flux.form_factors import get_form_factor_matrix
from flux.compressed_form_factors import FormFactorQuadtreeBlock
from flux.compressed_form_factors import CompressedFormFactorMatrix


# This function will likely need to be used when working with compressed matrices
# When used, ensure to add the args section of collect_data.py
def assemble_compression(args, shape_model, parts=None):
    #if there is no tolorance given
    if args.tol is None:
        tic() #starting a timer
        FF = get_form_factor_matrix(shape_model) #form factor matrix is calculated given shape model
        t_FF = toc() #calculated time it takes to make form factor matrix
        FF_nbytes = FF.data.nbytes + FF.indptr.nbytes + FF.indices.nbytes 
    else:
        tic()
        FF = CompressedFormFactorMatrix(
            shape_model, tol=args.tol, RootBlock=FormFactorQuadtreeBlock)
        t_FF = toc()
        FF_nbytes = FF.nbytes
    return FF, t_FF, FF_nbytes #returns matrix, time of calculated, bytes needed

def assemble(shape_model, parts=None):
    tic() #starting a timer
    FF = get_form_factor_matrix(shape_model) #form factor matrix is calculated given shape model
    t_FF = toc() #calculated time it takes to make form factor matrix
    FF_nbytes = FF.data.nbytes + FF.indptr.nbytes + FF.indices.nbytes 
    return FF, t_FF, FF_nbytes #returns matrix, time of calculated, bytes needed

def obj_to_shapemodel(filename, outdir):
    """ Function takes in a filename (string) and outputs
    a shape_model (CgalTrimeshShapeModel)
    """
    #Import OBJ file
    mesh = trimesh.exchange.load.load(os.path.join(outdir, filename))
    # Extract the array of vertices and faces
    V_t = mesh.vertices
    F_t = mesh.faces
    # Compute the normals
    N_t = get_surface_normals(V_t, F_t)
    # Flip an upside down triangles
    N_t[N_t[:, 2] < 0] *= -1
    # Create the shape model
    shape_model_t = CgalTrimeshShapeModel(V_t, F_t, N_t)
    return shape_model_t

if __name__ == '__main__':
    #if there is not a directory to write output files too, make a directory
    outdir = '.'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Loading the OBJ File
    filename = 'hemispherical_mesh_5.obj'
    shape_model = obj_to_shapemodel(filename, outdir)
    
    # Compute Form Factor Matrix
    FF, t_FF, FF_nbytes = assemble(shape_model)

    print('- finished assembly (%1.1f Mb) [%1.2f s]' %
          (FF_nbytes/1024**2, t_FF))
    
    # When using compressed matrices, tolarance will come into play
    # Ensure to readd the args section from collect_data.py
    tol = None
    filename = 'FF_midpoint_5.npz'
    if tol is None:
        scipy.sparse.save_npz(os.path.join(outdir, filename), FF)
        print('- wrote ' + filename)
    else:
        FF.save(os.path.join(outdir, 'FF_midpoint.bin'))
        print('- wrote FF_midpoint.bin')
