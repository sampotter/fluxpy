# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:23:14 2023

@author: heath
"""

# This file will compare the Form Factor Matrices that are outputted by PyViewFactor
# and by the midpoint approximation

import numpy as np
import scipy.sparse
from make_ffm_midpoint import obj_to_shapemodel
import matplotlib.pyplot as plt
    
def make_cent_dist(shape_model):
    """ Function that calculated the distances between the centiods of 
    all faces in a given mesh.
    
    Parameters
    ----------
    shape_model : CgalTrimeshShapeModel
    """
    P = shape_model.P
    cent_dist = []
    num_faces = len(P)
    for i in range(num_faces):
        row = []
        for j in range(num_faces):
            dist = np.linalg.norm(P[i] - P[j])
            row.append(round(dist,2))
        cent_dist.append(row)
    return cent_dist

def make_error_mat(FF_exp, FF_analy):
    """ Function that, given the an experimental Form Factor Matrix
    and an analytical Form Factor Matrix, will compute the error between
    each form factor and organize it in a matrix
    
    Parameters
    ---------
    FF_exp : scipy.sparse.csr_matric
            The experimental form factor matrix that needs to be tested.
            
    FF_analy : scipy.sparse.csr_matrix
            The analytical form factor matrix to compare.
    """
    error_mat = []
    #error_abs = []
    num_faces = len(FF_exp.toarray()[0])
    for i in range(num_faces):
        row = []
        #row_a = []
        for j in range(num_faces):
            exp = FF_exp.toarray()[i][j]
            analy = FF_analy.toarray()[i][j]
            if analy == 0 and exp == 0:
                error = 0
                #abs_error = 0
            elif analy == 0:
                error == 100
                #abs_error = .1
            else:
                error = abs((exp-analy)/analy)
                #abs_error = abs(exp-analy)
            row.append(error)
            #row_a.append(abs_error)
        error_mat.append(row)
        #error_abs.append(row_a)
    return error_mat

if __name__ == '__main__':  
    # Loading the FF_midpoint and FF_PyViewFactor
    FF_midpoint = scipy.sparse.load_npz('FF_midpoint_5.npz')
    FF_pvf = scipy.sparse.load_npz('FF_pvf_5.npz')
    
    # Computing Error Matrix
    error_mat = make_error_mat(FF_midpoint, FF_pvf)
    #abs_error = make_error_mat(FF_midpoint, FF_pvf)[1]
    #print(max(abs_error[400]))

    # Computing the Centriod Distance Matrix
    shape_model = obj_to_shapemodel('hemispherical_mesh_5.obj', '.')
    cent_dist = make_cent_dist(shape_model)
    
    # # Plot #1
    # plt.plot(cent_dist[200], error_mat[200], 'bo')
    # plt.title('Midpoint Approximation Error vs. Surface Centriod Distance')
    # plt.xlabel('Centriod Distance')
    # plt.ylabel('Error (%)')
    # plt.show()
    
    # Finding the maximum view factor from the center Triangle (N=)
    print('Max View Factor to Triangle 500 - PVF:', max(FF_pvf.toarray()[500]))
    print('Max View Factor to Triangle 500 - Midpoint:', max(FF_midpoint.toarray()[500]))
    print('Location of Max View Factor - PVF', np.argmax(FF_pvf.toarray()[500]))
    print('Location of Max View Factor - Midpoint', np.argmax(FF_midpoint.toarray()[500]))
    print('PVF Value at Midpoint Max', FF_pvf.toarray()[500][309])
    
    # Plot #2
    verts = shape_model.V[:, :2]
    faces = shape_model.F
    plt.tripcolor(*verts.T, error_mat[500], triangles=faces)
    plt.scatter(*shape_model.P[500, :2], s=30, color='red', zorder=2)
    plt.scatter(*shape_model.P[491, :2], s=30, color='green', zorder=2)
    plt.scatter(*shape_model.P[309, :2], s=30, color='orange', zorder=2)
    plt.colorbar()
    plt.show()
    
    # # Plot #3
    # plt.plot(FF_pvf.toarray()[41], error_mat[41], 'bo')
    # plt.title('Midpoint Approximation Error vs. Magnitude of View Factor')
    # plt.xlabel('View Factor Magnitude')
    # plt.ylabel('Error (%)')
    # plt.show()



